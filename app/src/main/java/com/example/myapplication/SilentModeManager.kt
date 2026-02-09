package com.example.myapplication

import android.content.Context
import android.content.SharedPreferences
import android.media.AudioManager
import android.provider.Settings
import android.util.Log
import java.io.File

/**
 * Silenciador del sistema para Samsung Galaxy S10e (One UI / Android 12+).
 *
 * FILOSOFÍA — SILENCIO SIN DnD:
 *
 *   NO usamos `zen_mode` (DnD). Total Silence (`zen_mode=2`) en One UI filtra
 *   también la vibración de la PROPIA app — no existe whitelist real. El "bypass
 *   DnD" via `cmd notification allow_dnd $pkg` aplica solo a canales de
 *   notificación bajo `zen_mode=1` (Priority Only), no a `vibrator.vibrate(...)`
 *   directo. Por eso ahora forzamos `zen_mode=0` explícitamente en cada silencio:
 *   sobreescribe cualquier estado heredado de instalaciones previas que dejaran
 *   zen_mode=2 y garantiza que el VibratorService no esté restringido por DnD.
 *
 *   El silencio se consigue con:
 *     - `mode_ringer=0` (perfil silencio del SO — preserva vibración propia)
 *     - `cmd audio set-stream-volume X 0` + `set-stream-mute X true` por stream
 *     - `cmd audio set-master-mute true`
 *     - `all_sound_off=1`, `sound_effects_enabled=0`
 *
 *   Estos comandos NO afectan al motor de vibración, así que `vibrator.vibrate(...)`
 *   sigue funcionando para nosotros. Como red secundaria, `vibrateViaSysfs`
 *   escribe directo a los nodos del kernel.
 *
 * FLUJO:
 *   1. applySilence (al pulsar ACTIVAR SISTEMA): backup + comandos de silencio.
 *   2. reinforceSilence (bucle inmortal, cada 60s): re-aplica el mismo silencio.
 *   3. restoreAudio (SOLO al pulsar DETENER SISTEMA): restaura el estado backupeado.
 *
 *   En cualquier otro evento (crash, USB disconnect, force-stop, reinicio) NO se
 *   restaura nada — el silencio es STICKY por diseño (caso de uso espía).
 *
 * Vibración por sysfs: [vibrateViaSysfs] escribe directamente en los nodos del
 * kernel como red secundaria, llamada desde `vibrate(ms)` del servicio.
 */
object SilentModeManager {
    private const val TAG = "SilentMode"

    // Claves de SharedPreferences donde guardamos el estado original del usuario
    private const val PREF_BACKUP_DONE = "audio_backup_done"
    private const val PREF_BAK_MODE_RINGER = "audio_bak_mode_ringer"
    private const val PREF_BAK_ALL_SOUND_OFF = "audio_bak_all_sound_off"
    private const val PREF_BAK_SOUND_EFFECTS = "audio_bak_sound_effects"
    private const val PREF_BAK_LOW_POWER = "audio_bak_low_power"
    private const val PREF_BAK_ZEN_MODE = "audio_bak_zen_mode"
    private const val PREF_BAK_STREAM_PREFIX = "audio_bak_stream_"   // + streamId

    // Streams que silenciamos a 0. Si añades aquí, añade también en backup/restore.
    // 0=VOICE, 1=SYSTEM, 2=RING, 3=MUSIC, 4=ALARM, 5=NOTIFICATION, 8=DTMF
    private val SILENCED_STREAMS = intArrayOf(0, 1, 2, 3, 4, 5, 8)

    /**
     * Apps de sistema One UI que solíamos forzar a VIBRATE=ignore en versiones previas.
     * Se mantiene SOLO para limpiar en restoreAudio el estado de instalaciones antiguas
     * (que dejaron estas apps con VIBRATE=ignore persistente). En applySilence ya NO se
     * tocan — zen_mode=2 filtra sus vibraciones igualmente.
     */
    private val LEGACY_SYSTEM_VIBRATORS_TO_RESTORE = listOf(
        "com.samsung.android.app.telephonyui",
        "com.samsung.android.dialer",
        "com.samsung.android.incallui",
        "com.android.server.telecom",
        "com.samsung.android.messaging",
        "com.samsung.android.calendar",
        "com.samsung.android.app.clockpackage",
        "com.android.systemui",
        "com.samsung.android.bixby.agent",
        "com.google.android.googlequicksearchbox",
        "com.samsung.knox.securefolder",
        // Apps de ajustes (Samsung S10e: com.android.settings; algunos
        // modelos también tienen com.samsung.android.settings o
        // com.samsung.android.app.settings). El slider de intensidad de
        // vibración hace `vibrator.vibrate()` desde estos paquetes como
        // preview — bloqueándoles VIBRATE evita que vibre al arrastrar.
        "com.android.settings",
        "com.samsung.android.settings",
        "com.samsung.android.app.settings",
        // Launcher / one-handed gestures (haptic al swipe up, etc.)
        "com.sec.android.app.launcher",
        // IME (teclado Samsung) — long-press haptic
        "com.sec.android.inputmethod",
        "com.samsung.android.honeyboard",
    )

    /**
     * Silencia el dispositivo manteniendo la vibración de NUESTRA app disponible.
     *
     * Aplica DOS lotes:
     *   1. Settings globales + master-mute + bloqueo de notifs de SystemUI (cheap,
     *      idempotente; se reaplica en reinforceSilence cada 60s).
     *   2. AppOps por paquete (caro, ~5-10s con pm list): VIBRATE ignore,
     *      PLAY_AUDIO deny, TAKE_AUDIO_FOCUS ignore para TODAS las apps de
     *      terceros excepto la nuestra. Esto cierra las vías que master-mute
     *      no cierra: apps con AudioAttributes especiales, IMPORTANCE_HIGH
     *      sound URIs, y vibraciones de WhatsApp/Telegram/etc. Solo se ejecuta
     *      en este apply inicial — el bucle reinforce NO lo repite porque
     *      iterar pm list cada 60s sería un drain. Apps recién instaladas
     *      mientras la app está activa quedarían fuera; si se vuelve a pulsar
     *      ACTIVAR SISTEMA se aplica de nuevo.
     */
    fun applySilence(
        ctx: Context,
        runRoot: (cmd: String, tag: String) -> Unit,
        alarmVibrationIntensity: Int = 3,
    ) {
        val prefs = ctx.getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
        if (!prefs.getBoolean(PREF_BACKUP_DONE, false)) {
            backupCurrentState(ctx, prefs)
            Log.i(TAG, "Backup audio del usuario hecho")
        }
        runRoot(buildSilenceCmds(ctx.packageName, alarmVibrationIntensity), "silence-on")
        runRoot(buildAppOpsRestrictCmds(ctx.packageName), "silence-appops-restrict")
        Log.i(TAG, "Silencio aplicado (alarm_vib=$alarmVibrationIntensity, zen=0, ringer=0, streams=0, master-mute, appops 3rd-party, vibrate-self)")
    }

    /**
     * Reaplica silencio periódicamente desde el bucle inmortal. Misma cadena que
     * applySilence — ya no usamos versión "ligera" porque la cadena entera son
     * ~15 comandos cheap (sin loops sobre `pm list packages`).
     *
     * Recibe ctx para poder reaplicar los AppOps VIBRATE/PLAY_AUDIO=allow de
     * nuestro propio paquete. Sin esto, si Samsung resetea alarm_vibration_intensity
     * o algún proceso pone VIBRATE en ignore para nosotros, dejaríamos de vibrar
     * sin enterarnos hasta el próximo DETENER+ACTIVAR.
     *
     * Recibe `alarmVibrationIntensity` (1..3 escala Samsung) para que el bucle
     * inmortal NO sobreescriba la intensidad elegida por el usuario en el slider.
     * Antes esto venía hardcoded a 3 → cada 60s pisaba el slider, por eso las
     * vibraciones se sentían siempre a máxima fuerza.
     */
    fun reinforceSilence(
        ctx: Context,
        runRoot: (cmd: String, tag: String) -> Unit,
        alarmVibrationIntensity: Int = 3,
    ) {
        runRoot(buildSilenceCmds(ctx.packageName, alarmVibrationIntensity), "silence-reinforce")
    }

    /**
     * Cadena MÍNIMA para reaplicar master-mute rápidamente (cada 15s, no 60s).
     *
     * En Samsung One UI 4 (Android 12) el `set-master-mute` se desactiva cuando
     * el usuario toca el volumen físico, conecta auriculares, o el sistema hace
     * "ringer rinse". Si solo lo reaplicamos cada 60s (bucle inmortal), hay una
     * ventana de hasta 60s donde sí sale audio. Esta cadena ligera (sólo 4
     * comandos, sin `pm list`) se puede ejecutar cada 15s sin coste de batería
     * apreciable.
     *
     * NO incluye los settings de intensidad de vibración (esos se reaplican
     * cada 60s en reinforceSilence) ni los AppOps de 3rd parties (one-shot
     * en applySilence). Sólo bloquea el audio.
     */
    fun fastReinforceMute(runRoot: (cmd: String, tag: String) -> Unit) {
        val cmds = buildString {
            append("cmd audio set-master-mute true 2>/dev/null;")
            append("service call audio 16 i32 1 i32 0 >/dev/null 2>&1;")
            append("settings put global mode_ringer 0;")
            // STREAM_MUSIC (3) específicamente: el más vulnerable a reset por
            // user-input. STREAM_NOTIFICATION (5) por si Samsung lo re-abre.
            append("cmd audio set-stream-volume 3 0 2>/dev/null;")
            append("cmd audio set-stream-mute 3 true 2>/dev/null;")
            append("cmd audio set-stream-volume 5 0 2>/dev/null;")
            append("cmd audio set-stream-mute 5 true 2>/dev/null;")
            // Haptic del sistema: si Samsung lo reactiva entre ciclos largos
            // (cada 60s), el botón atrás y los switches volverían a vibrar.
            // Lo bajamos también aquí (cada 15s) para cerrar esa ventana.
            // NO afecta a vibrator.vibrate() de nuestra app (USAGE_ALARM).
            append("settings put system haptic_feedback_enabled 0;")
        }
        runRoot(cmds, "silence-fast")
    }

    /**
     * Restaura el estado de audio que el usuario tenía antes de silenciar.
     * Solo se ejecuta desde el botón DETENER SISTEMA (no en crash, no en onDestroy,
     * no en USB disconnect — el silencio es sticky por diseño).
     */
    fun restoreAudio(ctx: Context, runRoot: (cmd: String, tag: String) -> Boolean) {
        val prefs = ctx.getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
        if (!prefs.getBoolean(PREF_BACKUP_DONE, false)) {
            Log.i(TAG, "No había backup; nada que restaurar")
            return
        }
        val ringer = prefs.getString(PREF_BAK_MODE_RINGER, "2") ?: "2"
        val allOff = prefs.getString(PREF_BAK_ALL_SOUND_OFF, "0") ?: "0"
        val effects = prefs.getString(PREF_BAK_SOUND_EFFECTS, "1") ?: "1"
        val lowPower = prefs.getString(PREF_BAK_LOW_POWER, "0") ?: "0"
        val zenMode = prefs.getString(PREF_BAK_ZEN_MODE, "0") ?: "0"
        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as? AudioManager

        val cmds = buildString {
            append("settings put global mode_ringer $ringer;")
            append("settings put system all_sound_off $allOff;")
            append("settings put system sound_effects_enabled $effects;")
            append("settings put global low_power $lowPower;")
            append("settings put global zen_mode $zenMode;")
            append("cmd audio set-master-mute false 2>/dev/null;")
            append("settings delete global zen_mode_config_etag 2>/dev/null;")

            // Restaurar volúmenes + quitar stream-mute por si quedó activo.
            for (s in SILENCED_STREAMS) {
                append("cmd audio set-stream-mute $s false 2>/dev/null;")
                val saved = prefs.getInt(PREF_BAK_STREAM_PREFIX + s, -1)
                val target = if (saved >= 0) saved else {
                    try { (am?.getStreamMaxVolume(s) ?: 6) / 2 } catch (_: Exception) { 3 }
                }
                append("cmd audio set-stream-volume $s $target 2>/dev/null;")
            }
        }
        val ok1 = runRoot(cmds, "silence-restore")
        // Cleanup one-shot del estado heredado de versiones previas (cuando sí
        // iterábamos pm list packages -3). En instalaciones nuevas esto es no-op.
        val ok2 = runRoot(buildLegacyRestoreVibrateCmds(), "silence-restore-vibrate-legacy")
        // Restaurar VIBRATE/PLAY_AUDIO/TAKE_AUDIO_FOCUS + POST_NOTIFICATION
        // que applySilence aplicó por paquete a terceros y SystemUI.
        val ok3 = runRoot(buildAppOpsRestoreCmds(), "silence-restore-appops")

        if (ok1 && ok2 && ok3) {
            prefs.edit().putBoolean(PREF_BACKUP_DONE, false).apply()
            Log.i(TAG, "Audio restaurado (ringer=$ringer, zen_mode=$zenMode, appops default)")
        } else {
            Log.w(TAG, "restoreAudio: runRoot falló (ok1=$ok1, ok2=$ok2, ok3=$ok3); backup conservado para reintento")
        }
    }

    /**
     * Vibración directa por sysfs (kernel), saltándose VibratorManagerService.
     * Útil como ruta blindada si el sistema bloquea por DnD/Doze.
     */
    fun vibrateViaSysfs(durationMs: Long, runRoot: (cmd: String, tag: String) -> Unit): Boolean {
        val candidatos = listOf(
            "/sys/class/timed_output/vibrator/enable",
            "/sys/class/leds/vibrator/duration",
            "/sys/class/leds/vibrator/state",
        )
        var alguno = false
        for (path in candidatos) {
            if (File(path).exists()) {
                runRoot("echo $durationMs > $path", "sysfs-vib")
                alguno = true
                break
            }
        }
        if (!alguno) {
            runRoot(
                "echo $durationMs > /sys/class/timed_output/vibrator/enable 2>/dev/null",
                "sysfs-vib-blind",
            )
        }
        return alguno
    }

    // ---------- internos ----------

    /** Lee el estado actual con la API de Settings (no requiere root) y lo guarda. */
    private fun backupCurrentState(ctx: Context, prefs: SharedPreferences) {
        val cr = ctx.contentResolver
        val ringer = readGlobal(cr, "mode_ringer", "2")
        val allOff = readSystem(cr, "all_sound_off", "0")
        val effects = readSystem(cr, "sound_effects_enabled", "1")
        val lowPower = readGlobal(cr, "low_power", "0")
        val zenMode = readGlobal(cr, "zen_mode", "0")
        val editor = prefs.edit()
            .putString(PREF_BAK_MODE_RINGER, ringer)
            .putString(PREF_BAK_ALL_SOUND_OFF, allOff)
            .putString(PREF_BAK_SOUND_EFFECTS, effects)
            .putString(PREF_BAK_LOW_POWER, lowPower)
            .putString(PREF_BAK_ZEN_MODE, zenMode)
            .putBoolean(PREF_BACKUP_DONE, true)

        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        if (am != null) {
            for (s in SILENCED_STREAMS) {
                val vol = try { am.getStreamVolume(s) } catch (_: Exception) { -1 }
                editor.putInt(PREF_BAK_STREAM_PREFIX + s, vol)
            }
        }
        editor.apply()
    }

    private fun readGlobal(cr: android.content.ContentResolver, key: String, def: String): String {
        return try {
            Settings.Global.getString(cr, key) ?: def
        } catch (e: Exception) { def }
    }

    private fun readSystem(cr: android.content.ContentResolver, key: String, def: String): String {
        return try {
            Settings.System.getString(cr, key) ?: def
        } catch (e: Exception) { def }
    }

    /**
     * Cadena de silencio. Aplicada en applySilence inicial Y en reinforce periódico.
     *
     * Capas (sin DnD, para no bloquear nuestra propia vibración):
     *   1. zen_mode=0 — Forzado a OFF. Sobreescribe cualquier zen_mode=2 que pudiera
     *      haber quedado de versiones previas y garantiza que el VibratorService
     *      no esté restringido por DnD.
     *   2. mode_ringer=0 — Perfil silencio del SO. Vibración propia preservada.
     *   3. Settings de silencio (all_sound_off, sound_effects_enabled).
     *   4. Volúmenes a 0 + stream-mute + master-mute — silencia el motor de audio
     *      sin tocar el de vibración. (Stream-mute es notablemente unreliable en
     *      Samsung — Galaxy S8 reports — pero el comando es cheap y 2>/dev/null
     *      lo ignora si falla.)
     */
    private fun buildSilenceCmds(myPackage: String, alarmVibrationIntensity: Int = 3): String = buildString {
        // Clamp defensivo: la escala Samsung es 1..3. Si llega algo fuera de rango,
        // mantenemos el comportamiento histórico (3 = MAX).
        val vibLevel = alarmVibrationIntensity.coerceIn(1, 3)
        // ── DnD OFF: imprescindible para que nuestra propia vibración funcione ─
        // Si una instalación previa dejó zen_mode=2 persistido, lo forzamos a 0
        // aquí en cada apply/reinforce.
        append("settings put global zen_mode 0;")
        append("settings delete global zen_mode_config_etag 2>/dev/null;")

        // ── REFUERZO de vibración PROPIA — VA PRIMERO ─────────────────────
        // Nuestras vibraciones usan VibrationEffect + AudioAttributes(USAGE_ALARM),
        // gobernado por `alarm_vibration_intensity` en Samsung One UI. Si está a
        // 0, no vibran aunque el AppOp VIBRATE esté en allow.
        //
        // Importante:
        // - `alarm_vibration_intensity = 3`  → MAX, lo único que necesita
        //   nuestro vibrator.vibrate(effect, USAGE_ALARM).
        // - `haptic_feedback_enabled = 0`    → DESACTIVA el haptic del sistema
        //   (botón atrás, switches en Ajustes, gestos del launcher). NO afecta
        //   a vibrator.vibrate() con un VibrationEffect explícito como hacemos
        //   nosotros — esa API tiene su propio canal por AudioAttributes.
        // - `haptic_feedback_intensity = 0`  → defensa adicional, intensidad 0
        //   por si algún componente respeta intensidad pero no enabled.
        // - `notification_vibration_intensity = 0` / `ring_vibration_intensity = 0`
        //   → elimina la vibración del slider "preview" en Ajustes y la
        //   vibración de llamadas/notificaciones entrantes. Cuando los valores
        //   son 0 (no > 0), Samsung NO dispara auto-corrección de mode_ringer
        //   porque es consistente con silent mode.
        // FIX: usar la intensidad elegida por el usuario en el slider en vez
        // del hardcoded 3. Sin esto el reinforce loop (cada 60s) pisaba el
        // slider y la vibración salía siempre a máxima fuerza.
        append("settings put system alarm_vibration_intensity $vibLevel;")
        append("settings put system haptic_feedback_enabled 0;")
        append("settings put system haptic_feedback_intensity 0;")
        append("settings put system notification_vibration_intensity 0;")
        append("settings put system ring_vibration_intensity 0;")
        // Por si alguna versión previa dejó VIBRATE en ignore para nosotros.
        append("cmd appops set $myPackage VIBRATE allow 2>/dev/null;")
        append("cmd appops set $myPackage PLAY_AUDIO allow 2>/dev/null;")
        append("cmd appops set $myPackage TAKE_AUDIO_FOCUS allow 2>/dev/null;")

        // ── REFUERZO de bloqueos de notificación del sistema ──────────────
        // Sin esto, Samsung One UI re-habilita silenciosamente las alertas
        // críticas (batería 15%/5%, USB, captura) tras unos minutos.
        append("cmd appops set android POST_NOTIFICATION ignore 2>/dev/null;")
        append("cmd appops set com.android.systemui POST_NOTIFICATION ignore 2>/dev/null;")
        append("cmd appops set com.android.systemui TOAST_WINDOW deny 2>/dev/null;")
        append("settings put global heads_up_notifications_enabled 0;")

        // ── SILENCIO — VA AL FINAL ────────────────────────────────────────
        // Importante: el orden importa. Aplicar mode_ringer=0 + master-mute
        // como los ÚLTIMOS comandos garantiza que si Samsung procesa cambios
        // fuera de orden, el estado final sea "silencio total".
        append("settings put system sound_effects_enabled 0;")
        append("settings put system all_sound_off 1;")
        append("settings put global low_power 0;")
        // Streams a 0 + mute por stream. En Samsung One UI 4 (Android 12),
        // `cmd audio set-stream-mute` es notoriamente unreliable — Samsung
        // re-habilita el stream cuando el usuario toca el volumen físico o
        // tras "ringer rinse" interno. Lo metemos aquí defensivamente pero
        // la línea CRÍTICA es el binder directo del master-mute más abajo.
        for (s in SILENCED_STREAMS) {
            append("cmd audio set-stream-volume $s 0 2>/dev/null;")
            append("cmd audio set-stream-mute $s true 2>/dev/null;")
        }
        // mode_ringer=0 (Silent) + master-mute — el último estado es silencio.
        append("settings put global mode_ringer 0;")
        // Master mute por DOS vías para máxima robustez en Samsung One UI 4:
        //  (1) Wrapper `cmd audio set-master-mute true` — interface oficial.
        //  (2) Binder directo: service call audio 16 i32 1 i32 0
        //      = IAudioService.setMasterMute(boolean true, int flags=0)
        //      `16` es el transaction code en Android 12 (AOSP r34).
        //      Esta vía pasa por encima de cualquier wrapper Samsung que
        //      pueda estar interceptando `cmd audio`.
        append("cmd audio set-master-mute true 2>/dev/null;")
        append("service call audio 16 i32 1 i32 0 >/dev/null 2>&1;")
    }

    /**
     * Cleanup one-shot ejecutado solo en restoreAudio. Vuelve VIBRATE a default
     * para las apps que versiones previas de SilentModeManager pudieron haber dejado
     * en VIBRATE=ignore. En instalaciones nuevas no hay apps así, es no-op funcional.
     */
    private fun buildLegacyRestoreVibrateCmds(): String = buildString {
        append("for pkg in \$(pm list packages -3 | cut -d: -f2); do")
        append(" cmd appops set \$pkg VIBRATE default 2>/dev/null; ")
        append("done;")
        for (sys in LEGACY_SYSTEM_VIBRATORS_TO_RESTORE) {
            append("cmd appops set $sys VIBRATE default 2>/dev/null;")
        }
    }

    /**
     * Restringe por AppOps las 3 vías por las que apps de terceros pueden
     * interrumpirnos pese a master-mute + zen_mode=0:
     *   - VIBRATE ignore        → no pueden invocar el VibratorService
     *   - PLAY_AUDIO deny       → AudioTrack/MediaPlayer rechazado
     *   - TAKE_AUDIO_FOCUS      → no nos roban el foco
     *
     * Excepciones (no se tocan):
     *   - Nuestro propio paquete (mantiene vibración + audio)
     *   - Apps de sistema críticas (Knox, biometrics, IME, keyguard) — tocarlas
     *     puede congelar la UI de One UI o impedir desbloquear el teléfono.
     *
     * Se ejecuta como un único lote shell: el for-loop + cut hace una sola
     * traversal del listado de paquetes (3rd-party = pm -3) en lugar de N
     * invocaciones de cmd appops.
     */
    private fun buildAppOpsRestrictCmds(myPackage: String): String = buildString {
        // -3 = solo terceros (no incluye system); rápido aunque haya 200 apps.
        // Bloqueamos 5 AppOps por paquete:
        //   - VIBRATE                → no pueden invocar VibratorService
        //   - PLAY_AUDIO             → AudioTrack/MediaPlayer rechazado
        //   - TAKE_AUDIO_FOCUS       → no nos roban el foco
        //   - AUDIO_MEDIA_VOLUME     → no pueden subir/desmutar STREAM_MUSIC
        //   - AUDIO_MASTER_VOLUME    → no pueden subir/desmutar master
        // Las últimas dos son las que faltaban: en Samsung One UI 4, apps de
        // música usan `adjustStreamVolume(STREAM_MUSIC, ADJUST_RAISE)` cuando
        // el usuario toca play, lo que automáticamente DESMUTEA STREAM_MUSIC
        // (Samsung lo trata como "user-initiated"). Con AUDIO_MEDIA_VOLUME
        // ignore, esa llamada se descarta silenciosamente.
        append("for pkg in \$(pm list packages -3 | cut -d: -f2); do")
        append("  if [ \"\$pkg\" != \"$myPackage\" ]; then")
        append("    cmd appops set \$pkg VIBRATE ignore 2>/dev/null;")
        append("    cmd appops set \$pkg PLAY_AUDIO deny 2>/dev/null;")
        append("    cmd appops set \$pkg TAKE_AUDIO_FOCUS ignore 2>/dev/null;")
        append("    cmd appops set \$pkg AUDIO_MEDIA_VOLUME ignore 2>/dev/null;")
        append("    cmd appops set \$pkg AUDIO_MASTER_VOLUME ignore 2>/dev/null;")
        append("  fi;")
        append("done;")
        // Apps de sistema Samsung que históricamente vibran / suenan pese a
        // mute (telefonía, dialer, mensajería, calendario, reloj, Bixby).
        // No están en pm -3 (son /system) así que las tratamos a mano.
        for (sys in LEGACY_SYSTEM_VIBRATORS_TO_RESTORE) {
            append("cmd appops set $sys VIBRATE ignore 2>/dev/null;")
            append("cmd appops set $sys PLAY_AUDIO deny 2>/dev/null;")
            append("cmd appops set $sys TAKE_AUDIO_FOCUS ignore 2>/dev/null;")
            append("cmd appops set $sys AUDIO_MEDIA_VOLUME ignore 2>/dev/null;")
            append("cmd appops set $sys AUDIO_MASTER_VOLUME ignore 2>/dev/null;")
        }
    }

    /**
     * Inverso de buildAppOpsRestrictCmds: devuelve VIBRATE/PLAY_AUDIO/
     * TAKE_AUDIO_FOCUS a "default" para todas las apps de terceros y para
     * las apps de sistema Samsung que tocamos. Solo se invoca desde
     * restoreAudio (DETENER SISTEMA).
     */
    private fun buildAppOpsRestoreCmds(): String = buildString {
        append("for pkg in \$(pm list packages -3 | cut -d: -f2); do")
        append("  cmd appops set \$pkg VIBRATE default 2>/dev/null;")
        append("  cmd appops set \$pkg PLAY_AUDIO default 2>/dev/null;")
        append("  cmd appops set \$pkg TAKE_AUDIO_FOCUS default 2>/dev/null;")
        append("  cmd appops set \$pkg AUDIO_MEDIA_VOLUME default 2>/dev/null;")
        append("  cmd appops set \$pkg AUDIO_MASTER_VOLUME default 2>/dev/null;")
        append("done;")
        for (sys in LEGACY_SYSTEM_VIBRATORS_TO_RESTORE) {
            append("cmd appops set $sys VIBRATE default 2>/dev/null;")
            append("cmd appops set $sys PLAY_AUDIO default 2>/dev/null;")
            append("cmd appops set $sys TAKE_AUDIO_FOCUS default 2>/dev/null;")
            append("cmd appops set $sys AUDIO_MEDIA_VOLUME default 2>/dev/null;")
            append("cmd appops set $sys AUDIO_MASTER_VOLUME default 2>/dev/null;")
        }
        // POST_NOTIFICATION y heads_up: devolverlos a allow / 1 para que el
        // teléfono vuelva a comportarse normal tras DETENER SISTEMA.
        append("cmd appops set android POST_NOTIFICATION allow 2>/dev/null;")
        append("cmd appops set com.android.systemui POST_NOTIFICATION allow 2>/dev/null;")
        append("cmd appops set com.android.systemui TOAST_WINDOW allow 2>/dev/null;")
        append("settings put global heads_up_notifications_enabled 1;")
        // Restaurar settings de vibración que pusimos a 0 mientras silenciábamos
        // (haptic del botón atrás, switches, sliders de Ajustes, llamadas, etc.).
        // Valor 3 = MAX (default de Samsung). 1 = enabled.
        append("settings put system haptic_feedback_enabled 1;")
        append("settings put system haptic_feedback_intensity 3;")
        append("settings put system notification_vibration_intensity 3;")
        append("settings put system ring_vibration_intensity 3;")
        // Desactivar master-mute (la doble vía igual que al activar)
        append("cmd audio set-master-mute false 2>/dev/null;")
        append("service call audio 16 i32 0 i32 0 >/dev/null 2>&1;")
    }
}
