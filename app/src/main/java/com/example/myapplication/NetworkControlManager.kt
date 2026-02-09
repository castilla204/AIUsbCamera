package com.example.myapplication

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.File
import java.io.InputStreamReader
import java.util.concurrent.TimeUnit

object NetworkControlManager {
    private const val TAG = "NetworkControlManager"

    private fun getSuPath(): String {
        val paths = arrayOf(
            "/data/adb/ksu/bin/su",
            "/data/adb/ap/bin/su",
            "/system/bin/su",
            "/system/xbin/su",
            "/su/bin/su",
            "/sbin/su"
        )
        for (path in paths) {
            if (File(path).exists()) return path
        }
        return "su"
    }

    suspend fun executeRootCommands(commands: Array<String>): Boolean = withContext(Dispatchers.IO) {
        val suPath = getSuPath()
        try {
            val process = Runtime.getRuntime().exec(suPath)
            DataOutputStream(process.outputStream).use { os ->
                for (command in commands) {
                    os.writeBytes("$command\n")
                }
                os.writeBytes("exit\n")
                os.flush()
            }
            // 12 s: am broadcast puede tardar varios segundos en ROMs cargadas.
            // No comprobamos exitValue(): muchas shells root devuelven != 0 aunque
            // los comandos se ejecutaron correctamente.
            val finished = process.waitFor(12, TimeUnit.SECONDS)
            if (!finished) {
                PersistentErrorLog.logError(
                    TAG, "Root exec TIMEOUT 12s (suPath=$suPath). Comando #1: ${commands.firstOrNull()?.take(80)}",
                )
                try { process.destroy() } catch (_: Exception) {}
            }
            finished
        } catch (e: Exception) {
            Log.e(TAG, "Root execution failed: ${e.message}")
            PersistentErrorLog.logError(
                TAG, "Root exec falló (suPath=$suPath): ${e.javaClass.simpleName}: ${e.message}. Comando #1: ${commands.firstOrNull()?.take(80)}", e,
            )
            false
        }
    }

    /**
     * Ejecuta un comando como root y devuelve su stdout (o null si falla).
     * Pensado para consultas rápidas tipo dumpsys.
     */
    suspend fun executeRootCommandWithOutput(
        command: String,
        timeoutSec: Long = 4,
    ): String? = withContext(Dispatchers.IO) {
        val suPath = getSuPath()
        try {
            val process = Runtime.getRuntime().exec(suPath)
            DataOutputStream(process.outputStream).use { os ->
                os.writeBytes("$command\n")
                os.writeBytes("exit\n")
                os.flush()
            }
            val out = BufferedReader(InputStreamReader(process.inputStream)).use { it.readText() }
            val finished = process.waitFor(timeoutSec, TimeUnit.SECONDS)
            if (!finished) {
                try { process.destroy() } catch (_: Exception) {}
                return@withContext null
            }
            out
        } catch (e: Exception) {
            Log.e(TAG, "Root output execution failed: ${e.message}")
            null
        }
    }

    /**
     * Activa o desactiva el modo avión y VERIFICA con el setting global tras los
     * comandos. Reintenta hasta 3 veces (con 300ms entre intentos) si el setting
     * no se aplicó — Samsung One UI ignora silenciosamente el primer cmd con
     * frecuencia (race con DataConnectionTracker, ringer rinse interno, etc.).
     *
     * Devuelve true SOLO si el setting confirmado coincide con [enable]. False
     * significa "los comandos no surtieron efecto" — el caller debe asumir que
     * la radio puede haber quedado en estado inverso al esperado y actuar en
     * consecuencia (logear, vibrar error, abortar el ciclo de RF-stealth).
     *
     * FIX DE FIABILIDAD: antes devolvía true en cuanto el shell completaba, sin
     * verificar que airplane_mode_on hubiera cambiado realmente. Eso permitía
     * ciclos enteros con la radio expuesta sin que nadie lo detectara. Ahora
     * se verifica el setting + se persiste el fallo en PersistentErrorLog para
     * que el usuario lo vea en el botón VER ERRORES.
     */
    suspend fun toggleAirplaneMode(enable: Boolean): Boolean {
        val state = if (enable) "1" else "0"
        val stateBool = if (enable) "true" else "false"
        val commands = arrayOf(
            "settings put global airplane_mode_on $state",
            "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state $stateBool",
            "cmd connectivity airplane-mode ${if (enable) "enable" else "disable"}"
        )
        val expected = if (enable) "1" else "0"
        repeat(3) { attempt ->
            val shellOk = executeRootCommands(commands)
            // Damos 300ms para que el setting se propague antes de leerlo.
            delay(300L)
            val current = executeRootCommandWithOutput(
                "settings get global airplane_mode_on", timeoutSec = 3,
            )?.trim()
            if (current == expected) {
                if (attempt > 0) {
                    Log.w(TAG, "toggleAirplaneMode($enable) OK tras ${attempt + 1} intentos")
                }
                return true
            }
            Log.w(TAG, "toggleAirplaneMode($enable): intento ${attempt + 1}/3 setting=$current (esperado=$expected, shell_ok=$shellOk)")
        }
        PersistentErrorLog.logError(
            TAG, "toggleAirplaneMode($enable) FALLÓ tras 3 intentos · setting global no cambió. " +
                 (if (enable) "RF-STEALTH COMPROMETIDA — la radio puede estar expuesta."
                  else "La radio puede seguir en avión cuando debería estar ON."),
        )
        return false
    }

    /**
     * Comandos shell para apagar Wi-Fi, Bluetooth, NFC y la pila de ubicación
     * (GPS/GNSS/aGPS y la red proveedora de localización), y cortar los sondeos
     * pasivos (wifi_scan_always_enabled, ble_scan_always_enabled).
     *
     * NO toca datos móviles ni el Modo Avión: la celda WWAN sigue siendo el único
     * canal autorizado. `2>/dev/null` evita que un comando no soportado en una
     * versión de Android (p.ej. `cmd location` en <11) reviente la cadena.
     *
     * Se expone como cadena única para reusarse sobre la shell root persistente
     * de HeadlessUvcService (sin spawnear un proceso "su" nuevo cada vez).
     */
    val DISABLE_NON_CELLULAR_RADIOS_CMD: String =
        "svc wifi disable;" +
        "svc bluetooth disable;" +                          // funciona en todos los Android
        "cmd bluetooth_manager disable 2>/dev/null;" +      // fallback por si acaso
        "cmd nfc disable 2>/dev/null;" +                    // Android 10+ (reemplaza svc nfc)
        "svc nfc disable 2>/dev/null;" +                    // Android <10
        "settings put global wifi_on 0;" +
        "settings put global bluetooth_on 0;" +
        "settings put global wifi_scan_always_enabled 0;" +
        "settings put global ble_scan_always_enabled 0;" +
        "cmd location set-location-enabled false 2>/dev/null;" +
        "settings put secure location_mode 0;" +
        "settings put secure location_providers_allowed -gps 2>/dev/null;" +
        "settings put secure location_providers_allowed -network 2>/dev/null;"

    /**
     * Apaga Wi-Fi, Bluetooth, NFC y ubicación (GPS/aGPS) dejando intactos los
     * datos móviles. Se usa cuando el modo de captura es "A" y el usuario ha
     * armado la protección RF (botón ACTIVAR SISTEMA). En modo C y TEST nunca
     * debe llamarse.
     */
    suspend fun disableNonCellularRadios(): Boolean {
        val commands = arrayOf(
            "svc wifi disable",
            "svc bluetooth disable",                        // funciona en todos los Android
            "cmd bluetooth_manager disable 2>/dev/null",    // fallback Android <12
            "cmd nfc disable 2>/dev/null",                  // Android 10+
            "svc nfc disable 2>/dev/null",                  // Android <10
            "settings put global wifi_on 0",
            "settings put global bluetooth_on 0",
            "settings put global wifi_scan_always_enabled 0",
            "settings put global ble_scan_always_enabled 0",
            "cmd location set-location-enabled false 2>/dev/null",
            "settings put secure location_mode 0",
            "settings put secure location_providers_allowed -gps 2>/dev/null",
            "settings put secure location_providers_allowed -network 2>/dev/null",
        )
        return executeRootCommands(commands)
    }

    /**
     * Tras desactivar el modo avión NO basta con quitarlo: si el toggle de
     * "Datos móviles" estaba en OFF (mobile_data=0) la pila WWAN no negocia un
     * contexto PDP y dumpsys mDataConnectionState se queda colgado en 0, con
     * lo que waitForConnectivity timeoutea y caemos al Modo B sin red.
     *
     * Esta función fuerza el toggle a ON y pide al ConnectivityService que
     * habilite datos. Llamar SIEMPRE justo después de avionOff() en Modo A.
     */
    suspend fun enableCellularData(): Boolean {
        val commands = arrayOf(
            "svc data enable",
            "settings put global mobile_data 1",
            "settings put global mobile_data1 1", // multi-SIM (Samsung)
        )
        return executeRootCommands(commands)
    }

    /**
     * Conmuta la SIM de DATOS por defecto a la suscripción [subId] (el valor es el
     * subscriptionId, NO el índice de slot). Comandos verificados en Samsung dual-SIM:
     * `user_preferred_data_sub` + `multi_sim_data_call` + broadcast SUB_DEFAULT_CHANGED.
     *
     * - [bounce] = false: usar con la radio APAGADA (avión ON) en la PRE-SELECCIÓN.
     *   No reinicia datos porque no hay sesión activa; al encender la radio, los datos
     *   engancharán directamente en la nueva SIM. Es lo más fiable: evita el estado
     *   "selección requerida" que aparece al conmutar en caliente en Exynos.
     * - [bounce] = true: usar con la radio ENCENDIDA para forzar que la sesión de
     *   datos migre AHORA (svc data disable→enable). Implica un reattach (~3-8 s),
     *   por eso sólo se hace cuando la cobertura cambió de verdad.
     */
    suspend fun setDataSubscription(subId: Int, bounce: Boolean): Boolean {
        if (subId < 0) return false
        val base = arrayOf(
            "settings put global user_preferred_data_sub $subId",
            "settings put global multi_sim_data_call $subId",
            "am broadcast -a android.intent.action.SUB_DEFAULT_CHANGED",
        )
        val commands = if (bounce) base + arrayOf("svc data disable", "sleep 1", "svc data enable") else base
        Log.i(TAG, "setDataSubscription(subId=$subId, bounce=$bounce)")
        return executeRootCommands(commands)
    }

    // ---------- Espera activa de conectividad ----------

    private val DATA_STATE_REGEX = Regex("""mDataConnectionState\s*=\s*(-?\d+)""")
    // Línea típica: "NetworkAgentInfo{ ... MOBILE ... CONNECTED/CONNECTED ...}" o WIFI análogo.
    private val NET_AGENT_CONNECTED_REGEX =
        Regex("""NetworkAgentInfo.*(MOBILE|WIFI).*CONNECTED""")

    /**
     * Comprueba el estado de datos móviles vía `dumpsys telephony.registry`.
     * Devuelve true sólo cuando mDataConnectionState == 2 (CONECTADO).
     */
    private suspend fun isMobileDataConnected(): Boolean {
        val out = executeRootCommandWithOutput(
            "dumpsys telephony.registry | grep mDataConnectionState"
        ) ?: return false
        // Puede haber varias líneas (multi-SIM); basta con una que sea 2.
        return DATA_STATE_REGEX.findAll(out).any { it.groupValues[1] == "2" }
    }

    /**
     * Comprueba si hay alguna red (móvil o WiFi) en estado CONNECTED a nivel de
     * ConnectivityService. Más genérico que mDataConnectionState.
     */
    private suspend fun isAnyNetworkConnected(): Boolean {
        val out = executeRootCommandWithOutput(
            "dumpsys connectivity | grep -E 'NetworkAgentInfo.*(MOBILE|WIFI).*CONNECTED'"
        ) ?: return false
        return NET_AGENT_CONNECTED_REGEX.containsMatchIn(out)
    }

    /**
     * Bloquea hasta que la radio confirme conectividad real (datos móviles o WiFi).
     * Combina las dos comprobaciones en una sola llamada su por poll para reducir
     * el overhead de arrancar procesos root.
     */
    suspend fun waitForConnectivity(
        timeoutMs: Long = 15_000L,
        pollIntervalMs: Long = 500L,
    ): Boolean {
        val t0 = System.currentTimeMillis()
        val deadline = t0 + timeoutMs
        var intentos = 0
        while (System.currentTimeMillis() < deadline) {
            intentos++
            val out = executeRootCommandWithOutput(
                "dumpsys telephony.registry | grep mDataConnectionState; dumpsys connectivity | grep -E 'NetworkAgentInfo.*(MOBILE|WIFI).*CONNECTED' | head -1"
            )
            if (out != null) {
                val mobileOk = DATA_STATE_REGEX.findAll(out).any { it.groupValues[1] == "2" }
                val netOk = NET_AGENT_CONNECTED_REGEX.containsMatchIn(out)
                if (mobileOk || netOk) {
                    Log.i(TAG, "Conectividad confirmada en intento $intentos (${System.currentTimeMillis() - t0}ms)")
                    return true
                }
            }
            delay(pollIntervalMs)
        }
        Log.w(TAG, "waitForConnectivity: timeout tras ${timeoutMs}ms ($intentos intentos)")
        PersistentErrorLog.logError(
            TAG, "waitForConnectivity TIMEOUT tras ${timeoutMs}ms ($intentos intentos). Datos móviles no negociaron contexto PDP — posible problema de SIM, cobertura o ROM",
        )
        return false
    }
}