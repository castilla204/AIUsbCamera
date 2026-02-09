package com.example.myapplication

import android.content.Context
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * Orquestador para el flujo de IMAGEN.
 *
 *  Modo A (relay con doble ciclo de modo avión):
 *    1. Desactiva avión
 *    2. POST imagen al relay (devuelve job_id en ~1-2 s)
 *    3. Activa avión
 *    4. Espera ~60 s mientras la nube procesa la imagen
 *    5. Desactiva avión
 *    6. GET resultado
 *    7. Activa avión
 *
 *    Si CUALQUIER paso falla, se ejecuta automáticamente Modo B.
 *
 *  Modo B (fallback interno, no seleccionable por el usuario):
 *    Ciclo simple - llamada DIRECTA a Gemini desde el móvil (sin relay).
 *    Si vino desde Modo A toca avión OFF -> Gemini -> avión ON. Si vino
 *    desde Modo C no toca avión.
 *
 *  Modo C (IAs directas con ciclo de avión, backup si el relay cae):
 *    1. Desactiva avión (igual que Modo A)
 *    2. Llama DIRECTAMENTE a Anthropic → OpenAI → Gemini (sin pasar por el relay)
 *    3. Activa avión
 *    Más rápido que Modo A (sin espera de 60 s) y funciona aunque el relay esté caído.
 */
class BolsilloIaOrquestador(
    private val client: BolsilloIaClient = BolsilloIaClient(),
    private val debugSinAvion: Boolean = false,
    // Necesario para leer la cobertura de las SIM (API de telefonía). Si es null
    // (p. ej. en tests), la selección automática de SIM se omite silenciosamente.
    private val appContext: Context? = null,
) {
    companion object {
        private const val TAG = "BolsilloIaOrq"

        // ─── Contador GLOBAL de ventanas RF intencionales ─────────────────────
        // Incrementa en cada avionOff, decrementa en cada avionOn (en finally,
        // garantizado aunque el toggle lance). El guardián RF del servicio
        // (HeadlessUvcService.rfStealthGuardCheck) lee este contador para
        // distinguir "avión OFF esperado" (envío/recogida en curso) de "avión
        // OFF anómalo" (setAirplaneMode silencioso, usuario lo tocó, otra app...).
        //
        // Mientras este contador sea >0, el guardián NO toca avión. Cuando vuelve
        // a 0, el guardián verifica que el setting global == 1; si no, lo
        // fuerza. Esto cierra cualquier ventana de exposición RF inesperada a
        // ≤15s garantizado, pase lo que pase aguas arriba.
        //
        // Atomic en vez de boolean: si dos orquestadores se solapan en un edge
        // case (oneShot + ciclo arrancando casi a la vez), el contador queda
        // coherente. En la práctica solo hay UNO activo, pero el contador
        // siempre es seguro frente a una falsa baja prematura.
        private val openRfWindows = java.util.concurrent.atomic.AtomicInteger(0)

        /** Timestamp de la última apertura de ventana RF. Se usa para detectar
         *  "ventana stale" — si el contador sigue >0 más de 90s después de
         *  la última apertura, asumimos que un cleanup se perdió y reseteamos
         *  para que el guardián pueda actuar. 90s es ~10% más que el peor caso
         *  realista de una ventana (polling de recogida Modo A ≈ 90s). Si el
         *  ciclo legítimo termina y abre otra ventana <90s después, el
         *  timestamp se actualiza y NO se considera stale. */
        @Volatile private var lastWindowOpenedMs: Long = 0L
        // Debe superar con margen la ventana máxima de recogida en Modo A:
        // conectividad (15s) + polling (36*2.5s=90s) + jitter/latencia.
        private const val WINDOW_STALE_MS = 135_000L

        /** True si hay AL MENOS una ventana de RF intencional abierta
         *  (alguien está entre avionOff y avionOn en este momento). El
         *  guardián RF del servicio respeta este flag — NO toca avión
         *  mientras sea true.
         *
         *  Defensa anti-stale: si el contador sigue >0 más de WINDOW_STALE_MS
         *  después de la última apertura, asumimos cleanup perdido (corutina
         *  cancelada antes del avionOn, excepción no capturada, etc.) y
         *  reseteamos. El guardián procederá a verificar el setting real. */
        @JvmStatic
        val isAnyRfWindowOpen: Boolean get() {
            if (openRfWindows.get() <= 0) return false
            val opened = lastWindowOpenedMs
            if (opened != 0L && System.currentTimeMillis() - opened > WINDOW_STALE_MS) {
                Log.w(TAG, "[RF-window] stale detectada (>${WINDOW_STALE_MS / 1000}s desde última apertura) — reseteando contador a 0")
                openRfWindows.set(0)
                lastWindowOpenedMs = 0L
                return false
            }
            return true
        }

        // tiempo mientras la API procesa en la nube (Modo A)
        const val MODO_A_ESPERA_MS = 60_000L
        // Modo C polea desde el segundo cero: damos margen total al relay
        // (HARD_TIMEOUT del servidor = 70 s) + un pequeño colchón.
        const val MODO_C_TIMEOUT_MS = 80_000L
        // Tope duro solicitado: cada fase RF (envío / recogida) no puede pasar
        // de 15s totales. Si se supera, se corta la fase y el ciclo seguirá en
        // la siguiente iteración (sin fallback adicional en la misma ventana).
        const val MAX_ENVIO_MS = 15_000L
        const val MAX_RECOGIDA_MS = 15_000L
        // En video, una ventana ligeramente mayor reduce falsos fallback por
        // relay "casi listo" sin disparar un coste RF descontrolado.
        const val MAX_RECOGIDA_VIDEO_MS = 30_000L
        // Tope máximo para que la radio confirme conectividad tras quitar avión.
        // Es solo un fallback: en la práctica salimos en cuanto mDataConnectionState=2.
        const val ESPERA_CONECTIVIDAD_MAX_MS = 15_000L
        const val ESPERA_CONECTIVIDAD_POLL_MS = 500L
        const val POLL_INTERVAL_MS = 2_500L
        // 6 intentos × 2.5s = 15s era demasiado corto para video: si la fase OCR
        // del relay tardaba >15s, el cliente se rendía ANTES de que el server
        // auto-aprobase y caía a fallback. Subido a 36 → 90s de ventana, con
        // detección de "Done" inmediata (no espera al 36 si llega antes).
        // El coste real es radio-on más larga solo cuando hace falta, gracias
        // al short-circuit en cuanto fetchResultDetailed devuelve Done.
        const val POLL_MAX_INTENTOS = 36

        /**
         * Devuelve un bloque de texto listo para concatenar a la respuesta cruda
         * con la duración exacta de cada ventana en la que el módem celular
         * estuvo emitiendo (avión OFF) más la suma total.
         * Si no hubo exposición (Modo C / TEST) devuelve cadena vacía.
         */
        fun formatearExposicion(ventanasMs: List<Long>, totalMs: Long): String {
            if (ventanasMs.isEmpty()) return ""
            val sb = StringBuilder("Exposición RF (avión OFF):")
            val labels = listOf("envío", "recogida")
            ventanasMs.forEachIndexed { i, ms ->
                val etiqueta = labels.getOrElse(i) { "ventana ${i + 1}" }
                sb.append("\n  • $etiqueta: ${formatMs(ms)}")
            }
            if (ventanasMs.size > 1) {
                sb.append("\n  • TOTAL: ${formatMs(totalMs)}")
            }
            return sb.toString()
        }

        private fun formatMs(ms: Long): String =
            if (ms >= 1000) "%.2fs".format(ms / 1000.0) else "${ms}ms"
    }

    // --- tracking de exposición de radiofrecuencia ---
    // Cada vez que se desactiva el modo avión arrancamos una ventana; al volver
    // a activarlo la cerramos. Cubre Modo A puro y los fallbacks a Modo B desde
    // Modo A; Modo C nunca toca avión y por tanto no genera ventanas.
    private var ventanaInicioMs: Long = 0L
    private val ventanasMs: MutableList<Long> = mutableListOf()
    val ventanasExposicionMs: List<Long> get() = ventanasMs.toList()
    val totalExposicionMs: Long get() = ventanasMs.sum()

    /**
     * Desactiva avión (best-effort) y fuerza datos móviles ON.
     * Si debugSinAvion=true, NO toca el avión bajo ningún concepto — defensive
     * coding para evitar que algún call path olvide consultar el flag.
     * Siempre abre la ventana de exposición y habilita datos aunque
     * toggleAirplaneMode devuelva false (timeout de shell != fallo real).
     */
    private suspend fun avionOff(): Boolean {
        if (debugSinAvion) {
            Log.i(TAG, "[avionOff] debugSinAvion=true → SALTANDO toggle avión y disableNonCellular")
            // Aún así abrimos la "ventana" para que el tracking de exposición sea coherente.
            if (ventanaInicioMs == 0L) ventanaInicioMs = System.currentTimeMillis()
            return true
        }
        // Pre-selección de SIM (Opción B): con la radio AÚN apagada, fijamos la SIM
        // de datos a la de mejor cobertura medida en la ventana anterior. Así, al
        // encender la radio justo abajo, los datos enganchan directamente en ella
        // sin reattach (≈0 ms extra). Si no hay lectura previa, es un no-op.
        SimCoverageSelector.preseleccionar(appContext)
        Log.i(TAG, "[avionOff] desactivando modo avión + habilitando datos móviles")
        // Marcar ventana RF INTENCIONAL abierta ANTES del toggle: el guardián RF
        // del servicio mira este contador para saber si una "avión OFF" es
        // esperada (envío/recogida en curso) o anómala (debe reactivar).
        // Lo subimos AQUÍ aunque toggleAirplaneMode falle — si el toggle falla,
        // la radio sigue ON y todo OK; cuando avionOn() decremente, el guardián
        // verificará el setting real. Si SÍ se desactivó, openRfWindows>0
        // impide que el guardián la reactive durante el envío.
        val opened = openRfWindows.incrementAndGet()
        lastWindowOpenedMs = System.currentTimeMillis()
        val ok = NetworkControlManager.toggleAirplaneMode(false)
        NetworkControlManager.enableCellularData()
        if (ventanaInicioMs == 0L) {
            ventanaInicioMs = System.currentTimeMillis()
        }
        Log.i(TAG, "[avionOff] toggle ok=$ok, ventanaInicioMs=$ventanaInicioMs, openRfWindows=$opened")
        return ok
    }

    /** Activa avión y cierra la ventana abierta (si la había).
     *  Si debugSinAvion=true, cierra la ventana pero NO toca el modo avión.
     *
     *  CRÍTICO PARA SEGURIDAD RF: este método debe SIEMPRE tratar de activar
     *  avión aunque alguna parte explote. Si `toggleAirplaneMode` lanza
     *  excepción, capturamos y reintentamos 1 vez con un pequeño backoff.
     *  Sin esto, el avión podría quedar OFF indefinidamente exponiendo al
     *  usuario a RF cuando debería estar cerrado. */
    private suspend fun avionOn() {
        // Cerrar ventana SIEMPRE primero (tracking de exposición RF).
        if (ventanaInicioMs > 0L) {
            ventanasMs.add(System.currentTimeMillis() - ventanaInicioMs)
            ventanaInicioMs = 0L
        }
        if (debugSinAvion) {
            Log.i(TAG, "[avionOn] debugSinAvion=true → SALTANDO toggle avión (solo cierro ventana tracking)")
            return
        }
        Log.i(TAG, "[avionOn] activando modo avión")
        // Reintento defensivo a DOS niveles:
        //   - NetworkControlManager.toggleAirplaneMode hace 3 verificaciones
        //     internas con el setting global; devuelve true SOLO si confirmado.
        //   - Aquí damos 2 vueltas más por si una excepción inesperada (OOM,
        //     shell root muerto, broadcast colgado) impide que toggle complete.
        // La seguridad RF del usuario no puede depender de un single shot ok.
        //
        // CRÍTICO: el decrement de openRfWindows VA EN FINALLY — pase lo que
        // pase aquí dentro (toggle OK, fallo, excepción, return), el contador
        // tiene que bajar para que el guardián RF del servicio pueda actuar.
        // Si quedara positivo el guardián jamás dispararía y, si el toggle
        // falló, la radio quedaría expuesta indefinidamente.
        var lastErr: Throwable? = null
        try {
            for (attempt in 1..2) {
                try {
                    if (NetworkControlManager.toggleAirplaneMode(true)) {
                        if (attempt > 1) Log.w(TAG, "[avionOn] OK tras retry $attempt")
                        return
                    }
                    // toggleAirplaneMode devolvió false: setting no verificado tras
                    // sus 3 reintentos internos. Damos UN intento más al nivel del
                    // orquestador con un pequeño backoff por si fue un glitch
                    // transitorio (samsung ringer rinse, etc.).
                    Log.w(TAG, "[avionOn] toggleAirplaneMode devolvió false en intento $attempt/2")
                    if (attempt == 1) kotlinx.coroutines.delay(500)
                } catch (t: Throwable) {
                    lastErr = t
                    Log.e(TAG, "[avionOn] toggleAirplaneMode(true) lanzó (intento $attempt/2): ${t.message}", t)
                    if (attempt == 1) kotlinx.coroutines.delay(300)
                }
            }
            // Si llegamos aquí, los 2 reintentos del orquestador fallaron (y
            // NetworkControlManager ya hizo 3 verificaciones internas + persistió
            // su propio error). El guardián RF del servicio reactivará el avión
            // en su próximo tick (≤15s) — la exposición queda acotada.
            try {
                val motivo = lastErr?.let { "excepción: ${it.message}" }
                             ?: "setting global no cambió tras 3+2 intentos"
                PersistentErrorLog.logError(
                    TAG, "avionOn() FALLÓ ($motivo). El guardián RF del servicio reactivará el avión en ≤15s — exposición acotada.",
                    lastErr as? Exception,
                )
            } catch (_: Throwable) {}
        } finally {
            // SIEMPRE decrementar. Si no llegamos aquí (caso teórico — no
            // debería pasar porque el try no tiene return fuera de los OK),
            // el contador quedaría positivo y el guardián no podría reactivar.
            val remaining = openRfWindows.decrementAndGet()
            if (remaining < 0) {
                // Defensa: si por algún bug llegamos a negativo, lo subimos a 0.
                openRfWindows.set(0)
                Log.w(TAG, "[avionOn] openRfWindows < 0 detectado — reseteado a 0 (bug en pair off/on?)")
            }
        }
    }

    /** Wrapper defensivo para disableNonCellularRadios. */
    private suspend fun disableNonCellularRadiosSafe() {
        if (debugSinAvion) {
            Log.i(TAG, "[radios] debugSinAvion=true → SALTANDO disableNonCellularRadios")
            return
        }
        NetworkControlManager.disableNonCellularRadios()
    }

    enum class Modo { A_RELAY, C_DIRECTO }

    sealed class Estado {
        object Inicio : Estado()
        object DesactivandoAvionEnvio : Estado()
        object EnviandoSolicitud : Estado()
        object ActivandoAvionEspera : Estado()
        data class Esperando(val segundosRestantes: Int) : Estado()
        object DesactivandoAvionRecogida : Estado()
        object RecogiendoResultado : Estado()
        object ActivandoAvionFin : Estado()
        data class FallbackB(val motivo: String) : Estado()
        object ModoCDirecto : Estado()
        data class Listo(val resultado: BolsilloIaClient.IaResultado, val ms: Long) : Estado()
    }

    /** True si el último resetContext() consiguió contactar al relay.
     *  El caller usa esto para NO consumir el flag pendingRelayReset si la llamada falló. */
    var lastResetContextOk: Boolean = false
        private set

    /**
     * Punto de entrada principal para imagen. Recibe los bytes JPEG ya comprimidos
     * (lo que devuelve BurstStackingProcessor.process).
     */
    suspend fun procesarImagen(
        imageJpegBytes: ByteArray,
        prompt: String,
        modo: Modo,
        system: String? = null,
        contextImageB64: String? = null,
        resetRelayContext: Boolean = false,
        esperaProcesadoMs: Long = MODO_A_ESPERA_MS,
        reviewTimeoutSeconds: Int = 90,
        onEstado: (Estado) -> Unit = {},
    ): BolsilloIaClient.IaResultado {
        val t0 = System.currentTimeMillis()
        onEstado(Estado.Inicio)

        // Validar tamaño antes de cifrar — el relay tiene límite 5 MB en base64.
        if (imageJpegBytes.size > 3_750_000) {  // 3.75 MB binario → ~5 MB en base64
            Log.w(TAG, "Imagen JPEG demasiado grande (${imageJpegBytes.size} bytes), saltando flujo nube")
            val r = BolsilloIaClient.IaResultado(
                answer = "Imagen demasiado grande (${imageJpegBytes.size / 1024} KB > 3750 KB)",
                provider = "fallback-fail",
                model = "",
                viaRelay = false,
            )
            onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
            return r
        }

        val b64 = Base64.encodeToString(imageJpegBytes, Base64.NO_WRAP)

        return when (modo) {
            Modo.A_RELAY -> ejecutarModoA(b64, null, prompt, system, contextImageB64, resetRelayContext, esperaProcesadoMs, reviewTimeoutSeconds, t0, onEstado)
            Modo.C_DIRECTO -> ejecutarModoC(b64, prompt, system, contextImageB64, t0, onEstado)
        }
    }

    /**
     * Punto de entrada alternativo para VIDEO MP4 (en lugar de imagen estática).
     *
     * El video lo procesa el RELAY en fase OCR (Qwen3-VL-Plus + Gemini 2.5 Pro)
     * y luego pasa el texto a los 4 analizadores (Claude + GPT + Gemini + DeepSeek).
     * Modo C (directo IAs sin relay) NO soporta video — degradamos a fallback-fail
     * porque ninguna IA del modo directo procesa video de forma fiable.
     *
     * Límite: 25 MB binario (≈ 33 MB base64). El relay acepta hasta 50 MB de body
     * pero damos margen para el resto del payload (prompt, system, etc.).
     */
    suspend fun procesarVideo(
        videoMp4Bytes: ByteArray,
        prompt: String,
        modo: Modo,
        system: String? = null,
        contextImageB64: String? = null,
        resetRelayContext: Boolean = false,
        esperaProcesadoMs: Long = MODO_A_ESPERA_MS,
        reviewTimeoutSeconds: Int = 120,
        onEstado: (Estado) -> Unit = {},
    ): BolsilloIaClient.IaResultado {
        val t0 = System.currentTimeMillis()
        onEstado(Estado.Inicio)

        if (videoMp4Bytes.size > 25_000_000) {
            Log.w(TAG, "Video MP4 demasiado grande (${videoMp4Bytes.size} bytes), saltando")
            val r = BolsilloIaClient.IaResultado(
                answer = "Video demasiado grande (${videoMp4Bytes.size / 1024 / 1024} MB > 25 MB)",
                provider = "fallback-fail",
                model = "",
                viaRelay = false,
            )
            onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
            return r
        }

        val b64 = Base64.encodeToString(videoMp4Bytes, Base64.NO_WRAP)

        return when (modo) {
            Modo.A_RELAY -> ejecutarModoA(null, b64, prompt, system, contextImageB64, resetRelayContext, esperaProcesadoMs, reviewTimeoutSeconds, t0, onEstado)
            Modo.C_DIRECTO -> {
                // Modo C no soporta video. Devolvemos fallback-fail explicativo.
                Log.w(TAG, "Modo C_DIRECTO no soporta video — devolviendo fallback-fail")
                val r = BolsilloIaClient.IaResultado(
                    answer = "Modo directo no soporta video",
                    provider = "fallback-fail",
                    model = "",
                    viaRelay = false,
                )
                onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
                r
            }
        }
    }

    /** Fallback explícito cuando el flujo VIDEO falla. NO degradamos a Modo B
     *  porque Modo B solo soporta texto/imagen y respondería sin ver el video,
     *  causando respuestas inventadas. Cerramos la ventana de avión limpiamente
     *  y devolvemos fallback-fail con motivo claro para que el UI lo refleje.
     */
    private suspend fun fallbackVideoFail(
        motivo: String,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        Log.w(TAG, "[Video FAIL] $motivo — NO degradando a Modo B (no soporta video)")
        // Cerrar avión si está OFF (asegurar estado limpio post-fallo)
        if (!debugSinAvion) {
            withContext(NonCancellable) { avionOn() }
        }
        val r = BolsilloIaClient.IaResultado(
            answer = "",
            provider = "fallback-fail-video",
            model = "",
            viaRelay = false,
        )
        onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
        return r
    }

    private suspend fun failFastNoFallback(
        esVideo: Boolean,
        motivo: String,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        PersistentErrorLog.logError(TAG, "[Modo A] corte por límite/fiabilidad: $motivo")
        return if (esVideo) {
            fallbackVideoFail(motivo, t0, onEstado)
        } else {
            if (!debugSinAvion) withContext(NonCancellable) { avionOn() }
            val r = BolsilloIaClient.IaResultado(
                answer = motivo,
                provider = "fallback-fail",
                model = "",
                viaRelay = false,
            )
            onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
            r
        }
    }

    // ---------- MODO A ----------

    private suspend fun ejecutarModoA(
        imageB64: String?,
        videoB64: String?,
        prompt: String,
        system: String?,
        contextImageB64: String?,
        resetRelayContext: Boolean,
        esperaProcesadoMs: Long,
        reviewTimeoutSeconds: Int,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        // GARANTÍA RF #1: snapshot del contador AL ENTRAR. En cualquier ruta de
        // salida (return normal, return por fallback, excepción), el finally de
        // abajo detecta si dejamos ventanas abiertas y fuerza avionOn por cada
        // una. Sin esto, las rutas que caen a Modo B con tocarAvion=true (default)
        // dejan el contador permanentemente positivo y el guardián RF del
        // servicio queda DESACTIVADO de por vida — un único fallo de envío
        // suficiente para perder la protección de RF para siempre.
        val baseWindows = openRfWindows.get()
        try {
            return ejecutarModoAImpl(imageB64, videoB64, prompt, system, contextImageB64,
                                      resetRelayContext, esperaProcesadoMs, reviewTimeoutSeconds,
                                      t0, onEstado)
        } finally {
            val leaked = openRfWindows.get() - baseWindows
            if (leaked > 0 && !debugSinAvion) {
                withContext(NonCancellable) {
                    Log.w(TAG, "[Modo A] cleanup forzado: $leaked ventana(s) RF sin cerrar al salir → forzando avionOn x$leaked")
                    PersistentErrorLog.logError(
                        TAG, "[Modo A] cleanup forzado de $leaked ventana(s) RF al salir (probablemente fallback a Modo B con avionOff duplicado). Reactivando avión.",
                    )
                    repeat(leaked) { avionOn() }
                }
            }
        }
    }

    private suspend fun ejecutarModoAImpl(
        imageB64: String?,
        videoB64: String?,
        prompt: String,
        system: String?,
        contextImageB64: String?,
        resetRelayContext: Boolean,
        esperaProcesadoMs: Long,
        reviewTimeoutSeconds: Int,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        val esVideo = videoB64 != null
        Log.i(TAG, "[Modo A] arrancando esVideo=$esVideo, debugSinAvion=$debugSinAvion, payload=${if (esVideo) "${videoB64!!.length} chars b64" else "${imageB64?.length ?: 0} chars b64"}")
        fun msRestantes(deadlineMs: Long): Long =
            (deadlineMs - android.os.SystemClock.elapsedRealtime()).coerceAtLeast(0L)

        // 1) desactivar avión y esperar conectividad (saltado en modo debug).
        onEstado(Estado.DesactivandoAvionEnvio)
        val envioDeadlineMs = android.os.SystemClock.elapsedRealtime() + MAX_ENVIO_MS
        if (!debugSinAvion) {
            avionOff()
            disableNonCellularRadiosSafe()
            val tConnEnvio = minOf(ESPERA_CONECTIVIDAD_MAX_MS, msRestantes(envioDeadlineMs))
            if (tConnEnvio <= 0L) {
                return failFastNoFallback(esVideo, "timeout fase ENVÍO (>$MAX_ENVIO_MS ms) antes de conectividad", t0, onEstado)
            }
            if (!NetworkControlManager.waitForConnectivity(
                    timeoutMs = tConnEnvio,
                    pollIntervalMs = ESPERA_CONECTIVIDAD_POLL_MS,
                )
            ) {
                val motivo = "timeout esperando conectividad tras quitar avión (envío)"
                return failFastNoFallback(esVideo, motivo, t0, onEstado)
            }

            // Radio ON y datos confirmados: medir cobertura de ambas SIM (gratis, la
            // radio ya está encendida) y conmutar a la mejor si la cobertura cambió
            // desde la última ventana. Si conmuta en caliente, re-esperar conectividad.
            // Sólo aquí (envío de la foto), nunca en la recogida del resultado.
            val simSwitch = SimCoverageSelector.medirYConmutarSiProcedeVerificado(appContext)
            if (simSwitch.conmuto) {
                if (!simSwitch.verificada) {
                    val motivo = "conmutación SIM no verificada (objetivo=sub${simSwitch.subObjetivo}, final=sub${simSwitch.subActualFinal})"
                    return failFastNoFallback(esVideo, motivo, t0, onEstado)
                }
                val tConnPostSwitch = minOf(ESPERA_CONECTIVIDAD_MAX_MS, msRestantes(envioDeadlineMs))
                if (tConnPostSwitch <= 0L) {
                    return failFastNoFallback(esVideo, "timeout fase ENVÍO (>$MAX_ENVIO_MS ms) tras conmutar SIM", t0, onEstado)
                }
                val conectividadTrasConmutar = NetworkControlManager.waitForConnectivity(
                    timeoutMs = tConnPostSwitch,
                    pollIntervalMs = ESPERA_CONECTIVIDAD_POLL_MS,
                )
                if (!conectividadTrasConmutar) {
                    val motivo = "timeout conectividad tras conmutar SIM (sub${simSwitch.subAnterior}→sub${simSwitch.subObjetivo})"
                    return failFastNoFallback(esVideo, motivo, t0, onEstado)
                }
            }
        }

        // 1b) Si es el inicio de un examen nuevo, borrar el diagrama en el relay
        // Timeout de 10s: 5s por servidor × 2 — si falla no importa, el relay auto-detecta
        // el diagrama en el siguiente job. Lo importante es no bloquear la ventana de radio.
        // Reportamos al caller si la llamada funcionó (lastResetContextOk) para que NO
        // consuma su flag pendingRelayReset si el reset realmente falló.
        if (resetRelayContext) {
            val tReset = minOf(10_000L, msRestantes(envioDeadlineMs))
            if (tReset <= 0L) {
                return failFastNoFallback(esVideo, "timeout fase ENVÍO (>$MAX_ENVIO_MS ms) antes de resetContext", t0, onEstado)
            }
            lastResetContextOk = try {
                withTimeout(tReset) { client.resetContext() }
            } catch (e: Exception) {
                PersistentErrorLog.logError(TAG, "resetContext falló: ${e.message}", e)
                false
            }
            Log.i(TAG, "resetContext: ok=$lastResetContextOk")
        }

        // 2) POST imagen O video (con diagrama de contexto opcional)
        onEstado(Estado.EnviandoSolicitud)
        Log.i(TAG, "[Modo A] startJob: enviando ${if (esVideo) "VIDEO" else "IMAGEN"} al relay…")
        val tUploadStart = System.currentTimeMillis()
        // Snapshot SIM justo antes del POST: la elección está cacheada de la última
        // `medirYConmutarSiProcede` (o de una ventana anterior si ya estábamos en la
        // mejor). Si nunca se ha medido (p.ej. debugSinAvion), `ultimaLectura` es null.
        val simLect = SimCoverageSelector.ultimaLectura
        val simSummary = SimCoverageSelector.ultimoResumen.takeIf { it.isNotBlank() }
        val tStartJob = msRestantes(envioDeadlineMs)
        if (tStartJob <= 0L) {
            return failFastNoFallback(esVideo, "timeout fase ENVÍO (>$MAX_ENVIO_MS ms) antes de POST /ask", t0, onEstado)
        }
        val jobId = try {
            withTimeout(tStartJob) {
                client.startJob(
                    prompt = prompt,
                    system = system,
                    imageB64 = imageB64,
                    videoB64 = videoB64,
                    // Diagrama de contexto: el relay lo adjunta como IMAGEN 1 a TODOS
                    // los analyzers/OCRs multimodales del job. Si null, no se envía
                    // (campo opcional en /ask).
                    contextImageB64 = contextImageB64,
                    reviewTimeoutSeconds = reviewTimeoutSeconds,
                    simOperator = simLect?.operador,
                    simDbm = simLect?.dbm,
                    simSlot = simLect?.slot,
                    simSummary = simSummary,
                )
            }
        } catch (e: TimeoutCancellationException) {
            null
        }
        val tUploadEnd = System.currentTimeMillis()
        val uploadMs = tUploadEnd - tUploadStart
        Log.i(TAG, "[Modo A] ⏱ envío al relay: ${uploadMs}ms")
        if (jobId == null) {
            val motivo = if (msRestantes(envioDeadlineMs) <= 0L)
                "timeout fase ENVÍO (>$MAX_ENVIO_MS ms) en POST /ask"
            else
                "relay /ask no respondió (jobId=null)"
            Log.w(TAG, "[Modo A] $motivo")
            return failFastNoFallback(esVideo, motivo, t0, onEstado)
        }
        Log.i(TAG, "[Modo A] jobId obtenido: $jobId (esperaProcesadoMs=$esperaProcesadoMs)")

        // 3) activar avión (saltado en modo debug)
        onEstado(Estado.ActivandoAvionEspera)
        if (!debugSinAvion) withContext(NonCancellable) { avionOn() }

        // 4) esperar
        var transcurrido = 0L
        while (transcurrido < esperaProcesadoMs) {
            val restante = ((esperaProcesadoMs - transcurrido) / 1000).toInt()
            onEstado(Estado.Esperando(restante))
            delay(1000L)
            transcurrido += 1000L
        }

        // 5) desactivar avión y esperar conectividad (saltado en modo debug)
        onEstado(Estado.DesactivandoAvionRecogida)
        val maxRecogidaMs = if (esVideo) MAX_RECOGIDA_VIDEO_MS else MAX_RECOGIDA_MS
        val recogidaDeadlineMs = android.os.SystemClock.elapsedRealtime() + maxRecogidaMs
        if (!debugSinAvion) {
            avionOff()
            disableNonCellularRadiosSafe()
            val tConnRecogida = minOf(ESPERA_CONECTIVIDAD_MAX_MS, msRestantes(recogidaDeadlineMs))
            if (tConnRecogida <= 0L) {
                return failFastNoFallback(esVideo, "timeout fase RECOGIDA (>$maxRecogidaMs ms) antes de conectividad", t0, onEstado)
            }
            if (!NetworkControlManager.waitForConnectivity(
                    timeoutMs = tConnRecogida,
                    pollIntervalMs = ESPERA_CONECTIVIDAD_POLL_MS,
                )
            ) {
                val motivo = "timeout esperando conectividad tras quitar avión (recogida)"
                return failFastNoFallback(esVideo, motivo, t0, onEstado)
            }
        }

        // 6) GET resultado — con reintentos para errores de RED transitorios
        //    (a diferencia de "pending del relay" que también devolvía null antes).
        onEstado(Estado.RecogiendoResultado)
        Log.i(TAG, "[Modo A] recogiendo resultado de jobId=$jobId (max ${POLL_MAX_INTENTOS} intentos)")
        val tPollStart = System.currentTimeMillis()
        var resultado: BolsilloIaClient.IaResultado? = null
        var intentos = 0
        var consecutiveNetErrs = 0
        while (intentos < POLL_MAX_INTENTOS && msRestantes(recogidaDeadlineMs) > 0L) {
            when (val o = client.fetchResultDetailed(jobId)) {
                is BolsilloIaClient.FetchOutcome.Done -> {
                    resultado = o.res
                    Log.i(TAG, "[Modo A] fetchResult intento $intentos → DONE provider=${resultado.provider} answer='${resultado.answer.take(60)}'")
                    // provider=="merged-partial" significa que el relay marcó
                    // status=error PERO mandó un merged_answer útil — el cliente
                    // lo aprovecha como éxito (fix en BolsilloIaClient para
                    // recuperar respuestas parciales que antes se descartaban).
                    if (resultado.provider != "relay") break  // éxito (incluye merged, merged-partial, etc.)
                    // provider=relay → error del propio relay SIN merged útil, caer a fallback
                    Log.w(TAG, "[Modo A] provider=relay (error del relay sin merged útil) → caer a fallback")
                    break
                }
                BolsilloIaClient.FetchOutcome.Pending -> {
                    Log.i(TAG, "[Modo A] fetchResult intento $intentos → PENDING")
                    consecutiveNetErrs = 0
                }
                BolsilloIaClient.FetchOutcome.NetworkError -> {
                    consecutiveNetErrs++
                    Log.w(TAG, "[Modo A] fetchResult intento $intentos → NETWORK ERROR #$consecutiveNetErrs")
                    // 3 errores seguidos → asumimos relay caído, ir a Modo B
                    if (consecutiveNetErrs >= 3) break
                }
            }
            val waitMs = minOf(POLL_INTERVAL_MS, msRestantes(recogidaDeadlineMs))
            if (waitMs <= 0L) break
            delay(waitMs)
            intentos++
        }
        val pollMs = System.currentTimeMillis() - tPollStart
        Log.i(TAG, "[Modo A] ⏱ recogida (polling): ${pollMs}ms en ${intentos} intentos")

        // 7) activar avión:
        //    • Si hay resultado → cerrar ventana y devolver.
        //    • Si NO hay resultado → NO cerrar avión todavía; Modo B reutiliza
        //      la ventana 2 ya abierta (evita abrir una 3ª ventana de RF).
        onEstado(Estado.ActivandoAvionFin)
        val fallbackNeeded = resultado == null || resultado.provider == "relay"
        if (!debugSinAvion && !fallbackNeeded) withContext(NonCancellable) { avionOn() }

        if (fallbackNeeded) {
            val motivo = if (msRestantes(recogidaDeadlineMs) <= 0L)
                "timeout fase RECOGIDA (>$maxRecogidaMs ms) sin resultado"
            else
                (resultado?.answer ?: "timeout esperando resultado (sin respuesta del relay)")
            Log.w(TAG, "[Modo A] fallback necesario (esVideo=$esVideo): $motivo")
            return failFastNoFallback(esVideo, motivo, t0, onEstado)
        }

        // Adjuntar timings finos al resultado (envío/recogida/total). Si el
        // resultado ya tenía algún timing (no debería en Modo A directo) se
        // preserva el original; aquí solo rellenamos los que están a 0.
        val totalMs = System.currentTimeMillis() - t0
        val resultadoFinal = resultado!!.copy(
            uploadMs = if (resultado!!.uploadMs > 0L) resultado!!.uploadMs else uploadMs,
            pollMs   = if (resultado!!.pollMs   > 0L) resultado!!.pollMs   else pollMs,
            totalMs  = if (resultado!!.totalMs  > 0L) resultado!!.totalMs  else totalMs,
        )
        Log.i(TAG, "[Modo A] ⏱ TIMINGS · envío=${uploadMs}ms · recogida=${pollMs}ms · total=${totalMs}ms")
        onEstado(Estado.Listo(resultadoFinal, totalMs))
        return resultadoFinal
    }

    // ---------- MODO B (fallback ciclo simple) ----------

    private suspend fun ejecutarModoB(
        imageB64: String?,
        prompt: String,
        system: String?,
        contextImageB64: String?,
        t0: Long,
        onEstado: (Estado) -> Unit,
        motivo: String,
        tocarAvion: Boolean = true,
    ): BolsilloIaClient.IaResultado {
        // Si debugSinAvion=true, FORZAR tocarAvion=false defensive (avionOff/avionOn
        // ya son no-op internamente, pero ahorramos el waitForConnectivity inútil).
        val realmenteTocaAvion = tocarAvion && !debugSinAvion
        Log.w(TAG, "[Modo B] FALLBACK directo (tocarAvion=$tocarAvion, debugSinAvion=$debugSinAvion, real=$realmenteTocaAvion): $motivo")
        onEstado(Estado.FallbackB(motivo))

        if (realmenteTocaAvion) {
            avionOff()
            disableNonCellularRadiosSafe()
            val conectividadOk = NetworkControlManager.waitForConnectivity(
                timeoutMs = ESPERA_CONECTIVIDAD_MAX_MS,
                pollIntervalMs = ESPERA_CONECTIVIDAD_POLL_MS,
            )
            if (!conectividadOk) {
                val fail = BolsilloIaClient.IaResultado(
                    answer = "Modo B cancelado: sin conectividad real",
                    provider = "fallback-fail",
                    model = "",
                    viaRelay = false,
                )
                PersistentErrorLog.logError(TAG, "[Modo B] abortado por falta de conectividad: $motivo")
                withContext(NonCancellable) { avionOn() }
                onEstado(Estado.Listo(fail, System.currentTimeMillis() - t0))
                return fail
            }
        }

        Log.i(TAG, "[Modo B] llamando askConFallback (imageB64=${imageB64 != null}, ctxB64=${contextImageB64 != null})")
        val r = try {
            client.askConFallback(
                prompt = prompt,
                system = system,
                webSearch = false,
                imageB64 = imageB64,
                contextImageB64 = contextImageB64,
                soloDirecto = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "[Modo B] askConFallback lanzó excepción: ${e.message}", e)
            PersistentErrorLog.logError(TAG, "Modo B askConFallback lanzó: ${e.message}", e)
            null
        }
        // Si las 3 IAs fallaron (fallback-fail), loguear para diagnóstico — esto es
        // el escenario peor caso donde el examen NO recibe respuesta y queremos saber por qué.
        if (r != null && r.provider == "fallback-fail") {
            PersistentErrorLog.logError(
                TAG, "Modo B FALLBACK-FAIL (3 IAs directas fallaron). Intentos: ${r.intentos.joinToString(" | ")}",
            )
        }

        if (realmenteTocaAvion) {
            withContext(NonCancellable) { avionOn() }
        }

        val final = r ?: BolsilloIaClient.IaResultado(
            answer = "Modo B también falló: $motivo",
            provider = "fallback-fail",
            model = "",
            viaRelay = false,
        )
        Log.i(TAG, "[Modo B] resultado provider=${final.provider}, answer='${final.answer.take(80)}'")
        onEstado(Estado.Listo(final, System.currentTimeMillis() - t0))
        return final
    }

    // ---------- MODO C (IAs directas con ciclo de avión, backup si relay cae) ----------

    private suspend fun ejecutarModoC(
        imageB64: String,
        prompt: String,
        system: String?,
        contextImageB64: String?,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        val baseWindows = openRfWindows.get()
        try {
            return ejecutarModoCImpl(imageB64, prompt, system, contextImageB64, t0, onEstado)
        } finally {
            val leaked = openRfWindows.get() - baseWindows
            if (leaked > 0 && !debugSinAvion) {
                withContext(NonCancellable) {
                    Log.w(TAG, "[Modo C] cleanup forzado: $leaked ventana(s) RF sin cerrar al salir → forzando avionOn x$leaked")
                    PersistentErrorLog.logError(
                        TAG, "[Modo C] cleanup forzado de $leaked ventana(s) RF al salir. Reactivando avión.",
                    )
                    repeat(leaked) { avionOn() }
                }
            }
        }
    }

    private suspend fun ejecutarModoCImpl(
        imageB64: String,
        prompt: String,
        system: String?,
        contextImageB64: String?,
        t0: Long,
        onEstado: (Estado) -> Unit,
    ): BolsilloIaClient.IaResultado {
        onEstado(Estado.ModoCDirecto)

        // 1) avión OFF + esperar conectividad (saltado en modo debug).
        onEstado(Estado.DesactivandoAvionEnvio)
        if (!debugSinAvion) {
            avionOff()
            disableNonCellularRadiosSafe()
            if (!NetworkControlManager.waitForConnectivity(
                    timeoutMs = ESPERA_CONECTIVIDAD_MAX_MS,
                    pollIntervalMs = ESPERA_CONECTIVIDAD_POLL_MS,
                )
            ) {
                PersistentErrorLog.logError(TAG, "Modo C: timeout esperando conectividad tras quitar avión")
                withContext(NonCancellable) { avionOn() }
                val r = BolsilloIaClient.IaResultado(
                    answer = "Timeout conectividad (Modo C)",
                    provider = "fallback-fail", model = "", viaRelay = false,
                )
                onEstado(Estado.Listo(r, System.currentTimeMillis() - t0))
                return r
            }
        }

        // 2) Llamada DIRECTA: Anthropic → OpenAI → Gemini (sin relay)
        onEstado(Estado.EnviandoSolicitud)
        val resultado = try {
            withTimeout(MODO_C_TIMEOUT_MS) {
                client.askConFallback(
                    prompt = prompt,
                    system = system,
                    webSearch = false,
                    imageB64 = imageB64,
                    contextImageB64 = contextImageB64,
                    soloDirecto = true,
                )
            }
        } catch (e: TimeoutCancellationException) {
            PersistentErrorLog.logError(TAG, "Modo C timeout global (${MODO_C_TIMEOUT_MS}ms) en IAs directas", e)
            BolsilloIaClient.IaResultado(
                answer = "Timeout IAs directas (Modo C)",
                provider = "fallback-fail", model = "", viaRelay = false,
            )
        } catch (e: Exception) {
            PersistentErrorLog.logError(TAG, "Modo C askConFallback lanzó: ${e.message}", e)
            BolsilloIaClient.IaResultado(
                answer = "Error IAs directas: ${e.message}",
                provider = "fallback-fail", model = "", viaRelay = false,
            )
        }
        // Log si las 3 IAs directas fallaron (peor caso para el usuario)
        if (resultado.provider == "fallback-fail") {
            PersistentErrorLog.logError(
                TAG, "Modo C FALLBACK-FAIL. Intentos: ${resultado.intentos.joinToString(" | ")}",
            )
        }

        // 3) avión ON — siempre, igual que Modo A (saltado en modo debug)
        if (!debugSinAvion) withContext(NonCancellable) { avionOn() }

        onEstado(Estado.Listo(resultado, System.currentTimeMillis() - t0))
        return resultado
    }
}