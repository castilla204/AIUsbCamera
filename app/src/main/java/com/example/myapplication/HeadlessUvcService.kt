package com.example.myapplication

import android.app.*
import android.content.*
import android.graphics.Bitmap
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.media.MediaScannerConnection
import android.os.*
import android.provider.Settings
import android.util.Base64
import android.util.Log
import android.view.Surface
import androidx.core.app.NotificationCompat
import com.serenegiant.usb.IFrameCallback
import com.serenegiant.usb.USBMonitor
import com.serenegiant.usb.UVCCamera
import com.serenegiant.usb.Size
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class HeadlessUvcService : Service(), USBMonitor.OnDeviceConnectListener, IFrameCallback {

    private val TAG = "HeadlessUvcService"

    companion object {
        // Tiempo de espera CORTO entre intentos cuando la captura falla por causa
        // recuperable (USB hiccup, cámara no abre, excepción durante burst).
        // Antes: el siguiente ciclo arrancaba con el wait COMPLETO (~4 min) → el
        // usuario veía el countdown saltar de 0 a 4:00. Ahora: salta a 0:05.
        const val CYCLE_RETRY_DELAY_MS = 5_000L

        // ÚNICA fuente de verdad del cycleSec base (el tiempo que damos al relay
        // para procesar = lo que el móvil espera tras enviar la foto, antes de
        // recoger la respuesta). 300s = 5 min: desde que el cliente manda la foto
        // hasta que recoge la respuesta pasan ~5 min, holgado para que el OCR
        // (≤90s) + Tavily (≤8s) + los analyzers de razonamiento (GPT-5 web,
        // DeepSeek-R1 ~90-120s) + meta-judge (60s) terminen sin presión:
        // ventana de analyzers = (300−15) − 98 ≈ 187s. NO se regula desde la UI
        // (el slider se retiró); este es el valor acordado. Cambiar AQUÍ y solo
        // aquí. El relay es agnóstico: usa el review_timeout que el móvil envía
        // (=cycleSec−15), así que no hay que tocar nada server-side.
        const val CYCLE_SECONDS_BASE = 300L

        // Delay LOCAL entre que termina la última vibración de respuesta y la
        // siguiente captura (el "aviso de foto"). Es INDEPENDIENTE del cycleSec
        // (CYCLE_SECONDS_BASE) que se le da al relay para procesar.
        //
        // El waitMs antes de la siguiente captura NO debe ser igual al cycleSec
        // (ese bug hacía ciclo total = wait(5min) + API(5min) ≈ 10 min). Es solo
        // POST_RESPONSE_DELAY_S, que arranca tras la última vibración de respuesta.
        // Total entre respuestas ≈ cycleSec(~5min) + 60s ≈ 6-7 min.
        const val POST_RESPONSE_DELAY_S = 60L
        // Jitter ±10s sobre el delay local para no producir un patrón temporal
        // perfecto entre capturas (un observador no debería ver capturas
        // exactamente cada N segundos).
        const val POST_RESPONSE_DELAY_JITTER_S = 10L
    }

    @Volatile private var actualWidth = 1280
    @Volatile private var actualHeight = 720
    // FPS realmente negociado con la cámara para la resolución activa.
    // Lo rellena pickBestPreviewMode() durante openCameraLocked*(). Es el ÚNICO
    // FPS que usa el encoder de video — no hay configuración manual. Así el MP4
    // siempre va al ritmo REAL que entrega la cámara (sin slow-motion ni timing roto).
    @Volatile private var actualCameraFps = 30
    private val frameBurstCount = 30 // Aumentado a 30 frames para captura de 1.5s (AdvancedBurstProcessor)

    // 2500 ms (subido de 1200): la IMX179 entrega los primeros 15-30 frames con
    // AE/AWB convergiendo (oscuros, washed-out o casi negros). En MJPEG, además,
    // los frames iniciales suelen ser los que más sufren truncamiento por USB
    // OTG mientras se estabiliza el isochronous endpoint. Esperar 2.5 s garantiza
    // que la captura útil empiece con sensor estable y bus USB en régimen.
    private val sensorWarmupMs = 2500L
    // Margen extra para el reinicio después de la respuesta (simula apagar/encender pantalla)
    private val postProcessingReopenDelayMs = 3000L

    // Usar BolsilloIaClient.GEMINI_KEY (configurada desde la UI / prefs).

    private lateinit var usbMonitor: USBMonitor
    @Volatile private var uvcCamera: UVCCamera? = null
    @Volatile private var usbCtrlBlock: USBMonitor.UsbControlBlock? = null
    private var usbMonitorRegistered = false
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val cameraMutex = Mutex()
    private val modeMutex = Mutex()

    // Coordinador de eventos USB: cuando llegan attach/detach rápidos (mal
    // contacto, reconexión <200 ms), serviceScope.launch NO garantiza FIFO →
    // un connect nuevo puede ejecutarse antes que el cleanup del disconnect
    // viejo, dejando ctrlBlocks cruzados o cámara con referencia inválida.
    // Manteniendo aquí el último Job, podemos esperarlo (`join()`) antes de
    // procesar el siguiente evento y serializar la secuencia.
    @Volatile private var usbJob: Job? = null

    private val _cameraState = MutableStateFlow<CameraState>(CameraState.Idle)
    val cameraState = _cameraState.asStateFlow()

    private val frameChannel = Channel<Bitmap>(frameBurstCount * 2)
    private val isCapturingForBurst = AtomicBoolean(false)

    // === Video buffer-mode (NV12 → MediaCodec.queueInputBuffer) ====================
    // Cuando isCapturingForVideo=true, onFrame() ruta los bytes NV12 que entrega
    // libuvc (via setFrameCallback(YUV420SP)) directamente al encoder vía
    // VideoEncoder.queueNv12(). Se sustituye el antiguo path Surface mode
    // (uvcCamera.startCapture(encoder.inputSurface)) que crashea en Samsung
    // Exynos S10e con SIGSEGV en copyToSurface — la inputSurface de MediaCodec
    // está respaldada por HAL_PIXEL_FORMAT_YCbCr_420_888 que Mali Gralloc no
    // soporta vía ANativeWindow_lock sin LOCK_FLEX (no expuesto en NDK).
    private val isCapturingForVideo = AtomicBoolean(false)
    @Volatile private var videoEncoderRef: VideoEncoder? = null
    // Candado global: evita iniciar una nueva captura mientras el resultado de la
    // anterior aún se está procesando/enviando/recogiendo.
    private val iaResponseInFlight = AtomicBoolean(false)
    @Volatile private var iaResponseInFlightSinceMs: Long = 0L
    private fun markIaResponseStart() {
        iaResponseInFlightSinceMs = SystemClock.elapsedRealtime()
        iaResponseInFlight.set(true)
    }
    private fun markIaResponseEnd() {
        iaResponseInFlight.set(false)
        iaResponseInFlightSinceMs = 0L
    }
    private fun isIaResponseStale(nowElapsedMs: Long = SystemClock.elapsedRealtime()): Boolean {
        val startedAt = iaResponseInFlightSinceMs
        if (startedAt <= 0L) return false
        // Alineado con el timeout real de iteración IA para evitar soltar el candado
        // mientras aún hay trabajo válido en curso (video + relay + polling).
        val staleThresholdMs = iterationWorkTimeoutMs() + 30_000L
        return (nowElapsedMs - startedAt) > staleThresholdMs
    }
    @Volatile private var videoFramesQueued: Int = 0
    @Volatile private var videoFramesDropped: Int = 0
    @Volatile private var videoFramesFilteredBlur: Int = 0  // descartados por desenfoque

    // === Detección "cámara tapada" (kill switch del ciclo) ========================
    // Si el usuario pone el brazo (u otra cosa opaca) delante de la cámara durante
    // >= CAMERA_COVERED_THRESHOLD_MS, el ciclo se aborta:
    //   - se DETIENE la captura inmediatamente (corta el loop de captureVideoFrames)
    //   - NO se manda el MP4 al relay (no HTTP, no base64, sin radio)
    //   - NO se vibra patrón de respuesta (no hay respuesta)
    //   - el siguiente ciclo arranca normal en su intervalo
    //
    // Heurística: el plano Y de NV12 tiene los primeros w*h bytes con luminancia.
    // Muestreamos ~256 valores espaciados (resto barato) y si la MEDIA es <= umbral
    // marcamos el frame como "dark". Contamos frames dark consecutivos con su
    // timestamp del primero — si supera el umbral temporal, set flag y abort.
    //
    // Umbral Y=25 sobre 255: el brazo a cm de distancia produce Y casi 0; una
    // habitación a oscuras "normal" rara vez baja de Y=40-50 con sensor ganancia
    // alta. 25 deja margen para no abortar por "luz pobre legítima".
    private val CAMERA_COVERED_Y_THRESHOLD = 25     // 0..255, plano Y NV12
    private val CAMERA_COVERED_THRESHOLD_MS = 3000L  // 3 s tapada → kill
    @Volatile private var cameraCoveredAbort: Boolean = false
    @Volatile private var darkFrameRunStartMs: Long = 0L  // 0 = no estamos en racha dark
    @Volatile private var darkFrameRunCount: Int = 0
    @Volatile private var videoAbortedByUsbDisconnect: Boolean = false
    // Contador de cuántos ciclos abortamos por tapado — para diagnóstico
    @Volatile private var cyclesAbortedByCover: Long = 0L

    // === Pre-filtro de nitidez (Laplacian variance sobre plano Y) ==================
    // Buffer móvil de los últimos N valores de sharpness para fijar umbral
    // adaptativo. No usamos un threshold absoluto porque la "nitidez típica"
    // varía mucho con iluminación/contenido: lo que es nítido bajo luz pobre
    // sería borroso bajo luz brillante. La mediana del propio stream se
    // adapta sola.
    private val SHARP_HISTORY_SIZE = 30
    private val sharpHistory = DoubleArray(SHARP_HISTORY_SIZE)
    @Volatile private var sharpHistoryCount: Int = 0
    @Volatile private var sharpHistoryIdx: Int = 0
    // Si el filtro tira > MAX_CONSECUTIVE_DROPS seguidos, deja pasar el
    // siguiente aunque sea malo — evita un MP4 vacío en condiciones adversas.
    @Volatile private var consecutiveBlurDrops: Int = 0
    private val MAX_CONSECUTIVE_DROPS = 5

    // === Best frame para fallback de video (cámara en pecho apuntando al folio) =====
    // Si el relay de video cae, en vez de subir el primer frame del MP4 (que con
    // MediaMetadataRetriever es lento Y puede pillar un frame malo del arranque
    // de cámara), subimos el MEJOR frame del periodo según score compuesto:
    //   score = sharpness × completeness × stability
    //     sharpness    = Laplacian variance (ya se calcula, mide nitidez)
    //     completeness = score 0..1 basado en histograma del plano Y; alto cuando
    //                    el frame tiene distribución bimodal (folio + texto/borde),
    //                    bajo cuando la cámara está pegada al folio (todo blanco)
    //                    o demasiado lejos (poca señal de folio)
    //     stability    = score 0..1 basado en diferencia media respecto al frame
    //                    anterior; bajo si hay movimiento abrupto
    // Guardamos en RAM la copia NV12 entera del mejor candidato (1.4 MB para 720p).
    // Al finalizar el video, lo convertimos a JPEG 85 (~150-300 KB) y queda listo
    // para subir como fallback sin abrir nueva ventana de RF.
    /** Estado del "best frame" agrupado en un objeto inmutable único. Así
     *  garantizamos atomicidad entre los 4 campos al leerlos desde otros
     *  hilos: la referencia @Volatile se intercambia de golpe (CAS) en vez
     *  de actualizar 4 campos independientes que un lector podría observar
     *  parcialmente actualizados.
     *
     *  null = aún no hay candidato (capture sin frames procesados).
     */
    private data class BestFrame(
        val nv12: ByteArray,
        val w: Int,
        val h: Int,
        val score: Double,
    )
    @Volatile private var bestFrame: BestFrame? = null
    @Volatile private var bestFrameJpeg: ByteArray? = null
    // Suma de pixeles del frame anterior para detectar movimiento (sin guardar
    // el frame entero — solo necesitamos un escalar como proxy).
    @Volatile private var prevFrameYSum: Long = -1L
    @Volatile private var prevFrameYAvg: Double = 0.0

    @Volatile private var isCycleActiveAndUsingCamera = false
    @Volatile private var isCapturingCycle = false

    @Volatile private var currentPreviewSurface: Surface? = null
    @Volatile private var attachedSurface: Surface? = null


    private lateinit var vibrator: Vibrator

    private val alarmAudioAttrs: android.media.AudioAttributes by lazy {
        android.media.AudioAttributes.Builder()
            .setUsage(android.media.AudioAttributes.USAGE_ALARM)
            .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
    }

    @Volatile private var currentVibrationIntensity = 255

    // === Sysfs intensity directo (Samsung S10e / sec_haptic driver) ============
    // CRÍTICO descubierto investigando el kernel exynos9820 (whatawurst/cruel
    // kernels): el driver sec_haptic.c expone
    //   /sys/class/timed_output/vibrator/intensity   (rango 1..10000)
    // que controla el motor DIRECTAMENTE. Escribir ahí salta TODO el stack:
    //   - Settings.System (MOVED_TO_SECURE → IllegalArgumentException desde Java)
    //   - Samsung HAL (que clampa amplitude según alarm_vibration_intensity)
    //   - VibratorService (que cachea valores en su propia cola)
    // El motor sale con la intensidad escrita en cuanto se dispare el siguiente
    // `enable`. Latencia <5ms (echo via shell persistente).
    //
    // MAX_INTENSITY = 10000 según include/linux/sec_haptic.h del kernel
    // exynos9820. Mapeo del slider del usuario (1..255) a sysfs (1..10000).
    private val VIB_SYSFS_INTENSITY_CANDIDATES = listOf(
        "/sys/class/timed_output/vibrator/intensity",
        "/sys/devices/virtual/timed_output/vibrator/intensity",
        "/sys/class/sec_class/motor/intensity",
    )
    private val VIB_SYSFS_MAX = 10000
    @Volatile private var vibSysfsIntensityPath: String? = null
    @Volatile private var vibSysfsChecked: Boolean = false

    @Volatile private var currentCycleDelaySeconds = CYCLE_SECONDS_BASE
    // Delay real aplicado en la iteración EN CURSO (con jitter ±25% sobre la media).
    // Lo escribe runSystemCycle() justo antes del delay y lo leen processBurstAndSend
    // y processVideoAndSend para reportar al relay el tiempo real (review_timeout_seconds),
    // garantizando que la API auto-aprueba ANTES del próximo ciclo aunque ese ciclo
    // haya tocado un delay corto (3 min) en lugar de la media (4 min).
    @Volatile private var currentIterationCycleSeconds = CYCLE_SECONDS_BASE
    // Payload único en producción: VIDEO (MP4 + OCR remoto).
    @Volatile private var currentCapturePayload = "VIDEO"
    // Duración del video grabado en cada ciclo (segundos) cuando payload=VIDEO.
    @Volatile private var videoDurationSeconds  = 5
    // NB: el FPS del video NO es configurable. Siempre se usa `actualCameraFps`
    // (el máximo negociado con la cámara para la resolución activa) — así el MP4
    // va al ritmo REAL y nunca queda lento o desincronizado.
    // Techo de bitrate (kbps). Es un TECHO, no un mínimo: el bitrate final =
    // max(pref usuario, computeBitrateKbps(res,fps)). Default 1500 kbps =
    // 1.5 Mbps → ~1.9 MB para 10s a 720p H.265 (≈ subida 7-10 s en 4G regular).
    // Editable por el usuario en el rango 600..50000 kbps (0.6-50 Mbps).
    @Volatile private var videoBitrateKbps      = 1500
    private var toggleAirplaneModeEnabled = false
    @Volatile private var currentCaptureMode = "A"
    @Volatile private var debugSinAvion: Boolean = false
    @Volatile private var suppressNextCaptureSignalVibration: Boolean = false

    private var vibrationDurationMs = 200L
    private var intraBeepDelayMs = 400L
    private var interLetterDelayMs = 800L
    @Volatile private var preBurstVibrationDelayMs = 0L
    @Volatile private var startDelayMinutes: Int = 0
    @Volatile private var initialDelayEndAtMs: Long = 0L

    // Prompt para páginas normales (sin diagrama de contexto aún almacenado).
    //
    // Diseño basado en research 2025-2026:
    //   • XML tags estructurados (Anthropic best practice — Claude Opus 4.7
    //     responde +20-30% mejor que con texto plano).
    //   • Few-shot con UN solo ejemplo (over-prompting degrada accuracy).
    //   • Process of Elimination (POE) — técnica probada que mejora MCQ en LLMs.
    //   • Self-check explícito al final (caza el bug "responde 3 veces").
    //
    // Soporta dos tipos de entrada:
    //   • IMAGEN: una foto de la hoja de examen (modo ráfaga / fallback foto).
    //   • TEXTO OCR: el server antepone <readings> con N transcripciones del
    //     MISMO examen + un <protocol> específico. Aquí solo definimos las
    //     reglas generales — el protocol del relay las complementa.
    private val SYSTEM_PROMPT_UNIFICADO = """
        <role>
        Eres un experto en resolución de exámenes tipo test de opción múltiple
        (A/B/C/D). Eres meticuloso al leer (OCR si hace falta) y al razonar:
        eliminas opciones obviamente erróneas antes de elegir, y verificas tu
        respuesta final antes de escribirla.
        </role>

        <input_kinds>
        Tu entrada puede ser de DOS tipos:
          • IMAGEN: foto de la hoja de examen — haces OCR + resuelves.
          • TEXTO OCR: te llega un bloque <readings> con N transcripciones del
            MISMO examen filmado, más un <protocol> que detalla cómo fusionarlas.
            NO son N exámenes distintos: es UN examen leído N veces.
        Identifica el tipo al verlo y aplica las reglas correspondientes.
        </input_kinds>

        <visual_context_marker>
        SOLO en modo IMAGEN: la 1ª línea de tu respuesta DEBE ser una de estas:
          • CONTEXTO_VISUAL:SI   (si la hoja contiene diagrama/figura/gráfico)
          • CONTEXTO_VISUAL:NO   (si solo es texto)
        En modo TEXTO OCR: OMITE esta línea y arranca por el PASO 1.
        </visual_context_marker>

        <process>
        PASO 1 — INVENTARIO ÚNICO DE PREGUNTAS
          Cuenta las preguntas DISTINTAS del examen y anótalo:
            "He identificado K preguntas."
          • Modo IMAGEN: cuenta lo que ves en la hoja.
          • Modo TEXTO OCR: si hay N lecturas, una pregunta que aparezca en
            varias cuenta como UNA. K es el número del examen REAL, NUNCA N×K.
            Si las lecturas discrepan, gana el consenso (mayoría). Si todas son
            distintas, gana la más completa.

        PASO 2 — LECTURA AL MÁXIMO
          Para cada pregunta del inventario:
            • Modo IMAGEN: incluso si está movida/oscura/recortada, analiza
              trazos, contexto, palabras parciales. Las marcas a mano del
              usuario (círculos, palomitas, letras rellenadas) NO cuentan —
              respondes tú desde cero.
            • Modo TEXTO OCR: si una palabra es [?] en una lectura pero clara
              en otra, usa la clara. Si las 3 tienen [?], deduce por contexto.

        PASO 3 — RESOLUCIÓN POR ELIMINACIÓN (POE)
          Para CADA pregunta, en orden:
            a) Cita la pregunta en 1 línea (formato: "P{n}: <enunciado breve>").
            b) Elimina 1-3 opciones obviamente erróneas y di por qué (1 frase).
            c) Escoge la más correcta de las que quedan.
            d) Si ninguna elimina → tu mejor apuesta razonada (NUNCA X).
            e) X SOLO si la pregunta es 100% ilegible incluso fusionando lecturas.

        PASO 4 — AUTO-VERIFICACIÓN
          Antes de escribir <FINAL>:
            • Cuenta las letras que vas a poner. Debe coincidir EXACTAMENTE con
              la K declarada en PASO 1.
            • Si no coincide, recuenta tus preguntas y respuestas, corrige.
            • Verifica orden: la letra i-ésima responde a la pregunta i-ésima.
        </process>

        <output_format>
        Termina SIEMPRE con esta etiqueta en su propia línea:
            <FINAL>letras</FINAL>
        donde "letras" = exactamente K caracteres en {A,B,C,D,X}, sin separadores.

        El número de letras DEBE ser igual a K. Si te sale más o menos, has
        contado mal — revisa antes de cerrar.
        </output_format>

        <example>
        Caso: modo TEXTO OCR con 3 lecturas del mismo examen de 4 preguntas.

        CONTEXTO_VISUAL:NO   ← (incorrecto aquí, omitir en modo TEXTO OCR)

        He identificado 4 preguntas.

        P1: ¿Capital de Francia? → A=Madrid, B=París, C=Berlín, D=Roma.
            Madrid (España), Berlín (Alemania), Roma (Italia) → eliminados.
            Respuesta: B.

        P2: Símbolo químico del oro. → A=Plata, B=Oro, C=Cobre, D=Hierro.
            La opción nombra el elemento, no el símbolo, pero la pregunta del
            enunciado es semántica: oro → B.

        P3: 2+2 = ? → A=3, B=4, C=5, D=22.
            Suma básica: B.

        P4: Año del descubrimiento de América. → A=1492, B=1500, C=1453, D=1620.
            Conocido: A.

        Auto-check: 4 letras (B,B,B,A) = K=4. ✓
        <FINAL>BBBA</FINAL>
        </example>

        <forbidden>
          • Saltarse preguntas del inventario.
          • Repetir el set de letras N veces porque hay N lecturas (ej:
            <FINAL>BBBABBBABBBA</FINAL> con K=4 sería 3× → PROHIBIDO).
          • Devolver más letras que K, o menos que K.
          • Poner X cuando solo es difícil — X SOLO si imposible.
          • Texto después del </FINAL>.
        </forbidden>
    """.trimIndent()

    // Prompt para páginas de la Parte 2 cuando ya hay un diagrama de contexto almacenado.
    // La hoja de preguntas puede llegar como IMAGEN o como TEXTO OCR (modo video,
    // con N lecturas del MISMO set). El diagrama de contexto siempre viaja como
    // IMAGEN aparte. Mismas técnicas que SYSTEM_PROMPT_UNIFICADO + diagrama.
    private val SYSTEM_PROMPT_CON_CONTEXTO = """
        <role>
        Eres un experto en exámenes tipo test (A/B/C/D) con apoyo de caso
        práctico. Razonas por eliminación, usas el diagrama como contexto, y
        verificas tu respuesta antes de escribirla.
        </role>

        <inputs>
        Recibes DOS entradas:
          • DIAGRAMA (siempre imagen): la 1ª imagen — el caso práctico de
            referencia. NO contiene preguntas, solo contexto.
          • HOJA DE PREGUNTAS: 2ª imagen, O un bloque <readings> con N
            transcripciones OCR del MISMO examen (modo video).

        Las respuestas son SIEMPRE sobre la hoja de preguntas. El diagrama es
        SOLO contexto para razonar.
        </inputs>

        <process>
        PASO 1 — INVENTARIO ÚNICO DE PREGUNTAS
          Cuenta las preguntas DISTINTAS y declara:
            "He identificado K preguntas."
          • Imagen: cuenta lo que ves en la 2ª imagen.
          • TEXTO OCR con N lecturas: una pregunta repetida cuenta UNA vez.
            K es el número REAL del examen, NUNCA N×K. En desacuerdo entre
            lecturas, gana la mayoría.

        PASO 2 — LECTURA AL MÁXIMO
          Imagen: análisis exhaustivo incluso si está movida/oscura. Las marcas
            a mano del usuario NO cuentan.
          TEXTO OCR: si una palabra es [?] en una lectura pero clara en otra,
            usa la clara.

        PASO 3 — RESOLUCIÓN POR ELIMINACIÓN (POE) CON DIAGRAMA
          Para cada pregunta, en orden:
            a) Cita la pregunta en 1 línea ("P{n}: <enunciado breve>").
            b) Usa el DIAGRAMA para entender el caso práctico cuando aplique.
            c) Elimina 1-3 opciones erróneas y di por qué (1 frase).
            d) Escoge la más correcta de las que quedan.
            e) Mejor apuesta razonada si dudas. X SOLO si 100% ilegible.

        PASO 4 — AUTO-VERIFICACIÓN
          Antes de <FINAL>: cuenta letras. Debe ser EXACTAMENTE K. Si no
          coincide, recuenta y corrige.
        </process>

        <output_format>
        Termina SIEMPRE con la etiqueta:
            <FINAL>letras</FINAL>
        donde "letras" = exactamente K caracteres en {A,B,C,D,X}, sin separadores.
        </output_format>

        <example>
        Caso: diagrama de un circuito eléctrico + hoja con 3 preguntas sobre él.

        He identificado 3 preguntas.

        P1: ¿Cuál es la resistencia equivalente? Diagrama muestra 2 resistencias
            de 10Ω en paralelo. Eliminadas A=20Ω y D=15Ω (no resultan de 10‖10).
            Paralelo 10‖10 = 5Ω → B.

        P2: ¿Intensidad por la rama principal? V=10V, Req=5Ω → I=2A. Eliminadas
            opciones distintas de 2A. → C.

        P3: ¿El circuito es serie o paralelo? Diagrama claro → paralelo. → A.

        Auto-check: 3 letras (B,C,A) = K=3. ✓
        <FINAL>BCA</FINAL>
        </example>

        <forbidden>
          • Saltarse preguntas o duplicar letras por haber visto la misma
            pregunta en varias lecturas (ej: <FINAL>BCABCABCA</FINAL> con K=3
            sería 3× → PROHIBIDO).
          • Más o menos letras que K en <FINAL>.
          • Responder sobre el diagrama en vez de sobre las preguntas.
          • Texto después del </FINAL>.
        </forbidden>
    """.trimIndent()

    private var geminiPrompt = SYSTEM_PROMPT_UNIFICADO

    // Imagen de contexto del caso práctico (Parte 2 del examen).
    // Se guarda en disco para usarla en fallbacks directos (Modo B/C) si el relay no responde.
    // El relay gestiona su propia copia: la detecta automáticamente y la borra vía /reset.
    @Volatile private var contextImageB64: String? = null
    private val CONTEXT_IMAGE_FILE = "ctx_diagram.jpg"
    // true = hay que llamar a /reset en el relay al inicio del siguiente ciclo de radio
    private val pendingRelayReset = java.util.concurrent.atomic.AtomicBoolean(false)

    private fun loadContextImage() {
        try {
            val f = java.io.File(filesDir, CONTEXT_IMAGE_FILE)
            if (f.exists()) {
                contextImageB64 = android.util.Base64.encodeToString(f.readBytes(), android.util.Base64.NO_WRAP)
                Log.i(TAG, "Diagrama de contexto cargado de disco (${f.length()} bytes).")
            }
        } catch (e: Exception) {
            Log.e(TAG, "loadContextImage error: ${e.message}")
        }
    }

    private fun saveContextImage(jpeg: ByteArray) {
        try {
            java.io.File(filesDir, CONTEXT_IMAGE_FILE).writeBytes(jpeg)
            contextImageB64 = android.util.Base64.encodeToString(jpeg, android.util.Base64.NO_WRAP)
            Log.i(TAG, "Diagrama de contexto guardado (${jpeg.size} bytes). Proximas paginas lo usaran.")
            broadcastContextChanged(true)
        } catch (e: Exception) {
            Log.e(TAG, "saveContextImage error: ${e.message}")
        }
    }

    private fun clearContextImage() {
        contextImageB64 = null
        pendingRelayReset.set(true) // relay debe borrar su copia en el próximo ciclo de radio
        try { java.io.File(filesDir, CONTEXT_IMAGE_FILE).delete() } catch (_: Exception) {}
        Log.i(TAG, "Diagrama de contexto borrado — examen nuevo.")
        broadcastContextChanged(false)
    }

    private lateinit var sharedPrefs: SharedPreferences

    private var wakeLock: PowerManager.WakeLock? = null

    @Volatile private var isPocketMode: Boolean = false
    @Volatile private var isAppInForeground: Boolean = false
    @Volatile private var screenInteractive: Boolean = true
    @Volatile private var cameraOpen: Boolean = false
    
    // CAMBIO IMPORTANTE: Inicia systemEnabled a FALSE para que no arranque solo
    @Volatile private var systemEnabled: Boolean = false 
    
    private var cycleJob: Job? = null
    private var screenReceiverRegistered = false

    private val intentionalCameraClose = AtomicBoolean(false)
    private val immortalLoopStarted = AtomicBoolean(false)

    @Volatile private var lastCameraOpenAtMs: Long = 0L

    private var pendingDeactivateJob: Job? = null
    private var vibrationTestJob: Job? = null

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_ON, Intent.ACTION_USER_PRESENT, Intent.ACTION_SCREEN_OFF -> {
                    updateInteractiveState()
                }
            }
        }
    }

    // ── Guardián RF event-driven (broadcast ACTION_AIRPLANE_MODE_CHANGED) ────
    // El bucle rápido de 15s es el safeguard final, pero queremos reaccionar
    // <200ms cuando el cambio ocurre. Samsung One UI emite este broadcast cada
    // vez que airplane_mode_on cambia — tanto por nuestro propio `am broadcast`
    // como por toques del usuario en Quick Settings, otra app, o el sistema
    // (Doze, etc.). El receiver compara el nuevo estado con la INTENCIÓN
    // (openRfWindows): si avión queda OFF pero no había ventana intencional,
    // reactivamos inmediatamente.
    @Volatile private var airplaneModeReceiverRegistered = false
    private val airplaneModeReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != Intent.ACTION_AIRPLANE_MODE_CHANGED) return
            val state = intent.getBooleanExtra("state", false)  // true = avión ON
            if (state) return  // cambio a ON → exactamente lo que queremos
            // Cambio a OFF: ¿intencional o anómalo?
            if (!systemEnabled) return
            if (debugSinAvion) return
            if (currentCaptureMode != "A" && currentCaptureMode != "C") return
            if (BolsilloIaOrquestador.isAnyRfWindowOpen) return  // OFF esperado
            // ANOMALY: avión OFF sin ventana intencional → reactivar YA.
            serviceScope.launch {
                try {
                    Log.w(TAG, "[RF-guard] broadcast: avión OFF sin ventana intencional → reactivando inmediato")
                    PersistentErrorLog.logError(
                        TAG, "Guardián RF (broadcast): avión OFF anómalo detectado <1s tras el cambio. Reactivando inmediatamente.",
                    )
                    setAirplaneMode(true)
                } catch (t: Throwable) {
                    Log.w(TAG, "[RF-guard] receiver setAirplaneMode(true) lanzó: ${t.message}")
                }
            }
        }
    }

    // ── Auto-rechazo silencioso de llamadas entrantes ────────────────────────
    // Si el sistema está habilitado y entra una llamada (típicamente cuando
    // debugSinAvion=true o entre ciclos con avión ON pero la cámara no está
    // activa para abrir avión OFF), colgamos vía root inmediatamente. Como el
    // Total Silence DnD + ringer=0 + master-mute están aplicados desde antes,
    // ni siquiera llega a sonar — sólo es la última red de seguridad.
    @Volatile private var phoneStateRegistered = false
    private var phoneStateListener: android.telephony.PhoneStateListener? = null

    // ── Recovery USB tras llamada / zombie state ────────────────────────────────
    // Samsung S10e + Android 12 + One UI 4 sufre un bug confirmado: cuando entra
    // una llamada, el subsistema audio reconfigura el puerto USB OTG (dwc3-exynos).
    // Tras la llamada el device USB queda en estado "zombie":
    //   - libusb retorna LIBUSB_ERROR_NO_DEVICE (-99) al open()
    //   - release_interface → EINVAL (errno 22)
    //   - UsbManager NO emite ACTION_USB_DEVICE_DETACHED (cree que sigue conectado)
    // Sin esto, el ciclo de captura entra en bucle infinito de retries cada 5s,
    // vibrando antes de cada intento → "vibra cada 5s y nunca recupera".
    //
    // Fix:
    //   1) Detectar transición CALL_STATE_IDLE tras RINGING/OFFHOOK → marcar
    //      pendingUsbResetAfterCall y reset preventivo en el próximo retry.
    //   2) Contar fallos consecutivos de openCameraLockedHidden. Si llega a
    //      USB_RECOVERY_THRESHOLD, forzar reset USB vía sysfs aunque no haya
    //      habido llamada (cubre otras causas de zombie: USB hub flaky, etc.)
    //   3) recoverUsbBusViaSysfs() escribe `0 > authorized; sleep; 1 > authorized`
    //      en todos los devices USB no-hub → el kernel hace re-enumerate y
    //      UsbAutoGrant recibe el ATTACHED de vuelta sin desenchufar físicamente.
    private val USB_RECOVERY_THRESHOLD = 3
    @Volatile private var consecutiveUsbOpenFails = 0
    private val pendingUsbResetAfterCall = AtomicBoolean(false)
    @Volatile private var lastCallStateChangedAtMs = 0L
    @Volatile private var prevCallState = android.telephony.TelephonyManager.CALL_STATE_IDLE

    /**
     * Fuerza un re-bind de los devices USB no-hub vía sysfs como root.
     * Resuelve el "USB zombie state" del Samsung S10e tras llamada (donde el
     * device sigue listado pero el pipe USB está roto a nivel kernel).
     *
     * Equivalente a desenchufar y enchufar la cámara, pero sin tocar el hardware:
     * el kernel hace un soft-reset del puerto y emite los uevents disconnect/connect
     * → UsbAutoGrant recibe ATTACHED → onAttach() reabre el usbCtrlBlock limpio.
     *
     * Es NO-BLOCKING: dispara el comando y vuelve. El recovery real ocurre cuando
     * llegue el ACTION_USB_DEVICE_ATTACHED ~500-1500ms después.
     */
    private fun recoverUsbBusViaSysfs() {
        // Buscar TODOS los devices USB que no sean hubs (idVendor existe). En el
        // S10e OTG hay típicamente: usb1 (hub root) + 1-1 (cámara UVC). Si hay
        // más devices (hub externo + cámara + mic), se reauthorizan todos.
        val cmd =
            "for d in /sys/bus/usb/devices/*/; do " +
            "  if [ -f \"\${d}idVendor\" ]; then " +
            "    n=\$(basename \"\$d\"); " +
            "    case \"\$n\" in usb*) ;; *) " +
            "      echo 0 > \"\${d}authorized\" 2>/dev/null; " +
            "    esac; " +
            "  fi; " +
            "done; " +
            "sleep 0.3; " +
            "for d in /sys/bus/usb/devices/*/; do " +
            "  if [ -f \"\${d}idVendor\" ]; then " +
            "    n=\$(basename \"\$d\"); " +
            "    case \"\$n\" in usb*) ;; *) " +
            "      echo 1 > \"\${d}authorized\" 2>/dev/null; " +
            "    esac; " +
            "  fi; " +
            "done"
        val ok = runRootCommand(cmd, "usb-bus-recover")
        Log.w(TAG, "[usb-recover] reset USB vía sysfs (authorized 0/1) · ok=$ok · " +
                   "consecutiveFails=$consecutiveUsbOpenFails · pendingAfterCall=${pendingUsbResetAfterCall.get()}")
        try {
            PersistentErrorLog.logError(
                TAG,
                "USB bus zombie detectado — recovery por sysfs (consecutiveFails=$consecutiveUsbOpenFails, " +
                "afterCall=${pendingUsbResetAfterCall.get()})",
            )
        } catch (_: Throwable) {}
        // Limpiar flags: si el reset funciona, UsbAutoGrant emitirá ATTACHED y
        // el ciclo recuperará. Si no, el counter volverá a subir y reintentaremos.
        consecutiveUsbOpenFails = 0
        pendingUsbResetAfterCall.set(false)
    }

    private fun registerPhoneStateListener() {
        if (phoneStateRegistered) return
        // El constructor de PhoneStateListener llama internamente `Looper.myLooper()` y
        // luego accede a `looper.mQueue` — si lo invocamos desde un hilo Binder (que es
        // el caso típico: setSystemEnabled viene por AIDL), `myLooper()` devuelve null
        // y casca con NPE silencioso (queda atrapado en el catch de abajo y nos quedamos
        // sin auto-rechazo de llamadas). Lo construimos en el Main looper para garantizar
        // que tenga un MessageQueue válido.
        Handler(Looper.getMainLooper()).post {
            if (phoneStateRegistered) return@post
            try {
                val tm = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
                phoneStateListener = object : android.telephony.PhoneStateListener() {
                    override fun onCallStateChanged(state: Int, phoneNumber: String?) {
                        // Track de transiciones para detectar IDLE tras llamada/rechazo.
                        // El USB OTG del S10e queda zombie cuando el audio routing del call
                        // toca el puerto. Necesitamos reset preventivo al volver a IDLE
                        // INDEPENDIENTEMENTE de si la llamada fue entrante (RINGING) o
                        // saliente (OFFHOOK), porque el bug aplica a ambos casos.
                        val prev = prevCallState
                        prevCallState = state
                        lastCallStateChangedAtMs = System.currentTimeMillis()

                        // Sólo reaccionar cuando el sistema está enabled y la modo NO es TEST
                        if (!systemEnabled || currentCaptureMode == "TEST") return

                        if (state == android.telephony.TelephonyManager.CALL_STATE_RINGING) {
                            Log.w(TAG, "[silencio] Llamada entrante detectada (state=RINGING) → rechazando silenciosamente vía root")
                            // service call phone 6 = ITelephony.endCall() en S10e One UI
                            // Fallback: KEYCODE_ENDCALL por si endCall code cambia entre versiones
                            runRootCommand(
                                "service call phone 6 2>/dev/null; input keyevent KEYCODE_ENDCALL 2>/dev/null",
                                "reject-call"
                            )
                            // Marcar que tendremos que recuperar el USB cuando vuelva a IDLE.
                            // El reset NO se hace ahora — durante RINGING el bus aún funciona;
                            // el problema aparece DESPUÉS de que el call manager toque el
                            // routing audio. Lo hacemos cuando llegue IDLE.
                            pendingUsbResetAfterCall.set(true)
                        } else if (state == android.telephony.TelephonyManager.CALL_STATE_OFFHOOK) {
                            // Llamada en curso (entrante contestada o saliente activa).
                            // Mismo problema: el audio routing va a tocar el USB OTG.
                            pendingUsbResetAfterCall.set(true)
                        } else if (state == android.telephony.TelephonyManager.CALL_STATE_IDLE &&
                                   prev != android.telephony.TelephonyManager.CALL_STATE_IDLE) {
                            // Transición OFFHOOK/RINGING → IDLE: la llamada terminó.
                            // Es CRÍTICO esperar ~1.5s antes del reset: el audio routing del
                            // S10e tarda en "soltar" el bus tras IDLE. Si reseteamos inmediato,
                            // a veces el bus se queda zombie OTRA VEZ.
                            //
                            // Lanzamos en serviceScope para no bloquear el callback. NonCancellable
                            // por si el sistema se desactiva entre medias — el recovery debe
                            // completarse para dejar el USB en estado limpio.
                            Log.w(TAG, "[usb-recover] Llamada terminó (transición → IDLE). Reset USB preventivo en 1500ms.")
                            serviceScope.launch {
                                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                                    kotlinx.coroutines.delay(1500)
                                    if (pendingUsbResetAfterCall.get()) {
                                        recoverUsbBusViaSysfs()
                                    }
                                }
                            }
                        }
                    }
                }
                @Suppress("DEPRECATION")
                tm.listen(phoneStateListener, android.telephony.PhoneStateListener.LISTEN_CALL_STATE)
                phoneStateRegistered = true
                Log.i(TAG, "[silencio] PhoneStateListener registrado — auto-rechazo de llamadas activo")
            } catch (e: Exception) {
                Log.w(TAG, "[silencio] No se pudo registrar PhoneStateListener: ${e.message}")
            }
        }
    }

    // ── CallScreeningService: bloqueo pre-RINGING de llamadas entrantes ──────
    // El PhoneStateListener llega DESPUÉS de que la llamada haya pasado por
    // RINGING (aunque sea milisegundos). En Samsung S10e eso ya dispara el bug
    // del audio routing → USB OTG zombie. Con CallScreeningService Android
    // consulta a nuestro service ANTES de hacer sonar → rechazamos a nivel
    // telecom-stack, no toca audio, no toca USB.
    //
    // Requiere ser el holder del role CALL_SCREENING. Solo hay UN holder por
    // sistema, así que al desactivar restauramos el default (Samsung Smart Call
    // Protection o vacío). Los comandos `cmd role` requieren root o `shell`,
    // ambos disponibles vía runRootCommand.
    private fun enableCallScreeningRole() {
        // CRÍTICO: setBlockingActive escribe a SharedPreferences con commit()
        // SÍNCRONO. Hacemos esto ANTES del add-role-holder para que cuando el
        // framework empiece a invocar a BlockAllCallScreeningService, ya
        // encuentre el flag en true. Si el orden fuera al revés habría una
        // ventana de race donde llamadas que entren entre add-role y commit
        // pasarían sin bloqueo.
        BlockAllCallScreeningService.setBlockingActive(applicationContext, true)
        // Asignar el role para que el framework Android llame a nuestro service.
        // Si ya éramos holder (re-activación rápida), no-op. Si había otro
        // holder (Samsung), lo SOBREESCRIBIMOS — se restaura en disable.
        val ok = runRootCommand(
            "cmd role add-role-holder android.app.role.CALL_SCREENING $packageName",
            "role-add-screening",
        )
        Log.i(TAG, "[call-screening] Bloqueo ACTIVADO · role add-role-holder ok=$ok · prefs=true")
        try {
            PersistentErrorLog.logError(
                TAG, "CallScreening role asignado a $packageName · ok=$ok",
            )
        } catch (_: Throwable) {}
    }

    private fun disableCallScreeningRole() {
        // Orden inverso al enable: PRIMERO quitamos el rol (así nuevas llamadas
        // siguen el flujo Samsung por defecto), DESPUÉS marcamos el flag false.
        // Si quedara una llamada in-flight invocando a nuestro service entre
        // medias, mejor que rechace (flag aún true) que no que pase (flag false
        // antes de quitar rol → comportamiento incoherente).
        val ok = runRootCommand(
            "cmd role remove-role-holder android.app.role.CALL_SCREENING $packageName",
            "role-rm-screening",
        )
        BlockAllCallScreeningService.setBlockingActive(applicationContext, false)
        Log.i(TAG, "[call-screening] Bloqueo DESACTIVADO · role remove-role-holder ok=$ok · prefs=false")
    }

    private fun unregisterPhoneStateListener() {
        if (!phoneStateRegistered) return
        // Misma simetría que register: ejecutamos en el Main para que el unsubscribe
        // se procese en el mismo Looper donde el listener fue creado.
        Handler(Looper.getMainLooper()).post {
            if (!phoneStateRegistered) return@post
            try {
                val tm = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
                @Suppress("DEPRECATION")
                tm.listen(phoneStateListener, android.telephony.PhoneStateListener.LISTEN_NONE)
            } catch (_: Exception) {}
            phoneStateListener = null
            phoneStateRegistered = false
        }
    }

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    // ---------- AIDL binder ----------
    private val binder: IHeadlessUvcService.Stub = object : IHeadlessUvcService.Stub() {
        override fun setPreviewSurface(surface: Surface?) =
            this@HeadlessUvcService.setPreviewSurface(surface)
        override fun getPreviewSize(): IntArray {
            val w = actualWidth
            val h = actualHeight
            return if (w > 0 && h > 0) intArrayOf(w, h) else IntArray(0)
        }
        override fun sendHapticFeedback() = this@HeadlessUvcService.sendHapticFeedback()
        override fun setVibrationIntensity(amplitude: Int) =
            this@HeadlessUvcService.setVibrationIntensity(amplitude)
        override fun setCycleDurationSeconds(seconds: Long) =
            this@HeadlessUvcService.setCycleDurationSeconds(seconds)
        override fun setBeepDuration(ms: Long) = this@HeadlessUvcService.setBeepDuration(ms)
        override fun setIntraBeepDelay(ms: Long) = this@HeadlessUvcService.setIntraBeepDelay(ms)
        override fun setInterLetterDelay(ms: Long) = this@HeadlessUvcService.setInterLetterDelay(ms)
        override fun setPreBurstVibrationDelay(ms: Long) =
            this@HeadlessUvcService.setPreBurstVibrationDelay(ms)
        override fun setStartDelayMinutes(minutes: Int) =
            this@HeadlessUvcService.setStartDelayMinutes(minutes)
        override fun getInitialDelayEndAtMs(): Long = this@HeadlessUvcService.getInitialDelayEndAtMs()
        override fun toggleVibrationTestLoop(): Boolean = this@HeadlessUvcService.toggleVibrationTestLoop()
        override fun setGeminiPrompt(prompt: String) = this@HeadlessUvcService.setGeminiPrompt(prompt)
        override fun setPocketMode(enabled: Boolean) = this@HeadlessUvcService.setPocketMode(enabled)
        override fun setToggleAirplaneModeEnabled(enabled: Boolean) =
            this@HeadlessUvcService.setToggleAirplaneModeEnabled(enabled)
        override fun setAppInForeground(isForeground: Boolean) =
            this@HeadlessUvcService.setAppInForeground(isForeground)
        override fun setSystemEnabled(enabled: Boolean) = this@HeadlessUvcService.setSystemEnabled(enabled)
        override fun isSystemEnabled(): Boolean = this@HeadlessUvcService.isSystemEnabled()
        override fun getCameraStateName(): String = _cameraState.value.javaClass.simpleName
        override fun setCaptureMode(modo: String) = this@HeadlessUvcService.setCaptureMode(modo)
        override fun setCapturePayload(payload: String) = this@HeadlessUvcService.setCapturePayload(payload)
        override fun getCapturePayload(): String = this@HeadlessUvcService.getCapturePayload()
        override fun setVideoParams(durationSeconds: Int) = this@HeadlessUvcService.setVideoParams(durationSeconds)
        override fun setVideoBitrateKbps(kbps: Int) = this@HeadlessUvcService.setVideoBitrateKbps(kbps)
        override fun getVideoBitrateKbps(): Int = this@HeadlessUvcService.getVideoBitrateKbps()
        override fun forceAirplaneOff() = this@HeadlessUvcService.forceAirplaneOff()
        override fun setDebugSinAvion(enabled: Boolean) = this@HeadlessUvcService.setDebugSinAvion(enabled)
        override fun isDebugSinAvion(): Boolean = this@HeadlessUvcService.isDebugSinAvion()
        override fun triggerOneShotCapture(): Boolean = this@HeadlessUvcService.triggerOneShotCapture()
        override fun toggleVideoTestLoop(): Boolean = this@HeadlessUvcService.toggleVideoTestLoop()
        override fun isVideoTestLoopRunning(): Boolean = this@HeadlessUvcService.isVideoTestLoopRunning()
    }
    override fun onBind(intent: Intent?): IBinder = binder

    private fun iterationWorkTimeoutMs(): Long {
        // Timeout global por iteración IA: evita que una llamada de red colgada
        // deje el ciclo detenido durante muchos minutos.
        val cycleSec = currentIterationCycleSeconds.coerceAtLeast(120L)
        return cycleSec * 1000L + 240_000L
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "ACTION_STOP_SYSTEM") {
            Log.i(TAG, "Notificación camuflada presionada: Apagando sistema...")
            
            // Primero, desactivamos el estado lógico (y las radios si procedía)
            setSystemEnabled(false)
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(Service.STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
            
            // CORRECCIÓN VITAL: No destruimos el servicio inmediatamente. 
            // Le damos 1 segundo al hilo Root para ejecutar por completo las órdenes
            // de restauración del WiFi y de audio antes de aniquilar el proceso.
            serviceScope.launch {
                delay(1000)
                stopSelf() 
            }
            return START_NOT_STICKY
        }
        return START_STICKY
    }

    override fun onCreate() {
        super.onCreate()

        // Log persistente: ANTES que nada — si el servicio arranca por BootReceiver
        // (sin pasar por MainActivity), igual queremos capturar fallos tempranos.
        // PersistentErrorLog.init es idempotente: si MainActivity ya lo llamó, no-op.
        try { PersistentErrorLog.init(applicationContext) } catch (_: Exception) {}

        val previousCrashHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            Log.e(TAG, "UncaughtException en hilo ${t.name}: ${e.message}", e)
            try { PersistentErrorLog.logError(TAG, "Uncaught en hilo '${t.name}'", e) } catch (_: Throwable) {}
            try { previousCrashHandler?.uncaughtException(t, e) } catch (_: Throwable) {}
        }

        applyRootProtection()

        // Cargar API keys persistidas (relay + fallbacks + backups) antes de cualquier llamada
        try { BolsilloIaClient.loadKeysFromPrefs(this) } catch (e: Exception) {
            Log.w(TAG, "loadKeysFromPrefs falló: ${e.message}")
            PersistentErrorLog.logError(TAG, "loadKeysFromPrefs falló en servicio: ${e.message}", e)
        }

        sharedPrefs = getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
        loadSettings()
        loadContextImage()

        // Conservar la intención del usuario tras reinicios/crash del proceso :uvc.
        // Si no existe aún la preferencia (primera ejecución), arrancamos desactivado.
        systemEnabled = sharedPrefs.getBoolean("system_enabled", false)

        // STICKY RF-OFF (decisión del usuario):
        //
        // Si la sesión anterior terminó por una vía AJENA al botón de detener
        // (crash, OOM-kill, USB attach/detach rápido que provoque service teardown,
        // force-stop, reboot inesperado, etc.) NO debemos reactivar WiFi/Bluetooth.
        // El único camino que puede levantar las radios es el botón explícito de
        // DETENER SISTEMA (UI o notificación camuflada → setSystemEnabled(false)).
        //
        // Por eso aquí:
        //   • NO llamamos a setAirplaneMode(false): mantenemos el avión tal como
        //     estaba (típicamente ON, porque el crash no toggea settings).
        //   • NO limpiamos rf_armed: queda persistido para que la próxima
        //     activación o reconexión sepa que la protección RF sigue armada.
        //   • Re-forzamos la política de radios (WiFi/BT/NFC/location OFF) de
        //     forma defensiva por si algún subsistema del SO las hubiera
        //     reactivado durante el teardown.
        val wasArmedOnStart = sharedPrefs.getBoolean("rf_armed", false)
        if (wasArmedOnStart && !debugSinAvion) {
            Log.i(TAG, "onCreate: rf_armed=true detectado tras restart involuntario — manteniendo avión y re-forzando radios OFF (sticky)")
            serviceScope.launch {
                try { enforceRadioPolicy() }
                catch (t: Throwable) { Log.w(TAG, "enforceRadioPolicy en onCreate falló: ${t.message}") }
            }
        }

        // DECISIÓN DELIBERADA: NO restauramos audio en onCreate aunque haya backup pendiente.
        // El silencio debe ser sticky a cualquier fallo (crash, OOM, USB disconnect, reinicio).
        // Si Android mata el servicio o el usuario hace force-stop, el móvil sigue mudo hasta
        // que reactive el sistema y le dé EXPLÍCITAMENTE al botón de detener. Restaurar aquí
        // arruinaría el caso de uso real (espía silencioso): el dispositivo empezaría a sonar
        // justo cuando el servicio se cae a mitad de operación.

        // FIX M7: el delay inicial no se reanuda tras rearranque (línea ~2207 sobreescribe
        // endAt al reactivar el sistema). Limpiamos la pref para evitar dejar valores
        // colgados que cualquier consulta vía sharedPrefs interpretaría como delay activo.
        sharedPrefs.edit().putLong("initial_delay_end_at", 0L).apply()

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "UVC:SafeWakeLock").apply {
            setReferenceCounted(false)
            acquire()
        }

        updateInteractiveState()

        // ── NOTIFICACIÓN CAMUFLADA (REVOLUT) ──
        val channel = NotificationChannel(
            "REVOLUT_CHANNEL", "Revolut", NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Alertas de cuenta y pagos"
            setShowBadge(true)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)

        val notification = NotificationCompat.Builder(this, "REVOLUT_CHANNEL")
            .setContentTitle("Revolut")
            .setContentText("Pago recibido: +85,00 EUR de Carlos G.")
            .setSmallIcon(R.drawable.ic_notif_revolut)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_STATUS)
            .setOnlyAlertOnce(true)
            .setAutoCancel(false)
            .setShowWhen(false)
            .build()

        startForeground(1, notification)

        usbMonitor = USBMonitor(this, this)
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (!usbMonitorRegistered) { usbMonitor.register(); usbMonitorRegistered = true }

        // FIX MEDIO-7: si el dispositivo arrancó con el cable ya enchufado, el
        // intent ACTION_USB_DEVICE_ATTACHED NO se dispara (sólo se envía cuando
        // el cable se conecta físicamente DESPUÉS de boot). El mDeviceCheckRunnable
        // interno de USBMonitor poll cada 2 s y nos llamará onAttach, PERO si
        // UsbAutoGrant ya concedió permiso vía root, el sistema no rebroadcasteará
        // → onConnect nunca llega. Forzamos un escaneo manual aquí: pedimos
        // permiso explícitamente para cualquier dispositivo USB ya presente.
        // Si la permiso ya está concedida, USBMonitor procesa el connect síncrono.
        serviceScope.launch {
            try {
                kotlinx.coroutines.delay(500L) // pequeño margen para que USBMonitor inicialice
                val devices = usbMonitor.deviceList
                if (devices.isNotEmpty()) {
                    Log.i(TAG, "[usb] ${devices.size} dispositivo(s) USB ya conectado(s) al arrancar servicio — solicitando permiso")
                    for (d in devices) {
                        val granted = try {
                            val um = getSystemService(Context.USB_SERVICE) as UsbManager
                            UsbAutoGrant.grantViaApi(um, d, packageName)
                        } catch (_: Exception) { false }
                        if (!granted) {
                            try { UsbAutoGrant.grant(d, packageName, ::runRootCommand) } catch (_: Exception) {}
                        }
                        try { usbMonitor.requestPermission(d) } catch (e: Exception) {
                            Log.w(TAG, "[usb] requestPermission falló para ${d.deviceName}: ${e.message}")
                        }
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "[usb] escaneo inicial falló: ${e.message}")
            }
        }

        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        }
        registerReceiver(screenReceiver, filter)
        screenReceiverRegistered = true

        // Receiver event-driven para AIRPLANE_MODE_CHANGED: reactivamos avión
        // <200ms si detectamos OFF sin ventana intencional. Complementa al poll
        // de 15s del bucle rápido.
        try {
            registerReceiver(airplaneModeReceiver,
                IntentFilter(Intent.ACTION_AIRPLANE_MODE_CHANGED))
            airplaneModeReceiverRegistered = true
            Log.i(TAG, "[RF-guard] receiver AIRPLANE_MODE_CHANGED registrado (reacción <200ms)")
        } catch (e: Exception) {
            Log.w(TAG, "[RF-guard] no se pudo registrar receiver: ${e.message}")
        }

        // Reanudación headless robusta tras reboot/crash del proceso :uvc.
        // Si el usuario lo dejó activado, reaplicamos side-effects críticos y
        // relanzamos ciclo sin depender de abrir la UI ni de eventos externos.
        if (systemEnabled) {
            Log.i(TAG, "[resume] system_enabled=true en onCreate → reanudando ciclo y side-effects")
            serviceScope.launch {
                try {
                    if (currentCaptureMode != "TEST") {
                        try {
                            SilentModeManager.reinforceSilence(
                                this@HeadlessUvcService,
                                ::runRootCommand,
                                alarmVibrationIntensity = samsungIntensityFor(currentVibrationIntensity),
                            )
                        } catch (t: Throwable) {
                            Log.w(TAG, "[resume] reinforceSilence falló: ${t.message}")
                        }
                        try { registerPhoneStateListener() } catch (t: Throwable) {
                            Log.w(TAG, "[resume] registerPhoneStateListener falló: ${t.message}")
                        }
                        try { enableCallScreeningRole() } catch (t: Throwable) {
                            Log.w(TAG, "[resume] enableCallScreeningRole falló: ${t.message}")
                        }
                    }
                    if ((currentCaptureMode == "A" || currentCaptureMode == "C") && !debugSinAvion) {
                        try { enforceRadioPolicy() } catch (t: Throwable) {
                            Log.w(TAG, "[resume] enforceRadioPolicy falló: ${t.message}")
                        }
                    }
                    ensureCorrectMode()
                } catch (t: Throwable) {
                    Log.w(TAG, "[resume] reanudación headless falló: ${t.message}")
                }
            }
        }
    }

    // ── Shell root (completo) ──
    private val immortalScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    @Volatile private var rootShellProc: java.lang.Process? = null
    @Volatile private var rootShellWriter: java.io.BufferedWriter? = null
    private val rootShellLock = Any()

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

    private fun ensureRootShell(): Boolean {
        val current = rootShellProc
        if (current != null && current.isAlive) return true
        synchronized(rootShellLock) {
            val again = rootShellProc
            if (again != null && again.isAlive) return true
            return try {
                val suPath = getSuPath()
                val proc: java.lang.Process = Runtime.getRuntime().exec(arrayOf(suPath))
                val writer = java.io.BufferedWriter(java.io.OutputStreamWriter(proc.outputStream))
                Thread {
                    try { proc.inputStream.use { ins -> while (ins.read() != -1) {} } } catch (_: Exception) {}
                }.apply { isDaemon = true; name = "root-shell-stdout"; start() }
                Thread {
                    try { proc.errorStream.use { ins -> while (ins.read() != -1) {} } } catch (_: Exception) {}
                }.apply { isDaemon = true; name = "root-shell-stderr"; start() }
                rootShellProc = proc
                rootShellWriter = writer
                Log.i(TAG, "Shell root persistente abierta con éxito vía: $suPath")
                true
            } catch (e: Exception) {
                Log.e(TAG, "No se pudo abrir shell root: ${e.message}")
                rootShellProc = null
                rootShellWriter = null
                false
            }
        }
    }

    private fun runRootCommand(cmd: String, tag: String): Boolean {
        if (!ensureRootShell()) return false
        return try {
            synchronized(rootShellLock) {
                rootShellWriter?.apply {
                    write(cmd)
                    if (!cmd.endsWith("\n")) write("\n")
                    flush()
                }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Root cmd[$tag] falló: ${e.message}; reintentaremos en próximo ciclo")
            try { rootShellProc?.destroy() } catch (_: Exception) {}
            rootShellProc = null
            rootShellWriter = null
            false
        }
    }

    private fun applyRootProtection() {
        val pkg = packageName
        val pid = android.os.Process.myPid()

        val oneShot = buildString {
            append("dumpsys deviceidle whitelist +$pkg;")
            append("cmd deviceidle whitelist +$pkg;")
            append("am set-standby-bucket $pkg active;")
            append("cmd appops set $pkg RUN_IN_BACKGROUND allow;")
            append("cmd appops set $pkg RUN_ANY_IN_BACKGROUND allow;")
            append("device_config put activity_manager max_phantom_processes 2147483647;")
            append("settings put global settings_enable_monitor_phantom_procs false;")
            append("echo -17 > /proc/$pid/oom_adj;")
            append("echo -1000 > /proc/$pid/oom_score_adj;")
            append("renice -n -20 -p $pid;")
            append("settings put system haptic_feedback_enabled 1;")
            append("settings put system vibrate_when_ringing 1;")
            append("settings put system vibrate_on 1;")
            // FIX: usar la intensidad elegida por el usuario (mapeada a la escala
            // Samsung 1..3) en lugar del hardcoded 3. Antes applyRootProtection
            // pisaba el slider al arrancar el servicio.
            val vibLvl = samsungIntensityFor(currentVibrationIntensity)
            append("settings put system haptic_feedback_intensity $vibLvl;")
            append("settings put system notification_vibration_intensity $vibLvl;")
            append("settings put system ring_vibration_intensity $vibLvl;")
            append("settings put system alarm_vibration_intensity $vibLvl;")
            append("settings put system vib_call_magnitude_v2 5;")
            append("settings put system vib_notification_magnitude_v2 5;")
            append("settings put system vib_haptic_magnitude_v2 5;")
            append("settings put system vib_system_magnitude_v2 5;")
            append("cmd notification allow_dnd $pkg;")
            append("cmd appops set android POST_NOTIFICATION ignore;")
            append("cmd appops set com.android.systemui POST_NOTIFICATION ignore;")
            append("cmd appops set com.android.systemui TOAST_WINDOW deny;")
            append("settings put global heads_up_notifications_enabled 0;")
            append("setprop persist.adb.notify 0;")
            // FIX retardo slider: dar a nuestra app WRITE_SETTINGS y
            // WRITE_SECURE_SETTINGS para poder escribir alarm_vibration_intensity
            // directamente desde Java vía Settings.System.putInt — instantáneo,
            // sin roundtrip al shell de su. Sin esto, cada cambio de intensidad
            // forzaba un `settings put` por shell + sleep 150ms para esperar al
            // shell, lo que el usuario percibía como retardo.
            append("pm grant $pkg android.permission.WRITE_SETTINGS 2>/dev/null;")
            append("pm grant $pkg android.permission.WRITE_SECURE_SETTINGS 2>/dev/null;")
            // READ_PHONE_STATE: necesario para leer la cobertura de cada SIM por API
            // (SubscriptionManager/TelephonyManager) desde el servicio headless.
            append("pm grant $pkg android.permission.READ_PHONE_STATE 2>/dev/null;")
        }
        runRootCommand(oneShot, "one-shot")

        if (immortalLoopStarted.compareAndSet(false, true)) {
            immortalScope.launch {
                // BULLETPROOF: el bucle inmortal es la red de seguridad — NUNCA debe morir.
                // - Catch Throwable (no solo Exception): protege ante OOM, StackOverflow, Error.
                // - Cada paso del trabajo va en su propio try: un fallo en runRoot no impide
                //   el reinforce, y viceversa.
                while (isActive) {
                    try {
                        delay(60_000L)
                    } catch (_: CancellationException) {
                        break // cancel del scope → salir limpio
                    } catch (_: Throwable) {
                        // delay no debería fallar de otra forma, pero por seguridad…
                        continue
                    }

                    try {
                        val curPid = android.os.Process.myPid()
                        runRootCommand(
                            "echo -17 > /proc/$curPid/oom_adj; echo -1000 > /proc/$curPid/oom_score_adj;",
                            "loop"
                        )
                    } catch (_: Throwable) {}

                    try {
                        val currentMode = currentCaptureMode
                        if (currentMode != "TEST") {
                            // Refuerzo activo: si en algún momento aplicamos silencio y el usuario
                            // NO ha pulsado el botón stop (único path que limpia audio_backup_done),
                            // mantenemos el silencio aunque systemEnabled=false. Esto cubre el caso
                            // post-crash: onCreate fuerza systemEnabled=false pero el backup sigue
                            // pendiente; sin esta línea Samsung podría revertir settings a horas vista
                            // y el móvil empezaría a sonar sin que nada lo contrarreste.
                            if (systemEnabled || sharedPrefs.getBoolean("audio_backup_done", false)) {
                                SilentModeManager.reinforceSilence(this@HeadlessUvcService, ::runRootCommand,
                                    alarmVibrationIntensity = samsungIntensityFor(currentVibrationIntensity))
                            }
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "Bucle: fallo en reinforceSilence: ${t.message}")
                    }

                    try {
                        if (systemEnabled && currentCaptureMode == "A" && !debugSinAvion) {
                            enforceRadioPolicy()
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "Bucle: fallo en enforceRadioPolicy: ${t.message}")
                    }

                    // Watchdog de ciclo: si systemEnabled sigue true pero el job murió
                    // por un fallo fatal fuera del while, relanzarlo automáticamente.
                    try {
                        if (systemEnabled && (cycleJob == null || cycleJob?.isActive != true)) {
                            ensureCorrectMode()
                        }
                    } catch (_: Throwable) {}
                }
            }
            Log.i(TAG, "Bucle inmortal lanzado (shell persistente, cada 60s) — bulletproof Throwable")

            // ── Bucle rápido de RE-MUTE (15s) ────────────────────────────────
            // En Samsung One UI 4 (Android 12) `set-master-mute` se desactiva
            // cuando el usuario toca el volumen físico, conecta auriculares,
            // o el "ringer rinse" interno. El bucle inmortal de 60s deja una
            // ventana de hasta 60s donde sí sale audio. Este bucle dedicado
            // reaplica la cadena MÍNIMA (master-mute por binder + STREAM_MUSIC=0)
            // cada 15s. Coste: 4 escrituras al shell root cada 15s — despreciable.
            immortalScope.launch {
                while (isActive) {
                    try {
                        delay(15_000L)
                    } catch (_: CancellationException) {
                        break
                    } catch (_: Throwable) {
                        continue
                    }
                    try {
                        if (currentCaptureMode != "TEST" &&
                            (systemEnabled || sharedPrefs.getBoolean("audio_backup_done", false))) {
                            SilentModeManager.fastReinforceMute(::runRootCommand)
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "Bucle rápido re-mute falló: ${t.message}")
                    }
                    // ── Guardián RF (cada 15s) ─────────────────────────────
                    // Verifica que el avión esté ON cuando NO hay ventana RF
                    // intencional abierta. Si detecta avión OFF anómalo (algún
                    // setAirplaneMode silencioso, usuario tocando el SO, otra
                    // app), fuerza ON. Esto acota CUALQUIER exposición RF
                    // inesperada a ≤15s pase lo que pase.
                    try {
                        if (systemEnabled && !debugSinAvion &&
                            (currentCaptureMode == "A" || currentCaptureMode == "C")) {
                            rfStealthGuardCheck()
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "Bucle rápido RF-guard falló: ${t.message}")
                    }
                }
            }
            Log.i(TAG, "Bucle rápido re-mute + RF-guard lanzado (cada 15s)")
        }
    }

    private fun getLongSafe(key: String, default: Long): Long {
        return try {
            sharedPrefs.getLong(key, default)
        } catch (e: ClassCastException) {
            sharedPrefs.getInt(key, default.toInt()).toLong()
        }
    }

    private fun loadSettings() {
        currentVibrationIntensity = sharedPrefs.getInt("vibration_intensity", 255)
        // cycle_duration ya NO es regulable desde ningún lado: el ciclo está
        // FIJADO a CYCLE_SECONDS_BASE (300s = 5 min, coordinado server-side con
        // stacking + Topaz + OCRs + analyzers + meta). El slider se retiró de la
        // UI. Forzamos el valor acordado e ignoramos cualquier valor stale que una
        // instalación previa (slider, o el viejo 240s) hubiera dejado en prefs
        // (migración silenciosa, igual que capture_payload→VIDEO). El AIDL
        // setCycleDurationSeconds queda dormante.
        currentCycleDelaySeconds = CYCLE_SECONDS_BASE
        if (getLongSafe("cycle_duration", CYCLE_SECONDS_BASE) != CYCLE_SECONDS_BASE) {
            sharedPrefs.edit().putLong("cycle_duration", CYCLE_SECONDS_BASE).apply()
        }
        vibrationDurationMs = getLongSafe("vibration_duration", 200L).coerceAtLeast(100L)
        intraBeepDelayMs = getLongSafe("intra_beep_delay", 400L).coerceAtLeast(100L)
        interLetterDelayMs = getLongSafe("inter_letter_delay", 800L)
        preBurstVibrationDelayMs = getLongSafe("pre_burst_vibration_delay", 0L)
        startDelayMinutes = sharedPrefs.getInt("start_delay_minutes", 0).coerceIn(0, 60)
        toggleAirplaneModeEnabled = sharedPrefs.getBoolean("toggle_airplane_mode", false)
        // Solo modo relay A en producción (sin ruta directa local).
        currentCaptureMode = "A"
        try {
            if (sharedPrefs.getString("capture_mode", "A") != "A") {
                sharedPrefs.edit().putString("capture_mode", "A").commit()
            }
        } catch (_: Throwable) {}
        // Solo VIDEO: migración forzada si quedaba BURST en prefs de versiones antiguas.
        currentCapturePayload = "VIDEO"
        try {
            if (sharedPrefs.getString("capture_payload", "VIDEO") != "VIDEO") {
                sharedPrefs.edit().putString("capture_payload", "VIDEO").commit()
            }
        } catch (_: Throwable) {}
        // Duración fija por fiabilidad operativa: siempre 5s.
        // Ignoramos cualquier valor legacy en prefs para evitar variaciones.
        videoDurationSeconds = 5
        // Rango ampliado: hasta 50 Mbps (50000 kbps) para que el usuario pueda
        // pedir calidad alta de verdad (p.ej. 15-20 Mbps a 4K). El valor que se
        // aplica en realidad es max(este pref, calculado por resolución).
        videoBitrateKbps     = sharedPrefs.getInt("video_bitrate_kbps", 1500).coerceIn(600, 50_000)
        // Restaurar estado del switch SIN MODO AVIÓN tras reinicio del servicio.
        // Si no existe la pref, default = false (comportamiento legacy: SÍ tocar avión).
        debugSinAvion        = sharedPrefs.getBoolean("debug_sin_avion", false)
        Log.i(TAG, "loadSettings: debugSinAvion restaurado a $debugSinAvion")
    }

    fun setCaptureMode(modo: String) {
        // Producción: solo relay (Modo A). Evita rutas directas locales.
        currentCaptureMode = "A"
        sharedPrefs.edit().putString("capture_mode", "A").apply()
        if (modo != "A") {
            Log.w(TAG, "setCaptureMode('$modo') ignorado: se fuerza modo A")
        } else {
            Log.i(TAG, "Modo captura (IA) confirmado → A")
        }
    }

    /** Payload fijo: VIDEO. Se ignoran valores no-video por compatibilidad AIDL. */
    fun setCapturePayload(payload: String) {
        currentCapturePayload = "VIDEO"
        sharedPrefs.edit().putString("capture_payload", "VIDEO").apply()
        if (payload != "VIDEO") {
            Log.w(TAG, "setCapturePayload('$payload') ignorado: modo FOTO/BURST deshabilitado, se fuerza VIDEO")
        } else {
            Log.i(TAG, "Payload de captura confirmado → VIDEO")
        }
    }

    fun getCapturePayload(): String = currentCapturePayload

    fun setVideoParams(durationSeconds: Int) {
        // Compatibilidad AIDL: ignoramos valores externos y forzamos 5s fijos.
        videoDurationSeconds = 5
        try { sharedPrefs.edit().putInt("video_duration_s", 5).apply() } catch (_: Throwable) {}
        Log.i(TAG, "Video params fijos: 5s (se ignora petición=$durationSeconds, fps=actualCameraFps)")
    }

    /** Ajusta el bitrate objetivo del encoder de video (en kbps).
     *  Rango ampliado: 800..50000 (0.8-50 Mbps). El valor real que se aplica
     *  es max(este pref, calculado dinámicamente por resolución/fps), para
     *  garantizar calidad mínima en resoluciones altas aunque el usuario lo
     *  baje. Persiste para que sobreviva reinicios del servicio. */
    fun setVideoBitrateKbps(kbps: Int) {
        videoBitrateKbps = kbps.coerceIn(600, 50_000)
        sharedPrefs.edit().putInt("video_bitrate_kbps", videoBitrateKbps).apply()
        Log.i(TAG, "Video bitrate actualizado → $videoBitrateKbps kbps")
    }

    /**
     * Bitrate "saludable" para H.265/H.264 según resolución y fps.
     *
     * Fórmula: bits_por_pixel_por_segundo × pixels × fps ÷ 1000 → kbps.
     *
     * Factor 0.04 bpp·s = "agresiva" para H.265 con contenido tipo texto
     * estático en pantalla móvil (preguntas tipo test). Para este caso de
     * uso (OCR de texto, no entretenimiento), HEVC con I-frames espaciados
     * comprime brutalmente: el codec ve fotogramas casi idénticos y emite
     * P-frames de pocos bytes. Android 12+ enforces un "VBR quality floor"
     * en resoluciones >320×240, así que aunque pidamos poco, el SO sube
     * automáticamente si detecta degradación → red de seguridad gratis.
     *
     * Ejemplos (con bpp=0.04):
     *  • 640×480  @ 30fps → ~370 kbps  → ~470 KB en 10 s
     *  • 1280×720 @ 30fps → ~1.1 Mbps  → ~1.4 MB en 10 s
     *  • 1920×1080 @ 30fps → ~2.5 Mbps → ~3.1 MB en 10 s
     *  • 2592×1944 @ 15fps → ~3.0 Mbps → ~3.8 MB en 10 s
     *
     * Para subir el bpp si la legibilidad se resiente, subir el slider de
     * "Bitrate del video" en el panel — el max(pref, computed) hace que el
     * pref del usuario suba el techo sin tocar este código.
     */
    private fun computeBitrateKbps(width: Int, height: Int, fps: Int): Int {
        val bpp = 0.04
        val bps = (width.toLong() * height * fps * bpp).toLong()
        // mínimo 600 kbps: incluso 320×240 mantiene texto legible a este
        // bitrate con HEVC (~1 bpp efectivo tras la compresión espacial)
        return (bps / 1000L).toInt().coerceIn(600, 50_000)
    }

    fun getVideoBitrateKbps(): Int = videoBitrateKbps

    fun forceAirplaneOff() {
        Log.i(TAG, "[DEBUG] forceAirplaneOff: desactivando avión y restaurando radios")
        val cmd = "settings put global airplane_mode_on 0;" +
            "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false;" +
            "cmd connectivity airplane-mode disable 2>/dev/null;" +
            "settings put global mobile_data 1;" +
            "settings put global mobile_data1 1;" +
            "svc data enable 2>/dev/null;" +
            "svc wifi enable;" +
            "svc bluetooth enable;"
        runRootCommand(cmd, "force-airplane-off")
    }

    fun setDebugSinAvion(enabled: Boolean) {
        debugSinAvion = enabled
        sharedPrefs.edit().putBoolean("debug_sin_avion", enabled).apply()
        Log.i(TAG, "[DEBUG] debugSinAvion → $enabled (persistido)")
    }

    fun isDebugSinAvion(): Boolean = debugSinAvion

    // Job activo del oneshot (botón "TEST"). Solo permitimos uno a la vez.
    @Volatile private var oneShotJob: Job? = null

    /**
     * Lanza UNA captura inmediata (sin esperar el ciclo de 4 min) y la envía a la
     * IA igual que el ciclo normal Modo A. Pensado para probar la API sin esperar.
     *
     * Reglas:
     *   - Solo si el USB está conectado (usbCtrlBlock != null)
     *   - Solo si no hay otra captura activa (isCycleActiveAndUsingCamera == false
     *     y oneShotJob no está corriendo)
     *   - Respeta debugSinAvion + el bitrate/duración configurados
     *   - Usa el payload actual (BURST o VIDEO)
     *
     * Devuelve true si se ha encolado, false si está ocupado o el USB no está listo.
     */
    fun triggerOneShotCapture(): Boolean {
        if (usbCtrlBlock == null) {
            Log.w(TAG, "[oneshot] Cancelado: USB no conectado")
            return false
        }
        if (iaResponseInFlight.get()) {
            if (isIaResponseStale()) {
                Log.w(TAG, "[oneshot] inFlight IA stale detectado — cancelando calls activas y limpiando candado")
                try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
                markIaResponseEnd()
            } else {
                Log.w(TAG, "[oneshot] Cancelado: aún hay una respuesta IA en curso")
                return false
            }
        }
        if (iaResponseInFlight.get()) {
            Log.w(TAG, "[oneshot] Cancelado: aún hay una respuesta IA en curso")
            return false
        }
        val existing = oneShotJob
        if (existing != null && existing.isActive) {
            Log.w(TAG, "[oneshot] Cancelado: ya hay un oneshot en curso")
            return false
        }
        if (isCycleActiveAndUsingCamera) {
            Log.w(TAG, "[oneshot] Cancelado: el ciclo periódico está capturando ahora mismo")
            return false
        }
        Log.i(TAG, "[oneshot] ▶ Disparando captura única (payload=$currentCapturePayload, debugSinAvion=$debugSinAvion)")
        oneShotJob = serviceScope.launch {
            try {
                // BUG-FIX: timeout global de 5 min para oneShot. Sin esto, si algo
                // dentro de runOneShotIteration() se cuelga (relay no responde,
                // delay() infinito por race), la coroutine vive hasta que el OS
                // mate el serviceScope — lo que aparecía en logs como
                // "StandaloneCoroutine was cancelled" sin contexto.
                // 5 min cubre con margen el peor caso real:
                //   warmup(2s) + captura(10s) + procesarVideo(~165s en oneShot
                //   con cycleSec=75s + polling=90s) + cleanup(~5s) ≈ 185s
                // El timeout extra permite reintentos internos del orquestador.
                kotlinx.coroutines.withTimeout(300_000L) {
                    runOneShotIteration()
                }
            } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                Log.e(TAG, "[oneshot] TIMEOUT global 5 min — algo se colgó", e)
                PersistentErrorLog.logError(
                    TAG, "OneShot timeout 5 min — runOneShotIteration nunca terminó", e,
                )
                try { vibrarPatronError() } catch (_: Exception) {}
            } catch (e: kotlinx.coroutines.CancellationException) {
                Log.w(TAG, "[oneshot] Coroutine cancelada (probablemente serviceScope.cancel o destroy): ${e.message}")
                PersistentErrorLog.logError(
                    TAG, "OneShot cancelado externamente: ${e.message}", e,
                )
                throw e  // re-lanzar: política coroutines (no swallow CancellationException)
            } catch (e: Exception) {
                Log.e(TAG, "[oneshot] Excepción: ${e.message}", e)
            } finally {
                oneShotJob = null
            }
        }
        return true
    }

    /** Cuerpo de la captura one-shot: idéntico al ciclo (abrir cámara hidden →
     *  vibrar señal → grabar/burst → cerrar cámara → procesar y enviar a IA).
     *  NO toca el ciclo periódico ni el flag systemEnabled. */
    private suspend fun runOneShotIteration() {
        if (iaResponseInFlight.get()) {
            if (isIaResponseStale()) {
                Log.w(TAG, "[oneshot] inFlight IA stale detectado — cancelando calls activas y limpiando candado")
                try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
                markIaResponseEnd()
            } else {
                Log.w(TAG, "[oneshot] Abortado: respuesta IA previa todavía en curso")
                return
            }
        }
        if (iaResponseInFlight.get()) {
            Log.w(TAG, "[oneshot] Abortado: respuesta IA previa todavía en curso")
            return
        }
        // Payload fijo en producción: solo VIDEO.
        val payload = "VIDEO"
        // Marcar ocupación para que ensureCorrectMode y el ciclo periódico se mantengan
        // fuera de la cámara mientras dura esta captura.
        isCycleActiveAndUsingCamera = true
        val processor = null
        var videoEncoder: VideoEncoder? = null
        var videoFile: java.io.File? = null
        try {
            vibrate(vibrationDurationMs)
            delay(vibrationDurationMs + intraBeepDelayMs)
            vibrate(vibrationDurationMs)
            if (preBurstVibrationDelayMs > 0) delay(preBurstVibrationDelayMs)

            cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
            delay(500)
            cameraMutex.withLock { openCameraLockedHidden() }
            if (!cameraOpen) {
                Log.w(TAG, "[oneshot] No se pudo abrir cámara — abortando")
                return
            }
            delay(sensorWarmupMs)

            videoFile = java.io.File(cacheDir, "vid_${System.currentTimeMillis()}.mp4")
            // FPS del encoder = FPS REAL negociado con la cámara para la
            // resolución activa (lo rellenó pickBestPreviewMode al abrir).
            // No hay configuración manual — siempre vamos al máximo soportado.
            val encFps = actualCameraFps.coerceAtLeast(5)
            // Bitrate dinámico — escala con resolución y fps para mantener
            // calidad alta. Factor 0.18 bpp·s es generoso (≈ "high quality"
            // para H.265). Se sobreescribe por el override del usuario si
            // éste pidió MÁS bitrate del calculado (nunca menos).
            val computedKbps = computeBitrateKbps(actualWidth, actualHeight, encFps)
            val effectiveKbps = maxOf(videoBitrateKbps, computedKbps).coerceAtMost(50_000)
            videoEncoder = try {
                VideoEncoder(
                    outputFile  = videoFile,
                    width       = actualWidth,
                    height      = actualHeight,
                    fps         = encFps,
                    bitrateKbps = effectiveKbps,
                    preferHevc  = true,
                )
            } catch (e: Exception) {
                Log.e(TAG, "[oneshot] VideoEncoder error: ${e.message}", e)
                // Si el encoder no se pudo crear (codec no soporta esta combinación
                // de res/fps/bitrate), borramos el archivo vacío de cache para no
                // acumular basura ciclo tras ciclo en /data/data/.../cache/.
                try { videoFile?.delete() } catch (_: Exception) {}
                videoFile = null
                null
            }
            val enc = videoEncoder
            if (enc != null) {
                Log.i(TAG, "[oneshot][video] grabando ${actualWidth}x${actualHeight}@${encFps}fps · ${effectiveKbps} kbps (calc=$computedKbps · pref=$videoBitrateKbps)")
                captureVideoFrames(enc, videoDurationSeconds, encFps)
            }
        } finally {
            withContext(NonCancellable) {
                delay(200)
                cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
            }
            isCycleActiveAndUsingCamera = false
            // Procesar y enviar a la IA. Se hace FUERA del scope que mantiene
            // la cámara abierta para no bloquear futuras capturas.
            val encFinal  = videoEncoder
            val fileFinal = videoFile
            val procFinal = processor
            if (payload == "VIDEO" && encFinal != null && fileFinal != null) {
                if (!iaResponseInFlight.compareAndSet(false, true)) {
                    Log.w(TAG, "[oneshot] Saltando envío: respuesta IA previa sigue en curso")
                } else {
                    markIaResponseStart()
                    try {
                        // isOneShot=true → fuerza debugSinAvion en el orquestador. El botón TEST
                        // NUNCA debe activar modo avión: es para probar la captura+IA, no para
                        // operación real. La exposición RF debe quedar a 0 ventanas.
                        processVideoAndSend(encFinal, fileFinal, isOneShot = true)
                    } catch (e: Exception) {
                        Log.e(TAG, "[oneshot] processVideoAndSend falló: ${e.message}", e)
                    } finally {
                        markIaResponseEnd()
                    }
                }
            }
            broadcastCameraState(CameraState.ReadyForCapture)
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // MODO TEST CONTINUO DE VIDEO
    // ────────────────────────────────────────────────────────────────────────
    // Loop de verificación del pipeline de video. Reproduce la pipeline de
    // producción (open cámara hidden → VideoEncoder NV12 → setFrameCallback →
    // captureVideoFrames → close) en bucle, SIN tocar IA, radios, silencio,
    // ni broadcasts hacia la UI de respuestas. Cada MP4 se guarda en la
    // galería bajo Movies/UVC_TEST/ para que el usuario pueda comprobar
    // visualmente que no salen frames grises.
    //
    // Tag distintivo en logcat: [video-test].
    // Pensado para verificar el fix del stride padding del Exynos 9820 (S10e)
    // observando 30-60 grabaciones seguidas + abriendo los MP4 en la galería.
    @Volatile private var videoTestJob: Job? = null
    private val videoTestStopFlag = AtomicBoolean(false)

    fun isVideoTestLoopRunning(): Boolean {
        val j = videoTestJob
        return j != null && j.isActive
    }

    /** Toggle on/off. Devuelve true si quedó CORRIENDO tras la llamada. */
    fun toggleVideoTestLoop(): Boolean {
        val existing = videoTestJob
        if (existing != null && existing.isActive) {
            Log.i(TAG, "[video-test] ⏹  Deteniendo loop por petición del usuario")
            videoTestStopFlag.set(true)
            existing.cancel()
            videoTestJob = null
            return false
        }
        if (usbCtrlBlock == null) {
            Log.w(TAG, "[video-test] ✗ No se puede iniciar: USB no conectado")
            return false
        }
        if (isCycleActiveAndUsingCamera || isCapturingForBurst.get() || isCapturingForVideo.get()) {
            Log.w(TAG, "[video-test] ✗ No se puede iniciar: ya hay una captura en curso (ciclo/burst/video)")
            return false
        }
        if (systemEnabled) {
            Log.w(TAG, "[video-test] ✗ No se puede iniciar: el sistema principal está activo — desactívalo primero")
            return false
        }
        val one = oneShotJob
        if (one != null && one.isActive) {
            Log.w(TAG, "[video-test] ✗ No se puede iniciar: hay un oneshot en curso")
            return false
        }
        videoTestStopFlag.set(false)
        videoTestJob = serviceScope.launch {
            try {
                runVideoTestLoop()
            } catch (e: CancellationException) {
                Log.i(TAG, "[video-test] Loop cancelado limpio")
            } catch (e: Throwable) {
                Log.e(TAG, "[video-test] Loop terminó con excepción: ${e.message}", e)
            } finally {
                videoTestJob = null
                isCycleActiveAndUsingCamera = false
                isCapturingForVideo.set(false)
            }
        }
        Log.i(TAG, "[video-test] ▶  Loop INICIADO (sin IA, sin radios, sin silencio). Logcat: filtro [video-test]")
        return true
    }

    /** Una iteración = abrir cámara hidden + grabar videoDurationSeconds + cerrar + copiar a galería. */
    private suspend fun runVideoTestLoop() = coroutineScope {
        val outDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
            "UVC_TEST",
        )
        try { outDir.mkdirs() } catch (_: Throwable) {}
        Log.i(TAG, "[video-test] Carpeta destino: ${outDir.absolutePath}")

        var iter = 0
        var iterOk = 0
        var iterFail = 0
        val tLoopStart = System.currentTimeMillis()

        while (isActive && !videoTestStopFlag.get()) {
            iter++
            val tIterStart = System.currentTimeMillis()
            Log.i(TAG, "[video-test] ───── Iteración #$iter (USB=${usbCtrlBlock != null}) ─────")

            if (usbCtrlBlock == null) {
                Log.w(TAG, "[video-test] USB no conectado; esperando 2 s y reintentando")
                delay(2000)
                continue
            }

            isCycleActiveAndUsingCamera = true
            var encoder: VideoEncoder? = null
            var rawFile: File? = null

            try {
                // Cerrar cualquier estado previo de cámara (igual que producción).
                cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
                delay(300)
                cameraMutex.withLock { openCameraLockedHidden() }
                if (!cameraOpen) {
                    Log.w(TAG, "[video-test] No se pudo abrir cámara — esperando 1 s y reintentando")
                    iterFail++
                    delay(1000)
                    continue
                }
                Log.i(TAG, "[video-test] Cámara abierta hidden · ${actualWidth}x${actualHeight}@${actualCameraFps}fps · warmup ${sensorWarmupMs}ms")
                delay(sensorWarmupMs)

                rawFile = File(cacheDir, "vidtest_${System.currentTimeMillis()}.mp4")
                val encFps = actualCameraFps.coerceAtLeast(5)
                val computedKbps = computeBitrateKbps(actualWidth, actualHeight, encFps)
                val effectiveKbps = maxOf(videoBitrateKbps, computedKbps).coerceAtMost(50_000)

                encoder = try {
                    VideoEncoder(
                        outputFile = rawFile,
                        width = actualWidth,
                        height = actualHeight,
                        fps = encFps,
                        bitrateKbps = effectiveKbps,
                        preferHevc = true,
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "[video-test] VideoEncoder constructor falló: ${e.message}", e)
                    try { rawFile.delete() } catch (_: Throwable) {}
                    rawFile = null
                    null
                }

                if (encoder == null) {
                    iterFail++
                } else {
                    Log.i(TAG, "[video-test] Grabando ${videoDurationSeconds}s · codec=${encoder.actualMime} · ${effectiveKbps} kbps")
                    captureVideoFrames(encoder, videoDurationSeconds, encFps)
                }
            } catch (e: CancellationException) {
                Log.i(TAG, "[video-test] Iteración #$iter cancelada (stop solicitado)")
                throw e
            } catch (e: Throwable) {
                iterFail++
                Log.e(TAG, "[video-test] Iteración #$iter excepción: ${e.message}", e)
            } finally {
                withContext(NonCancellable) {
                    try {
                        delay(120)
                        cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
                    } catch (e: Throwable) {
                        Log.w(TAG, "[video-test] closeCamera falló: ${e.message}")
                    }
                    // Cerrar encoder y mover el MP4 a la galería para inspección visual.
                    val mp4 = try { encoder?.finish() } catch (e: Throwable) {
                        Log.w(TAG, "[video-test] encoder.finish falló: ${e.message}"); null
                    }
                    if (mp4 != null && mp4.exists() && mp4.length() > 0) {
                        try {
                            val dest = File(outDir, "UVC_TEST_${System.currentTimeMillis()}.mp4")
                            mp4.inputStream().use { input ->
                                FileOutputStream(dest).use { output -> input.copyTo(output) }
                            }
                            try {
                                MediaScannerConnection.scanFile(
                                    this@HeadlessUvcService,
                                    arrayOf(dest.absolutePath),
                                    arrayOf("video/mp4"),
                                    null,
                                )
                            } catch (_: Throwable) {}
                            val durMs = System.currentTimeMillis() - tIterStart
                            Log.i(TAG, "[video-test] ✓ Iter #$iter OK · ${dest.name} · ${dest.length() / 1024} KB · ${durMs} ms total")
                            iterOk++
                        } catch (e: Throwable) {
                            Log.w(TAG, "[video-test] No se pudo copiar MP4 a galería: ${e.message}")
                            iterFail++
                        }
                    } else {
                        Log.w(TAG, "[video-test] ✗ Iter #$iter: encoder devolvió MP4 vacío o nulo")
                        if (iter == 1 || iter % 5 == 0) iterFail++  // no inflar el contador por reintentos rápidos
                    }
                    try { rawFile?.delete() } catch (_: Throwable) {}

                    // Rotar carpeta: conservar solo los últimos 10 MP4 (~50-80 MB
                    // típico). Se acumularían varios GB en 1 hora si no rotamos.
                    try {
                        val files = outDir.listFiles { f -> f.name.startsWith("UVC_TEST_") && f.name.endsWith(".mp4") }
                            ?.sortedByDescending { it.lastModified() } ?: emptyList()
                        files.drop(10).forEach { try { it.delete() } catch (_: Throwable) {} }
                    } catch (_: Throwable) {}

                    isCycleActiveAndUsingCamera = false
                }
            }

            // Pequeño respiro entre iteraciones (idéntico al respiro entre
            // bursts del ciclo de producción: deja al bus USB y al codec
            // descansar 400 ms antes del siguiente abrir cámara).
            if (!videoTestStopFlag.get() && isActive) {
                delay(400)
            }
        }

        val total = System.currentTimeMillis() - tLoopStart
        Log.i(TAG, "[video-test] ⏹  Loop terminado · iters=$iter · ok=$iterOk · fail=$iterFail · total=${total / 1000}s")
    }

    fun setToggleAirplaneModeEnabled(enabled: Boolean) {
        toggleAirplaneModeEnabled = enabled
        sharedPrefs.edit().putBoolean("toggle_airplane_mode", enabled).apply()
        Log.i(TAG, "Automatización de Modo Avión cambiada a: $enabled")
    }

    /**
     * Activa o desactiva el modo avión con verificación del setting global y
     * hasta 3 reintentos. Devuelve true SOLO si el setting confirmado coincide
     * con [enable]. False significa "el setting no cambió" — el caller debe
     * actuar (vibrar error, persistir el fallo) porque la RF-stealth puede
     * estar comprometida si esperábamos enable=true.
     *
     * FIX DE FIABILIDAD: antes era `Unit` y solo loggeaba `Log.e` cuando los 3
     * intentos se agotaban. El sistema seguía operando como si todo hubiera
     * ido bien — la radio podía quedar expuesta sin que el usuario se enterara
     * salvo abriendo logcat. Ahora propagamos el resultado al caller Y
     * persistimos a PersistentErrorLog para que aparezca en "VER ERRORES".
     */
    private suspend fun setAirplaneMode(enable: Boolean): Boolean {
        val modo = currentCaptureMode

        if (enable && modo == "TEST") {
            Log.i(TAG, "setAirplaneMode(true): Modo TEST activo → ignorado.")
            return true
        }

        Log.i(TAG, "setAirplaneMode($enable): modo=$modo, buscando shell root…")
        if (!ensureRootShell()) {
            Log.e(TAG, "setAirplaneMode($enable): NO hay acceso ROOT.")
            PersistentErrorLog.logError(
                TAG, "setAirplaneMode($enable) sin acceso root — radio queda en estado actual" +
                     (if (enable) " (RF-STEALTH COMPROMETIDA)" else ""),
            )
            return false
        }

        val cmd = if (enable) {
            // Samsung Android 12+: cmd connectivity airplane-mode enable no corta datos móviles.
            // Doble seguro con settings + svc data disable.
            "settings put global airplane_mode_on 1;" +
            "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state true;" +
            "cmd connectivity airplane-mode enable 2>/dev/null;" +
            "settings put global mobile_data 0;" +
            "settings put global mobile_data1 0;" +
            "svc data disable 2>/dev/null;" +
            NetworkControlManager.DISABLE_NON_CELLULAR_RADIOS_CMD
        } else {
            "settings put global airplane_mode_on 0;" +
            "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false;" +
            "cmd connectivity airplane-mode disable 2>/dev/null;" +
            "svc wifi enable;" +
            "svc bluetooth enable;" +
            "settings put global mobile_data 1;" +
            "settings put global mobile_data1 1;" +
            "svc data enable 2>/dev/null;"
        }

        // Reintentamos hasta 3 veces verificando que el setting global se haya
        // aplicado. Samsung One UI a veces ignora silenciosamente el primer cmd
        // (race con DataConnectionTracker), un segundo intento tras 300 ms suele
        // pegar. Sin esto teníamos casos en producción donde "avión enviado OK"
        // logueaba pero el setting seguía sin cambiar y todo el ciclo de radio
        // se rompía sin diagnóstico.
        val expected = if (enable) 1 else 0
        repeat(3) { attempt ->
            try {
                Log.i(TAG, "setAirplaneMode($enable): intento ${attempt + 1}/3 → enviando cmd")
                runRootCommand(cmd, "airplane-$attempt")
            } catch (t: Throwable) {
                Log.w(TAG, "setAirplaneMode($enable): excepción en intento ${attempt + 1}: ${t.message}")
            }
            // Esperar a que el setting se propague antes de verificar.
            try { delay(300L) } catch (_: Throwable) {}
            val current = try {
                Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, -1)
            } catch (t: Throwable) { Log.w(TAG, "setAirplaneMode($enable): no se pudo leer setting: ${t.message}"); -1 }
            if (current == expected) {
                Log.i(TAG, "setAirplaneMode($enable): VERIFICADO (setting=$current) en intento ${attempt + 1}")
                return true
            }
            Log.w(TAG, "setAirplaneMode($enable): tras intento ${attempt + 1} setting=$current (esperado=$expected) — reintentando")
        }
        Log.e(TAG, "setAirplaneMode($enable): 3 intentos agotados, setting sigue sin coincidir con esperado=$expected.")
        PersistentErrorLog.logError(
            TAG, "setAirplaneMode($enable) FALLÓ tras 3 intentos · setting global no cambió. " +
                 (if (enable) "RF-STEALTH COMPROMETIDA — la radio puede haber quedado expuesta al activar el sistema."
                  else "La radio puede seguir en avión cuando debería estar ON."),
        )
        return false
    }

    /**
     * Guardián RF: garantiza que el modo avión NUNCA quede OFF fuera de una
     * ventana de envío/recogida intencional. Lo invoca el bucle rápido del
     * servicio cada 15s. Si detecta avión=0 cuando el orquestador NO está en
     * mitad de un ciclo de radio (openRfWindows == 0), fuerza setAirplaneMode
     * (true) y persiste el incidente a "VER ERRORES".
     *
     * Por qué este guardián existe: en producción con Samsung One UI 4 se
     * observaron casos donde `settings put global airplane_mode_on 1` completaba
     * pero el setting NO cambiaba (race con DataConnectionTracker, ringer rinse,
     * etc.). Antes el cliente registraba el ciclo como "RF stealth OK" mientras
     * la radio seguía emitiendo hasta el SIGUIENTE ciclo (5 min después).
     * Inaceptable para el caso de uso. Ahora la exposición queda acotada a ≤15s
     * GARANTIZADO pase lo que pase aguas arriba.
     *
     * Coste por tick: 1 lectura de Settings.Global.AIRPLANE_MODE_ON (memoria
     * del SO, ~µs). Solo entra al toggle si REALMENTE detecta avión OFF anómalo.
     */
    private suspend fun rfStealthGuardCheck() {
        // Si el orquestador está en mitad de envío/recogida, avión OFF es
        // ESPERADO — no tocamos. El contador se decrementa en finally cuando
        // avionOn termine (con o sin éxito).
        if (BolsilloIaOrquestador.isAnyRfWindowOpen) return
        val current = try {
            Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, -1)
        } catch (_: Throwable) { return }
        if (current == 1) return  // avión confirmado ON, todo bien
        Log.w(TAG, "[RF-guard] avión OFF cuando NO hay ventana intencional (setting=$current) — forzando ON")
        try {
            PersistentErrorLog.logError(
                TAG, "Guardián RF: avión OFF inesperado (setting=$current) — reactivando. " +
                     "Causa posible: setAirplaneMode silencioso por race del SO, otra app que lo desactivó, " +
                     "o usuario que tocó el toggle del SO mientras el sistema estaba activo.",
            )
        } catch (_: Throwable) {}
        // setAirplaneMode internamente hace 3 reintentos con verificación. Si
        // sigue fallando aquí, el siguiente tick del guardián (15s) lo
        // reintentará — no quedamos sin red de seguridad.
        setAirplaneMode(true)
    }

    private fun enforceRadioPolicy() {
        val modo = currentCaptureMode
        val armed = sharedPrefs.getBoolean("rf_armed", false)
        Log.i(TAG, "enforceRadioPolicy: modo=$modo, rf_armed=$armed")
        if (modo == "TEST") {
            Log.i(TAG, "enforceRadioPolicy: modo TEST → sin acción.")
            return
        }
        if (!armed) {
            Log.w(TAG, "enforceRadioPolicy: rf_armed=false → sin acción (race condition?).")
            return
        }
        Log.i(TAG, "enforceRadioPolicy: enviando DISABLE_NON_CELLULAR_RADIOS_CMD")
        runRootCommand(NetworkControlManager.DISABLE_NON_CELLULAR_RADIOS_CMD, "radio-policy")
        Log.i(TAG, "enforceRadioPolicy: cmd enviado.")
    }

    fun setPocketMode(enabled: Boolean) {
    }

    fun setAppInForeground(isForeground: Boolean) {
        if (isAppInForeground != isForeground) {
            isAppInForeground = isForeground
            Log.i(TAG, "App en primer plano cambiado a: $isForeground")
            updateInteractiveState()
        }
    }

    private fun updateInteractiveState() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val isScreenOn = pm.isInteractive

        val newInteractive = isScreenOn && isAppInForeground

        if (screenInteractive != newInteractive) {
            screenInteractive = newInteractive
            Log.i(TAG, "Estado interactivo cambiado a: $screenInteractive (Pantalla=$isScreenOn, Foreground=$isAppInForeground)")

            if (newInteractive) {
                pendingDeactivateJob?.cancel()
                pendingDeactivateJob = null
                ensureCorrectMode()
            } else {
                pendingDeactivateJob?.cancel()
                val sinceOpenMs = System.currentTimeMillis() - lastCameraOpenAtMs
                val graceMs = 400L
                val waitMs = (graceMs - sinceOpenMs).coerceAtLeast(0L)
                if (waitMs == 0L) {
                    pendingDeactivateJob = null
                    ensureCorrectMode()
                } else {
                    pendingDeactivateJob = serviceScope.launch {
                        try {
                            delay(waitMs)
                            if (!screenInteractive) ensureCorrectMode()
                        } catch (_: CancellationException) {
                        }
                    }
                }
            }
        }
    }

    override fun onConnect(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?, createNew: Boolean) {
        if (ctrlBlock == null) return
        // Encadenamos a usbJob: si hay un disconnect/cleanup en curso, esperamos
        // a que termine antes de abrir cámara nueva. Sin esto, una reconexión
        // <200 ms (cable con mal contacto) puede ejecutar onConnect antes del
        // cleanup → ctrlBlock viejo cerrado mientras uvcCamera aún lo apunta.
        val previous = usbJob
        usbJob = serviceScope.launch {
            try { previous?.join() } catch (_: Throwable) {}
            cameraMutex.withLock {
                // FIX CRÍTICO-2: si ya hay un ciclo activo con su propia cámara
                // abierta vía openCameraLockedHidden, NO la sobrescribimos: sólo
                // actualizamos la referencia al ctrlBlock para que el siguiente
                // ciclo lo encuentre, y dejamos que el ciclo actual termine.
                if (isCycleActiveAndUsingCamera || cameraOpen) {
                    Log.w(TAG, "[usb] onConnect ignorado (cycle activo o cámara abierta) — solo actualizo ctrlBlock")
                    usbCtrlBlock = ctrlBlock
                    return@withLock
                }
                usbCtrlBlock = ctrlBlock
                actualWidth = 1280
                actualHeight = 720
                if (screenInteractive) {
                    if (!openCameraLocked()) {
                        broadcastCameraState(CameraState.Error("Open failed"))
                        return@withLock
                    }
                }
                broadcastCameraState(CameraState.ReadyForCapture)
            }
            ensureCorrectMode()
        }
    }

    override fun onFrame(frame: ByteBuffer?) {
        if (frame == null) return

        // Ruta VIDEO: bytes NV12 (Y plane + UV interleaved, w·h·1.5) → encoder.
        // Tiene prioridad sobre burst y NUNCA recae en el path RGBX de abajo: el
        // formato de bytes es distinto y el size check del burst los descartaría
        // silenciosamente, perdiendo el frame.
        if (isCapturingForVideo.get()) {
            // try/catch envolvente OBLIGATORIO: este callback corre en un hilo
            // JNI de libuvc. Una excepción no capturada propaga al código nativo
            // y aborta el proceso :uvc completo (CheckJNI). Cualquier fallo en
            // scoring/encoder debe quedar contenido aquí.
            try {
                // 0) KILL SWITCH: si la cámara lleva tapada >3s, dejamos de
                // alimentar el encoder. Los frames que sigan llegando se
                // descartan silenciosamente. El loop de captureVideoFrames
                // detectará el flag en su tick de 200ms y cerrará la cámara
                // antes de que la radio toque el aire.
                try {
                    updateCameraCoveredState(frame, actualWidth, actualHeight)
                } catch (_: Throwable) { /* defensive: jamás propagar al JNI */ }
                if (cameraCoveredAbort) return

                val enc = videoEncoderRef
                if (enc != null) {
                    // 1) Sharpness UNA SOLA VEZ (Laplaciano franja central 200×200)
                    val sharpness = try {
                        laplacianVarianceCenter(frame, actualWidth, actualHeight)
                    } catch (_: Throwable) { 0.0 }

                    // 2) Best-frame scoring (con su propio try interno por si
                    //    OpenCV o el memcpy fallan en condiciones adversas).
                    try {
                        updateBestFrameIfBetter(frame, actualWidth, actualHeight, sharpness)
                    } catch (e: Throwable) {
                        Log.w(TAG, "[video] updateBestFrameIfBetter falló: ${e.message}")
                    }

                    // 3) Filtro de nitidez para el MP4 (mediana móvil + safety net).
                    val pass = try {
                        if (sharpness <= 0) true else shouldEncodeForSharpness(sharpness)
                    } catch (_: Throwable) { true }

                    if (pass) {
                        if (enc.queueNv12(frame)) videoFramesQueued++ else videoFramesDropped++
                    } else {
                        videoFramesFilteredBlur++
                    }
                }
            } catch (e: Throwable) {
                // Última red de seguridad. Si llegamos aquí, algo muy raro pasó;
                // logueamos pero NO dejamos que la excepción salga del callback.
                Log.e(TAG, "[video] onFrame ruta video lanzó excepción (contenida): ${e.message}", e)
            }
            return
        }

        if (!isCapturingForBurst.get()) return

        try {
            val w = actualWidth
            val h = actualHeight
            val expectedSize = w * h * 4
            if (frame.remaining() < expectedSize) return

            // KILL SWITCH ruta BURST: si el usuario tapa la cámara >3s durante
            // el burst, dejamos de encolar bitmaps al processor. processBurstAndSend
            // detecta el flag y salta el envío a la IA SIN hacer stacking ni HTTP
            // (idéntica semántica que la ruta de vídeo). Antes solo se miraba el
            // JPEG final con isBrazoTapandoCamara (umbral demasiado laxo) y el
            // ciclo se enviaba igual aunque la cámara estuviera tapada.
            try {
                updateCameraCoveredStateRgbx(frame, w, h)
            } catch (_: Throwable) { /* defensive: jamás propagar al JNI */ }
            if (cameraCoveredAbort) return

            // libuvc entrega bytes en orden R,G,B,X (X=padding, suele ser 0 sin alpha).
            // Bitmap.Config.ARGB_8888 + copyPixelsFromBuffer espera bytes en formato
            // de int (alpha en MSB) → si pasamos RGBX directamente el byte X (0)
            // acaba como ALPHA → la imagen queda transparente y se ve gris (color
            // del fondo del Surface). Convertimos explícitamente a ARGB con A=255.
            frame.rewind()
            val pixels = IntArray(w * h)
            val ib = frame.order(java.nio.ByteOrder.LITTLE_ENDIAN).asIntBuffer()
            ib.get(pixels)
            // En little-endian un int leído de bytes (R,G,B,X) es: X<<24 | B<<16 | G<<8 | R.
            // Queremos ARGB int = 0xFF<<24 | R<<16 | G<<8 | B → reordenamos por canal.
            val opaque = 0xFF shl 24
            for (i in pixels.indices) {
                val v = pixels[i]
                val r = v and 0xff
                val g = (v shr 8) and 0xff
                val b = (v shr 16) and 0xff
                pixels[i] = opaque or (r shl 16) or (g shl 8) or b
            }
            val bitmap = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)

            if (!frameChannel.trySend(bitmap).isSuccess) bitmap.recycle()
        } catch (e: Throwable) {
            // No dejar escapar errores (incl. OOM) al hilo JNI de libuvc.
            Log.e(TAG, "[burst] onFrame lanzó (contenido): ${e.message}", e)
        }
    }

    /**
     * Calcula la varianza del Laplaciano sobre una franja central 200×200 del
     * plano Y del NV12. Es una métrica clásica de nitidez: más varianza =
     * bordes más marcados = imagen más enfocada. Devuelve 0.0 si el frame es
     * demasiado pequeño para la franja.
     *
     * Mira sólo el plano Y (primeros w·h bytes del NV12) — el chroma no aporta
     * info de nitidez en texto. Y sólo en una franja central para no gastar
     * CPU procesando bordes que típicamente tienen distorsión óptica.
     *
     * Coste: ~40 K iteraciones · ~10 ns/iter ≈ 0.4 ms por frame en un S10e.
     * Despreciable comparado con la cadencia de 30 fps (33 ms entre frames).
     */
    private fun laplacianVarianceCenter(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        val W = 200; val H = 200
        if (frameW < W + 2 || frameH < H + 2) return 0.0
        val startX = (frameW - W) / 2
        val startY = (frameH - H) / 2
        val basePos = buf.position()  // donde empieza el plano Y
        val rowStride = frameW

        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        // Bordes evitados (i=0 y i=H-1 no tienen vecino arriba/abajo dentro)
        for (y in 1 until H - 1) {
            val py = startY + y
            val rowPos = basePos + py * rowStride + startX
            for (x in 1 until W - 1) {
                val pos = rowPos + x
                val c   = buf.get(pos).toInt()             and 0xff
                val top = buf.get(pos - rowStride).toInt() and 0xff
                val bot = buf.get(pos + rowStride).toInt() and 0xff
                val lef = buf.get(pos - 1).toInt()         and 0xff
                val rig = buf.get(pos + 1).toInt()         and 0xff
                val lap = (4 * c - top - bot - lef - rig).toDouble()
                sum   += lap
                sumSq += lap * lap
                count++
            }
        }
        if (count == 0) return 0.0
        val mean = sum / count
        return (sumSq / count) - mean * mean
    }

    /**
     * Detección "cámara tapada": calcula la luminancia media del plano Y de UN
     * frame NV12 muestreando ~256 puntos en grilla. Devuelve un valor 0..255.
     *
     * NV12: primeros frameW·frameH bytes son Y (luminancia). Una cámara con un
     * brazo encima produce Y casi 0 en TODO el plano; una habitación a oscuras
     * normal raramente baja de 40-50 con AGC alto.
     *
     * Coste: ~256 lecturas · ~5 ns ≈ 1-2 µs por frame. Despreciable.
     */
    private fun computeFrameBrightnessY(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        val basePos = buf.position()
        val stride = frameW
        // Muestreo en grilla 16×16 = 256 puntos. Mucho más rápido que recorrer
        // todos los píxeles y suficientemente representativo para "tapada vs no".
        val gridN = 16
        val stepX = (frameW / gridN).coerceAtLeast(1)
        val stepY = (frameH / gridN).coerceAtLeast(1)
        var sum = 0L
        var count = 0
        var y = stepY / 2
        while (y < frameH) {
            var x = stepX / 2
            while (x < frameW) {
                sum += buf.get(basePos + y * stride + x).toInt() and 0xff
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count > 0) sum.toDouble() / count else 255.0
    }

    /**
     * Variante RGBX para la ruta de BURST. libuvc entrega bytes en orden R,G,B,X
     * (4 bytes por píxel). Calculamos luminancia ITU-R BT.601 = 0.299·R + 0.587·G
     * + 0.114·B sobre un muestreo 16×16 = 256 píxeles. Sirve al mismo propósito
     * que computeFrameBrightnessY pero sin necesitar plano Y separado.
     *
     * Coste: ~256·3 lecturas ≈ 4 µs por frame. Despreciable.
     */
    private fun computeFrameBrightnessRgbx(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        val basePos = buf.position()
        val rowBytes = frameW * 4
        val gridN = 16
        val stepX = (frameW / gridN).coerceAtLeast(1)
        val stepY = (frameH / gridN).coerceAtLeast(1)
        var sum = 0L
        var count = 0
        var y = stepY / 2
        while (y < frameH) {
            var x = stepX / 2
            while (x < frameW) {
                val off = basePos + y * rowBytes + x * 4
                val r = buf.get(off).toInt() and 0xff
                val g = buf.get(off + 1).toInt() and 0xff
                val b = buf.get(off + 2).toInt() and 0xff
                // ITU-R BT.601 luma · enteros para evitar float en hot path.
                sum += (r * 299 + g * 587 + b * 114) / 1000
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count > 0) sum.toDouble() / count else 255.0
    }

    /**
     * Llamado desde onFrame() en la ruta de video. Mantiene la racha de frames
     * "dark" y, si supera CAMERA_COVERED_THRESHOLD_MS consecutivos, activa el
     * flag de aborto. Idempotente: una vez seteado el flag, no se vuelve a
     * resetear hasta el próximo `captureVideoFrames` (reset explícito al inicio).
     *
     * Devuelve true si acabamos de activar el flag (para loguearlo UNA vez).
     */
    private fun updateCameraCoveredState(buf: ByteBuffer, frameW: Int, frameH: Int): Boolean {
        if (cameraCoveredAbort) return false  // ya está activado, nada que hacer
        val avgY = try {
            computeFrameBrightnessY(buf, frameW, frameH)
        } catch (_: Throwable) { return false }
        val now = System.currentTimeMillis()
        if (avgY <= CAMERA_COVERED_Y_THRESHOLD) {
            if (darkFrameRunStartMs == 0L) {
                darkFrameRunStartMs = now
                darkFrameRunCount = 1
            } else {
                darkFrameRunCount++
                if (now - darkFrameRunStartMs >= CAMERA_COVERED_THRESHOLD_MS) {
                    cameraCoveredAbort = true
                    Log.w(TAG, "[video] 🛑 KILL SWITCH activado: cámara tapada >${CAMERA_COVERED_THRESHOLD_MS}ms " +
                            "(${darkFrameRunCount} frames con Y<=${CAMERA_COVERED_Y_THRESHOLD}, avgY=%.1f)".format(avgY))
                    return true
                }
            }
        } else {
            // El frame está iluminado → reset de la racha. Permitimos un "blink"
            // momentáneo sin penalizar. La condición de abort requiere 3s SEGUIDOS.
            if (darkFrameRunStartMs != 0L) {
                darkFrameRunStartMs = 0L
                darkFrameRunCount = 0
            }
        }
        return false
    }

    /**
     * Equivalente a updateCameraCoveredState pero leyendo el frame en formato
     * RGBX (la ruta de BURST recibe esos bytes, no NV12). Comparte el mismo
     * estado de racha y el mismo flag `cameraCoveredAbort` — así el orden de
     * los modos (foto vs vídeo) no cambia la semántica del kill switch.
     *
     * Antes el burst NO tenía detección DURANTE la captura: solo se miraba el
     * JPEG final tras stacking, y `isBrazoTapandoCamara` usaba un umbral muy
     * laxo (<1% píxeles >80 brillo) que el AGC del sensor con cámara tapada
     * superaba por ruido → el ciclo se enviaba a la IA igualmente.
     */
    private fun updateCameraCoveredStateRgbx(buf: ByteBuffer, frameW: Int, frameH: Int): Boolean {
        if (cameraCoveredAbort) return false
        val avgY = try {
            computeFrameBrightnessRgbx(buf, frameW, frameH)
        } catch (_: Throwable) { return false }
        val now = System.currentTimeMillis()
        if (avgY <= CAMERA_COVERED_Y_THRESHOLD) {
            if (darkFrameRunStartMs == 0L) {
                darkFrameRunStartMs = now
                darkFrameRunCount = 1
            } else {
                darkFrameRunCount++
                if (now - darkFrameRunStartMs >= CAMERA_COVERED_THRESHOLD_MS) {
                    cameraCoveredAbort = true
                    Log.w(TAG, "[burst] 🛑 KILL SWITCH activado: cámara tapada >${CAMERA_COVERED_THRESHOLD_MS}ms " +
                            "(${darkFrameRunCount} frames con luma<=${CAMERA_COVERED_Y_THRESHOLD}, avg=%.1f)".format(avgY))
                    return true
                }
            }
        } else {
            if (darkFrameRunStartMs != 0L) {
                darkFrameRunStartMs = 0L
                darkFrameRunCount = 0
            }
        }
        return false
    }

    /**
     * Decide si el frame se encolca al encoder. Política:
     *  1. Primeros SHARP_HISTORY_SIZE frames pasan SIEMPRE — necesarios para
     *     llenar el buffer estadístico y para que el encoder tenga keyframes.
     *  2. Después: calcula sharpness y compara contra la mediana móvil. Si
     *     está bajo `mediana × 0.6`, se descarta como "borroso".
     *  3. Safety: si llevamos > MAX_CONSECUTIVE_DROPS seguidos descartados,
     *     dejamos pasar el siguiente para no quedar con MP4 vacío.
     *
     * El frame SIEMPRE se contabiliza en el histórico aunque se descarte —
     * así el umbral se adapta también a "rachas de borrosidad" en lugar de
     * fijarse en una mediana inflada que excluye los descartes.
     */
    /**
     * Mide cómo "completo" se ve el folio en el frame, a partir de la
     * distribución de luminancias del plano Y. Devuelve 0..1 (más alto = más
     * probable que el folio entero esté en cuadro con texto visible).
     *
     * Casos:
     *  - Cámara MUY cerca del folio → todo blanco, casi nada oscuro → score bajo
     *  - Cámara LEJOS / fuera del folio → poco blanco, mucha cosa oscura → score bajo
     *  - Cámara a buena distancia → bimodal (blanco del folio + oscuro del texto/borde) → score alto
     *
     * Métrica: porcentaje de pixeles "oscuros" (Y < 80, sobre 255). Para folio
     * blanco con texto típico, esto suele caer en 8-25%. Fuera de ese rango
     * penalizamos linealmente.
     */
    private fun completenessScore(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        val basePos = buf.position()
        val yPlaneSize = frameW * frameH
        // Submuestreo: 1 de cada 16 pixels = 57 K muestras en 720p, suficiente para
        // estimar histograma estable y costo despreciable (~1 ms).
        val step = 16
        var darkCount = 0
        var totalCount = 0
        var i = 0
        while (i < yPlaneSize) {
            val v = buf.get(basePos + i).toInt() and 0xff
            if (v < 80) darkCount++
            totalCount++
            i += step
        }
        if (totalCount == 0) return 0.0
        val darkPct = darkCount.toDouble() / totalCount
        // Curva: 1.0 cuando darkPct está en [0.08, 0.25]; decae linealmente fuera.
        return when {
            darkPct < 0.02  -> 0.10                                // casi todo blanco → cámara pegada
            darkPct < 0.08  -> 0.20 + (darkPct - 0.02) / 0.06 * 0.80  // 0.20→1.00 lineal
            darkPct <= 0.25 -> 1.00                                // rango ideal
            darkPct <= 0.40 -> 1.00 - (darkPct - 0.25) / 0.15 * 0.60 // 1.00→0.40 lineal
            else            -> 0.20                                // demasiado oscuro / lejos
        }
    }

    /**
     * Estima estabilidad comparando media de luminancia del frame actual con la
     * del anterior. Si la cámara se movió de golpe, la media cambia mucho.
     * Devuelve 0..1 (1 = estable, 0 = mucho movimiento). Primer frame siempre 1.0.
     *
     * Métrica simple pero barata. No detecta micro-movimientos (eso ya lo
     * penaliza el sharpness) pero sí saltos bruscos (que coinciden con frames
     * desenfocados / con motion blur).
     */
    private fun stabilityScore(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        val basePos = buf.position()
        val yPlaneSize = frameW * frameH
        val step = 32  // muestreo aún más raro: ~29 K samples
        var sum = 0L
        var n = 0
        var i = 0
        while (i < yPlaneSize) {
            sum += (buf.get(basePos + i).toInt() and 0xff)
            n++
            i += step
        }
        if (n == 0) return 1.0
        val curAvg = sum.toDouble() / n
        val score = if (prevFrameYSum < 0) {
            1.0  // primer frame: sin referencia, asumimos estable
        } else {
            val delta = kotlin.math.abs(curAvg - prevFrameYAvg)
            // delta > 10 en escala 0..255 ya es bastante movimiento; >30 es brusco.
            // Mapeo: delta=0 → 1.0, delta=10 → 0.7, delta=30 → 0.1
            (1.0 - (delta / 30.0)).coerceIn(0.1, 1.0)
        }
        prevFrameYSum = sum
        prevFrameYAvg = curAvg
        return score
    }

    /**
     * Calcula score compuesto del frame y, si supera el mejor previo, guarda
     * una copia del frame NV12 entero (Y + UV) en RAM. Llamado desde onFrame
     * en la ruta de video.
     *
     * @param sharpness ya calculada por laplacianVarianceCenter(); evitamos
     *                  recomputarla.
     */
    private fun updateBestFrameIfBetter(
        buf: ByteBuffer, frameW: Int, frameH: Int, sharpness: Double,
    ) {
        if (sharpness <= 0) return  // frame sin info útil
        val complete = completenessScore(buf, frameW, frameH)
        val stable   = stabilityScore(buf, frameW, frameH)

        // Score parcial sin detector de rectángulo (rápido, ~7 ms).
        val quickScore = sharpness * complete * stable

        // Snapshot del best previo: una sola lectura del @Volatile garantiza
        // que el umbral y la comparación final usen el MISMO valor.
        val prev = bestFrame
        val prevScore = prev?.score ?: -1.0

        // Sólo aplicamos el detector de rectángulo (Canny + findContours,
        // ~10-15 ms extra) si el quickScore es competitivo respecto al mejor
        // actual. Para los frames claramente malos (borrosos, cámara pegada,
        // movimiento) nos ahorramos el coste — y son la mayoría.
        val rectFactor = if (prevScore < 0 || quickScore >= prevScore * 0.7) {
            detectFolioRectangleFactor(buf, frameW, frameH)
        } else {
            1.0
        }
        val score = quickScore * rectFactor

        if (score > prevScore) {
            // Copia del NV12 entero (Y plane + UV interleaved = w·h·1.5 bytes)
            val nv12Size = frameW * frameH * 3 / 2
            val avail = buf.remaining()
            if (avail < nv12Size) return  // buffer raro, mejor no tocar
            val copy = ByteArray(nv12Size)
            val basePos = buf.position()
            for (k in 0 until nv12Size) {
                copy[k] = buf.get(basePos + k)
            }
            // Intercambio ATÓMICO de la referencia: el lector siempre ve un
            // BestFrame consistente (los 4 campos del previo o los 4 del nuevo,
            // nunca una mezcla).
            bestFrame = BestFrame(copy, frameW, frameH, score)
        }
    }

    /**
     * Convierte un NV12 (Y plane + UV interleaved) a JPEG. Reaprovecha el codec
     * nativo YuvImage de Android, que sólo come NV21 → swap UV→VU inline.
     * Llamar SIEMPRE desde Dispatchers.IO. Devuelve null si algo falla.
     */
    private fun nv12ToJpeg(nv12: ByteArray, w: Int, h: Int, quality: Int = 85): ByteArray? {
        return try {
            val ySize = w * h
            val total = ySize * 3 / 2
            if (nv12.size < total) return null
            // YuvImage soporta NV21 (Y + VU). NV12 es Y + UV. Swap bytes UV.
            val nv21 = ByteArray(total)
            System.arraycopy(nv12, 0, nv21, 0, ySize)
            var k = ySize
            while (k + 1 < total) {
                nv21[k]     = nv12[k + 1]   // V = U_nv12
                nv21[k + 1] = nv12[k]       // U = V_nv12
                k += 2
            }
            val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, w, h, null)
            val baos = java.io.ByteArrayOutputStream()
            yuv.compressToJpeg(android.graphics.Rect(0, 0, w, h), quality, baos)
            baos.toByteArray()
        } catch (e: Exception) {
            Log.w(TAG, "nv12ToJpeg falló: ${e.message}")
            null
        }
    }

    // OpenCV se carga lazy la primera vez que se necesita el detector de
    // rectángulo. Si la lib no está disponible (build sin OpenCV, etc.), el
    // detector devolverá factor=1.0 sin romper nada.
    @Volatile private var opencvLoaded: Boolean? = null

    private fun ensureOpenCv(): Boolean {
        opencvLoaded?.let { return it }
        return synchronized(this) {
            opencvLoaded?.let { return@synchronized it }
            val ok = try {
                System.loadLibrary("opencv_java4")
                Log.i(TAG, "[opencv] cargado para detector de rectángulo del folio")
                true
            } catch (e: Throwable) {
                Log.w(TAG, "[opencv] no se pudo cargar: ${e.message} — detector de rectángulo desactivado")
                false
            }
            opencvLoaded = ok
            ok
        }
    }

    /**
     * Detecta si el plano Y del frame contiene un rectángulo grande (el folio)
     * usando Canny + findContours sobre versión reescalada a 320×180.
     *
     * Devuelve un factor multiplicativo para el score:
     *   1.5 si encuentra rectángulo grande (>50% del área del frame)
     *   1.2 si encuentra rectángulo mediano (30-50%)
     *   1.0 si encuentra rectángulo pequeño (10-30%)
     *   0.7 si no encuentra rectángulo grande (penalty leve, no descarta)
     *
     * Coste: ~10-15 ms por frame en S10e (resize 1280×720→320×180 + Canny +
     * findContours + filtrado). A 30 fps es asumible.
     *
     * Limitaciones (heurísticas, no perfectas):
     *  - Folio sobre mesa de madera oscura sin contraste de borde → puede fallar
     *  - Folio doblado / arrugado → contornos no son cuadriláteros limpios
     *  - Cualquier rectángulo en cuadro (libro, marco, pantalla) será confundido
     *    con el folio. La heurística de completeness ya filtra estos casos
     *    cuando dan distribución no-bimodal.
     */
    private fun detectFolioRectangleFactor(buf: ByteBuffer, frameW: Int, frameH: Int): Double {
        if (!ensureOpenCv()) return 1.0  // sin OpenCV, no afecta al score

        val ySize = frameW * frameH
        val basePos = buf.position()
        val yMat = org.opencv.core.Mat(frameH, frameW, org.opencv.core.CvType.CV_8UC1)
        // Copia plano Y a un ByteArray temporal y luego al Mat. Mat.put acepta
        // ByteArray directamente — más rápido que iterar pixel a pixel.
        try {
            val yBytes = ByteArray(ySize)
            for (i in 0 until ySize) yBytes[i] = buf.get(basePos + i)
            yMat.put(0, 0, yBytes)
        } catch (e: Throwable) {
            yMat.release()
            Log.w(TAG, "[rect] copia Y plane falló: ${e.message}")
            return 1.0
        }

        var result = 1.0
        val small = org.opencv.core.Mat()
        val blurred = org.opencv.core.Mat()
        val edges = org.opencv.core.Mat()
        val hierarchy = org.opencv.core.Mat()
        val contours = ArrayList<org.opencv.core.MatOfPoint>()
        var approx: org.opencv.core.MatOfPoint2f? = null
        var contour2f: org.opencv.core.MatOfPoint2f? = null

        try {
            // 1) Resize a 320×180 para acelerar Canny + findContours (16× menos pixeles)
            org.opencv.imgproc.Imgproc.resize(
                yMat, small, org.opencv.core.Size(320.0, 180.0),
                0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_AREA,
            )
            // 2) Gaussian blur 3×3 para reducir ruido antes de Canny
            org.opencv.imgproc.Imgproc.GaussianBlur(small, blurred, org.opencv.core.Size(3.0, 3.0), 0.0)
            // 3) Canny edge detection — thresholds ajustados para texto en folio
            org.opencv.imgproc.Imgproc.Canny(blurred, edges, 50.0, 150.0)
            // 4) Find external contours
            org.opencv.imgproc.Imgproc.findContours(
                edges, contours, hierarchy,
                org.opencv.imgproc.Imgproc.RETR_EXTERNAL,
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE,
            )

            val smallArea = 320.0 * 180.0
            var bestRectArea = 0.0
            approx = org.opencv.core.MatOfPoint2f()
            contour2f = org.opencv.core.MatOfPoint2f()
            for (c in contours) {
                c.convertTo(contour2f, org.opencv.core.CvType.CV_32F)
                val peri = org.opencv.imgproc.Imgproc.arcLength(contour2f, true)
                org.opencv.imgproc.Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true)
                // Buscamos cuadriláteros (4 vértices) — los folios suelen serlo en cuadro
                if (approx.total() == 4L) {
                    val area = org.opencv.imgproc.Imgproc.contourArea(approx)
                    if (area > bestRectArea) bestRectArea = area
                }
                c.release()
            }
            val ratio = bestRectArea / smallArea
            result = when {
                ratio > 0.50 -> 1.5   // folio grande detectado → boost fuerte
                ratio > 0.30 -> 1.2   // mediano → boost leve
                ratio > 0.10 -> 1.0   // pequeño → neutral
                else         -> 0.7   // no se detecta rectángulo grande → penalty leve
            }
        } catch (e: Throwable) {
            Log.w(TAG, "[rect] detectFolioRectangleFactor falló: ${e.message}")
            result = 1.0
        } finally {
            yMat.release()
            small.release()
            blurred.release()
            edges.release()
            hierarchy.release()
            approx?.release()
            contour2f?.release()
        }
        return result
    }

    /**
     * Variante que recibe sharpness ya calculado para evitar recomputar el
     * Laplaciano dos veces (una para filtrar y otra para best-frame scoring).
     */
    private fun shouldEncodeForSharpness(sharpness: Double): Boolean {
        val s = sharpness
        // Append al histórico circular
        sharpHistory[sharpHistoryIdx] = s
        sharpHistoryIdx = (sharpHistoryIdx + 1) % SHARP_HISTORY_SIZE
        if (sharpHistoryCount < SHARP_HISTORY_SIZE) sharpHistoryCount++

        // Warm-up: deja pasar los primeros frames sin filtrar (necesarios para
        // que el encoder arranque y para tener estadística representativa).
        if (sharpHistoryCount < SHARP_HISTORY_SIZE) {
            consecutiveBlurDrops = 0
            return true
        }

        // Mediana móvil de los últimos SHARP_HISTORY_SIZE valores
        val sorted = sharpHistory.copyOf().also { it.sort() }
        val median = sorted[SHARP_HISTORY_SIZE / 2]
        val threshold = median * 0.6   // 40% por debajo de la mediana = borroso

        val pass = s >= threshold
        if (pass) {
            consecutiveBlurDrops = 0
        } else {
            consecutiveBlurDrops++
            // Safety net: si llevamos demasiados seguidos descartados, dejar
            // pasar el siguiente aunque sea malo — preferible MP4 borroso a
            // MP4 vacío en condiciones adversas (poca luz, movimiento, etc.).
            if (consecutiveBlurDrops > MAX_CONSECUTIVE_DROPS) {
                consecutiveBlurDrops = 0
                return true
            }
        }
        return pass
    }

    private suspend fun captureBurstFrames(processor: AdvancedBurstProcessor) {
        if (!isCapturingForBurst.compareAndSet(false, true)) {
            Log.w(TAG, "[captura] Ya hay una captura en curso, se omite")
            return
        }

        // Reset del kill switch "cámara tapada". Cada burst empieza con racha=0;
        // si esta captura sufre tapado >=3s, updateCameraCoveredStateRgbx lo
        // activará y processBurstAndSend lo verá antes de invocar stacking/IA.
        cameraCoveredAbort = false
        darkFrameRunStartMs = 0L
        darkFrameRunCount = 0

        cameraMutex.withLock {
            try { uvcCamera?.setFrameCallback(this@HeadlessUvcService, UVCCamera.PIXEL_FORMAT_RGBX) } catch (_: Exception) {}
        }

        Log.i(TAG, "[captura] Iniciando burst de $frameBurstCount frames")
        try {
            broadcastCameraState(CameraState.CapturingBurst(0, frameBurstCount))
            while (frameChannel.tryReceive().isSuccess) { }

            repeat(frameBurstCount) { idx ->
                if (!kotlinx.coroutines.currentCoroutineContext().isActive || !systemEnabled) {
                    Log.w(TAG, "[captura] Burst cancelado por stop del sistema")
                    return@repeat
                }
                if (usbCtrlBlock == null) {
                    Log.w(TAG, "[captura] USB desconectado durante burst, abortando captura")
                    videoAbortedByUsbDisconnect = true
                    return@repeat
                }
                try {
                    val bmp = withTimeout(5000) { frameChannel.receive() }
                    processor.addFrame(bmp)
                } catch (e: Exception) {
                    Log.w(TAG, "[captura] frame $idx timeout/error: ${e.message}")
                }
            }
        } finally {
            // NO llamamos setFrameCallback(null, 0) aquí: esa llamada nula el jmethodID nativo
            // bajo un mutex diferente al que usa do_capture_callback → race → SIGABRT (CheckJNI).
            // isCapturingForBurst=false ya hace que onFrame() descarte los frames; destroy() en
            // closeCameraLocked los detiene limpiamente bajo el mutex correcto (mFrameCallbackMutex).
            isCapturingForBurst.set(false)
            while (frameChannel.tryReceive().isSuccess) { }
            Log.i(TAG, "[captura] Burst terminado: ${processor.getFrameCount()} frames recibidos de $frameBurstCount")
        }
    }

    /**
     * Captura frames de la cámara durante `durationSeconds` y los entrega al
     * encoder en BUFFER mode (NV12 → MediaCodec.queueInputBuffer).
     *
     * Reemplaza el antiguo SURFACE mode (uvcCamera.startCapture(encoder.inputSurface))
     * que crashea en Samsung Exynos S10e con SIGSEGV en libUVCCamera/copyToSurface:
     * el inputSurface del encoder HEVC está respaldado por un buffer
     * HAL_PIXEL_FORMAT_YCbCr_420_888 que Mali Gralloc NO soporta vía
     * ANativeWindow_lock sin LOCK_FLEX (API no expuesta al NDK). libuvc escribía
     * filas de 5120 B asumiendo RGBA en un plano Y mapeado sólo 1 página → SEGV.
     *
     * Aquí:
     *   1. encoder.start() arranca el drain loop
     *   2. setFrameCallback(this, PIXEL_FORMAT_YUV420SP) → libuvc convierte
     *      YUYV→NV12 nativamente (uvc_yuyv2yuv420SP) y entrega bytes vía JNI
     *   3. isCapturingForVideo=true → onFrame() ruta cada NV12 al encoder
     *   4. Tras durationSeconds, isCapturingForVideo=false detiene el routing
     *   5. closeCameraLocked() (más tarde) destruirá la cámara y detendrá el
     *      callback de forma limpia bajo el mFrameCallbackMutex nativo
     *
     * La cámara debe estar abierta y warm ANTES de llamar a esta función.
     */
    private suspend fun captureVideoFrames(encoder: VideoEncoder, durationSeconds: Int, fps: Int) {
        if (!isCapturingForBurst.compareAndSet(false, true)) {
            Log.w(TAG, "[video] Ya hay una captura en curso, se omite")
            return
        }

        Log.i(TAG, "[video] Iniciando captura BUFFER NV12: ${durationSeconds}s @ ${fps}fps (negociado con cámara)")
        try {
            broadcastCameraState(CameraState.CapturingBurst(0, durationSeconds * fps))

            // 1) Arrancar el drain loop del encoder (extrae buffers codificados y los muxea)
            encoder.start()

            val cameraReady = cameraMutex.withLock {
                val cam = uvcCamera
                if (cam == null) {
                    Log.w(TAG, "[video] uvcCamera == null al iniciar captura; abortando")
                    return@withLock false
                }

                // Publicar encoder y resetear contadores ANTES de activar la ruta
                // de video en onFrame: si el callback ya está activo desde un burst
                // anterior, no queremos un frame fantasma con videoEncoderRef=null.
                videoEncoderRef = encoder
                videoFramesQueued = 0
                videoFramesDropped = 0
                videoFramesFilteredBlur = 0
                // Reset del histórico de nitidez: cada captura empieza con su
                // propia mediana adaptativa (la iluminación/contenido cambian
                // entre ciclos, no queremos arrastrar baseline obsoleto).
                sharpHistoryCount = 0
                sharpHistoryIdx = 0
                consecutiveBlurDrops = 0
                // Reset del best-frame state — empezamos sin candidato.
                bestFrame = null
                bestFrameJpeg = null
                prevFrameYSum = -1L
                prevFrameYAvg = 0.0
                // Reset del kill switch "cámara tapada". Cada captura empieza
                // con racha=0; si esta captura sufre tapado >=3s, se activará
                // y el loop de abajo lo detectará para cortar el ciclo.
                cameraCoveredAbort = false
                videoAbortedByUsbDisconnect = false
                darkFrameRunStartMs = 0L
                darkFrameRunCount = 0

                try {
                    // Cambiar el callback a NV12. libuvc usará uvc_yuyv2yuv420SP
                    // internamente para convertir cada frame antes de entregarlo.
                    cam.setFrameCallback(this@HeadlessUvcService, UVCCamera.PIXEL_FORMAT_YUV420SP)
                } catch (e: Exception) {
                    Log.e(TAG, "[video] setFrameCallback(NV12) falló: ${e.message}", e)
                    videoEncoderRef = null
                    // encoder.start() ya se llamó arriba → su drain loop daemon
                    // queda corriendo sin nadie llamando finish(). Liberamos
                    // explícitamente para evitar leak del codec + muxer.
                    try { encoder.release() } catch (_: Throwable) {}
                    return@withLock false
                }

                // Activar el routing en onFrame DESPUÉS de cambiar el callback,
                // para no procesar bytes RGBX residuales como NV12.
                isCapturingForVideo.set(true)
                true
            }
            if (!cameraReady) return

            val t0 = System.currentTimeMillis()
            val deadlineMs = t0 + durationSeconds * 1000L
            var lastTick = 0L
            var abortedByDisconnect = false
            try {
                while (System.currentTimeMillis() < deadlineMs) {
                    if (!kotlinx.coroutines.currentCoroutineContext().isActive || !systemEnabled) {
                        Log.w(TAG, "[video] Captura cancelada por stop del sistema")
                        break
                    }
                    delay(200)
                    // FIX CRÍTICO-1: si el usuario quita el USB mid-capture,
                    // onDettach pone usbCtrlBlock=null al instante. Sin este
                    // chequeo el loop se quedaría 10 s alimentando frames a
                    // un encoder cuya cámara ya no existe → MP4 truncado y
                    // ventana de UAF en JNI hasta que termine el delay.
                    if (usbCtrlBlock == null) {
                        Log.w(TAG, "[video] USB desenchufado mid-capture en t=${(System.currentTimeMillis()-t0)/1000}s — abortando")
                        abortedByDisconnect = true
                        videoAbortedByUsbDisconnect = true
                        break
                    }
                    // KILL SWITCH: cámara tapada >=3s → cortamos ya. No
                    // mandaremos nada al relay (processVideoAndSend lo
                    // detecta y retorna antes del HTTP/base64).
                    if (cameraCoveredAbort) {
                        Log.w(TAG, "[video] 🛑 Cámara tapada >${CAMERA_COVERED_THRESHOLD_MS}ms en t=${(System.currentTimeMillis()-t0)/1000}s — abortando captura SIN enviar")
                        break
                    }
                    val elapsedSec = ((System.currentTimeMillis() - t0) / 1000).toInt()
                    if (elapsedSec != lastTick.toInt()) {
                        lastTick = elapsedSec.toLong()
                        broadcastCameraState(CameraState.CapturingBurst(elapsedSec, durationSeconds))
                    }
                }
            } finally {
                if (abortedByDisconnect) {
                    Log.w(TAG, "[video] Capture aborted by USB disconnect")
                }
                // Apagar el routing primero para que onFrame() empiece a
                // descartar frames (sin tocar el callback nativo, que se
                // detendrá limpio cuando closeCameraLocked() haga destroy).
                isCapturingForVideo.set(false)
                // Pequeño margen para que el último frame "fantasma" que
                // pueda estar en vuelo en el callback termine antes de que
                // leamos bestFrame fuera del mutex.
                delay(50)
                videoEncoderRef = null
                val durMs = System.currentTimeMillis() - t0
                val grayFrames = try { encoder.grayFramesDetected } catch (_: Throwable) { -1L }
                Log.i(TAG, "[video] Captura BUFFER terminada: ${durMs}ms reales, " +
                        "queued=$videoFramesQueued, dropped=$videoFramesDropped, " +
                        "filtered_blur=$videoFramesFilteredBlur · gray_frames=$grayFrames · " +
                        "bestScore=%.1f".format(bestFrame?.score ?: -1.0))
                if (grayFrames > 0L) {
                    Log.w(TAG, "[video] ⚠ Detectados $grayFrames frames grises sobre $videoFramesQueued queued — MJPEG corrupto vía USB (libuvc#122)")
                }
            }

            // Convertir el best NV12 a JPEG AQUÍ (fuera del cameraMutex, sin
            // bloquear el siguiente ciclo). Hacerlo en cuanto termina la
            // captura, no en el momento del fallback, para que el JPEG ya
            // esté listo si processVideoAndSend lo necesita más tarde.
            val bf = bestFrame  // snapshot atómico de la referencia
            if (bf != null) {
                val jpegBytes = withContext(Dispatchers.IO) { nv12ToJpeg(bf.nv12, bf.w, bf.h, 85) }
                bestFrameJpeg = jpegBytes
                // Liberar el buffer NV12 (~1.4 MB) tras usarlo: no lo
                // necesitamos más, solo el JPEG mucho más ligero (~150-300 KB).
                bestFrame = null
                if (jpegBytes != null) {
                    Log.i(TAG, "[video] Best frame fallback listo: ${jpegBytes.size / 1024} KB " +
                            "(score=%.1f, ${bf.w}x${bf.h})".format(bf.score))
                }
            } else {
                Log.w(TAG, "[video] No hay best frame candidato — fallback usará primer frame del MP4")
            }
        } finally {
            isCapturingForBurst.set(false)
            // Defensa anti-leak: si capture abortó tempranamente (excepción,
            // cancelación), aseguramos que el NV12 (~1.4 MB) no quede colgado.
            // bestFrameJpeg se libera en processVideoAndSend tras usarlo.
            bestFrame = null
        }
    }

    /**
     * Finaliza el encoder, lee el MP4 a memoria, valida tamaño y lo envía al
     * orquestador en modo VIDEO (relay → OCR Qwen+Gemini → análisis 4 IAs).
     */
    private suspend fun processVideoAndSend(encoder: VideoEncoder, file: java.io.File, isOneShot: Boolean = false) {
        try {
            if (!isOneShot && !systemEnabled) {
                Log.w(TAG, "[video] Sistema detenido antes de enviar — NO se manda MP4")
                try { withContext(Dispatchers.IO) { encoder.finish() } } catch (_: Exception) {}
                try { withContext(Dispatchers.IO) { if (file.exists()) file.delete() } } catch (_: Exception) {}
                return
            }
            if (videoAbortedByUsbDisconnect) {
                Log.w(TAG, "[video] 🛑 Captura abortada por desconexión USB — NO se envía MP4 parcial")
                try { withContext(Dispatchers.IO) { encoder.finish() } } catch (e: Exception) {
                    Log.w(TAG, "[video] encoder.finish() en abort USB lanzó: ${e.message}")
                }
                try { withContext(Dispatchers.IO) { if (file.exists()) file.delete() } } catch (_: Exception) {}
                videoAbortedByUsbDisconnect = false
                return
            }
            // KILL SWITCH: si la captura fue abortada por cámara tapada, NO
            // se manda nada al exterior. Cerramos el encoder (libera codec +
            // muxer) pero saltamos todo el path de subida → no HTTP, no base64,
            // no DNS, no nada que dispare la radio. El ciclo termina silencioso
            // y el siguiente arranca normal cuando toque.
            if (cameraCoveredAbort) {
                Log.w(TAG, "[video] 🛑 Cámara tapada — ABORT del ciclo. NO se manda video, NO se activa radio.")
                cyclesAbortedByCover++
                try { withContext(Dispatchers.IO) { encoder.finish() } } catch (e: Exception) {
                    Log.w(TAG, "[video] encoder.finish() en abort lanzó: ${e.message}")
                }
                // Limpieza: borrar el MP4 parcial del cache (lo que se grabara
                // hasta el segundo del tapado). NO se persiste en galería.
                try { withContext(Dispatchers.IO) { if (file.exists()) file.delete() } } catch (_: Exception) {}
                // Silencioso por UX: si la cámara se tapó, descartamos sin avisar.
                // El usuario ya se dará cuenta en el siguiente aviso de captura.
                // Reset del flag para que el siguiente ciclo arranque limpio.
                cameraCoveredAbort = false
                return
            }
            val mp4 = withContext(Dispatchers.IO) { encoder.finish() }
            if (mp4 == null || !mp4.exists() || mp4.length() == 0L) {
                Log.w(TAG, "[video] Encoder terminó sin archivo válido")
                try { vibrarPatronError() } catch (_: Exception) {}
                return
            }
            val bytes = withContext(Dispatchers.IO) { mp4.readBytes() }
            Log.i(TAG, "[video] MP4 generado: ${bytes.size} bytes (${bytes.size / 1024} KB)")

            // Guardar copia persistente del vídeo en la galería (Movies/BolsilloIA/)
            // para que el usuario pueda verlo después. La copia es independiente
            // del archivo de cache: ese se borrará al final igual que antes.
            withContext(Dispatchers.IO) { saveVideoToGallery(mp4, mp4.name) }

            // Guardar también el "best frame" JPEG en Pictures/BolsilloIA/ con
            // el mismo timestamp que el MP4 (sufijo _best.jpg). Así el usuario
            // ve en la galería el par video+foto, y si el video falla y se
            // manda la foto como fallback, queda registro visual de qué se
            // envió a la IA.
            val bestJpeg = bestFrameJpeg
            if (bestJpeg != null) {
                withContext(Dispatchers.IO) { saveBestFrameToGallery(bestJpeg, mp4.name) }
            }

            val modoPref = currentCaptureMode
            if (modoPref == "TEST") {
                Log.i(TAG, "[video] Modo TEST → sin IA")
                sendResponseBroadcast(mp4.name, listOf('X'),
                    "MODO TEST:\nVideo grabado (${bytes.size / 1024} KB) sin enviar a IA.", 0L)
                // FIX SAMSUNG S10e: ruta sysfs (vía playLetterSequence → vibrate)
                // — Android API queda bloqueada por One UI / VibratorManagerService
                // cuando la app está en background, aunque AudioAttributes=ALARM.
                playLetterSequence(listOf('X'))
                return
            }

            // Solo relay: no usar ruta directa local (askConFallback) en producción.
            val modo = BolsilloIaOrquestador.Modo.A_RELAY
            // En oneShot forzamos SIN avión (botón TEST nunca debe exponer RF).
            val forzarSinAvion = isOneShot || debugSinAvion
            if (modoPref != "A") {
                Log.w(TAG, "[video] modo legado '$modoPref' detectado; se fuerza A_RELAY")
            }
            Log.i(TAG, "[video] modo=$modo · oneShot=$isOneShot · forzarSinAvion=$forzarSinAvion")
            val orq = BolsilloIaOrquestador(debugSinAvion = forzarSinAvion, appContext = applicationContext)
            val prompt = (if (contextImageB64 != null) SYSTEM_PROMPT_CON_CONTEXTO else SYSTEM_PROMPT_UNIFICADO)

            // Tiempo REAL del ciclo en curso (jitter aplicado), para que el server
            // auto-apruebe antes del próximo ciclo. Mínimo 120s por si el ciclo es
            // muy corto (3-5 min jitter + OCR del video que tarda 30-60s extra).
            //
            // BUG-FIX (oneShot): en oneShot NO hay ciclo periódico activo, así que
            // `currentIterationCycleSeconds` conserva su default (CYCLE_SECONDS_BASE
            // =300s, o el último valor del último ciclo) y oneShot terminaba
            // esperando 300s de delay() puro antes de pollear. Para oneShot usamos
            // un valor mucho más corto (75s) porque:
            //   - El relay típicamente termina el pipeline en 60-90s (OCR + analyzers)
            //   - El polling tiene 90s de ventana → tiempo total cap ~165s
            //   - debugSinAvion=true (oneShot fuerza esto) → no hace falta el
            //     margen de "esperar a que se cierre el avión"
            //   - Reduce a la mitad la ventana donde el OS puede matar la coroutine
            val cycleSec = if (isOneShot) 75L
                           else currentIterationCycleSeconds.coerceAtLeast(120L)
            // MARGEN DE SINCRONIZACIÓN — mismo razonamiento que en burst: el relay
            // auto-aprueba 15s ANTES de que el mobile termine su espera para que el
            // primer poll encuentre el resultado listo. En video el margen es más
            // valioso porque la fase 1 (OCR de 3 modelos en paralelo) tarda 20-60s
            // antes de que arranque la fase 2 (analyzers) — la 1ª IA analítica puede
            // tardar 40s post-OCR. Sin margen, sync se rompe en video grande.
            val reviewTimeout = (cycleSec - 15L).coerceAtLeast(10L).toInt()
            Log.i(TAG, "[video] ▶ enviando ${bytes.size / 1024} KB a orquestador · modo=$modo · debugSinAvion=$debugSinAvion · payload=VIDEO · ctx=${contextImageB64 != null} · mobile_espera=${cycleSec}s · review_timeout=${reviewTimeout}s (margen 15s)")
            val t0 = System.currentTimeMillis()
            val resultado = try {
                orq.procesarVideo(
                    videoMp4Bytes = bytes,
                    prompt = prompt,
                    modo = modo,
                    system = null,
                    // Diagrama de contexto: snapshot del global en este punto
                    // (Volatile read). Si la IA detectó un caso práctico en una
                    // página anterior, esta variable está rellena y el relay lo
                    // adjuntará como IMAGEN 1 a TODOS los analyzers/OCRs del job.
                    contextImageB64 = contextImageB64,
                    esperaProcesadoMs = cycleSec * 1000L,
                    reviewTimeoutSeconds = reviewTimeout,
                    onEstado = { e -> Log.i(TAG, "[video][orq] estado=$e") },
                )
            } catch (e: Exception) {
                Log.e(TAG, "[video] orquestador lanzó: ${e.message}", e)
                PersistentErrorLog.logError(TAG, "Orquestador VIDEO lanzó: ${e.message}", e)
                BolsilloIaClient.IaResultado(
                    answer = "Error orquestador video: ${e.message}",
                    provider = "fallback-fail", model = "", viaRelay = false,
                )
            }
            val ms0 = System.currentTimeMillis() - t0
            Log.i(TAG, "[video] ✓ Resultado: '${resultado.answer.take(80)}' via ${resultado.provider} en ${ms0}ms (esVideo=true)")

            // Solo VIDEO: no hay degradación a ruta foto.
            val resultadoFinal = resultado

            val ms = System.currentTimeMillis() - t0
            // Añadir bloque de exposición RF al final del raw text — el usuario lo ve en la UI
            // para auditar cuánto tiempo total estuvo el módem celular emitiendo. Vacío si
            // fue oneShot/Modo C / debugSinAvion (no hubo ventanas de RF).
            val expBlock = BolsilloIaOrquestador.formatearExposicion(
                orq.ventanasExposicionMs,
                orq.totalExposicionMs,
            )
            Log.i(TAG, "[video][exposicion] ventanas=${orq.ventanasExposicionMs} total=${orq.totalExposicionMs}ms")
            // Header de timings (envío/recogida/total). Los 3 valores vienen del
            // orquestador; en fallbacks (sin Modo A completo) puede haber valores
            // a 0 — los ocultamos en ese caso para no enseñar "0ms" confuso.
            val timingsHeader = formatearTimings(resultadoFinal, ms)
            val rawTextWithExp = buildString {
                if (timingsHeader.isNotEmpty()) {
                    append(timingsHeader)
                    append("\n\n")
                }
                append("Video → IA (${resultadoFinal.provider}, ${ms}ms)\n${resultadoFinal.answer}")
                if (expBlock.isNotEmpty()) {
                    append("\n\n")
                    append(expBlock)
                }
            }
            // Si no había contexto previo y la IA detectó un diagrama en este
            // video → guardar el best frame como diagrama del caso práctico.
            // Mismo patrón que el path BURST (procesarConFlujoNuevo:3107), pero
            // aquí la "foto" representativa del clip es bestFrameJpeg (mejor
            // sharpness/completeness/stability) en lugar del processed-burst.
            // Si bestFrameJpeg es null (raro: clip abortado antes de procesar
            // frames), saltamos el guardado — siguiente ciclo lo intentará.
            if (contextImageB64 == null &&
                resultadoFinal.answer.contains("CONTEXTO_VISUAL:SI", ignoreCase = true)) {
                val jpegToSave = bestFrameJpeg
                if (jpegToSave != null) {
                    Log.i(TAG, "[video] Diagrama detectado en este clip → guardando best frame (${jpegToSave.size / 1024} KB) como contexto del caso práctico.")
                    saveContextImage(jpegToSave)
                } else {
                    Log.w(TAG, "[video] CONTEXTO_VISUAL:SI detectado pero bestFrameJpeg null — no se guarda diagrama")
                }
            }

            // SI el cómplice marcó manualmente la casilla 🖼️ "Imagen en
            // pregunta" en el dashboard, el relay devuelve la mejor imagen
            // del job en `contextImageJpeg` (prioridad: fused Topaz >
            // stacking local > frame top-1 > burst). La guardamos como
            // diagrama de contexto sobreescribiendo el anterior — eso es lo
            // que el cómplice está pidiendo explícitamente: "esta página
            // tiene un diagrama relevante para las siguientes".
            val manualCtx = resultadoFinal.contextImageJpeg
            if (manualCtx != null && manualCtx.size > 1024) {
                Log.i(TAG, "[video] 🗂️ Cómplice marcó casilla 'Imagen en pregunta' → guardando ${manualCtx.size / 1024} KB como contexto (sobreescribe el anterior si lo había)")
                saveContextImage(manualCtx)
            }
            sendResponseBroadcast(mp4.name, resultadoFinal.answer.toList(), rawTextWithExp, ms)
            if (resultadoFinal.answer.isNotEmpty() &&
                resultadoFinal.provider != "fallback-fail" &&
                resultadoFinal.provider != "fallback-fail-video") {
                // FIX SAMSUNG S10e: usar playLetterSequence (ruta sysfs vía
                // vibrate()) en lugar de translateSequenceToVibration (Android
                // API). Bug reportado: en oneShot + app en background, el
                // VibratorManagerService de One UI ignora silenciosamente
                // VibrationEffect.createWaveform. Sysfs no tiene esa restricción.
                Log.i(TAG, "[video] 🔔 vibrando respuesta '${resultadoFinal.answer}' (${resultadoFinal.answer.length} letras)")
                playLetterSequence(resultadoFinal.answer.toList())
            } else {
                // Sin respuesta útil → log + vibración de error (el usuario recibe X y
                // sabe que algo falló durante la ventana de avión).
                PersistentErrorLog.logError(
                    TAG, "[video] SIN RESPUESTA UTIL · provider=${resultadoFinal.provider} · answer='${resultadoFinal.answer.take(40)}' · ${ms}ms",
                )
                try { vibrarPatronError() } catch (_: Exception) {}
            }
        } finally {
            try { encoder.release() } catch (_: Exception) {}
            try { file.delete() } catch (_: Exception) {}
            // Liberar el JPEG fallback (200-300 KB) ya consumido. Si se vuelve
            // a necesitar para el próximo ciclo, captureVideoFrames generará
            // uno nuevo desde el siguiente video. No reusar JPEGs entre ciclos.
            bestFrameJpeg = null
            broadcastCameraState(CameraState.ReadyForCapture)
        }
    }

    /**
     * Detecta si el brazo está tapando la cámara analizando el % de píxeles blancos.
     * Un folio de examen tiene >40% píxeles muy claros (brillo >210).
     * Un brazo tapando la vista reduce eso a <20% — umbral con margen amplio vs sombras.
     * Se decodifica a 1/8 de resolución para que sea instantáneo (~14k píxeles).
     */
    private fun isBrazoTapandoCamara(jpegBytes: ByteArray): Boolean {
        return try {
            val opts = android.graphics.BitmapFactory.Options().apply { inSampleSize = 8 }
            val bmp = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, opts)
                ?: return false
            val total = bmp.width * bmp.height
            // Doble criterio:
            //   (a) luminancia MEDIA del frame — el detector primario. Con la
            //       cámara tapada por brazo, la media baja a <30 incluso con AGC
            //       alto. Un folio normal está entre 120-220.
            //   (b) % de píxeles "no-negros" (brillo>80) — secundario. Antes era
            //       el único criterio y fallaba: pct<0.01 (umbral previo) lo
            //       superaba el ruido del sensor con AGC alto y la detección no
            //       saltaba aunque el brazo cubriera la lente.
            //
            // Esto es DEFENSA EN PROFUNDIDAD del kill switch en onFrame; si el
            // burst duró <3s y no llegó a activar el flag, este chequeo del JPEG
            // final pilla el caso. Si el flag YA disparó, processBurstAndSend
            // sale antes de llegar a llamar a esta función.
            var sumBrightness = 0L
            var noNegros = 0
            for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val pixel = bmp.getPixel(x, y)
                    val r = (pixel shr 16) and 0xFF
                    val g = (pixel shr 8) and 0xFF
                    val b = pixel and 0xFF
                    val brightness = (r * 299 + g * 587 + b * 114) / 1000
                    sumBrightness += brightness
                    if (brightness > 80) noNegros++
                }
            }
            bmp.recycle()
            val avgBrightness = sumBrightness.toDouble() / total
            val pct = noNegros.toDouble() / total
            // Brazo = (luminancia media muy baja) O (casi nada por encima de 80).
            // El umbral 35 deja margen vs ruido del sensor (<15 típico AGC alto
            // tapada) sin morder por luz muy pobre legítima (>50 incluso a oscuras).
            val esBrazo = avgBrightness < 35.0 || pct < 0.02
            Log.i(TAG, "[brazo] avgLuma=%.1f, %.2f%% píxeles >80 brillo, brazo=$esBrazo"
                .format(avgBrightness, pct * 100))
            esBrazo
        } catch (e: Exception) {
            Log.e(TAG, "isBrazoTapandoCamara error: ${e.message}")
            false
        }
    }

    private suspend fun processBurstAndSend(processor: AdvancedBurstProcessor, isOneShot: Boolean = false) {
        try {
            if (!isOneShot && !systemEnabled) {
                Log.w(TAG, "[proceso] Sistema detenido antes de procesar burst — NO se manda foto")
                return
            }
            if (videoAbortedByUsbDisconnect) {
                Log.w(TAG, "[proceso] Burst invalidado por desconexión USB — se descarta sin IA")
                videoAbortedByUsbDisconnect = false
                return
            }
            // KILL SWITCH: si la captura fue abortada por cámara tapada (el flag
            // se activa en onFrame vía updateCameraCoveredStateRgbx), saltamos
            // stacking + IA + radio. Cerramos el processor para liberar bitmaps
            // y seguimos silencioso (sin vibración). El siguiente ciclo
            // arrancará con racha=0 (lo resetea captureBurstFrames).
            if (cameraCoveredAbort) {
                Log.w(TAG, "[proceso] 🛑 Cámara tapada — ABORT del burst. NO se procesa stacking, NO se manda foto, NO se activa radio.")
                cyclesAbortedByCover++
                broadcastCameraState(CameraState.ReadyForCapture)
                sendBroadcast(Intent("BRAZO_DETECTADO").setPackage(packageName))
                cameraCoveredAbort = false
                return
            }
            val frameCount = processor.getFrameCount()
            Log.i(TAG, "[proceso] processBurstAndSend: frames=$frameCount")
            if (frameCount == 0) {
                Log.w(TAG, "[proceso] Sin frames capturados, saltando ciclo")
                broadcastCameraState(CameraState.ReadyForCapture)
                // Sin frames → cámara mal conectada o burst interrumpido.
                // Vibrar error para que el usuario no espere una respuesta que no llegará.
                try { vibrarPatronError() } catch (_: Exception) {}
                return
            }

            Log.i(TAG, "[proceso] Procesando stacking...")
            val processed = try { processor.process() } catch (e: Exception) {
                Log.e(TAG, "[proceso] processor.process() lanzó: ${e.message}", e)
                null
            }

            if (processed == null) {
                Log.w(TAG, "[proceso] processor.process() devolvió null, saltando ciclo")
                try { vibrarPatronError() } catch (_: Exception) {}
                return
            }

            Log.i(TAG, "[proceso] Imagen procesada: ${processed.size} bytes")
            val fileName = "BURST_${System.currentTimeMillis()}.jpg"
            saveImageToFile(processed, fileName)
            Log.i(TAG, "[proceso] Foto guardada: $fileName")

            if (isBrazoTapandoCamara(processed)) {
                Log.i(TAG, "[proceso] Brazo detectado → ciclo saltado sin llamar a IA")
                sendBroadcast(Intent("BRAZO_DETECTADO").setPackage(packageName))
                // Silencioso por UX: no avisamos con vibración en brazo detectado.
                return
            }

            val modoPref = currentCaptureMode
            Log.i(TAG, "[proceso] modoPref=$modoPref, debugSinAvion=$debugSinAvion")

            if (modoPref == "TEST") {
                Log.i(TAG, "[proceso] Modo TEST → sin IA")
                sendResponseBroadcast(fileName, listOf('X'), "MODO TEST:\nFoto guardada correctamente en galería sin usar IA.", 0L)
                // FIX SAMSUNG S10e: sysfs route
                playLetterSequence(listOf('X'))
                return
            }

            val modo = BolsilloIaOrquestador.Modo.A_RELAY
            Log.i(TAG, "[proceso] Llamando procesarConFlujoNuevo con modo=$modo (oneShot=$isOneShot)")
            procesarConFlujoNuevo(processed, fileName, modo, isOneShot)
        } finally {
            processor.release()
            broadcastCameraState(CameraState.ReadyForCapture)
        }
    }

    private suspend fun procesarConFlujoNuevo(
        processed: ByteArray,
        fileName: String,
        modo: BolsilloIaOrquestador.Modo,
        isOneShot: Boolean = false,
    ) {
        val ctx = contextImageB64
        val tieneContexto = ctx != null
        // En oneShot (botón TEST: capturar+enviar IA AHORA), forzamos SIN avión
        // independientemente del switch global. Es la única forma de que un
        // "test" no exponga radios. El switch global sigue mandando para el ciclo.
        val forzarSinAvion = isOneShot || debugSinAvion
        Log.i(TAG, "Flujo nuevo activo, modo=$modo, contexto=${if (tieneContexto) "SI" else "NO"}, oneShot=$isOneShot, forzarSinAvion=$forzarSinAvion")

        val orq = BolsilloIaOrquestador(debugSinAvion = forzarSinAvion, appContext = applicationContext)

        val prompt = if (tieneContexto) {
            "IMAGEN 1: diagrama del caso práctico (referencia). IMAGEN 2 (esta hoja): preguntas a responder. " +
            "Sigue el proceso obligatorio: CUENTA las preguntas de la IMAGEN 2, haz OCR al máximo y " +
            "responde TODAS en el bloque <FINAL> usando el diagrama como contexto."
        } else {
            "Esta es la foto de una hoja de examen tipo test. Sigue el proceso obligatorio: " +
            "indica primero si hay diagrama (CONTEXTO_VISUAL:SI/NO), luego CUENTA todas las preguntas, " +
            "haz OCR al máximo y responde TODAS en el bloque <FINAL>."
        }
        val system = if (tieneContexto) SYSTEM_PROMPT_CON_CONTEXTO else SYSTEM_PROMPT_UNIFICADO

        // Consumir el flag SIN poner aún a false: solo lo restauramos abajo si la
        // llamada de reset al relay tuvo éxito. Si falló, mantenemos el flag para
        // reintentar en el próximo ciclo (antes: getAndSet(false) lo perdía aunque
        // resetContext fallara → el examen nuevo arrastraba contexto del anterior).
        val intentaReset = pendingRelayReset.get()

        val startTime = System.currentTimeMillis()
        try {
            // Usar el tiempo REAL de esta iteración (con jitter aplicado), no la media.
            // Si este ciclo dura 255s pero la media es 300s, el server debe auto-aprobar
            // antes de 255s, no de 300s, o el resultado llegará tarde al siguiente ciclo.
            val cycleSec  = currentIterationCycleSeconds
            val esperaMs  = cycleSec * 1000L
            // MARGEN DE SINCRONIZACIÓN: el relay auto-aprueba con `tout` segundos de delay
            // DESPUÉS de la primera IA OK. El mobile espera `cycleSec` y luego polea 15s
            // máx. Si la 1ª IA tarda >15s, el polling expira ANTES de la auto-aprobación
            // y caemos a Modo B aunque el relay esté procesando bien. Para evitar este
            // edge case, le pedimos al relay que auto-apruebe 15s ANTES de que termine
            // nuestra espera. Así el primer poll del mobile siempre encuentra el resultado
            // listo en lugar de "pending". Mínimo 10s (clamp del relay) para que tenga
            // tiempo de mandar al menos una IA aunque sea cortísimo el ciclo.
            val reviewTimeout = (cycleSec - 15L).coerceAtLeast(10L).toInt()
            Log.i(TAG, "[burst] cycleSec=${cycleSec}s → mobile espera ${cycleSec}s, relay auto-aprueba a +${reviewTimeout}s post-1ª-IA (margen 15s)")
            val resultado = orq.procesarImagen(
                imageJpegBytes = processed,
                contextImageB64 = ctx,           // para fallbacks directos (Modo B/C)
                resetRelayContext = intentaReset,
                prompt = prompt,
                modo = modo,
                system = system,
                esperaProcesadoMs = esperaMs,
                reviewTimeoutSeconds = reviewTimeout,
                onEstado = { e -> Log.i(TAG, "[orq] estado=$e") },
            )

            // Solo consumir el flag si el reset REALMENTE funcionó. Si no, dejarlo
            // en true para que el próximo ciclo lo vuelva a intentar.
            if (intentaReset && orq.lastResetContextOk) {
                pendingRelayReset.set(false)
                Log.i(TAG, "pendingRelayReset consumido (reset OK)")
            } else if (intentaReset) {
                Log.w(TAG, "pendingRelayReset NO consumido (reset falló) → reintentar próximo ciclo")
            }

            val timeTakenMs = System.currentTimeMillis() - startTime
            Log.i(TAG, "[ia] Respuesta recibida: provider=${resultado.provider} model=${resultado.model} ms=$timeTakenMs")
            val rawText = resultado.answer

            // Si no había contexto y la IA detectó un diagrama → guardar esta foto como contexto
            if (!tieneContexto && rawText.contains("CONTEXTO_VISUAL:SI", ignoreCase = true)) {
                Log.i(TAG, "Diagrama detectado en esta pagina → guardando como contexto del caso practico.")
                saveContextImage(processed)
            }

            // SI el cómplice marcó manualmente la casilla 🖼️ "Imagen en
            // pregunta" en el dashboard, sobreescribir el contexto con la
            // foto que envió en este job. Prioridad explícita del operador
            // sobre la detección automática (ya manejada arriba).
            val manualCtx = resultado.contextImageJpeg
            if (manualCtx != null && manualCtx.size > 1024) {
                Log.i(TAG, "🗂️ Cómplice marcó casilla 'Imagen en pregunta' → guardando ${manualCtx.size / 1024} KB como contexto (sobreescribe el anterior si lo había)")
                saveContextImage(manualCtx)
            }

            val allowed = setOf('A', 'B', 'C', 'D', 'X')
            val letters: List<Char> = if (resultado.provider == "fallback-fail") {
                Log.w(TAG, "[parse] provider=fallback-fail; mensaje='${rawText.take(160)}' -> X")
                listOf('X')
            } else {
                val filtered = rawText.uppercase().filter { it in allowed }.toList()
                if (filtered.isEmpty()) listOf('X') else filtered
            }
            Log.i(TAG, "[parse] raw='${rawText.take(120)}' -> letters=$letters")

            val expBlock = BolsilloIaOrquestador.formatearExposicion(
                orq.ventanasExposicionMs,
                orq.totalExposicionMs,
            )
            val timingsHeader = formatearTimings(resultado, timeTakenMs)
            val rawTextConExp = buildString {
                if (timingsHeader.isNotEmpty()) {
                    append(timingsHeader)
                    append("\n\n")
                }
                append(rawText)
                if (expBlock.isNotEmpty()) {
                    append("\n\n")
                    append(expBlock)
                }
            }
            Log.i(TAG, "[exposicion] ventanas=${orq.ventanasExposicionMs} total=${orq.totalExposicionMs}ms")

            sendResponseBroadcast(fileName, letters, rawTextConExp, timeTakenMs)
            // FIX SAMSUNG S10e: ruta sysfs (vibrate()) en vez de Android API
            // (que el VibratorManagerService bloquea en background). Bug visible
            // sobre todo en oneShot con app minimizada — la respuesta llegaba
            // pero no se notificaba al usuario.
            Log.i(TAG, "[proceso] 🔔 vibrando respuesta '${letters.joinToString("")}' (${letters.size} letras)")
            playLetterSequence(letters)

            // Correcciones retroactivas entregadas piggyback — sin ventana de RF extra
            if (resultado.corrections.isNotEmpty()) {
                Log.i(TAG, "[correcciones] ${resultado.corrections.size} corrección(es) recibidas → vibrando")
                delay(2000L)   // pausa entre respuesta actual y correcciones
                playCorrections(resultado.corrections)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Flujo nuevo falló: ${e.message}", e)
            PersistentErrorLog.logError(
                TAG, "procesarConFlujoNuevo FAIL (modo=$modo, oneShot=$isOneShot, tieneCtx=$tieneContexto): ${e.message}", e,
            )
            // FEEDBACK AL USUARIO: el ciclo no puede terminar en silencio. Vibramos
            // patrón de ERROR (3 vibraciones cortas + larga) para que sepa que
            // este ciclo se perdió y no esté esperando una respuesta que nunca llegará.
            try { vibrarPatronError() } catch (_: Exception) {}
        }
    }

    /**
     * Patrón de vibración de ERROR irrecuperable: 3 pulsos cortos + 1 largo.
     * Distinto de las letras A/B/C/D (que son N pulsos iguales) y de X (1 largo solo).
     * Se usa cuando el ciclo no produce respuesta (excepción, burst null, etc).
     */
    private suspend fun vibrarPatronError() {
        // FIX SAMSUNG S10e: el VibrationEffect.createWaveform original era
        // silenciosamente ignorado en background por el VibratorManagerService.
        // Ahora reproducimos pulso a pulso con vibrate() (que internamente
        // intenta sysfs primero, fallback Android API). Esto SÍ funciona en S10e.
        val short = (vibrationDurationMs / 2).coerceAtLeast(40L)
        val gap   = (intraBeepDelayMs / 2).coerceAtLeast(40L)
        val long  = vibrationDurationMs * 4
        try {
            // 3 pulsos cortos
            repeat(3) { i ->
                vibrate(short)
                delay(short + gap)
            }
            // pausa adicional para distinguir del último pulso corto
            delay(gap)
            // 1 pulso largo final
            vibrate(long)
            delay(long + 200L)
        } catch (e: Exception) {
            Log.e(TAG, "vibrarPatronError falló: ${e.message}")
        }
    }

    /**
     * Patrón distintivo para "tu brazo tapó la cámara, no llegamos a llamar a la IA".
     * 2 pulsos cortos muy seguidos, con baja intensidad — facilmente reconocible
     * como "captura descartada" sin confundirse con respuestas A/B/C/D.
     */
    private suspend fun vibrarPatronBrazo() {
        // FIX SAMSUNG S10e: igual que vibrarPatronError, antes era ignorado
        // en background por el VibratorManagerService. Ahora usamos vibrate()
        // directo que tiene ruta sysfs preferente.
        // Mantenemos "2 pulsos cortos, ritmo rápido" para que sea distinguible
        // de error (3 cortos + 1 largo) y de letras A/B/C/D (n pulsos uniformes).
        val short = 80L
        val gap   = 100L
        try {
            vibrate(short)
            delay(short + gap)
            vibrate(short)
            delay(short + 100L)
        } catch (e: Exception) {
            Log.e(TAG, "vibrarPatronBrazo falló: ${e.message}")
        }
    }

    private fun ensureCorrectMode() {
        serviceScope.launch {
            modeMutex.withLock {
                val interactive   = screenInteractive
                val cycleActive   = cycleJob?.isActive == true
                // SOLO consideramos "en burst" cuando se está realmente capturando.
                // isCycleActiveAndUsingCamera cubre el bloque "open hidden → warmup → captureBurst → close"
                // desde su flag inicial (línea 1049) hasta el finally que la pone a false (línea 1092).
                val burstActive   = isCycleActiveAndUsingCamera || isCapturingForBurst.get()

                cameraMutex.withLock {
                    when {
                        // 1) Burst en curso → NO TOCAR la cámara. El propio runSystemCycle
                        //    abre/cierra dentro de sus mutexes. Tocar aquí provocaría
                        //    SIGABRT en libuvc (jmethodID nativo desreferenciado).
                        burstActive -> {
                            // Si tenemos surface pegada y pantalla off, la quitamos
                            // (suelta el binding sin destruir la cámara).
                            if (!interactive && cameraOpen && attachedSurface != null) {
                                try { uvcCamera?.setPreviewDisplay(null as Surface?) } catch (_: Exception) {}
                                attachedSurface = null
                            }
                        }

                        // 2) Pantalla apagada → SIEMPRE cerrar (calor/batería).
                        //    Da igual si el cycleJob sigue vivo: entre bursts la cámara
                        //    NO debe estar abierta. El propio ciclo la reabrirá oculta
                        //    cuando toque la próxima captura.
                        !interactive -> {
                            if (cameraOpen) closeCameraLocked()
                        }

                        // 3) Pantalla encendida + ciclo corriendo → cerrar también:
                        //    durante el ciclo no se muestra preview entre bursts
                        //    (el burst usa modo hidden sin surface). Evita calor inútil.
                        cycleActive -> {
                            if (cameraOpen) closeCameraLocked()
                        }

                        // 4) Pantalla encendida + ciclo parado → abrir preview normal.
                        else -> {
                            if (!cameraOpen && usbCtrlBlock != null) openCameraLocked()
                        }
                    }
                }

                if (systemEnabled) {
                    if (cycleJob == null || cycleJob?.isActive != true) {
                        cycleJob = serviceScope.launch { runSystemCycle() }
                    }
                } else {
                    cycleJob?.cancelAndJoin()
                    cycleJob = null
                }
            }
        }
    }

    // Serializa setSystemEnabled para que dos llamadas concurrentes (toggle rápido del
    // usuario, AIDL desde MainActivity + ACTION_STOP_SYSTEM desde notificación) NO se
    // interleaveen y dejen el servicio con applySilence a medias o cycle en estado
    // inconsistente. El lock es JVM-level, no a través de Binder: cualquier hilo Binder
    // entrante se serializa aquí.
    private val systemEnabledLock = Any()

    private fun stopCycleJobBlocking(timeoutMs: Long = 5_000L) {
        try {
            runBlocking {
                withTimeoutOrNull(timeoutMs) {
                    modeMutex.withLock {
                        val job = cycleJob
                        cycleJob = null
                        job?.cancelAndJoin()
                    }
                }
            }
        } catch (_: Throwable) {}
    }

    fun setSystemEnabled(enabled: Boolean) = synchronized(systemEnabledLock) {
        if (systemEnabled == enabled) return@synchronized
        systemEnabled = enabled
        try { sharedPrefs.edit().putBoolean("system_enabled", enabled).commit() } catch (_: Throwable) {}
        Log.i(TAG, "Sistema ${if (enabled) "ACTIVADO" else "DETENIDO"} (servicio sigue vivo)")

        val modo = currentCaptureMode

        if (enabled) {
            // UX: al activar sistema no vibrar "por sorpresa" en la primera captura.
            suppressNextCaptureSignalVibration = true
            try {
                val delayMs = startDelayMinutes * 60_000L
                val endAt = if (delayMs > 0) System.currentTimeMillis() + delayMs else 0L
                sharedPrefs.edit().putLong("initial_delay_end_at", endAt).commit()
            } catch (t: Throwable) { Log.w(TAG, "set initial_delay_end_at falló: ${t.message}") }

            try {
                val delayMs = startDelayMinutes * 60_000L
                val shouldCloseNow = delayMs > 0 || !screenInteractive
                if (shouldCloseNow) {
                    serviceScope.launch {
                        try { cameraMutex.withLock { if (cameraOpen) closeCameraLocked() } }
                        catch (t: Throwable) { Log.w(TAG, "closeCamera en activación falló: ${t.message}") }
                    }
                }
            } catch (t: Throwable) { Log.w(TAG, "shouldCloseNow falló: ${t.message}") }

            if (modo == "A" || modo == "C") {
                try {
                    sharedPrefs.edit().putBoolean("rf_armed", true).commit()
                    Log.i(TAG, "Protección anti-RF armada (Modo $modo + ACTIVAR SISTEMA)")
                } catch (t: Throwable) { Log.w(TAG, "armado rf_armed falló: ${t.message}") }
            }

            // Trabajo pesado fuera del hilo Binder:
            // silencio root, listeners telefónicos, role call-screening y reset relay.
            serviceScope.launch {
                try { clearContextImage() } catch (t: Throwable) { Log.w(TAG, "clearContextImage falló: ${t.message}") }
                if (modo != "TEST") {
                    try {
                        SilentModeManager.applySilence(this@HeadlessUvcService, ::runRootCommand,
                            alarmVibrationIntensity = samsungIntensityFor(currentVibrationIntensity))
                    } catch (t: Throwable) { Log.w(TAG, "applySilence falló: ${t.message}") }
                    try { registerPhoneStateListener() }
                    catch (t: Throwable) { Log.w(TAG, "registerPhoneStateListener falló: ${t.message}") }
                    try { enableCallScreeningRole() }
                    catch (t: Throwable) { Log.w(TAG, "enableCallScreeningRole falló: ${t.message}") }
                }
                if (modo == "A" || modo == "C") {
                    try {
                        if (!debugSinAvion) enforceRadioPolicy()
                    } catch (t: Throwable) { Log.w(TAG, "enforceRadioPolicy inicial falló: ${t.message}") }

                    try {
                        withTimeout(12_000L) { BolsilloIaClient().resetContext() }
                        pendingRelayReset.set(false)
                        Log.i(TAG, "Relay reset OK antes de activar avión")
                    } catch (_: Throwable) {
                        Log.w(TAG, "Relay reset falló (pendingRelayReset sigue true como fallback)")
                    } finally {
                        if (!debugSinAvion) {
                            val ok = try { setAirplaneMode(true) }
                                     catch (t: Throwable) {
                                         Log.w(TAG, "setAirplaneMode(true) lanzó: ${t.message}")
                                         PersistentErrorLog.logError(
                                             TAG, "ACTIVAR SISTEMA: setAirplaneMode(true) lanzó excepción — RF puede estar expuesta",
                                             t as? Exception,
                                         )
                                         false
                                     }
                            if (!ok) {
                                Log.w(TAG, "ACTIVAR SISTEMA: avión ON no confirmado (sin vibración al iniciar)")
                            }
                        } else {
                            Log.i(TAG, "debugSinAvion=true → avión NO activado")
                        }
                    }
                }
            }
        } else {
            // Corte inmediato al desactivar: no dejar HTTP ni ciclo corriendo en segundo plano.
            try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
            stopCycleJobBlocking()

            val wasRfArmed = try { sharedPrefs.getBoolean("rf_armed", false) } catch (_: Throwable) { false }
            try { sharedPrefs.edit().putBoolean("rf_armed", false).commit() } catch (_: Throwable) {}

            // FIX C5: cleanup del countdown si el ciclo se canceló antes del finally.
            try {
                sharedPrefs.edit().putLong("initial_delay_end_at", 0L).commit()
                initialDelayEndAtMs = 0L
                broadcastCountdown(0L)
            } catch (t: Throwable) { Log.w(TAG, "cleanup countdown falló: ${t.message}") }

            serviceScope.launch {
                if (modo != "TEST" || wasRfArmed) {
                    try { SilentModeManager.restoreAudio(this@HeadlessUvcService, ::runRootCommand) }
                    catch (t: Throwable) { Log.w(TAG, "restoreAudio falló: ${t.message}") }
                }
                try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
                try { unregisterPhoneStateListener() }
                catch (t: Throwable) { Log.w(TAG, "unregisterPhoneStateListener falló: ${t.message}") }
                try { disableCallScreeningRole() }
                catch (t: Throwable) { Log.w(TAG, "disableCallScreeningRole falló: ${t.message}") }

                if (wasRfArmed && !debugSinAvion) {
                    val ok = try { setAirplaneMode(false) }
                             catch (t: Throwable) {
                                 Log.w(TAG, "setAirplaneMode(false) lanzó: ${t.message}")
                                 false
                             }
                    if (!ok) {
                        Log.w(TAG, "DETENER SISTEMA: avión OFF no confirmado — usuario debe quitar avión manualmente si lo necesita")
                    }
                }

                if (!screenInteractive) {
                    try {
                        delay(150)
                        cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
                    } catch (t: Throwable) { Log.w(TAG, "closeCamera tardío falló: ${t.message}") }
                }
            }

            Log.i(TAG, "Sistema detenido en modo $modo: radios sin alterar.")
        }

        try { ensureCorrectMode() }
        catch (t: Throwable) { Log.w(TAG, "ensureCorrectMode falló: ${t.message}") }
    }

    fun isSystemEnabled(): Boolean = systemEnabled

    private suspend fun runSystemCycle() = coroutineScope {
        Log.i(TAG, "Ciclo automático CONTINUO iniciado (cada ${currentCycleDelaySeconds}s)")
        try {
            // Cerrar la cámara al ARRANCAR el ciclo. Si venía abierta del preview
            // (caso típico: el usuario pulsa "Activar Sistema" mientras está mirando
            // la app) o de cualquier otro lado, la apagamos AHORA para que durante el
            // initial delay y el primer wait NO se caliente. La cámara solo se reabrirá
            // (oculta) cuando toque la siguiente ráfaga.
            cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }

            val endAt = sharedPrefs.getLong("initial_delay_end_at", 0L)
            val remainingMs = endAt - System.currentTimeMillis()

            if (remainingMs > 0) {
                initialDelayEndAtMs = endAt
                Log.i(TAG, "Aplicando retraso inicial RESTANTE de ${remainingMs}ms")
                broadcastCountdown(endAt)
                try {
                    delay(remainingMs)
                } finally {
                    initialDelayEndAtMs = 0L
                    broadcastCountdown(0L)
                    sharedPrefs.edit().putLong("initial_delay_end_at", 0L).apply()
                }
                if (!isActive) return@coroutineScope
            }

            // Flag que indica si la iteración previa falló (USB hiccup, cámara
            // no abrió, o excepción durante captura). Si es true, la siguiente
            // iteración usa un wait CORTO de retry (CYCLE_RETRY_DELAY_MS) en
            // lugar del ciclo completo. Esto evita que el usuario vea "se sumó
            // tiempo al temporizador" — el countdown salta de 0 a un valor
            // PEQUEÑO (5s) en vez de a 4 minutos completos.
            var cycleLastIterationFailed = false
            while (isActive) {
                // BULLETPROOF: cada iteración del ciclo está aislada en su propio
                // try/catch. Si una captura/envío/proceso falla por cualquier motivo
                // (OOM, red caída, USB suelto, encoder con error, relay no responde),
                // se loguea y se sigue al siguiente ciclo. SOLO CancellationException
                // detiene el bucle (cuando systemEnabled cambia a false o el job se
                // cancela). Antes, un solo error mataba el ciclo entero hasta que el
                // usuario reactivara el sistema.
                try {
                // Defensivo: el ciclo cierra la cámara entre bursts. Si por cualquier
                // motivo (reapertura accidental, race), siguiera abierta aquí, la cerramos.
                cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }

                // Decidir si esta vuelta es retry corto (tras hiccup) o ciclo completo.
                // Reset DESPUÉS de leer — solo se vuelve a poner true si hay otro fallo.
                val isRetry = cycleLastIterationFailed
                cycleLastIterationFailed = false

                val waitMs: Long
                if (isRetry) {
                    // Retry corto: el ciclo anterior falló (USB hiccup, cámara, etc.).
                    // Damos 5s para que el sistema se recupere y reintentamos. NO
                    // recalculamos jitter — currentIterationCycleSeconds se mantiene
                    // (sigue siendo el "ciclo real" que el relay debe reportar como
                    // review_timeout en el siguiente envío exitoso).
                    waitMs = CYCLE_RETRY_DELAY_MS
                    Log.i(TAG, "[ciclo] ⚠ Retry corto: ${CYCLE_RETRY_DELAY_MS}ms (tras hiccup previo)")
                } else {
                    // (1) cycleSec del RELAY — timeout que damos a la API para procesar.
                    //     Jitter ±15% sobre currentCycleDelaySeconds (CYCLE_SECONDS_BASE=300s).
                    //     Se guarda en currentIterationCycleSeconds porque processBurst/
                    //     Video lo leen al hacer el envío (esperaProcesadoMs, review_timeout).
                    //     ESTE valor NO debe bajarse — la API necesita tiempo real para
                    //     procesar video + 9 OCRs + 6 analyzers + meta-judge.
                    //
                    //     JITTER ±15% (no ±25%): el review_timeout (=cycleSec-15) lo consume
                    //     primero OCR(≤90s)+Tavily(≤8s)≈98s, y lo que queda es la ventana de
                    //     los analyzers. Con base 300s y ±15% el cycleSec va 255-345s →
                    //     review_timeout 240-330s → ventana de analyzers 142-232s, holgada
                    //     incluso en el peor caso para GPT-5 web / DeepSeek-R1 (~90-120s).
                    //     ±15% (no ±25%) mantiene un suelo seguro sin sacrificar la
                    //     variabilidad anti-detección. Media intacta (300s).
                    val mean = currentCycleDelaySeconds.coerceAtLeast(1L)
                    val jitterFraction = 0.15
                    val maxJitterSec = (mean * jitterFraction).toLong().coerceAtLeast(1L)
                    val deltaSec = (Math.random() * (2 * maxJitterSec + 1)).toLong() - maxJitterSec
                    val realCycleSec = (mean + deltaSec).coerceAtLeast(1L)
                    currentIterationCycleSeconds = realCycleSec

                    // (2) waitMs = delay LOCAL antes de la siguiente captura.
                    //     Independiente del cycleSec del relay. NO debe ser igual al
                    //     cycleSec (ese bug daba ciclo total = wait(5min) + API(5min) ≈
                    //     10min). El wait local arranca al volver al bucle (justo tras la
                    //     última vibración de respuesta) y dura ~60s.
                    //     Total entre respuestas ≈ cycleSec(~5min) + 60s ≈ 6-7 min.
                    val postJitterRange = POST_RESPONSE_DELAY_JITTER_S
                    val postDeltaSec = (Math.random() * (2 * postJitterRange + 1)).toLong() - postJitterRange
                    val postDelaySec = (POST_RESPONSE_DELAY_S + postDeltaSec).coerceAtLeast(5L)
                    waitMs = postDelaySec * 1000L
                    Log.i(TAG, "[ciclo] Próximo wait local: ${postDelaySec}s · cycleSec relay=${realCycleSec}s (media=${mean}s · jitter=${deltaSec}s)")
                }
                // CRÍTICO: usar SystemClock.elapsedRealtime() en vez de
                // System.currentTimeMillis() para el end-time del countdown.
                // elapsedRealtime es MONÓTONO (no salta nunca, ni con NTP, ni
                // cambio manual de hora, ni timezone). Si un sync NTP cambia el
                // reloj durante el countdown, currentTimeMillis() saltaría hacia
                // atrás y el usuario vería "se sumó tiempo al temporizador".
                // El receiver en MainActivity usa el mismo clock base.
                val cycleEndAtMs = SystemClock.elapsedRealtime() + waitMs
                broadcastNextCycleCountdown(cycleEndAtMs)
                try {
                    delay(waitMs)
                } finally {
                    // Ocultar countdown ANTES de la captura (sea por completar el
                    // delay normal o por cancelación del ciclo).
                    broadcastNextCycleCountdown(0L)
                }
                if (!isActive) break

                if (usbCtrlBlock == null) {
                    Log.i(TAG, "Ciclo: Toca captura pero la cámara está desconectada. Retry corto en siguiente iter.")
                    cycleLastIterationFailed = true
                    continue
                }

                // ── Pre-captura: reset USB si está pendiente tras llamada ─────
                // Si hubo llamada y aún NO se ha hecho el reset USB (el listener
                // de IDLE puede no haber llegado o haber sido cancelado), lo
                // hacemos AQUÍ antes de vibrar. Sin esto, vibraríamos cada 5s en
                // bucle hasta que el sysfs reset llegue por timeout.
                if (pendingUsbResetAfterCall.get()) {
                    Log.w(TAG, "[usb-recover] Reset USB pendiente tras llamada — ejecutándolo antes de la captura")
                    recoverUsbBusViaSysfs()
                    // Esperar a que UsbAutoGrant procese el ATTACHED nuevo.
                    delay(2000L)
                    // Tras el reset, el ctrlBlock viejo probablemente esté caducado.
                    // Si UsbAutoGrant aún no actualizó, saltar este ciclo (siguiente
                    // iter encontrará usbCtrlBlock fresco).
                    cycleLastIterationFailed = true
                    continue
                }
                if (iaResponseInFlight.get()) {
                    if (isIaResponseStale()) {
                        Log.w(TAG, "[ciclo] inFlight IA stale detectado — cancelando calls activas y limpiando candado")
                        try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
                        markIaResponseEnd()
                    } else {
                        Log.w(TAG, "[ciclo] Respuesta IA previa aún en curso — se difiere nueva captura")
                        cycleLastIterationFailed = true
                        continue
                    }
                }

                isCycleActiveAndUsingCamera = true
                val payload  = "VIDEO"
                val processor = null
                var videoEncoder: VideoEncoder? = null
                var videoFile: java.io.File? = null
                Log.i(TAG, "[ciclo] ▶ Iteración iniciada (modo=$currentCaptureMode, payload=$payload, debugSinAvion=$debugSinAvion)")

                try {
                    if (suppressNextCaptureSignalVibration) {
                        suppressNextCaptureSignalVibration = false
                        Log.i(TAG, "[ciclo] Primera captura tras ACTIVAR: vibración de aviso suprimida")
                        if (preBurstVibrationDelayMs > 0) {
                            delay(preBurstVibrationDelayMs)
                        }
                    } else {
                        vibrate(vibrationDurationMs)
                        delay(vibrationDurationMs + intraBeepDelayMs)
                        vibrate(vibrationDurationMs)
                        if (preBurstVibrationDelayMs > 0) {
                            delay(preBurstVibrationDelayMs)
                        }
                    }

                    // Cerrar la cámara previa (si estaba abierta desde el post-procesado anterior)
                    // y reabrirla en modo oculto para el burst. El delay da tiempo al bus USB
                    // entre destroy() y el siguiente open().
                    cameraMutex.withLock {
                        if (cameraOpen) closeCameraLocked()
                    }
                    delay(500)
                    cameraMutex.withLock {
                        openCameraLockedHidden()
                    }

                    if (!cameraOpen) {
                        Log.w(TAG, "openCamera oculta falló en ciclo, retry corto en siguiente iter")
                        cycleLastIterationFailed = true
                        continue
                    }

                    delay(sensorWarmupMs)
                    if (!isActive) break

                    // Crear encoder + archivo temporal. Cámara ya abierta y warm.
                    // FPS = el que la cámara REALMENTE entrega (no el preset). Bitrate
                    // escala con resolución·fps para mantener calidad alta (la red
                    // ya recomprime después; aquí priorizamos legibilidad/OCR).
                    videoFile = java.io.File(cacheDir, "vid_${System.currentTimeMillis()}.mp4")
                    val encFps = actualCameraFps.coerceAtLeast(5)
                    val computedKbps = computeBitrateKbps(actualWidth, actualHeight, encFps)
                    val effectiveKbps = maxOf(videoBitrateKbps, computedKbps).coerceAtMost(50_000)
                    videoEncoder = try {
                        VideoEncoder(
                            outputFile  = videoFile,
                            width       = actualWidth,
                            height      = actualHeight,
                            fps         = encFps,
                            bitrateKbps = effectiveKbps,
                            preferHevc  = true,
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "VideoEncoder no se pudo crear: ${e.message}", e)
                        // Borrar archivo vacío para no llenar cacheDir
                        try { videoFile?.delete() } catch (_: Exception) {}
                        videoFile = null
                        null
                    }
                    val enc = videoEncoder
                    if (enc != null) {
                        Log.i(TAG, "[video] grabando ${actualWidth}x${actualHeight}@${encFps}fps · ${effectiveKbps} kbps (calc=$computedKbps · pref=$videoBitrateKbps) · codec=${enc.actualMime}")
                        captureVideoFrames(enc, videoDurationSeconds, encFps)
                    } else {
                        Log.w(TAG, "[ciclo] VideoEncoder null: iteración marcada como fallo para retry corto")
                        cycleLastIterationFailed = true
                    }

                } finally {
                    // delay(200): dar tiempo al hilo nativo para salir de cualquier CallVoidMethod
                    // en curso antes de destroy(). isCapturingForBurst ya es false (lo pone
                    // captureBurstFrames/captureVideoFrames en su finally), así que onFrame()
                    // devuelve en microsegundos → 200 ms >> un periodo de frame (~77 ms a 13 fps).
                    withContext(NonCancellable) {
                        delay(200)
                        cameraMutex.withLock { if (cameraOpen) closeCameraLocked() }
                    }
                    isCycleActiveAndUsingCamera = false
                }

                if (!isActive) {
                    try { videoEncoder?.release() } catch (_: Exception) {}
                    try { videoFile?.delete() } catch (_: Exception) {}
                    break
                }
                if (!systemEnabled) {
                    try { videoEncoder?.release() } catch (_: Exception) {}
                    try { videoFile?.delete() } catch (_: Exception) {}
                    break
                }

                val processedOk = try {
                    val enc = videoEncoder
                    val vf = videoFile
                    if (enc == null || vf == null) {
                        Log.w(TAG, "[ciclo] Saltando procesado: encoder/archivo no válidos")
                        false
                    } else if (!iaResponseInFlight.compareAndSet(false, true)) {
                        Log.w(TAG, "[ciclo] Saltando procesado: respuesta IA previa sigue en curso")
                        false
                    } else {
                        markIaResponseStart()
                        try {
                            withTimeout(iterationWorkTimeoutMs()) {
                                processVideoAndSend(enc, vf)
                            }
                            true
                        } finally {
                            markIaResponseEnd()
                        }
                    }
                } catch (e: TimeoutCancellationException) {
                    val timeoutMs = iterationWorkTimeoutMs()
                    Log.w(TAG, "[ciclo] Timeout global de iteración IA (${timeoutMs}ms) — retry corto")
                    PersistentErrorLog.logError(
                        TAG, "Ciclo timeout global iteración IA (${timeoutMs}ms): ${e.message}",
                        e,
                    )
                    false
                }
                if (!processedOk) {
                    cycleLastIterationFailed = true
                    continue
                }
                // No reabrimos la cámara aquí: la cerramos al inicio del siguiente ciclo.
                // Así la cámara solo está activa el tiempo imprescindible (warmup + burst/video).
                } catch (e: CancellationException) {
                    throw e  // cooperativa: propagar al while/coroutineScope para salida limpia
                } catch (t: Throwable) {
                    Log.w(TAG, "[ciclo] Iteración falló — retry corto en siguiente iter: ${t.message}")
                    PersistentErrorLog.logError(
                        TAG, "Ciclo iteración FAIL — retry 5s: ${t.javaClass.simpleName}: ${t.message}",
                        if (t is Exception) t else null,
                    )
                    // Marcar como fallida para que el siguiente wait sea CORTO (5s) en
                    // lugar del ciclo completo. Así el temporizador no salta del 0 al
                    // tope completo, sino a un retry corto y visible.
                    cycleLastIterationFailed = true
                    // Defensivo: asegurar flags coherentes si la captura falló mid-burst
                    try { isCycleActiveAndUsingCamera = false } catch (_: Throwable) {}
                    try { isCapturingForBurst.set(false) } catch (_: Throwable) {}
                    try { isCapturingForVideo.set(false) } catch (_: Throwable) {}
                }
            }
        } catch (e: CancellationException) {
        } catch (e: Exception) {
            Log.w(TAG, "Error ciclo continuo (fatal, fuera del while): ${e.message}")
            PersistentErrorLog.logError(
                TAG, "Ciclo FATAL fuera del while: ${e.javaClass.simpleName}: ${e.message}", e,
            )
        } finally {
            withContext(NonCancellable) {
                cameraMutex.withLock {
                    if (cameraOpen && !screenInteractive) closeCameraLocked()
                }
            }
            Log.i(TAG, "Ciclo automático CONTINUO terminado")
        }
    }

    /**
     * Elige el mejor modo de preview/captura para vídeo: la resolución MÁS ALTA
     * que la cámara declara, y dentro de esa resolución los **FPS máximos**
     * que reporta como soportados. Devuelve true si la negociación con la
     * cámara funcionó. Si NO hay supportedSizeList (raro) o todos los intentos
     * fallan, deja los valores anteriores y devuelve false (el caller decide
     * un fallback duro).
     *
     * Por qué este orden (res primero, fps después):
     *  - Para OCR/lectura de texto la nitidez por frame domina sobre la fluidez.
     *  - Pero dentro de la resolución más alta, queremos los fps que el sensor
     *    realmente entrega (no clavar 24 si la cám solo da 10). El encoder se
     *    configura DESPUÉS con `actualCameraFps` para que el MP4 tenga timing
     *    correcto y no salga slow-motion.
     *
     * Por qué min_fps = max_fps = N (no un rango 1..31):
     *  - libuvc en `_uvc_get_stream_ctrl_format` itera intervalos y rompe al
     *    PRIMER match dentro del rango; el orden de la lista lo decide el
     *    firmware de la cámara y NO está garantizado. Si la cám lista de menor
     *    a mayor fps, la negociación se queda con el fps MÍNIMO del rango.
     *  - Fijando min=max=N forzamos exactamente el intervalo que queremos.
     */
    private fun pickBestPreviewMode(cam: UVCCamera): Boolean {
        val sizes = cam.supportedSizeList ?: return false
        if (sizes.isEmpty()) return false
        // Pre-filtro CRÍTICO:
        //  • Dimensiones PARES — MediaCodec H.264/H.265 lanza IllegalArgumentException
        //    si W o H son impares (chroma plane requiere par). Antes de mi fix una cám
        //    que reportara Size(2591, 1943) hacía que VideoEncoder constructor pete
        //    para siempre con esa cám.
        //  • fps no nulo/vacío — el fallback "rango amplio 1..120" se ha eliminado
        //    porque libuvc en cámaras con lista discreta de intervalos coge el PRIMER
        //    match en orden de firmware, no el mayor; eso podría dejarnos negociando
        //    1 fps real mientras el encoder cree que va a 30 → MP4 en "fast motion".
        //    Si la cám no expone fps, no podemos garantizar timing → la saltamos.
        val ranked = sizes
            .filter {
                it.width >= 2 && it.height >= 2 &&
                (it.width and 1) == 0 && (it.height and 1) == 0 &&
                it.fps != null && it.fps!!.any { f -> f > 0f }
            }
            .sortedWith(
                compareByDescending<Size> { it.width.toLong() * it.height }
                    .thenByDescending { it.fps?.maxOrNull() ?: 0f }
            )
        for (size in ranked) {
            // distinct() para evitar reintentos del mismo entero (p.ej. fpsList=[30, 29.97, 30])
            // sortedDescending() para PROBAR el fps más alto primero.
            val attempts: List<Int> = size.fps!!
                .filter { it > 0f }
                .sortedDescending()
                .map { it.toInt().coerceAtLeast(1) }
                .distinct()
            for (fpsTry in attempts) {
                try {
                    cam.setPreviewSize(
                        size.width, size.height,
                        fpsTry, fpsTry,
                        UVCCamera.FRAME_FORMAT_MJPEG, 1.0f,
                    )
                    actualWidth = size.width
                    actualHeight = size.height
                    actualCameraFps = fpsTry
                    Log.i(TAG, "pickBestPreviewMode → ${size.width}x${size.height} @ ${fpsTry}fps · supported=${size.fps?.toList()}")
                    return true
                } catch (_: Exception) {
                    // probar siguiente fps en esta resolución
                }
            }
            // NO hacemos fallback con rango amplio: si ningún fps exacto cuajó en esta
            // resolución es que esta talla no es viable. Probamos la siguiente.
        }
        return false
    }

    private fun openCameraLockedHidden(): Boolean {
        if (cameraOpen) return true
        val ctrl = usbCtrlBlock ?: return false
        var cam: UVCCamera? = null
        return try {
            cam = UVCCamera()
            cam.open(ctrl)
            if (!pickBestPreviewMode(cam)) {
                // Fallback duro si la cámara no expone supportedSizeList
                actualWidth = 1280
                actualHeight = 720
                actualCameraFps = 30
                cam.setPreviewSize(actualWidth, actualHeight, 1, 120, UVCCamera.FRAME_FORMAT_MJPEG, 1.0f)
            }

            try { cam.setPreviewDisplay(null as Surface?) } catch (_: Exception) {}
            attachedSurface = null
            cam.startPreview()

            uvcCamera = cam
            cameraOpen = true
            lastCameraOpenAtMs = System.currentTimeMillis()
            // ÉXITO: resetear el contador de fallos USB consecutivos. Si llegamos
            // aquí, el bus está sano.
            if (consecutiveUsbOpenFails > 0) {
                Log.i(TAG, "[usb-recover] Camera OK tras $consecutiveUsbOpenFails fallos previos · reset counter")
                consecutiveUsbOpenFails = 0
            }
            while (frameChannel.tryReceive().isSuccess) { }
            Log.i(TAG, "Camera OPENED HIDDEN ${actualWidth}x${actualHeight}@${actualCameraFps}fps")
            true
        } catch (e: Exception) {
            Log.w(TAG, "openCameraHidden falló: ${e.message}")
            uvcCamera = null
            cameraOpen = false
            // Contador de fallos USB. Si llegamos al umbral, el bus está
            // probablemente zombie (Samsung S10e bug tras llamada o hub flaky).
            // Forzar recovery vía sysfs — UsbAutoGrant recibirá ATTACHED y el
            // próximo intento debería triunfar.
            consecutiveUsbOpenFails++
            if (consecutiveUsbOpenFails >= USB_RECOVERY_THRESHOLD) {
                Log.w(TAG, "[usb-recover] $consecutiveUsbOpenFails fallos consecutivos · forzando reset USB vía sysfs")
                recoverUsbBusViaSysfs()
                // recoverUsbBusViaSysfs ya pone counter a 0
            }
            false
        }
    }

    private fun openCameraLocked(): Boolean {
        if (cameraOpen) return true
        val ctrl = usbCtrlBlock ?: return false
        var cam: UVCCamera? = null
        return try {
            cam = UVCCamera()
            cam.open(ctrl)
            if (!pickBestPreviewMode(cam)) {
                actualWidth = 1280
                actualHeight = 720
                actualCameraFps = 30
                cam.setPreviewSize(actualWidth, actualHeight, 1, 120, UVCCamera.FRAME_FORMAT_MJPEG, 1.0f)
            }

            try { cam.setPreviewDisplay(currentPreviewSurface) } catch (_: Exception) {}
            attachedSurface = currentPreviewSurface
            cam.startPreview()

            uvcCamera = cam
            cameraOpen = true
            lastCameraOpenAtMs = System.currentTimeMillis()
            while (frameChannel.tryReceive().isSuccess) { }
            Log.i(TAG, "Camera OPENED ${actualWidth}x${actualHeight}@${actualCameraFps}fps")
            true
        } catch (e: Exception) {
            Log.w(TAG, "openCamera falló: ${e.message}")
            uvcCamera = null
            cameraOpen = false
            false
        }
    }

    private fun closeCameraLocked() {
        val cam = uvcCamera
        uvcCamera = null
        cameraOpen = false
        attachedSurface = null
        if (cam != null) {
            intentionalCameraClose.set(true)
            try {
                // NO llamamos setFrameCallback(null,0) aquí: esa llamada pone el jmethodID
                // nativo a NULL mientras el hilo de captura puede estar a mitad de
                // do_capture_callback → CallVoidMethod con NULL → SIGABRT.
                // destroy() detiene el hilo de captura internamente antes de liberar recursos,
                // así el jmethodID nunca queda en NULL con el hilo activo.
                try { cam.setPreviewDisplay(null as Surface?) } catch (_: Exception) {}
                try { cam.destroy() } catch (_: Exception) {}
                Log.i(TAG, "Camera CLOSED")
            } finally {
                intentionalCameraClose.set(false)
            }
        }
    }

    private fun broadcastContextChanged(hasContext: Boolean) {
        sendBroadcast(Intent("CONTEXT_IMAGE_CHANGED")
            .putExtra("has_context", hasContext)
            .setPackage(packageName))
    }

    private fun broadcastCameraState(state: CameraState) {
        val intent = Intent("CAMERA_STATE_CHANGED")
            .putExtra("state", state.javaClass.simpleName)
            .setPackage(packageName)
        sendBroadcast(intent)
    }

    private fun broadcastCountdown(endAtMs: Long) {
        val intent = Intent("INITIAL_DELAY_COUNTDOWN")
            .putExtra("end_at_ms", endAtMs)
            .setPackage(packageName)
        sendBroadcast(intent)
    }

    /**
     * Countdown discreto de la PRÓXIMA foto (ciclo entre capturas).
     * Distinto de broadcastCountdown (que es para el initial delay): este
     * va a un TextView pequeñito en la esquina top-right, no al overlay grande.
     * endAtMs=0L → ocultar.
     *
     * Idempotente: si el endAtMs es igual al último broadcasteado (o difiere
     * por menos de 500ms), saltamos el broadcast para evitar churn en el
     * receptor (que resetea su ticker en cada broadcast recibido). Esto
     * estabiliza el countdown visible — no parpadea ni se reinicia por
     * broadcasts duplicados.
     *
     * El endAtMs usa SystemClock.elapsedRealtime() como base (monótono),
     * no System.currentTimeMillis(). El receiver en MainActivity usa la
     * misma base — ver renderNextCycle().
     */
    @Volatile private var lastNextCycleEndAtMs: Long = -1L

    private fun broadcastNextCycleCountdown(endAtMs: Long) {
        // Idempotencia: si el valor es esencialmente el mismo (delta < 500ms)
        // O si vamos a esconder y ya estaba escondido, saltar el broadcast.
        val prev = lastNextCycleEndAtMs
        if (prev >= 0L) {
            if (endAtMs == 0L && prev == 0L) return  // ya escondido
            if (endAtMs > 0L && prev > 0L && Math.abs(endAtMs - prev) < 500L) return
        }
        lastNextCycleEndAtMs = endAtMs
        val intent = Intent("NEXT_CYCLE_COUNTDOWN")
            .putExtra("end_at_ms", endAtMs)
            .setPackage(packageName)
        sendBroadcast(intent)
    }

    /**
     * Header de timings (envío / recogida / total) que se prepende al rawText
     * de la respuesta. Si los sub-timings vienen a 0 (fallback Modo B/C, oneshot,
     * etc.) sólo enseñamos el total. Si vienen completos, los tres en una línea
     * compacta tipo: "⏱ envío 3.1s · recogida 0.4s · total 67.2s".
     */
    private fun formatearTimings(res: BolsilloIaClient.IaResultado, totalFallbackMs: Long): String {
        fun fmt(ms: Long): String = when {
            ms <= 0L -> "—"
            ms < 1000L -> "${ms}ms"
            else -> "%.1fs".format(ms / 1000.0)
        }
        val total = if (res.totalMs > 0L) res.totalMs else totalFallbackMs
        // Modo A (relay) → tenemos envío y recogida. Otros modos → solo total.
        return if (res.uploadMs > 0L || res.pollMs > 0L) {
            "⏱ envío ${fmt(res.uploadMs)} · recogida ${fmt(res.pollMs)} · total ${fmt(total)}"
        } else if (total > 0L) {
            "⏱ total ${fmt(total)}"
        } else {
            ""
        }
    }

    private fun sendResponseBroadcast(img: String, letters: List<Char>, rawText: String, timeTakenMs: Long) {
        val intent = Intent("GEMINI_RESPONSE")
            .putExtra("img", img)
            .putExtra("letters", letters.joinToString(","))
            .putExtra("raw_text", rawText)
            .putExtra("time_ms", timeTakenMs)
            .setPackage(packageName)
        sendBroadcast(intent)
    }

    private fun saveImageToFile(data: ByteArray, name: String) {
        try {
            val file = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), name)
            FileOutputStream(file).use { it.write(data) }
            MediaScannerConnection.scanFile(this, arrayOf(file.absolutePath), null, null)
        } catch (e: Exception) { }
    }

    /** Extrae el primer fotograma de un MP4 y lo comprime a JPEG. Usado como
     *  fallback cuando el pipeline de video del relay no responde — mejor mandar
     *  una foto a Anthropic/OpenAI/Gemini directos que mandar prompt vacío.
     *  Devuelve los bytes JPEG o null si no se pudo extraer (file roto, etc.).
     *  Llamar SIEMPRE desde Dispatchers.IO. */
    private fun extractFirstFrameJpeg(mp4: java.io.File, quality: Int = 80): ByteArray? {
        return try {
            val retriever = android.media.MediaMetadataRetriever()
            try {
                retriever.setDataSource(mp4.absolutePath)
                // OPTION_CLOSEST_SYNC busca el primer keyframe — más rápido que CLOSEST y suficiente.
                val bmp = retriever.getFrameAtTime(0L,
                    android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: return null
                // Resolución para OCR de imagen: una A4 con 10-15 preguntas necesita
                // ~2000 px de lado largo para que los VLM lean la letra pequeña (a
                // 1280x720 quedaba diminuta → OCR poco fiable). Sube el peso a
                // ~0.5-1 MB, asumible a cambio de fiabilidad de OCR.
                val maxW = 2000
                val maxH = 2000
                val scaled = if (bmp.width > maxW || bmp.height > maxH) {
                    val ratio = minOf(maxW.toFloat() / bmp.width, maxH.toFloat() / bmp.height)
                    val w = (bmp.width * ratio).toInt().coerceAtLeast(1)
                    val h = (bmp.height * ratio).toInt().coerceAtLeast(1)
                    Bitmap.createScaledBitmap(bmp, w, h, true).also { if (it !== bmp) bmp.recycle() }
                } else bmp
                val baos = java.io.ByteArrayOutputStream()
                scaled.compress(Bitmap.CompressFormat.JPEG, quality, baos)
                if (scaled !== bmp) scaled.recycle()
                baos.toByteArray()
            } finally {
                try { retriever.release() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            Log.w(TAG, "extractFirstFrameJpeg falló: ${e.message}")
            null
        }
    }

    /** Copia el MP4 procesado a una carpeta visible del móvil (Movies/BolsilloIA)
     *  y lo registra con MediaScanner para que aparezca en la galería. El archivo
     *  original en cache se borra igualmente — esta es una COPIA persistente. */
    private fun saveVideoToGallery(source: File, name: String) {
        try {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "BolsilloIA")
            if (!dir.exists()) dir.mkdirs()
            val dest = File(dir, name)
            source.inputStream().use { input ->
                FileOutputStream(dest).use { output -> input.copyTo(output) }
            }
            MediaScannerConnection.scanFile(this, arrayOf(dest.absolutePath), arrayOf("video/mp4"), null)
            Log.i(TAG, "[video] ✓ copiado a galería: ${dest.absolutePath} (${dest.length() / 1024} KB)")
        } catch (e: Exception) {
            Log.w(TAG, "[video] No se pudo copiar a galería: ${e.message}")
        }
    }

    /** Guarda el "best frame" JPEG (fallback de video) en Pictures/BolsilloIA/
     *  con el MISMO timestamp que el MP4 hermano, para que en la galería el par
     *  video+foto quede claramente asociado. Si la foto se usa como fallback, el
     *  usuario puede revisar luego qué se mandó a la IA. */
    private fun saveBestFrameToGallery(jpeg: ByteArray, baseName: String) {
        try {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "BolsilloIA")
            if (!dir.exists()) dir.mkdirs()
            // baseName típico: "vid_1778862197023.mp4" → "vid_1778862197023_best.jpg"
            val stem = baseName.substringBeforeLast('.', baseName)
            val dest = File(dir, "${stem}_best.jpg")
            FileOutputStream(dest).use { it.write(jpeg) }
            MediaScannerConnection.scanFile(this, arrayOf(dest.absolutePath), arrayOf("image/jpeg"), null)
            Log.i(TAG, "[video] ✓ best frame copiado a galería: ${dest.absolutePath} (${dest.length() / 1024} KB)")
        } catch (e: Exception) {
            Log.w(TAG, "[video] No se pudo copiar best frame a galería: ${e.message}")
        }
    }

    private fun vibrate(ms: Long) {
        // 🎯 Path PRINCIPAL en Samsung S10e: escribir intensity sysfs justo
        // antes (asegura que el motor tiene la intensidad correcta) Y disparar
        // ENABLE sysfs. Esto bypasea completamente Android+HAL — el motor
        // arranca con la intensidad EXACTA del slider.
        //
        // Si el sysfs no está disponible (otro device), caemos a la API Android
        // que ya sabemos que en Samsung tiene problemas pero funcionará en otros.
        val sysfsPath = detectVibSysfsIntensityPath()
        if (sysfsPath != null) {
            // Refrescar intensity por si Settings.System lo hubiera tocado.
            writeVibSysfsIntensity(currentVibrationIntensity)
            try { SilentModeManager.vibrateViaSysfs(ms, ::runRootCommand) } catch (_: Throwable) {}
            return
        }
        // Fallback Android API
        if (Build.VERSION.SDK_INT >= 26) {
            vibrator.vibrate(
                VibrationEffect.createOneShot(ms, currentVibrationIntensity),
                alarmAudioAttrs,
            )
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(ms)
        }
    }

    private suspend fun translateSequenceToVibration(letters: List<Char>) {
        if (letters.isEmpty()) return
        vibrator.cancel()

        val timings = mutableListOf<Long>()
        val amplitudes = mutableListOf<Int>()

        timings.add(50L)
        amplitudes.add(0)

        val longDuration = vibrationDurationMs * 3

        for (i in letters.indices) {
            val l = letters[i]
            val count = when(l) { 'A'->1; 'B'->2; 'C'->3; 'D'->4; else->0 }

            if (count == 0 && l == 'X') {
                timings.add(longDuration)
                amplitudes.add(currentVibrationIntensity)
            } else if (count > 0) {
                for (pulse in 0 until count) {
                    timings.add(vibrationDurationMs)
                    amplitudes.add(currentVibrationIntensity)

                    if (pulse < count - 1) {
                        timings.add(intraBeepDelayMs)
                        amplitudes.add(0)
                    }
                }
            } else {
                continue
            }

            if (i < letters.size - 1) {
                timings.add(interLetterDelayMs)
                amplitudes.add(0)
            }
        }

        try {
            if (Build.VERSION.SDK_INT >= 26) {
                val effect = VibrationEffect.createWaveform(timings.toLongArray(), amplitudes.toIntArray(), -1)
                vibrator.vibrate(effect, alarmAudioAttrs)
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(timings.toLongArray(), -1)
            }
            delay(timings.sum() + 100L)
        } catch (e: Exception) {
            // No tragar silenciosamente: si la vibración compuesta falla, intentamos
            // un fallback simple letra por letra. Si TODO falla, al menos un log.
            Log.e(TAG, "translateSequenceToVibration falló: ${e.message} — fallback letra a letra")
            try {
                for (l in letters) {
                    val count = when (l) { 'A'->1; 'B'->2; 'C'->3; 'D'->4; 'X'->-1; else -> 0 }
                    if (count == -1) {
                        vibrate(vibrationDurationMs * 3)
                        delay(vibrationDurationMs * 3 + interLetterDelayMs)
                    } else if (count > 0) {
                        repeat(count) {
                            vibrate(vibrationDurationMs)
                            delay(vibrationDurationMs + intraBeepDelayMs)
                        }
                        delay(interLetterDelayMs - intraBeepDelayMs)
                    }
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Fallback letra a letra también falló: ${e2.message}")
            }
        }
    }

    /**
     * Vibra las correcciones retroactivas entregadas piggyback en el GET /result.
     * Patrón por corrección:
     *   6 vibs (señal) · silencio · N vibs (nº página) · silencio · letras corregidas
     */
    private suspend fun playCorrections(corrections: List<BolsilloIaClient.Correction>) {
        val allowed = setOf('A', 'B', 'C', 'D', 'X')
        val longDuration = vibrationDurationMs * 3

        for (corr in corrections) {
            val letters = corr.answer.filter { it in allowed }.toList()
            if (letters.isEmpty()) continue

            val timings    = mutableListOf<Long>()
            val amplitudes = mutableListOf<Int>()
            timings.add(50L); amplitudes.add(0)           // silencio inicial

            // ── 6 vibraciones señal ──
            for (i in 0 until 6) {
                timings.add(vibrationDurationMs); amplitudes.add(currentVibrationIntensity)
                if (i < 5) { timings.add(intraBeepDelayMs); amplitudes.add(0) }
            }
            timings.add(interLetterDelayMs); amplitudes.add(0)

            // ── número de página (N vibs = número de página) ──
            val page = corr.page
            if (page != null && page > 0) {
                for (i in 0 until page) {
                    timings.add(vibrationDurationMs); amplitudes.add(currentVibrationIntensity)
                    if (i < page - 1) { timings.add(intraBeepDelayMs); amplitudes.add(0) }
                }
                timings.add(interLetterDelayMs); amplitudes.add(0)
            }

            // ── secuencia de letras corregidas ──
            for (i in letters.indices) {
                val l = letters[i]
                val count = when (l) { 'A' -> 1; 'B' -> 2; 'C' -> 3; 'D' -> 4; else -> 0 }
                if (l == 'X') {
                    timings.add(longDuration); amplitudes.add(currentVibrationIntensity)
                } else if (count > 0) {
                    for (pulse in 0 until count) {
                        timings.add(vibrationDurationMs); amplitudes.add(currentVibrationIntensity)
                        if (pulse < count - 1) { timings.add(intraBeepDelayMs); amplitudes.add(0) }
                    }
                }
                if (i < letters.size - 1) { timings.add(interLetterDelayMs); amplitudes.add(0) }
            }

            try {
                if (Build.VERSION.SDK_INT >= 26) {
                    val effect = VibrationEffect.createWaveform(
                        timings.toLongArray(), amplitudes.toIntArray(), -1)
                    vibrator.vibrate(effect, alarmAudioAttrs)
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(timings.toLongArray(), -1)
                }
                delay(timings.sum() + 800L)   // pausa entre correcciones
            } catch (e: Exception) {
                Log.e(TAG, "playCorrections error: ${e.message}")
            }
        }
    }

    override fun onDestroy() {
        try { BolsilloIaClient.cancelAllActiveCalls() } catch (_: Throwable) {}
        try { pendingDeactivateJob?.cancel() } catch (_: Throwable) {}
        pendingDeactivateJob = null
        try { oneShotJob?.cancel() } catch (_: Throwable) {}
        oneShotJob = null
        try { videoTestJob?.cancel() } catch (_: Throwable) {}
        videoTestJob = null
        try { usbJob?.cancel() } catch (_: Throwable) {}
        usbJob = null
        vibrationTestJob?.cancel()
        stopCycleJobBlocking(timeoutMs = 1_500L)
        if (screenReceiverRegistered) {
            try { unregisterReceiver(screenReceiver) } catch (e: Exception) {}
            screenReceiverRegistered = false
        }
        if (airplaneModeReceiverRegistered) {
            try { unregisterReceiver(airplaneModeReceiver) } catch (e: Exception) {}
            airplaneModeReceiverRegistered = false
        }
        wakeLock?.let { if (it.isHeld) it.release() }

        // FIX MEDIO-5: desregistrar USBMonitor ANTES del runBlocking. Si lo
        // dejamos al final, durante el runBlocking pueden llegar callbacks de
        // attach/detach (postados desde el handler interno de USBMonitor) que
        // posten serviceScope.launch que vamos a cancelar 3 líneas más abajo
        // → trabajo huérfano.
        // FIX CRÍTICO-1 (cleanup): cortar también los flags de captura por si
        // hay un onFrame en vuelo justo cuando destruimos el servicio.
        isCapturingForVideo.set(false)
        isCapturingForBurst.set(false)
        usbCtrlBlock = null
        // Soltar el listener de llamadas si quedó activo.
        unregisterPhoneStateListener()
        if (usbMonitorRegistered) {
            try { usbMonitor.unregister() } catch (_: Exception) {}
            try { usbMonitor.destroy()    } catch (_: Exception) {}
            usbMonitorRegistered = false
        }

        // Teardown síncrono con timeout corto: garantiza intento real de cierre
        // de cámara ANTES de cancelar serviceScope (evita cleanup huérfano).
        try {
            runBlocking {
                withContext(NonCancellable + Dispatchers.IO) {
                    withTimeoutOrNull(800L) {
                        cameraMutex.withLock { closeCameraLocked() }
                    }
                }
            }
        } catch (_: Throwable) {}

        // DECISIÓN DELIBERADA: NO restauramos audio en onDestroy. onDestroy puede
        // dispararse en escenarios INVOLUNTARIOS (swipe desde recents, force-stop,
        // OOM-kill, USB disconnect que provoque shutdown). En esos casos queremos
        // que el dispositivo siga mudo — restaurar arruinaría el caso de uso real
        // (servicio espía que NO puede empezar a sonar a mitad de operación).
        // La única vía válida de restaurar es el botón stop → setSystemEnabled(false).
        serviceScope.cancel()
        immortalScope.cancel()
        try {
            rootShellWriter?.apply { try { write("exit\n"); flush() } catch (_: Exception) {} }
            rootShellProc?.destroy()
        } catch (_: Exception) {}
        rootShellWriter = null
        rootShellProc = null
        super.onDestroy()
    }

    override fun onDisconnect(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?) {
        if (intentionalCameraClose.get()) return
        // FIX CRÍTICO-1: señalar AHORA, sin esperar al mutex. Si hay una
        // captura de video en curso, ya está reteniendo el cameraMutex por
        // varios segundos; poniendo a null usbCtrlBlock + cortando los flags
        // de captura, el bucle interno detecta el desenchufe y aborta limpio
        // en vez de seguir alimentando bytes a un ctrlBlock cerrado.
        usbCtrlBlock = null
        isCapturingForVideo.set(false)
        isCapturingForBurst.set(false)
        videoAbortedByUsbDisconnect = true
        // FIX CRÍTICO-3: encadenar con usbJob para que el siguiente connect
        // espere a este cleanup antes de abrir cámara nueva.
        val previous = usbJob
        usbJob = serviceScope.launch {
            try { previous?.join() } catch (_: Throwable) {}
            cameraMutex.withLock { closeCameraLocked() }
            broadcastCameraState(CameraState.Idle)
        }
    }

    override fun onDettach(device: UsbDevice?) {
        if (intentionalCameraClose.get()) return
        // Mismo patrón que onDisconnect — nullify inmediato + cleanup serializado.
        // onDettach llega cuando el cable se quita FÍSICAMENTE; el USBMonitor ya
        // ha cerrado el UsbDeviceConnection en su BroadcastReceiver antes de
        // notificarnos, así que cualquier operación sobre uvcCamera/ctrlBlock
        // tras este punto es UAF latente. Cortamos los flags YA.
        usbCtrlBlock = null
        isCapturingForVideo.set(false)
        isCapturingForBurst.set(false)
        videoAbortedByUsbDisconnect = true
        val previous = usbJob
        usbJob = serviceScope.launch {
            try { previous?.join() } catch (_: Throwable) {}
            cameraMutex.withLock { closeCameraLocked() }
            broadcastCameraState(CameraState.Idle)
        }
    }

    override fun onAttach(device: UsbDevice?) {
        if (device == null) return
        // VÍA PRINCIPAL: conceder el permiso en memoria con la API de sistema
        // MANAGE_USB. Si la app es privilegiada (módulo Magisk uvc-privapp), esto
        // evita el diálogo sin tocar la pantalla ni reiniciar.
        val granted = try {
            val um = getSystemService(Context.USB_SERVICE) as UsbManager
            UsbAutoGrant.grantViaApi(um, device, packageName)
        } catch (e: Exception) {
            Log.w(TAG, "grantViaApi excepción: ${e.message}"); false
        }
        // RESPALDO: si aún no hay MANAGE_USB (módulo no instalado), caer al método
        // antiguo (siembra XML al boot + watcher del diálogo).
        if (!granted) {
            try {
                UsbAutoGrant.grant(device, packageName, ::runRootCommand)
            } catch (e: Exception) {
                Log.w(TAG, "UsbAutoGrant fallback falló (sigue intentando vía libuvccamera): ${e.message}")
            }
        }
        usbMonitor.requestPermission(device)
    }
    override fun onCancel(device: UsbDevice?) { broadcastCameraState(CameraState.Idle) }

    fun setPreviewSurface(surface: Surface?) {
        currentPreviewSurface = surface
        serviceScope.launch {
            cameraMutex.withLock {
                try {
                    if (isCapturingForBurst.get()) return@withLock

                    if (attachedSurface === surface && surface != null) return@withLock

                    if (attachedSurface != null) {
                        try { uvcCamera?.setPreviewDisplay(null as Surface?) } catch (_: Exception) {}
                        attachedSurface = null
                    }

                    if (surface != null) {
                        uvcCamera?.setPreviewDisplay(surface)
                        attachedSurface = surface
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "setPreviewSurface error: ${e.message}", e)
                }
            }
        }
    }

    /**
     * Llamado por el slider del cliente cuando el usuario suelta (onStopTrackingTouch).
     * Sin la primera línea, la vibración de prueba se dispara ANTES de que el
     * push debounced de la intensidad llegue al sistema → Samsung HAL lee el
     * setting antiguo y vibra con la intensidad anterior. Lo forzamos ahora.
     */
    fun sendHapticFeedback() {
        flushPendingIntensityNow()
        vibrate(vibrationDurationMs)
    }

    /**
     * Cancela el debounce de setVibrationIntensity y aplica AHORA la intensidad
     * al sistema. Como ahora usamos Settings.System.putInt (instantáneo, sin
     * shell), el flush no necesita Thread.sleep — el setting está aplicado
     * antes de que volvamos. La siguiente vibrate() lee el valor nuevo.
     */
    private fun flushPendingIntensityNow() {
        pendingIntensityPush?.cancel()
        pendingIntensityPush = null
        try {
            pushSamsungVibrationIntensity(samsungIntensityFor(currentVibrationIntensity))
        } catch (e: Exception) {
            Log.w(TAG, "flushPendingIntensityNow falló: ${e.message}")
        }
    }

    /**
     * Mapea la intensidad interna (1..255) a la escala de Samsung One UI (1..3)
     * que el HAL Háptico de Samsung respeta:
     *   - 1 → "Débil"     (33%)
     *   - 2 → "Medio"     (66%)
     *   - 3 → "Fuerte"    (100%)
     *
     * Razón del fix: en Samsung Galaxy con One UI, el HAL táctil IGNORA el
     * parámetro `amplitude` de VibrationEffect.createWaveform y respeta SOLO
     * el setting `alarm_vibration_intensity` (porque usamos USAGE_ALARM en
     * alarmAudioAttrs). Por eso el slider del cliente no hacía nada: pasábamos
     * un amplitude=50 al API pero el sistema replicaba el setting hardcoded a 3.
     */
    private fun samsungIntensityFor(amp: Int): Int = when {
        amp <= 85  -> 1
        amp <= 170 -> 2
        else       -> 3
    }

    /**
     * Detecta el path del nodo sysfs intensity al primer uso. Lo cachea en
     * vibSysfsIntensityPath. Si no encuentra ningún path válido, deja null
     * (signal de que este device no es Samsung S10e o el driver es otro).
     *
     * No requiere root para `exists()` (FS read), pero los writes después sí.
     */
    private fun detectVibSysfsIntensityPath(): String? {
        if (vibSysfsChecked) return vibSysfsIntensityPath
        synchronized(this) {
            if (vibSysfsChecked) return vibSysfsIntensityPath
            for (path in VIB_SYSFS_INTENSITY_CANDIDATES) {
                try {
                    if (File(path).exists()) {
                        vibSysfsIntensityPath = path
                        Log.i(TAG, "[vib-sysfs] nodo intensity encontrado: $path (MAX=$VIB_SYSFS_MAX)")
                        vibSysfsChecked = true
                        return path
                    }
                } catch (_: Exception) {}
            }
            vibSysfsChecked = true
            Log.w(TAG, "[vib-sysfs] ningún nodo intensity encontrado en candidatos — caemos a Settings.System")
            return null
        }
    }

    /**
     * Mapea la intensidad del slider del usuario (1..255) al rango del sysfs
     * Samsung S10e (1..10000). Lineal con piso de 1 (el driver rechaza 0).
     */
    private fun sliderToSysfsIntensity(slider: Int): Int =
        ((slider.toLong() * VIB_SYSFS_MAX) / 255).toInt().coerceIn(1, VIB_SYSFS_MAX)

    /**
     * Escribe la intensidad al nodo sysfs Samsung directamente. Cuando el
     * driver dispare el motor (sea vía sysfs/enable o vía Android API), usará
     * esta intensidad. Latencia <5ms.
     *
     * Devuelve true si escribió OK, false si el nodo no existe o el shell
     * falló.
     */
    private fun writeVibSysfsIntensity(slider: Int): Boolean {
        val path = detectVibSysfsIntensityPath() ?: return false
        val sysfsVal = sliderToSysfsIntensity(slider)
        return try {
            runRootCommand("echo $sysfsVal > $path", "vib-sysfs-int")
        } catch (e: Exception) {
            Log.w(TAG, "[vib-sysfs] write falló: ${e.message}")
            false
        }
    }

    /**
     * Escribe la intensidad Samsung (1..3) en `Settings.System` y ESPERA
     * confirmación. Estrategia óptima en Samsung One UI:
     *
     * 1) Las keys `alarm_vibration_intensity`, etc. están en MOVED_TO_SECURE,
     *    así que `Settings.System.putInt` desde Java siempre falla.
     * 2) `Settings.Secure.putInt` "funciona" pero el HAL lee desde System,
     *    así que no tiene efecto en la vibración real.
     * 3) El shell `settings put system X Y` SÍ escribe en System (corre como
     *    UID 2000 que bypassa el check del provider).
     *
     * Para que sea RÁPIDO usamos el shell persistente (~5ms para escribir a
     * stdin) y luego POLLEAMOS `Settings.System.getInt` cada 5ms hasta ver
     * el valor nuevo (significa que el `settings put` ha completado su
     * llamada al ContentProvider y el HAL ya lo verá). Polling read no
     * requiere permisos y el get devuelve en microsegundos.
     *
     * Latencia típica: 10-50ms (escribe + 1-5 polls). Mucho mejor que
     * ProcessBuilder.waitFor (200-500ms por spawn de proceso) o que un
     * Thread.sleep ciego (siempre paga el peor caso).
     */
    private fun pushSamsungVibrationIntensity(samsungScale: Int) {
        val cmd = (
            "settings put system alarm_vibration_intensity $samsungScale;" +
            "settings put system haptic_feedback_intensity $samsungScale;" +
            "settings put system notification_vibration_intensity $samsungScale;" +
            "settings put system ring_vibration_intensity $samsungScale"
        )
        // Escribe al shell persistente — no espera. Vuelta en microsegundos.
        val written = try { runRootCommand(cmd, "vib-intensity") } catch (_: Exception) { false }
        if (!written) {
            Log.w(TAG, "vib-intensity: no se pudo escribir al shell root")
            return
        }
        // Poll hasta ver el valor aplicado. Lectura de Settings.System NO
        // requiere permisos; siempre devuelve el último valor escrito.
        val cr = contentResolver
        val deadlineMs = System.currentTimeMillis() + 500L
        var polls = 0
        while (System.currentTimeMillis() < deadlineMs) {
            polls++
            try {
                val v = android.provider.Settings.System.getInt(cr, "alarm_vibration_intensity", -1)
                if (v == samsungScale) {
                    if (polls > 1) {
                        Log.d(TAG, "vib-intensity aplicado tras $polls polls (~${polls * 5}ms)")
                    }
                    return  // 🎯 confirmado, podemos vibrar con el valor nuevo
                }
            } catch (_: Exception) {}
            try { Thread.sleep(5L) } catch (_: InterruptedException) { break }
        }
        Log.w(TAG, "vib-intensity: timeout 500ms sin ver el valor $samsungScale aplicado")
    }

    @Volatile private var pendingIntensityPush: kotlinx.coroutines.Job? = null
    private val intensityPushMutex = kotlinx.coroutines.sync.Mutex()

    fun setVibrationIntensity(amplitude: Int) {
        currentVibrationIntensity = amplitude.coerceIn(1, 255)
        // Multi-proceso (:uvc + UI): commit síncrono para que la Activity lea el
        // valor correcto al reabrir, sin ventana de pérdida por apply asíncrono.
        try { sharedPrefs.edit().putInt("vibration_intensity", currentVibrationIntensity).commit() } catch (_: Throwable) {}
        // ⚡ Path RÁPIDO (Samsung S10e): escribir directamente al sysfs del motor.
        // Es <5ms y la próxima vibración usa esta intensidad SIN pasar por Android
        // ni la HAL Samsung que clampa amplitudes. Lo hacemos SÍNCRONO en el
        // Binder thread porque cuesta nada (1 write a stdin del shell persistente).
        val sysfsOk = writeVibSysfsIntensity(currentVibrationIntensity)
        if (sysfsOk) return  // 🎯 path Samsung S10e — terminamos en <5ms

        // Path fallback (otros devices): empujar al sistema Android. Esto es
        // mucho más lento (~50-200ms) así que va debounced + en background.
        pendingIntensityPush?.cancel()
        pendingIntensityPush = serviceScope.launch {
            try {
                kotlinx.coroutines.delay(50)
                intensityPushMutex.lock()
                try {
                    pushSamsungVibrationIntensity(samsungIntensityFor(currentVibrationIntensity))
                } finally {
                    intensityPushMutex.unlock()
                }
            } catch (_: kotlinx.coroutines.CancellationException) {
                // Cancelado por otra llamada más reciente o por flushPendingIntensityNow.
            } catch (e: Exception) {
                Log.w(TAG, "No se pudo aplicar intensidad sistema: ${e.message}")
            }
        }
    }

    fun setCycleDurationSeconds(seconds: Long) {
        currentCycleDelaySeconds = seconds.coerceAtLeast(1L)
        sharedPrefs.edit().putLong("cycle_duration", currentCycleDelaySeconds).apply()
    }

    fun setBeepDuration(ms: Long) {
        vibrationDurationMs = ms.coerceAtLeast(100L)
        sharedPrefs.edit().putLong("vibration_duration", vibrationDurationMs).apply()
    }

    fun setIntraBeepDelay(ms: Long) {
        intraBeepDelayMs = ms.coerceAtLeast(100L)
        sharedPrefs.edit().putLong("intra_beep_delay", intraBeepDelayMs).apply()
    }

    fun setInterLetterDelay(ms: Long) {
        interLetterDelayMs = ms.coerceAtLeast(0L)
        sharedPrefs.edit().putLong("inter_letter_delay", interLetterDelayMs).apply()
    }

    fun setPreBurstVibrationDelay(ms: Long) {
        preBurstVibrationDelayMs = ms.coerceAtLeast(0L)
        sharedPrefs.edit().putLong("pre_burst_vibration_delay", preBurstVibrationDelayMs).apply()
    }

    // FIX M10: leer de prefs en vez de la variable @Volatile. La variable solo se
    // actualiza cuando runSystemCycle entra al bloque de delay (línea ~2308); entre
    // setSystemEnabled(true) y ese momento puede haber latencia (mutex de ensureCorrectMode),
    // y la UI consultando vía AIDL recibía 0 aunque ya hubiera un delay programado.
    // Prefs es el dato canónico desde el instante del enable.
    fun getInitialDelayEndAtMs(): Long = sharedPrefs.getLong("initial_delay_end_at", 0L)

    fun setStartDelayMinutes(minutes: Int) {
        startDelayMinutes = minutes.coerceIn(0, 60)
        sharedPrefs.edit().putInt("start_delay_minutes", startDelayMinutes).apply()
        Log.i(TAG, "Configuración de retraso inicial ajustada a: ${startDelayMinutes} min")
    }

    fun setGeminiPrompt(prompt: String) { }

    /**
     * Reproduce un pulso de N vibraciones cortas separadas por intra-beep.
     * Usa la función `vibrate(ms)` SIMPLE que internamente prefiere sysfs en
     * Samsung S10e (bypaseando VibratorManagerService que está bloqueado por
     * One UI). Cada llamada es un pulso real del motor — sin waveform compuesto
     * de la Android API (que NO vibra en S10e).
     */
    private suspend fun playPulses(count: Int) {
        if (count <= 0) return
        // Defensiva: si por config corrupta los timings son 0 o negativos, no
        // colapsa la coroutina (delay(0) o vibrate(0) son no-ops nativos pero
        // crearían un bucle CPU-bound). Pisamos un mínimo para que el motor
        // tenga tiempo de arrancar y los pulsos sean distinguibles.
        val dur = vibrationDurationMs.coerceAtLeast(50L)
        val gap = intraBeepDelayMs.coerceAtLeast(50L)
        for (i in 0 until count) {
            vibrate(dur)
            if (i < count - 1) {
                delay(dur + gap)
            } else {
                delay(dur)
            }
        }
    }

    /**
     * Reproduce una letra A/B/C/D/X como pulsos discretos (1/2/3/4 cortos o 1 largo
     * para X). Equivalente al patrón de [translateSequenceToVibration] pero usando
     * la ruta sysfs que sí funciona en Samsung S10e.
     */
    private suspend fun playLetterPulses(letter: Char) {
        when (letter) {
            'A' -> playPulses(1)
            'B' -> playPulses(2)
            'C' -> playPulses(3)
            'D' -> playPulses(4)
            'X' -> { vibrate(vibrationDurationMs * 3); delay(vibrationDurationMs * 3) }
        }
    }

    /**
     * Reproduce la secuencia A·B·C·D como respuesta de 4 preguntas, con
     * separaciones de [interLetterDelayMs] entre letras. Vía sysfs (S10e OK).
     */
    private suspend fun playLetterSequence(letters: List<Char>) {
        for (i in letters.indices) {
            playLetterPulses(letters[i])
            if (i < letters.size - 1) delay(interLetterDelayMs)
        }
    }

    /**
     * Reproduce el patrón de UNA corrección de ejemplo (página 3, respuesta "BD"):
     *   6 pulsos señal · pausa · 3 pulsos (página) · pausa · B (2) · D (4)
     *
     * Idéntica estructura que [playCorrections] pero pulse-by-pulse para que
     * funcione en Samsung S10e (donde el waveform compuesto no vibra).
     */
    private suspend fun playSampleCorrection() {
        // 6 pulsos señal de "viene una corrección"
        playPulses(6)
        delay(interLetterDelayMs)

        // Número de página (ejemplo: 3 → 3 pulsos)
        playPulses(3)
        delay(interLetterDelayMs)

        // Letras corregidas: B (2 pulsos) + sep + D (4 pulsos)
        playLetterSequence(listOf('B', 'D'))
    }

    fun toggleVibrationTestLoop(): Boolean {
        if (vibrationTestJob?.isActive == true) {
            vibrationTestJob?.cancel()
            vibrationTestJob = null
            // Cancelar el motor SIN esperar: vibrator.cancel() es asíncrono pero
            // muy rápido. Damos un pequeño respiro para que el último pulso
            // sysfs (echo a /sys/class/timed_output/.../enable) tenga tiempo de
            // procesarse — si llamamos cancel() exactamente mientras el shell
            // root está escribiendo el siguiente pulso, el shell puede quedarse
            // bloqueado. La cancelación del scheduler del kernel se encarga de
            // detener el motor en el siguiente tick.
            try { vibrator.cancel() } catch (_: Exception) {}
            // Reset por sysfs: escribir 0 al enable para garantizar que el
            // motor está parado, incluso si el pulso anterior aún no terminó.
            try {
                runRootCommand(
                    "echo 0 > /sys/class/timed_output/vibrator/enable 2>/dev/null; " +
                    "echo 0 > /sys/class/leds/vibrator/duration 2>/dev/null",
                    "vib-test-stop",
                )
            } catch (_: Throwable) {}
            Log.i(TAG, "Test de vibración DETENIDO")
            return false
        } else {
            vibrationTestJob = serviceScope.launch {
                Log.i(TAG, "Test de vibración INICIADO — A·B·C·D + corrección (BD pág 3) en bucle [sysfs S10e]")
                // ────────────────────────────────────────────────────────────
                // FIX SAMSUNG S10e: NO usar `translateSequenceToVibration` ni
                // `VibrationEffect.createWaveform`. La API Android del vibrador
                // está bloqueada por One UI / VibratorManagerService incluso con
                // AudioAttributes=ALARM. La ÚNICA ruta fiable es escribir a
                // sysfs (/sys/class/timed_output/vibrator/enable) — exactamente
                // lo que hace `vibrate(ms)` simple a través de SilentModeManager.
                //
                // Por eso reproducimos pulso a pulso con `vibrate()` + `delay()`,
                // respetando los parámetros del panel (intensidad, duración,
                // intra-beep, inter-letter). Cambiar un slider se nota en la
                // SIGUIENTE iteración del bucle.
                //
                // PATRÓN 1: A (1) · B (2) · C (3) · D (4)
                // PATRÓN 2: 6 señal · página 3 · B (2) · D (4)
                // ────────────────────────────────────────────────────────────
                while (isActive) {
                    // ── Patrón 1: respuesta normal de 4 preguntas ──
                    playLetterSequence(listOf('A', 'B', 'C', 'D'))
                    delay(interLetterDelayMs * 3)

                    // ── Patrón 2: corrección retroactiva (BD para página 3) ──
                    playSampleCorrection()
                    delay(interLetterDelayMs * 3)
                }
            }
            return true
        }
    }
}