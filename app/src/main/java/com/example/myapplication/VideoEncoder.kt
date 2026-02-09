package com.example.myapplication

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import java.io.File
import java.nio.ByteBuffer

/**
 * Encoder MP4 (H.265/HEVC con fallback H.264) en **BUFFER mode con NV12**.
 *
 * Historia: el camino "oficial saki4510t" era SURFACE mode (createInputSurface +
 * uvcCamera.startCapture). En Samsung Exynos (S10e con Mali GPU), la superficie
 * que devuelve createInputSurface está respaldada por un buffer
 * HAL_PIXEL_FORMAT_YCbCr_420_888 (YUV 420), no RGBA. ANativeWindow_lock sobre
 * ese buffer en Mali sólo expone la primera página del plano Y. libUVCCamera
 * escribe filas de 1280·4 = 5120 B asumiendo RGBA → SIGSEGV al cruzar la página.
 *
 * Solución: pasamos a BUFFER mode. El encoder se configura con
 * COLOR_FormatYUV420SemiPlanar y NO crea input surface. libuvc convierte
 * YUYV→NV12 nativamente (uvc_yuyv2yuv420SP — ya existe en frame.c) y entrega
 * los bytes vía IFrameCallback. El servicio los empaqueta en
 * queueInputBuffer. Una copia extra Java↔native pero compatible con TODOS los
 * SoC (Exynos/Snapdragon/Mediatek). A 1280x720@30fps son ~40 MB/s de NV12 — el
 * CPU lo procesa en <1 ms por frame.
 *
 * Uso típico:
 *   val enc = VideoEncoder(file, 1280, 720, fps = 30, bitrateKbps = 8000)
 *   enc.start()
 *   uvcCamera.setFrameCallback(callback, UVCCamera.PIXEL_FORMAT_YUV420SP)
 *   // callback.onFrame(buf) → enc.queueNv12(buf)
 *   delay(durationMs)
 *   // detener la cámara o cambiar el callback
 *   val mp4 = enc.finish()
 *
 * Notas:
 * - HEVC primero; si el chipset no lo soporta el constructor cae a AVC.
 * - finish() es idempotente.
 * - queueNv12() es thread-safe.
 */
class VideoEncoder(
    private val outputFile: File,
    private val width: Int,
    private val height: Int,
    private val fps: Int = 15,
    private val bitrateKbps: Int = 3000,
    private val preferHevc: Boolean = true,
) {
    companion object {
        private const val TAG = "VideoEncoder"
        private const val MIME_HEVC = MediaFormat.MIMETYPE_VIDEO_HEVC
        private const val MIME_AVC  = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val DEQUEUE_OUT_TIMEOUT_US = 10_000L  // 10 ms drain loop
        // 15 ms (subido de 5 ms): el callback de libuvc puede ser más burst-y
        // que el rate de drenado del encoder Exynos del S10e a 1080p — antes
        // dropeábamos 1-2 frames/s gratis. 15 ms sigue dejando margen para que
        // el hilo de callback nativo no se quede esperando >30 ms (1 frame@30fps).
        private const val DEQUEUE_IN_TIMEOUT_US  = 15_000L
        // I-frame cada 3 s. Para contenido tipo "texto estático en pantalla
        // móvil" (preguntas de test), HEVC emite I-frames pesados sólo cuando
        // toca y rellena con P-frames de pocos bytes que codifican el delta
        // mínimo entre fotogramas casi idénticos. Bajar a 1 s gasta ~3× más
        // bytes sin ganancia perceptible para OCR.
        private const val I_FRAME_INTERVAL_S = 3
    }

    // Constructor de compatibilidad: bitrateMbps entero (callers legacy).
    constructor(
        outputFile: File, width: Int, height: Int,
        fps: Int = 15, bitrateMbps: Int,
    ) : this(outputFile, width, height, fps, bitrateMbps * 1000, true)

    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private var muxerStarted = false
    private var trackIndex = -1
    @Volatile private var finished = false
    private var drainStarted = false

    /** Tamaño esperado del frame NV12 en bytes (Y plane + UV interleaved). */
    private val expectedNv12Size: Int = width * height * 3 / 2

    /** PTS base (ns/1000) tomado del primer frame para que el primer sample sea 0. */
    @Volatile private var startPtsUs: Long = -1L

    /** MIME real elegido (video/hevc o video/avc). Útil para logs. */
    val actualMime: String

    // Stride y slice-height REALES publicados por el encoder tras start(). En la
    // mayoría de chipsets (Snapdragon, MediaTek, y Exynos 9820 a 1280×720) son
    // iguales a width/height → ruta rápida con copia plana. En Exynos con
    // resoluciones no alineadas (p.ej. 1920×1080 → stride=2048) o algunos
    // dispositivos viejos, el encoder padea → ruta lenta row-by-row.
    @Volatile private var codecStride: Int = width
    @Volatile private var codecSliceHeight: Int = height
    @Volatile private var loggedSlowPathOnce: Boolean = false

    // ── Detector DIAGNÓSTICO de frames grises ─────────────────────────────────
    // Causa típica del bug "el video sale gris": libuvc decodifica un MJPEG
    // truncado vía libjpeg-turbo, devuelve UVC_SUCCESS aunque el JPEG estaba
    // malformado, y deja parte (o todo) el buffer YUYV sin escribir. Como el
    // pool de buffers reusa memoria, esos bytes valen normalmente 0 (calloc)
    // o ~128 (debug fill) → al convertirse a NV12 producen un frame gris.
    // libuvc#122 documenta este patrón. El detector muestrea 64 puntos del
    // plano Y y, si la media está cerca de 128 con varianza ~0, logea (NO
    // dropea) para que el usuario pueda contar cuántos frames grises llegan
    // al encoder. La fix REAL está en JNI (validación SOI/EOI del MJPEG).
    @Volatile var grayFramesDetected: Long = 0L
        private set
    private var totalFramesSeen: Long = 0L

    // Buffer reusable para copia row-by-row cuando hay stride padding. Tamaño:
    // width bytes (suficiente para una fila de Y o de UV interleaved). Se asigna
    // una vez por instancia y se reutiliza para no machacar al GC.
    private val rowCopyBuf = ByteArray(width)

    init {
        require(width % 2 == 0 && height % 2 == 0) {
            "Width/height deben ser pares (encoders H.26x lo exigen). Recibido ${width}x${height}"
        }
        require(bitrateKbps in 200..100_000) {
            "bitrateKbps fuera de rango sano (200..100000). Recibido $bitrateKbps"
        }

        // Intentar HEVC primero; si falla, cae a AVC con la misma config.
        val (mediaCodec, mediaMime) = createCodecWithFallback()
        codec      = mediaCodec
        actualMime = mediaMime

        outputFile.parentFile?.mkdirs()
        muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        Log.i(TAG, "Encoder listo (BUFFER NV12): ${width}x${height}@${fps}fps, " +
                "${bitrateKbps} kbps, codec=$actualMime, archivo=${outputFile.name}")
    }

    /** Crea MediaCodec con KEY_COLOR_FORMAT=COLOR_FormatYUV420SemiPlanar (NV12).
     *  Intenta HEVC primero (si preferHevc=true), si falla la configuración cae a AVC. */
    private fun createCodecWithFallback(): Pair<MediaCodec, String> {
        val order = if (preferHevc) listOf(MIME_HEVC, MIME_AVC) else listOf(MIME_AVC)
        var last: Exception? = null
        for (mime in order) {
            var c: MediaCodec? = null
            try {
                val format = MediaFormat.createVideoFormat(mime, width, height).apply {
                    // NV12 buffer mode. Samsung Exynos / Snapdragon / Mediatek lo soportan
                    // de forma uniforme. Es lo que libuvc nos entrega vía uvc_yuyv2yuv420SP.
                    setInteger(MediaFormat.KEY_COLOR_FORMAT,
                        MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
                    setInteger(MediaFormat.KEY_BIT_RATE,         bitrateKbps * 1000)
                    setInteger(MediaFormat.KEY_FRAME_RATE,       fps)
                    setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_S)
                    // VBR mejora calidad/peso para texto estático (OCR).
                    try {
                        setInteger(MediaFormat.KEY_BITRATE_MODE,
                            MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
                    } catch (_: Exception) {}
                }
                c = MediaCodec.createEncoderByType(mime)
                c.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                c.start()
                // DIAGNÓSTICO crítico Samsung Exynos: tras start(), el encoder
                // publica su stride y slice-height REALES. En Exynos 9820 (S10e)
                // el encoder NV12 SemiPlanar suele padded a múltiplo de 16/32
                // (ej. stride=1296 para width=1280, o 2048 para 1920). Si era
                // copia plana sin respetar este padding, las filas de la croma
                // caían en filas incorrectas → frame gris uniforme intermitente.
                // queueNv12 ahora usa getInputImage(idx) que respeta automáticamente
                // rowStride/pixelStride, pero conservamos el log para diagnóstico.
                val inFmt = try { c.inputFormat } catch (_: Exception) { null }
                val stride = inFmt?.let { try { it.getInteger("stride") } catch (_: Exception) { -1 } } ?: -1
                val slice  = inFmt?.let { try { it.getInteger("slice-height") } catch (_: Exception) { -1 } } ?: -1
                val name   = try { c.name } catch (_: Exception) { "?" }
                // Cachear stride/slice-height para queueNv12. Si la API no los expone
                // (-1), asumimos sin padding (== width/height). Coerce defensivo para
                // que nunca sean menores que el lógico, lo que rompería los offsets.
                codecStride      = if (stride > 0) stride.coerceAtLeast(width)   else width
                codecSliceHeight = if (slice > 0)  slice.coerceAtLeast(height)  else height
                Log.i(TAG, "Códec elegido: $mime · name=$name · ${width}x${height} · stride=$codecStride · slice-height=$codecSliceHeight")
                if (codecStride != width || codecSliceHeight != height) {
                    Log.w(TAG, "Codec con PADDING (stride=$codecStride vs width=$width, slice=$codecSliceHeight vs height=$height) — ruta lenta row-by-row activada")
                }
                return Pair(c, mime)
            } catch (e: Exception) {
                last = e
                Log.w(TAG, "Fallo configurando $mime: ${e.message} — probando siguiente")
                try { c?.release() } catch (_: Exception) {}
            }
        }
        throw RuntimeException(
            "No se pudo configurar ningún encoder (HEVC ni AVC): ${last?.message}", last,
        )
    }

    /**
     * Arranca el drain loop en un thread daemon que extrae los buffers
     * codificados del codec y los escribe al muxer. Llamar UNA vez, antes de
     * que la UVCCamera empiece a entregar frames a queueNv12().
     */
    @Synchronized
    fun start() {
        if (drainStarted) return
        drainStarted = true
        Thread({
            try { drainEncoder(endOfStream = false) }
            catch (e: Exception) { Log.e(TAG, "drain loop falló: ${e.message}", e) }
        }, "VideoEncoderDrain").apply { isDaemon = true }.start()
    }

    /**
     * Encola un frame NV12 al encoder. Thread-safe — diseñado para ser llamado
     * desde el hilo de IFrameCallback (JNI), que entrega un ByteBuffer directo
     * con bytes en orden: [Y plane (w·h bytes)] [UV interleaved (w·h/2 bytes)].
     *
     * Devuelve true si el frame se encoló, false si:
     *  - El encoder ya está cerrado (finish() llamado)
     *  - El buffer es más pequeño que w·h·1.5
     *  - No hay input buffer libre tras 5 ms (encoder saturado → drop frame)
     *
     * No bloquea más de DEQUEUE_IN_TIMEOUT_US (5 ms). Si el encoder se atasca,
     * dropea el frame en vez de bloquear el hilo de callback nativo y arrastrar
     * lag a la cámara.
     */
    fun queueNv12(src: ByteBuffer): Boolean {
        if (finished) return false
        val avail = src.remaining()
        if (avail < expectedNv12Size) {
            Log.w(TAG, "queueNv12: buffer demasiado pequeño ($avail < $expectedNv12Size) — drop")
            return false
        }

        // DIAGNÓSTICO: detectar frames grises (libuvc#122 — MJPEG truncado
        // decodeado parcialmente). Solo loguea, NO dropea: si dropeáramos
        // perderíamos sincronía de timing y sería más difícil reproducir.
        // El fix real está en la validación SOI/EOI del JNI.
        totalFramesSeen++
        if (isLikelyGrayFrame(src)) {
            grayFramesDetected++
            // Throttle: solo loguear los 3 primeros y cada 30 después para no
            // spammar logcat si todos los frames son grises.
            if (grayFramesDetected <= 3L || grayFramesDetected % 30L == 0L) {
                Log.w(TAG, "FRAME GRIS detectado · #$grayFramesDetected/$totalFramesSeen total — el MJPEG llegó truncado y libjpeg dejó el buffer YUYV con datos uniformes")
            }
        }

        val idx: Int = try {
            codec.dequeueInputBuffer(DEQUEUE_IN_TIMEOUT_US)
        } catch (e: IllegalStateException) {
            return false  // codec released
        }
        if (idx < 0) return false  // saturado → drop

        val dst: ByteBuffer = try {
            codec.getInputBuffer(idx) ?: run {
                try { codec.queueInputBuffer(idx, 0, 0, 0, 0) } catch (_: Exception) {}
                return false
            }
        } catch (e: IllegalStateException) {
            return false
        }

        return try {
            val bytesQueued: Int = if (codecStride == width && codecSliceHeight == height) {
                // FAST PATH (caso típico Samsung S10e + 1280×720 + Snapdragon/MediaTek):
                // El encoder NO padea → una copia plana directa al input buffer
                // funciona y es la ruta que ya estaba en producción. Mantener este
                // camino IDÉNTICO al legacy evita regresiones en dispositivos que
                // ya funcionaban.
                dst.clear()
                val srcOrigLimit = src.limit()
                val srcOrigPos   = src.position()
                src.limit(srcOrigPos + expectedNv12Size)
                dst.put(src)
                src.limit(srcOrigLimit)
                expectedNv12Size
            } else {
                // SLOW PATH (Exynos a 1920×1080 u otros chipsets con padding):
                // El encoder espera filas separadas por `codecStride` bytes con
                // hueco de `codecStride - width` bytes al final de cada fila, y
                // el plano UV empieza en `codecStride * codecSliceHeight`. Copia
                // row-by-row directamente al input buffer (NO usar getInputImage:
                // su .buffer del plano U tiene capacity 1 byte menor que la UV
                // interleaved total y produce BufferOverflowException).
                if (!loggedSlowPathOnce) {
                    Log.i(TAG, "queueNv12: ruta SLOW (stride=$codecStride/$width · slice=$codecSliceHeight/$height) — primera vez")
                    loggedSlowPathOnce = true
                }
                writeNv12WithStride(src, dst)
            }

            val nowUs = System.nanoTime() / 1000L
            val ptsUs = if (startPtsUs == -1L) {
                startPtsUs = nowUs
                0L
            } else nowUs - startPtsUs
            codec.queueInputBuffer(idx, 0, bytesQueued, ptsUs, 0)
            true
        } catch (e: Exception) {
            // Loguear tipo de excepción además del message: BufferOverflowException
            // y similares vienen con message=null y el log antiguo solo decía "null".
            Log.w(TAG, "queueNv12 falló: ${e.javaClass.simpleName}: ${e.message ?: "(sin mensaje)"}")
            try { codec.queueInputBuffer(idx, 0, 0, 0, 0) } catch (_: Exception) {}
            false
        }
    }

    /**
     * Copia un frame NV12 desde `src` (compacto: stride=width, slice-height=height)
     * a `dst` (input buffer del encoder con padding stride/slice-height). Hace una
     * copia row-by-row para Y, luego para UV interleaved. NO usa getInputImage()
     * porque en SemiPlanar el .buffer del plano U tiene capacidad insuficiente
     * para escribir UV interleaved (corta 1 byte antes del último V).
     *
     * Devuelve el total de bytes escritos al input buffer (incluye padding).
     * NO modifica `srcOriginal.position()` — usa duplicate().
     */
    private fun writeNv12WithStride(srcOriginal: ByteBuffer, dst: ByteBuffer): Int {
        val src = srcOriginal.duplicate()
        val srcStart = src.position()
        val stride = codecStride
        val slice = codecSliceHeight

        dst.clear()

        // ── Plano Y: ${height} filas, ${width} bytes útiles cada una a stride spacing ──
        for (row in 0 until height) {
            src.position(srcStart + row * width)
            src.get(rowCopyBuf, 0, width)
            dst.position(row * stride)
            dst.put(rowCopyBuf, 0, width)
        }

        // ── Plano UV interleaved: empieza en stride*slice, ${height/2} filas ──
        val uvDstStart = stride * slice
        val uvSrcStart = srcStart + width * height
        val uvHeight = height / 2
        for (row in 0 until uvHeight) {
            src.position(uvSrcStart + row * width)
            src.get(rowCopyBuf, 0, width)
            dst.position(uvDstStart + row * stride)
            dst.put(rowCopyBuf, 0, width)
        }

        // Total bytes en el input buffer (Y plane padded + UV plane padded).
        return uvDstStart + uvHeight * stride
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val info = MediaCodec.BufferInfo()
        // Cuando estamos cerrando, tope total de 5s para no colgar el ciclo de captura.
        val deadlineNs = if (endOfStream) System.nanoTime() + 5_000_000_000L else Long.MAX_VALUE
        while (true) {
            if (finished && !endOfStream) return  // bucle background termina al finish()
            val outIdx = try {
                codec.dequeueOutputBuffer(info, DEQUEUE_OUT_TIMEOUT_US)
            } catch (e: IllegalStateException) {
                // codec released
                return
            }
            when {
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) {
                        if (finished) return
                        continue
                    }
                    if (System.nanoTime() > deadlineNs) {
                        Log.w(TAG, "drainEncoder: timeout esperando EOS — saliendo con lo que hay")
                        return
                    }
                }
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) {
                        Log.w(TAG, "formato cambió 2 veces — ignorando segundo cambio")
                        continue
                    }
                    val newFmt = codec.outputFormat
                    trackIndex = muxer.addTrack(newFmt)
                    muxer.start()
                    muxerStarted = true
                    Log.i(TAG, "Muxer arrancado con track=$trackIndex")
                }
                outIdx >= 0 -> {
                    val encoded = codec.getOutputBuffer(outIdx)
                    if (encoded == null) {
                        try { codec.releaseOutputBuffer(outIdx, false) } catch (_: Exception) {}
                        continue
                    }
                    // BUFFER_FLAG_CODEC_CONFIG: SPS/PPS — MediaMuxer los toma del outputFormat.
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                        info.size = 0
                    }
                    if (info.size > 0 && muxerStarted) {
                        try {
                            encoded.position(info.offset)
                            encoded.limit(info.offset + info.size)
                            muxer.writeSampleData(trackIndex, encoded, info)
                        } catch (e: Exception) {
                            Log.w(TAG, "writeSampleData falló: ${e.message}")
                        }
                    }
                    try { codec.releaseOutputBuffer(outIdx, false) } catch (_: Exception) {}
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        return
                    }
                }
            }
        }
    }

    /**
     * Finaliza el encoder: signal EOS vía input buffer vacío con
     * BUFFER_FLAG_END_OF_STREAM, drena buffers restantes, cierra muxer y
     * libera recursos. Devuelve el File MP4 generado (o null si no hay datos).
     * Idempotente — múltiples llamadas son seguras.
     *
     * IMPORTANTE: el caller debe haber parado el flujo de frames (cambiar
     * callback o destroy cámara) ANTES de finish(), para que queueNv12 deje
     * de competir por input buffers.
     */
    @Synchronized
    fun finish(): File? {
        if (finished) return outputFile.takeIf { it.exists() && it.length() > 0 }
        finished = true
        try {
            // Signal EOS via empty input buffer con BUFFER_FLAG_END_OF_STREAM
            // (equivalente buffer-mode de codec.signalEndOfInputStream()).
            try {
                val idx = codec.dequeueInputBuffer(500_000L)  // 500 ms — debería haber buffer libre
                if (idx >= 0) {
                    val nowUs = System.nanoTime() / 1000L
                    val ptsUs = if (startPtsUs == -1L) 0L else nowUs - startPtsUs
                    codec.queueInputBuffer(idx, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                } else {
                    Log.w(TAG, "finish(): no input buffer disponible para EOS — drain igual")
                }
            } catch (e: Exception) { Log.w(TAG, "EOS via queueInputBuffer falló: ${e.message}") }
            // Drenar lo que quede en el codec
            drainEncoder(endOfStream = true)
        } catch (e: Exception) {
            Log.e(TAG, "Error en finish(): ${e.message}", e)
        }
        release()
        return outputFile.takeIf { it.exists() && it.length() > 0 }
    }

    /**
     * Heurística rápida (~16 µs) para detectar frames "gris uniforme" producidos
     * por libuvc/libjpeg-turbo al decodificar MJPEG truncado. Muestrea 64 bytes
     * del plano Y espaciados uniformemente y calcula media + varianza:
     *   - Media en rango [108..148] (gris medio, NV12 luma "neutral")
     *   - Varianza < 16 (desviación estándar < 4, esencialmente plano)
     *
     * NO marca como gris una pared blanca (Y≈235) ni una imagen oscura (Y<108).
     * Falsos positivos posibles: scene con cara/objeto centrado en gris perfecto
     * sin textura — extremadamente raro en captura real.
     */
    private fun isLikelyGrayFrame(src: ByteBuffer): Boolean {
        val srcPos = src.position()
        val ySize = width * height
        if (ySize < 64) return false
        val step = ySize / 64
        var sum = 0
        var sumSq = 0
        for (i in 0 until 64) {
            val off = srcPos + i * step
            val byte = try { src.get(off).toInt() and 0xFF } catch (_: Exception) { return false }
            sum += byte
            sumSq += byte * byte
        }
        val mean = sum / 64
        // Varianza poblacional = E[X²] - E[X]²
        val variance = (sumSq / 64) - (mean * mean)
        return mean in 108..148 && variance < 16
    }

    /** Libera todos los recursos. Seguro de llamar múltiples veces. */
    @Synchronized
    fun release() {
        try { codec.stop()    } catch (_: Exception) {}
        try { codec.release() } catch (_: Exception) {}
        try { if (muxerStarted) muxer.stop() } catch (_: Exception) {}
        try { muxer.release() } catch (_: Exception) {}
    }
}
