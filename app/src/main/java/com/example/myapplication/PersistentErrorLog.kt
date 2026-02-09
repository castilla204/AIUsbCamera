package com.example.myapplication

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Log persistente de errores del cliente Android.
 *
 * MOTIVACIÓN
 *   Cuando el orquestador activa modo avión, logcat por WiFi/ADB se corta y el
 *   usuario no puede ver si algo falla durante esa ventana. Este log escribe a
 *   disco interno de la app — sobrevive a desconexión de WiFi, reinicio del
 *   servicio, e incluso crashes del proceso.
 *
 * GARANTÍAS
 *   • Thread-safe (deque concurrente + escritura debounced en su propio thread).
 *   • NUNCA lanza: cada operación capturada en try/catch silencioso, así que un
 *     fallo del log no rompe el código instrumentado.
 *   • Cap absoluto de 500 entradas en RAM y disco — un loop de errores no infla
 *     el almacenamiento.
 *   • Best-effort en disco: si el FS está lleno o read-only, se sigue
 *     manteniendo el buffer en RAM hasta el próximo intento.
 *
 * USO
 *   PersistentErrorLog.init(context)                          // en MainActivity.onCreate
 *   PersistentErrorLog.logError("tag", "msg corto", throwable)
 *   PersistentErrorLog.getRecent(limit=50)                    // para mostrar en UI
 *   PersistentErrorLog.clear()                                // botón "limpiar"
 */
object PersistentErrorLog {
    private const val TAG = "PersistentErrorLog"
    private const val FILE_NAME = "error_log.json"
    private const val MAX_ENTRIES = 500
    private const val MAX_MSG_LEN = 800        // recorta msg larguísimos (stacktraces)
    private const val WRITE_DEBOUNCE_MS = 1500L  // agrupa escrituras en ráfaga

    data class Entry(
        val t: Long,         // epoch ms
        val tag: String,
        val msg: String,
        val stack: String?,  // primeras líneas del stack si hay throwable
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("t", t)
            put("tag", tag)
            put("msg", msg)
            if (stack != null) put("stack", stack)
        }

        companion object {
            fun fromJson(o: JSONObject): Entry? = try {
                Entry(
                    t = o.optLong("t"),
                    tag = o.optString("tag", "?"),
                    msg = o.optString("msg", ""),
                    stack = if (o.isNull("stack")) null
                            else o.optString("stack", "").takeIf { it.isNotBlank() },
                )
            } catch (_: Exception) { null }
        }
    }

    private val buffer = ConcurrentLinkedDeque<Entry>()
    @Volatile private var file: File? = null
    private val initialized = AtomicBoolean(false)
    private val dirty = AtomicBoolean(false)
    private val lastWriteAttempt = AtomicLong(0L)

    // Thread propio para escribir. Daemon → no impide el shutdown del proceso.
    private var writerThread: Thread? = null
    private var isWriterProcess: Boolean = false

    /**
     * Inicializa el log. Idempotente: múltiples llamadas son seguras y solo la
     * primera tiene efecto. Carga las entradas previas del disco al buffer.
     *
     * NUNCA lanza — si falla la carga, el buffer queda vacío y seguimos.
     */
    @JvmStatic
    fun init(ctx: Context) {
        if (!initialized.compareAndSet(false, true)) return
        try {
            file = File(ctx.filesDir, FILE_NAME)
            isWriterProcess = runCatching {
                android.os.Process.myProcessName().endsWith(":uvc")
            }.getOrDefault(false)
            loadFromDisk()
            if (isWriterProcess) startWriterThread()
            // Capturar excepciones no manejadas a nivel global (último recurso).
            installCrashHandler()
            Log.i(TAG, "Iniciado. ${buffer.size} entradas cargadas del disco.")
        } catch (e: Exception) {
            Log.w(TAG, "init falló: ${e.message}")
        }
    }

    /**
     * Registra un error. Llamar desde cualquier hilo. Devuelve inmediatamente
     * (la escritura a disco se hace en background, debounced).
     *
     * CANCELACIÓN COOPERATIVA DE COROUTINES:
     *   Si `error` es `CancellationException` (lanzada por Kotlin cuando un
     *   `Job`/`scope` es cancelado — p.ej. al detener el sistema, desconectar
     *   USB, o cancelar el cycleJob), NO se considera un error de aplicación
     *   sino señalización de control flow. La re-lanzamos para que la
     *   cancelación propague al scope padre como debe, y NO se guarda en el
     *   log (sería ruido — el usuario ve "ERROR" cuando realmente todo va bien).
     *
     *   Esto cubre `JobCancellationException`, `TimeoutCancellationException` y
     *   cualquier otra subclase. Patrón estándar Kotlin: nunca tragar
     *   `CancellationException` salvo intencionalmente con `NonCancellable`.
     *
     * @param tag etiqueta corta (ej. "Orquestador", "BolsilloIaClient")
     * @param msg mensaje legible (truncado a MAX_MSG_LEN chars)
     * @param error opcional — si se pasa, se incluyen las 5 primeras líneas del stack
     */
    @JvmStatic
    @JvmOverloads
    fun logError(tag: String, msg: String, error: Throwable? = null) {
        // FILTRO CRÍTICO: CancellationException NO es error de app. Re-lanzar
        // para propagar la cancelación al scope padre. Hacemos esto ANTES del
        // try/catch envolvente — el throw debe escapar de logError.
        if (error is java.util.concurrent.CancellationException) throw error
        try {
            val cleanMsg = msg.take(MAX_MSG_LEN)
            val stack = error?.let { extractStackPreview(it) }
            val entry = Entry(
                t = System.currentTimeMillis(),
                tag = tag.take(40),
                msg = cleanMsg,
                stack = stack,
            )
            buffer.addLast(entry)
            // Cap: si pasamos del límite, descartamos los más viejos.
            while (buffer.size > MAX_ENTRIES) buffer.pollFirst()
            dirty.set(true)
            // También al logcat para no perder visibilidad cuando hay ADB.
            if (error != null) Log.e(tag, cleanMsg, error)
            else Log.e(tag, cleanMsg)
        } catch (_: Exception) {
            // Cualquier fallo aquí se traga: el log NUNCA debe romper al caller.
        }
    }

    /** Devuelve las últimas N entradas, más recientes primero. */
    @JvmStatic
    fun getRecent(limit: Int = 50): List<Entry> {
        val n = limit.coerceIn(1, MAX_ENTRIES)
        return try {
            try {
                // Multi-proceso: refrescar desde disco para que la UI vea también
                // los errores que haya escrito el servicio :uvc en su propio proceso.
                loadFromDisk()
            } catch (_: Exception) {}
            buffer.toList().takeLast(n).reversed()
        } catch (_: Exception) {
            emptyList()
        }
    }

    /** Cuenta total actual en el buffer. */
    @JvmStatic
    fun size(): Int {
        return try {
            try { loadFromDisk() } catch (_: Exception) {}
            buffer.size
        } catch (_: Exception) {
            0
        }
    }

    /** Borra TODO (RAM + disco). Idempotente. */
    @JvmStatic
    fun clear() {
        try {
            buffer.clear()
            dirty.set(true)
            // Forzar escritura inmediata para que el disco quede vacío.
            file?.takeIf { it.exists() }?.let {
                try { it.delete() } catch (_: Exception) {}
            }
        } catch (_: Exception) {}
    }

    // ──── Internos ─────────────────────────────────────────────────────────────

    private fun extractStackPreview(t: Throwable): String {
        return try {
            val sb = StringBuilder()
            sb.append(t.javaClass.simpleName)
            t.message?.let { sb.append(": ").append(it.take(200)) }
            sb.append('\n')
            // Solo las primeras 5 frames — suficiente para localizar y no infla el JSON
            t.stackTrace.take(5).forEach { frame ->
                sb.append("  at ").append(frame.className).append('.')
                  .append(frame.methodName).append('(')
                  .append(frame.fileName ?: "?").append(':')
                  .append(frame.lineNumber).append(")\n")
            }
            t.cause?.let { c ->
                sb.append("Caused by: ").append(c.javaClass.simpleName)
                  .append(": ").append((c.message ?: "").take(200)).append('\n')
            }
            sb.toString().take(1500)
        } catch (_: Exception) {
            "(stack no disponible)"
        }
    }

    private fun loadFromDisk() {
        val f = file ?: return
        if (!f.exists()) {
            buffer.clear()
            return
        }
        try {
            val text = f.readText(Charsets.UTF_8)
            if (text.isBlank()) {
                buffer.clear()
                return
            }
            val arr = JSONArray(text)
            val loaded = ArrayList<Entry>(arr.length())
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                Entry.fromJson(o)?.let { loaded.add(it) }
            }
            buffer.clear()
            loaded.takeLast(MAX_ENTRIES).forEach { buffer.addLast(it) }
        } catch (e: Exception) {
            Log.w(TAG, "loadFromDisk falló: ${e.message} — empezando vacío")
            buffer.clear()
            // Backup del archivo corrupto para diagnóstico, sin bloquear.
            try {
                val bak = File(f.parentFile, "$FILE_NAME.corrupt")
                f.copyTo(bak, overwrite = true)
            } catch (_: Exception) {}
        }
    }

    private fun saveToDiskInternal() {
        val f = file ?: return
        try {
            // Snapshot atómico del buffer (concurrent deque permite iteración segura).
            val snapshot = buffer.toList()
            val arr = JSONArray()
            snapshot.forEach { entry ->
                try { arr.put(entry.toJson()) } catch (_: Exception) {}
            }
            // Escritura atómica: tmp → rename. Si peta a mitad, el archivo bueno
            // anterior queda intacto.
            val tmp = File(f.parentFile, "$FILE_NAME.tmp")
            tmp.writeText(arr.toString(), Charsets.UTF_8)
            if (!tmp.renameTo(f)) {
                // Fallback: copy + delete (Android puede rechazar rename entre FS).
                tmp.copyTo(f, overwrite = true)
                try { tmp.delete() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            Log.w(TAG, "saveToDisk falló: ${e.message} — el buffer en RAM queda intacto")
        }
    }

    /** Thread daemon que vuelca a disco con debounce. NUNCA muere. */
    private fun startWriterThread() {
        val t = Thread({
            while (true) {
                try {
                    Thread.sleep(WRITE_DEBOUNCE_MS)
                    if (dirty.compareAndSet(true, false)) {
                        lastWriteAttempt.set(System.currentTimeMillis())
                        saveToDiskInternal()
                    }
                } catch (_: InterruptedException) {
                    // Interrumpido: salimos limpiamente intentando un último flush.
                    try { if (dirty.get()) saveToDiskInternal() } catch (_: Exception) {}
                    return@Thread
                } catch (e: Exception) {
                    // Cualquier otra excepción inesperada: dormir un poco y seguir.
                    try { Thread.sleep(5000L) } catch (_: InterruptedException) { return@Thread }
                }
            }
        }, "PersistentErrorLog-writer")
        t.isDaemon = true
        t.start()
        writerThread = t
    }

    private fun installCrashHandler() {
        try {
            val previous = Thread.getDefaultUncaughtExceptionHandler()
            Thread.setDefaultUncaughtExceptionHandler { thread, ex ->
                try {
                    logError("UNCAUGHT", "Crash en thread '${thread.name}'", ex)
                    // Forzar flush sincrónico antes de morir.
                    if (dirty.get()) saveToDiskInternal()
                } catch (_: Throwable) {}
                // Delegar al handler anterior (Android default) para que el proceso
                // muera con el ANR/crash dialog estándar.
                try { previous?.uncaughtException(thread, ex) } catch (_: Throwable) {}
            }
        } catch (_: Exception) {}
    }
}
