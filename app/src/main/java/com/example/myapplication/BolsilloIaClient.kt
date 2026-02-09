package com.example.myapplication

import android.util.Log
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Cliente para el relay de IA "bolsillo".
 *
 * El parámetro relayUrls solo sirve como SEMILLA si SharedPreferences aún no
 * tiene URLs guardadas (primera ejecución / tras un wipe). En cuanto el usuario
 * llama a [loadKeysFromPrefs] o [setRelayOrder], el estado mutable del
 * companion (RELAY_URLS) manda. Los métodos del cliente leen siempre de ahí.
 */
class BolsilloIaClient(
    relayUrls: List<String> = listOfNotNull(
        URL_RENDER.takeIf { it.isNotBlank() },
        URL_RAILWAY.takeIf { it.isNotBlank() },
    ),
) {
    init {
        // Solo sembrar si nadie ha tocado el companion todavía.
        if (RELAY_URLS.isEmpty()) RELAY_URLS = relayUrls
    }

    /** URLs activas en orden (primary → backup). Se lee del companion mutable. */
    private val relayUrls: List<String> get() = RELAY_URLS

    @Volatile private var activeRelayUrl: String = RELAY_URLS.firstOrNull().orEmpty()

    companion object {
        // Endpoints: defaults desde BuildConfig (secrets.properties / CI). Editables en runtime.
        val URL_RENDER: String
            get() = BuildConfig.RELAY_URL_RENDER.ifBlank { "" }
        val URL_RAILWAY: String
            get() = BuildConfig.RELAY_URL_RAILWAY.ifBlank { "" }
        const val LABEL_RENDER  = "Render"
        const val LABEL_RAILWAY = "Railway"

        /** Orden actual de los relays (primary, backup, ...). Mutable. */
        @Volatile var RELAY_URLS: List<String> = listOfNotNull(
            BuildConfig.RELAY_URL_RENDER.takeIf { it.isNotBlank() },
            BuildConfig.RELAY_URL_RAILWAY.takeIf { it.isNotBlank() },
        )

        /** Etiqueta humana para una URL conocida. "?" si es desconocida. */
        @JvmStatic
        fun labelFor(url: String): String = when (url) {
            URL_RENDER  -> LABEL_RENDER
            URL_RAILWAY -> LABEL_RAILWAY
            else        -> "?"
        }

        // Claves: defaults desde BuildConfig (nunca hardcodear en el repo).
        // La UI / SharedPreferences pueden sobrescribirlas en runtime.
        @Volatile var API_KEY_RELAY = BuildConfig.API_KEY_RELAY

        @Volatile var ANTHROPIC_KEY = BuildConfig.ANTHROPIC_API_KEY
        @Volatile var OPENAI_KEY    = BuildConfig.OPENAI_API_KEY
        @Volatile var GEMINI_KEY    = BuildConfig.GEMINI_API_KEY

        // Respaldos (vacío = no usar)
        @Volatile var ANTHROPIC_KEY_BACKUP = ""
        @Volatile var OPENAI_KEY_BACKUP    = ""
        @Volatile var GEMINI_KEY_BACKUP    = ""

        // Pesos en la fusión por votación del relay (1 = básica, 2 = doble, 0 = ignorar).
        // Persistidos en SharedPreferences y replicados al servidor via /api/config.
        @Volatile var ANTHROPIC_WEIGHT = 1
        @Volatile var OPENAI_WEIGHT    = 1
        @Volatile var GEMINI_WEIGHT    = 1

        // Modelos RÁPIDOS para el fallback directo desde el móvil.
        // Mutables: se pueden editar desde la UI por si Anthropic/OpenAI/Google
        // retiran un modelo o sale uno nuevo. Defaults: los más rápidos disponibles
        // porque el fallback se usa cuando hay una ventana de RF abierta y hay que
        // minimizar tiempo expuesto.
        @Volatile var CLAUDE_MODEL   = "claude-3-5-haiku-20241022"        // ~1-2s
        @Volatile var OPENAI_MODEL   = "gpt-4o-mini"                      // ~1-2s
        @Volatile var GEMINI_MODEL   = "gemini-2.0-flash-lite-preview-02-25"  // ~1s

        private const val FAST_MAX_TOKENS = 2000
        private const val FAST_THINKING_BUDGET = 1024
        private const val OCR_MAX_TOKENS = 1500
        private val ACTIVE_CALLS = ConcurrentHashMap.newKeySet<Call>()

        @JvmStatic
        fun cancelAllActiveCalls() {
            for (c in ACTIVE_CALLS) {
                try { c.cancel() } catch (_: Exception) {}
            }
        }

        /**
         * Carga claves desde SharedPreferences y las publica en las variables companion.
         * Llamar al arrancar la app/servicio para que sobrevivan reinicios.
         */
        // Lista de keys obsoletas que deben migrarse al default actual cuando se
        // detectan en SharedPreferences. Útil cuando rotamos una key revocada y
        // no queremos depender de syncConfigFromRelay (que solo actualiza si el
        // server tiene un valor real). Mantener corta y solo añadir cuando se rota.
        private val DEPRECATED_GEMINI_KEYS = setOf<String>(
            // Añadir aquí fingerprints de keys revocadas si hace falta migrar prefs.
        )

        @JvmStatic
        fun loadKeysFromPrefs(ctx: android.content.Context) {
            val p = ctx.getSharedPreferences("BolsilloKeys", android.content.Context.MODE_PRIVATE)
            // Patrón: si la pref NO existe (1ª ejecución / wipe) → mantenemos el default
            // hardcoded del companion. Si existe (aunque sea ""), aplicamos lo guardado —
            // así el server puede revocar/limpiar una key vía syncConfigFromRelay.
            if (p.contains("relay_key"))     API_KEY_RELAY = p.getString("relay_key", "") ?: ""
            if (p.contains("anthropic_key")) ANTHROPIC_KEY = p.getString("anthropic_key", "") ?: ""
            if (p.contains("openai_key"))    OPENAI_KEY    = p.getString("openai_key",    "") ?: ""
            if (p.contains("gemini_key")) {
                val stored = p.getString("gemini_key", "") ?: ""
                if (stored in DEPRECATED_GEMINI_KEYS) {
                    // Migración: sobreescribimos la pref con el default actual.
                    // GEMINI_KEY ya tiene el valor nuevo (default del companion).
                    p.edit().putString("gemini_key", GEMINI_KEY).apply()
                    android.util.Log.w("BolsilloIaClient",
                        "GEMINI_KEY obsoleta detectada en prefs → migrada a default actual")
                } else {
                    GEMINI_KEY = stored
                }
            }
            // Los backups: si la pref no existe quedan "" (no hay default hardcoded).
            ANTHROPIC_KEY_BACKUP = p.getString("anthropic_key_backup", "") ?: ""
            OPENAI_KEY_BACKUP    = p.getString("openai_key_backup",    "") ?: ""
            GEMINI_KEY_BACKUP    = p.getString("gemini_key_backup",    "") ?: ""

            // Orden de relays (primary, backup). Si solo hay 1 → falta el otro como backup.
            val primary = p.getString("relay_primary", null)
            val backup  = p.getString("relay_backup",  null)
            if (!primary.isNullOrBlank()) {
                val ordered = mutableListOf(primary)
                if (!backup.isNullOrBlank() && backup != primary) ordered.add(backup)
                // Garantizamos que la URL no elegida siga disponible como último fallback.
                for (known in listOf(URL_RENDER, URL_RAILWAY)) {
                    if (known !in ordered) ordered.add(known)
                }
                RELAY_URLS = ordered
            }

            // Pesos por IA (clamp 0..10)
            ANTHROPIC_WEIGHT = p.getInt("anthropic_weight", 1).coerceIn(0, 10)
            OPENAI_WEIGHT    = p.getInt("openai_weight",    1).coerceIn(0, 10)
            GEMINI_WEIGHT    = p.getInt("gemini_weight",    1).coerceIn(0, 10)

            // Modelos del fallback directo (string vacío = mantener default).
            p.getString("claude_model",   null)?.takeIf { it.isNotBlank() }?.let { CLAUDE_MODEL   = it }
            p.getString("openai_model",   null)?.takeIf { it.isNotBlank() }?.let { OPENAI_MODEL   = it }
            p.getString("gemini_model",   null)?.takeIf { it.isNotBlank() }?.let { GEMINI_MODEL   = it }
        }

        /** Guarda claves en SharedPreferences y las activa en runtime. */
        @JvmStatic
        fun saveKeysToPrefs(
            ctx: android.content.Context,
            relayKey: String? = null,
            anthropicKey: String? = null, anthropicBackup: String? = null,
            openaiKey: String? = null,    openaiBackup:    String? = null,
            geminiKey: String? = null,    geminiBackup:    String? = null,
        ) {
            val p = ctx.getSharedPreferences("BolsilloKeys", android.content.Context.MODE_PRIVATE).edit()
            if (relayKey != null)        { p.putString("relay_key",            relayKey.trim());        API_KEY_RELAY        = relayKey.trim() }
            if (anthropicKey != null)    { p.putString("anthropic_key",        anthropicKey.trim());    ANTHROPIC_KEY        = anthropicKey.trim() }
            if (anthropicBackup != null) { p.putString("anthropic_key_backup", anthropicBackup.trim()); ANTHROPIC_KEY_BACKUP = anthropicBackup.trim() }
            if (openaiKey != null)       { p.putString("openai_key",           openaiKey.trim());       OPENAI_KEY           = openaiKey.trim() }
            if (openaiBackup != null)    { p.putString("openai_key_backup",    openaiBackup.trim());    OPENAI_KEY_BACKUP    = openaiBackup.trim() }
            if (geminiKey != null)       { p.putString("gemini_key",           geminiKey.trim());       GEMINI_KEY           = geminiKey.trim() }
            if (geminiBackup != null)    { p.putString("gemini_key_backup",    geminiBackup.trim());    GEMINI_KEY_BACKUP    = geminiBackup.trim() }
            p.apply()
        }

        /** Guarda los modelos del fallback directo (móvil) en SharedPreferences. */
        @JvmStatic
        fun saveModels(
            ctx: android.content.Context,
            claudeModel: String? = null,
            openaiModel: String? = null,
            geminiModel: String? = null,
        ) {
            val e = ctx.getSharedPreferences("BolsilloKeys", android.content.Context.MODE_PRIVATE).edit()
            if (claudeModel != null)   { val v = claudeModel.trim();   e.putString("claude_model",   v); if (v.isNotEmpty()) CLAUDE_MODEL   = v }
            if (openaiModel != null)   { val v = openaiModel.trim();   e.putString("openai_model",   v); if (v.isNotEmpty()) OPENAI_MODEL   = v }
            if (geminiModel != null)   { val v = geminiModel.trim();   e.putString("gemini_model",   v); if (v.isNotEmpty()) GEMINI_MODEL   = v }
            e.apply()
        }

        /**
         * Cambia el orden de los relays (qué API se llama primero) y lo persiste.
         * La UI le pasa una de las dos: URL_RENDER o URL_RAILWAY como primary; el
         * otro queda como backup automáticamente.
         */
        @JvmStatic
        fun setRelayOrder(ctx: android.content.Context, primaryUrl: String) {
            val backupUrl = if (primaryUrl == URL_RENDER) URL_RAILWAY else URL_RENDER
            RELAY_URLS = listOf(primaryUrl, backupUrl)
            ctx.getSharedPreferences("BolsilloKeys", android.content.Context.MODE_PRIVATE)
                .edit()
                .putString("relay_primary", primaryUrl)
                .putString("relay_backup",  backupUrl)
                .apply()
        }

        /**
         * Guarda los pesos por IA en SharedPreferences. Los publica en runtime
         * inmediatamente. Para que el servidor también los use en su fusión,
         * el caller debe llamar a [pushWeightsToRelay] después (o a /api/config).
         */
        @JvmStatic
        fun saveWeights(
            ctx: android.content.Context,
            anthropic: Int? = null, openai: Int? = null, gemini: Int? = null,
        ) {
            val e = ctx.getSharedPreferences("BolsilloKeys", android.content.Context.MODE_PRIVATE).edit()
            if (anthropic != null) { ANTHROPIC_WEIGHT = anthropic.coerceIn(0, 10); e.putInt("anthropic_weight", ANTHROPIC_WEIGHT) }
            if (openai    != null) { OPENAI_WEIGHT    = openai.coerceIn(0, 10);    e.putInt("openai_weight",    OPENAI_WEIGHT) }
            if (gemini    != null) { GEMINI_WEIGHT    = gemini.coerceIn(0, 10);    e.putInt("gemini_weight",    GEMINI_WEIGHT) }
            e.apply()
        }
    }

    /**
     * Ejecuta [fn] con la clave primaria. Si lanza excepción y existe una clave de respaldo no vacía,
     * reintenta con ésta. Garantiza que devolverá un resultado si alguna de las dos funciona.
     *
     * FIX DE FIABILIDAD: cuando AMBAS llamadas devuelven null (no excepción) o ambas
     * lanzan excepción, ahora logueamos el motivo en PersistentErrorLog para que el
     * operador pueda diagnosticar "fallback-fail" sin tener que adivinar. Antes esta
     * función devolvía null silencioso → diagnóstico imposible en producción.
     */
    private inline fun <T> tryWithBackup(primary: String, backup: String, fn: (String) -> T?): T? {
        val p = primary.trim()
        val b = backup.trim()
        if (p.isEmpty() && b.isEmpty()) {
            try { PersistentErrorLog.logError("BolsilloIaClient", "tryWithBackup: sin keys (ni primary ni backup)") } catch (_: Throwable) {}
            return null
        }
        var firstExc: Exception? = null
        if (p.isNotEmpty()) {
            try {
                val r = fn(p)
                if (r != null) return r
                // Primary devolvió null sin lanzar — caer a backup si existe.
            } catch (e: Exception) {
                firstExc = e
                if (b.isEmpty()) {
                    try { PersistentErrorLog.logError("BolsilloIaClient", "tryWithBackup: primary lanzó y sin backup: ${e.message}", e) } catch (_: Throwable) {}
                    return null
                }
            }
        }
        if (b.isEmpty()) {
            try { PersistentErrorLog.logError("BolsilloIaClient", "tryWithBackup: primary devolvió null y sin backup") } catch (_: Throwable) {}
            return null
        }
        return try {
            val r = fn(b)
            if (r == null) {
                try {
                    PersistentErrorLog.logError(
                        "BolsilloIaClient",
                        "tryWithBackup: AMBAS keys fallaron (primary=${if (firstExc != null) "exc:${firstExc.message}" else "null"}, backup=null)",
                    )
                } catch (_: Throwable) {}
            }
            r
        } catch (e: Exception) {
            try {
                PersistentErrorLog.logError(
                    "BolsilloIaClient",
                    "tryWithBackup: AMBAS keys lanzaron (primary=${firstExc?.message ?: "n/a"} · backup=${e.message})",
                    e,
                )
            } catch (_: Throwable) {}
            null
        }
    }

    // AISLAMIENTO DE DISPATCHERS / CONNECTION POOLS:
    //   Por defecto, todos los OkHttpClient comparten el ExecutorService global
    //   y el ConnectionPool del primer client creado (vía `.dispatcher()` y
    //   `.connectionPool()`). Si httpRelayBig satura con un upload de 4 MB
    //   (writeTimeout=120s), las conexiones se cuelgan ahí dentro del mismo
    //   pool — y http (connectTimeout=10s) puede timeout antes incluso de
    //   conseguir un thread del executor.
    //
    //   Cada client tiene su PROPIO dispatcher + connection pool independientes,
    //   con `maxRequests` y `maxRequestsPerHost` razonables. Resultado: los 3
    //   clients no se afectan entre sí — un upload pesado de relay no bloquea
    //   los polls de /api/jobs ni las llamadas directas a IAs.
    private fun makeIsolatedDispatcher(maxRequests: Int = 4): okhttp3.Dispatcher {
        return okhttp3.Dispatcher().apply {
            this.maxRequests = maxRequests
            this.maxRequestsPerHost = maxRequests  // mismo host casi siempre
        }
    }

    private val httpRelay = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(45, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .dispatcher(makeIsolatedDispatcher(maxRequests = 4))
        .connectionPool(okhttp3.ConnectionPool(4, 5, TimeUnit.MINUTES))
        .build()

    // Cliente específico para uploads pesados (video MP4 base64 ~33 MB en modo VIDEO).
    // En 4G/3G, 20s de writeTimeout no alcanza: necesitamos margen real para no
    // caer a fallback-fail por timeout de subida. La API a su vez tiene timeout
    // de procesamiento propio (AI_HARD_TIMEOUT=120s server-side).
    // Dispatcher SEPARADO: estos uploads largos NO deben bloquear los polls
    // rápidos a /api/jobs (httpRelay) ni las llamadas directas a IAs (http).
    private val httpRelayBig = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .callTimeout(180, TimeUnit.SECONDS)  // tope absoluto: subida + respuesta
        .dispatcher(makeIsolatedDispatcher(maxRequests = 2))  // sólo 2 uploads simultáneos
        .connectionPool(okhttp3.ConnectionPool(2, 2, TimeUnit.MINUTES))
        .build()

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(45, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .dispatcher(makeIsolatedDispatcher(maxRequests = 4))
        .connectionPool(okhttp3.ConnectionPool(4, 5, TimeUnit.MINUTES))
        .build()

    /** Elige el cliente HTTP según tamaño del body. Bodies >500KB (típicamente video,
     *  ya que un MP4 de 10s a 3 Mbps + base64 ronda los 4 MB) usan el cliente con
     *  writeTimeout largo. Umbral bajo: en 4G, 500 KB tardan ~5s que ya roza el
     *  writeTimeout=20s del cliente normal cuando hay latencia variable. */
    private fun pickRelayClient(bodyBytes: Int): OkHttpClient =
        if (bodyBytes > 500_000) httpRelayBig else httpRelay

    private val JSON = "application/json; charset=utf-8".toMediaType()

    private inline fun <T> executeTracked(call: Call, block: (Response) -> T): T {
        ACTIVE_CALLS.add(call)
        return try {
            call.execute().use { resp -> block(resp) }
        } finally {
            ACTIVE_CALLS.remove(call)
        }
    }

    private suspend inline fun <T> executeTrackedSuspend(
        call: Call,
        crossinline block: (Response) -> T,
    ): T = suspendCancellableCoroutine { cont ->
        ACTIVE_CALLS.add(call)
        cont.invokeOnCancellation {
            try { call.cancel() } catch (_: Exception) {}
            ACTIVE_CALLS.remove(call)
        }
        call.enqueue(object : okhttp3.Callback {
            override fun onFailure(call: Call, e: java.io.IOException) {
                ACTIVE_CALLS.remove(call)
                if (!cont.isActive) return
                cont.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    response.use { resp ->
                        val out = block(resp)
                        ACTIVE_CALLS.remove(call)
                        if (!cont.isActive) return
                        cont.resume(out)
                    }
                } catch (t: Throwable) {
                    ACTIVE_CALLS.remove(call)
                    if (!cont.isActive) return
                    cont.resumeWithException(t)
                }
            }
        })
    }

    private fun parseFetchOutcomeFromResponse(jobId: String, resp: Response): FetchOutcome {
        if (!resp.isSuccessful) {
            PersistentErrorLog.logError(
                "BolsilloIaClient",
                "GET /result/$jobId HTTP ${resp.code} en ${BolsilloIaClient.labelFor(activeRelayUrl)}",
            )
            return FetchOutcome.NetworkError
        }
        val body = resp.body?.string() ?: return FetchOutcome.NetworkError
        val json = JSONObject(body)
        val status = json.optString("status")
        val ctxJpegFromAnyStatus: ByteArray? = run {
            val ctxB64 = json.optString("context_image_b64")
            if (ctxB64.isEmpty()) return@run null
            try {
                android.util.Base64.decode(ctxB64, android.util.Base64.NO_WRAP)
            } catch (e: Exception) {
                Log.w("BolsilloIaClient", "context_image_b64 inválido en /result/$jobId: ${e.message}")
                null
            }
        }
        if (status == "pending") return FetchOutcome.Pending
        if (status == "done") {
            val correctionsArr = json.optJSONArray("corrections")
            val corrections = mutableListOf<Correction>()
            if (correctionsArr != null) {
                for (i in 0 until correctionsArr.length()) {
                    val o = correctionsArr.getJSONObject(i)
                    corrections.add(Correction(
                        sourceJobId = o.optString("source_job_id"),
                        answer      = o.optString("answer"),
                        page        = if (o.isNull("page")) null else o.optInt("page"),
                    ))
                }
            }
            return FetchOutcome.Done(IaResultado(
                answer    = json.optString("answer"),
                provider  = json.optString("provider").ifEmpty { "merged" },
                model     = json.optString("model"),
                viaRelay  = true,
                corrections = corrections,
                contextImageJpeg = ctxJpegFromAnyStatus,
            ))
        }
        if (status == "error") {
            val partialAnswer = json.optString("answer")
                .ifEmpty { json.optString("merged_answer") }
            val hasUsefulLetters = partialAnswer.any { it in "ABCDXabcdx" }
            if (hasUsefulLetters) {
                val errMsg = json.optString("error")
                Log.w("BolsilloIaClient",
                    "Relay status=error pero merged_answer útil ('${partialAnswer.take(40)}'): $errMsg — usando respuesta parcial")
                return FetchOutcome.Done(IaResultado(
                    answer    = partialAnswer,
                    provider  = json.optString("provider").ifEmpty { "merged-partial" },
                    model     = json.optString("model"),
                    viaRelay  = true,
                    intentos  = listOf("status=error con answer útil: $errMsg"),
                    contextImageJpeg = ctxJpegFromAnyStatus,
                ))
            }
            return FetchOutcome.Done(IaResultado(
                answer = "", provider = "relay", model = "",
                viaRelay = true,
                intentos = listOf(json.optString("error")),
                // Si el cómplice marcó la casilla pero el job cerró en error sin letras,
                // igual conservamos el diagrama para las siguientes preguntas.
                contextImageJpeg = ctxJpegFromAnyStatus,
            ))
        }
        return FetchOutcome.Pending
    }

    data class Correction(
        val sourceJobId: String,
        val answer: String,
        val page: Int?,
    )

    data class IaResultado(
        val answer: String,
        val provider: String,
        val model: String,
        val citations: List<String> = emptyList(),
        val viaRelay: Boolean,
        val intentos: List<String> = emptyList(),
        val corrections: List<Correction> = emptyList(),
        // Timings finos (ms). Defaults a 0 para no romper callers de fallback.
        // uploadMs = tiempo desde inicio de POST /ask hasta recibir job_id.
        // pollMs   = tiempo desde 1er GET /result hasta recibir status=done.
        // totalMs  = end-to-end del ciclo (envío inicial → respuesta en mano).
        val uploadMs: Long = 0L,
        val pollMs:   Long = 0L,
        val totalMs:  Long = 0L,
        // Si el cómplice marcó la casilla 🖼️ "Imagen en pregunta" en el panel,
        // el relay devuelve aquí el JPEG decodificado de la mejor imagen del
        // job (fused Topaz > stacking local > frame top-1 > burst). El móvil
        // lo guarda como diagrama de contexto del caso práctico y lo adjunta
        // a las siguientes solicitudes. Sólo no-null cuando el cómplice lo
        // pidió EXPLÍCITAMENTE; la detección automática vía "CONTEXTO_VISUAL:SI"
        // del rawText sigue funcionando en paralelo.
        val contextImageJpeg: ByteArray? = null,
    )

    // --- NUEVAS ESTRUCTURAS PARA POLLING PARCIAL ---

    data class PartialResponse(
        val provider: String,
        val answer: String,
        val status: String, // "pending", "processing", "done", "error"
        val error: String? = null
    )

    data class JobProgress(
        val jobId: String,
        val status: String,
        val responses: List<PartialResponse>,
        val fusion: String,
        val accomplice: String,
        val reviewDeadline: Long
    )

    /**
     * Llama POST /reset en TODOS los relays configurados. Devuelve true si al menos
     * uno respondió 200. Pensado para invocarse al iniciar un examen nuevo: limpia
     * jobs anteriores en el servidor pero NO toca la config (keys, modelos).
     */
    suspend fun resetContext(): Boolean {
        var anyOk = false
        for (url in relayUrls) {
            try {
                val req = Request.Builder()
                    .url("$url/reset")
                    .addHeader("X-API-Key", API_KEY_RELAY)
                    .post("".toRequestBody(JSON))
                    .build()
                executeTracked(httpRelay.newCall(req)) { resp ->
                    if (resp.isSuccessful) anyOk = true
                }
            } catch (_: Exception) { /* probar siguiente URL */ }
        }
        return anyOk
    }

    /**
     * Modo B/C: llama directamente a las 3 IAs en cascada (Anthropic → OpenAI → Gemini),
     * devolviendo la primera que responda con una respuesta no vacía. Usa internamente
     * el sistema de respaldo (key principal → backup) de cada función directXxx.
     *
     * Cuando soloDirecto=true, no intenta el relay (los callers lo usan después de que
     * el relay ya ha fallado o no se quiere usar).
     */
    suspend fun askConFallback(
        prompt: String,
        system: String?,
        webSearch: Boolean,
        imageB64: String?,
        imageMime: String = "image/jpeg",
        contextImageB64: String? = null,
        soloDirecto: Boolean = false,
    ): IaResultado {
        val intentos = mutableListOf<String>()
        // 1) Anthropic
        try {
            val r = directAnthropic(prompt, system, webSearch, imageB64, imageMime, contextImageB64)
            if (r != null && r.answer.isNotEmpty()) return r.copy(intentos = intentos + "anthropic-ok")
            intentos.add("anthropic-null")
        } catch (e: Exception) { intentos.add("anthropic-exc:${e.message?.take(60)}") }
        // 2) OpenAI
        try {
            val r = directOpenAI(prompt, system, webSearch, imageB64, imageMime, contextImageB64)
            if (r != null && r.answer.isNotEmpty()) return r.copy(intentos = intentos + "openai-ok")
            intentos.add("openai-null")
        } catch (e: Exception) { intentos.add("openai-exc:${e.message?.take(60)}") }
        // 3) Gemini
        try {
            val r = directGemini(prompt, system, webSearch, imageB64, imageMime, contextImageB64)
            if (r != null && r.answer.isNotEmpty()) return r.copy(intentos = intentos + "gemini-ok")
            intentos.add("gemini-null")
        } catch (e: Exception) { intentos.add("gemini-exc:${e.message?.take(60)}") }
        // Todas fallaron
        return IaResultado(
            answer    = "",
            provider  = "fallback-fail",
            model     = "",
            viaRelay  = false,
            intentos  = intentos,
        )
    }

    private fun extractFinalLetters(rawText: String): String {
        val match = Regex("<FINAL>(.*?)</FINAL>", RegexOption.IGNORE_CASE).find(rawText)
        val ans = if (match != null) {
            match.groupValues[1].replace(Regex("[^ABCDXabcdx]"), "").uppercase()
        } else {
            rawText.replace(Regex("[^ABCDXabcdx]"), "").uppercase()
        }
        return if (ans.isNotEmpty()) ans else "X"
    }

    suspend fun startJob(
        prompt: String,
        system: String? = null,
        webSearch: Boolean = true,
        imageB64: String? = null,
        imageMime: String = "image/jpeg",
        videoB64: String? = null,
        videoMime: String = "video/mp4",
        contextImageB64: String? = null,
        contextImageMime: String = "image/jpeg",
        reviewTimeoutSeconds: Int = 90,
        simOperator: String? = null,
        simDbm: Int? = null,
        simSlot: Int? = null,
        simSummary: String? = null,
    ): String? {
        val bodyJson = JSONObject().apply {
            put("prompt", prompt)
            if (system != null) put("system", system)
            put("web_search", webSearch)
            // Modo VIDEO tiene prioridad: si llega videoB64, NO mandamos image_b64
            // (el server detecta video_b64 y arranca el pipeline OCR + análisis).
            if (videoB64 != null) {
                put("video_b64", videoB64)
                put("video_mime", videoMime)
            } else if (imageB64 != null) {
                put("image_b64", imageB64)
                put("image_mime", imageMime)
            }
            // Diagrama de contexto (caso práctico): si el móvil tiene uno guardado
            // tras detectar CONTEXTO_VISUAL:SI en una respuesta anterior, lo
            // adjuntamos para que TODOS los analyzers/OCRs lo vean como IMAGEN 1.
            // El relay reusa el campo en cada solicitud — no se persiste server-side
            // (el móvil decide en qué jobs ir adjuntándolo).
            if (contextImageB64 != null) {
                put("context_image_b64", contextImageB64)
                put("context_image_mime", contextImageMime)
            }
            put("review_timeout_seconds", reviewTimeoutSeconds)
            // Telemetría SIM: operador + dBm de la SIM efectivamente elegida en esta
            // ventana, y resumen legible de ambas. Lo lee el panel del relay para
            // mostrar qué SIM se usó y cómo estaba la señal cuando se mandó la foto.
            if (simOperator != null) put("sim_operator", simOperator)
            if (simDbm != null) put("sim_dbm", simDbm)
            if (simSlot != null) put("sim_slot", simSlot)
            if (simSummary != null) put("sim_summary", simSummary)
        }.toString()

        for (url in relayUrls) {
            val jobId = tryStartJobOn(url, bodyJson) ?: continue
            activeRelayUrl = url
            return jobId
        }
        return null
    }

    private suspend fun tryStartJobOn(url: String, bodyJson: String): String? {
        return try {
            val bodyBytes = bodyJson.toByteArray(Charsets.UTF_8).size
            val client = pickRelayClient(bodyBytes)
            val hasContextImage = bodyJson.contains("\"context_image_b64\"")
            val contextKb = if (hasContextImage) {
                try {
                    val b64 = JSONObject(bodyJson).optString("context_image_b64")
                    (b64.length * 3 / 4) / 1024
                } catch (_: Exception) { -1 }
            } else 0
            android.util.Log.i("BolsilloIaClient",
                "startJob: POST $url/ask · body=${bodyBytes / 1024} KB · ctx=${if (hasContextImage) "SI" else "NO"}${if (hasContextImage) "(${if (contextKb >= 0) "${contextKb}KB" else "?"})" else ""} · client=${if (bodyBytes > 1_000_000) "BIG" else "SMALL"}")
            val req = Request.Builder()
                .url("$url/ask")
                .addHeader("X-API-Key", API_KEY_RELAY)
                .post(bodyJson.toRequestBody(JSON))
                .build()
            executeTrackedSuspend(client.newCall(req)) { resp ->
                if (!resp.isSuccessful) {
                    val bodyTxt = resp.body?.string()?.take(200) ?: ""
                    android.util.Log.w("BolsilloIaClient",
                        "startJob FAIL $url: HTTP ${resp.code} body='$bodyTxt'")
                    PersistentErrorLog.logError(
                        "BolsilloIaClient",
                        "POST /ask falló en ${BolsilloIaClient.labelFor(url)}: HTTP ${resp.code} · body=$bodyTxt",
                    )
                    null
                } else {
                    val raw = resp.body?.string()
                    if (raw.isNullOrEmpty()) {
                        null
                    } else {
                        val json = JSONObject(raw)
                        val jobId = json.optString("job_id").takeIf { it.isNotEmpty() }
                        android.util.Log.i("BolsilloIaClient", "startJob OK $url → jobId=$jobId")
                        jobId
                    }
                }
            }
        } catch (e: Exception) {
            android.util.Log.w("BolsilloIaClient", "startJob $url EXC: ${e.javaClass.simpleName}: ${e.message}")
            PersistentErrorLog.logError(
                "BolsilloIaClient",
                "POST /ask excepción en ${BolsilloIaClient.labelFor(url)}: ${e.javaClass.simpleName}: ${e.message}",
                e,
            )
            null
        }
    }

    /** Polling parcial para ver como llegan las respuestas */
    suspend fun fetchPartial(jobId: String): JobProgress? {
        return try {
            val req = Request.Builder()
                .url("$activeRelayUrl/api/partial/$jobId?key=$API_KEY_RELAY")
                .get()
                .build()

            executeTrackedSuspend(httpRelay.newCall(req)) { resp ->
                if (!resp.isSuccessful) {
                    null
                } else {
                    val raw = resp.body?.string()
                    if (raw.isNullOrEmpty()) return@executeTrackedSuspend null
                    val json = JSONObject(raw)
                
                    val resArr = json.optJSONArray("responses")
                    val responses = mutableListOf<PartialResponse>()
                    if (resArr != null) {
                        for (i in 0 until resArr.length()) {
                            val o = resArr.getJSONObject(i)
                            responses.add(PartialResponse(
                                provider = o.optString("provider"),
                                answer = o.optString("answer"),
                                status = o.optString("status"),
                                error = o.optString("error").takeIf { it.isNotEmpty() }
                            ))
                        }
                    }

                    JobProgress(
                        jobId = jobId,
                        status = json.optString("status"),
                        responses = responses,
                        fusion = json.optString("fusion"),
                        accomplice = json.optString("accomplice"),
                        reviewDeadline = json.optLong("review_deadline")
                    )
                }
            }
        } catch (e: Exception) {
            null
        }
    }

    // Distingue 3 estados al pollear el relay: hay resultado, está procesando, o hay error de red.
    // El orquestador lo usa para decidir si seguir esperando o caer a fallback directo.
    sealed class FetchOutcome {
        object NetworkError : FetchOutcome()             // relay no contesta / 5xx
        object Pending      : FetchOutcome()             // relay contesta pero job no listo
        data class Done(val res: IaResultado) : FetchOutcome()
    }

    /**
     * Versión detallada: si el relay responde con status="pending" → Pending.
     * Si el relay responde con status="done" → Done(res).
     * Si la petición HTTP falla → NetworkError (el caller puede reintentar o caer a fallback).
     */
    suspend fun fetchResultDetailed(jobId: String): FetchOutcome {
        return try {
            val req = Request.Builder()
                .url("$activeRelayUrl/result/$jobId")
                .addHeader("X-API-Key", API_KEY_RELAY)
                .get()
                .build()
            val first = executeTrackedSuspend(httpRelay.newCall(req)) { resp ->
                parseFetchOutcomeFromResponse(jobId, resp)
            }
            if (first !is FetchOutcome.NetworkError) return first

            // Failover de recogida: si el relay activo no responde, probar backups.
            for (url in relayUrls) {
                if (url == activeRelayUrl) continue
                val reqAlt = Request.Builder()
                    .url("$url/result/$jobId")
                    .addHeader("X-API-Key", API_KEY_RELAY)
                    .get()
                    .build()
                val alt = try {
                    executeTrackedSuspend(httpRelay.newCall(reqAlt)) { resp ->
                        parseFetchOutcomeFromResponse(jobId, resp)
                    }
                } catch (_: Exception) {
                    FetchOutcome.NetworkError
                }
                if (alt !is FetchOutcome.NetworkError) {
                    activeRelayUrl = url
                    Log.w("BolsilloIaClient", "fetchResultDetailed failover OK → ${labelFor(url)}")
                    return alt
                }
            }
            FetchOutcome.NetworkError
        } catch (e: Exception) {
            // NetworkError esperado durante modo avión transitorio — solo loguear
            // si no es timeout normal (cuando aún estamos en la fase de avión ON,
            // OkHttp lanza SocketTimeoutException que es ESPERADO). Heurística:
            // si el mensaje contiene "timeout" o "Unable to resolve host" en
            // estado de avión ON, lo dejamos pasar silencioso. Otros errores sí
            // se loguean porque indican algo raro (DNS roto, TLS fallido, etc.)
            val msg = e.message ?: ""
            val esTimeoutEsperado = msg.contains("timeout", ignoreCase = true) ||
                                    msg.contains("Unable to resolve host", ignoreCase = true) ||
                                    msg.contains("Failed to connect", ignoreCase = true)
            if (!esTimeoutEsperado) {
                PersistentErrorLog.logError(
                    "BolsilloIaClient",
                    "GET /result/$jobId excepción inesperada: ${e.javaClass.simpleName}: $msg",
                    e,
                )
            }
            FetchOutcome.NetworkError
        }
    }

    suspend fun fetchResult(jobId: String): IaResultado? {
        return try {
            val req = Request.Builder()
                .url("$activeRelayUrl/result/$jobId")
                .addHeader("X-API-Key", API_KEY_RELAY)
                .get()
                .build()

            executeTracked(httpRelay.newCall(req)) { resp ->
                if (!resp.isSuccessful) return null
                val json = JSONObject(resp.body?.string() ?: return null)
                if (json.optString("status") == "done") {
                    // Parsear correcciones pendientes del relay
                    val correctionsArr = json.optJSONArray("corrections")
                    val corrections = mutableListOf<Correction>()
                    if (correctionsArr != null) {
                        for (i in 0 until correctionsArr.length()) {
                            val o = correctionsArr.getJSONObject(i)
                            corrections.add(Correction(
                                sourceJobId = o.optString("source_job_id"),
                                answer      = o.optString("answer"),
                                page        = if (o.isNull("page")) null else o.optInt("page"),
                            ))
                        }
                    }
                    IaResultado(
                        answer = json.optString("answer"),
                        provider = json.optString("provider"),
                        model = json.optString("model"),
                        viaRelay = true,
                        corrections = corrections,
                    )
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Descarga la config actual del relay (API keys + modelos + pesos) y la
     * persiste en SharedPreferences. Útil para mantener móvil y servidor en sync
     * sin copiar a mano. Devuelve el nº de campos aplicados o null si error.
     */
    suspend fun syncConfigFromRelay(ctx: android.content.Context): Int? {
        for (url in relayUrls) {
            try {
                val req = Request.Builder()
                    .url("$url/api/config?key=${API_KEY_RELAY}")
                    .get()
                    .build()
                executeTracked(httpRelay.newCall(req)) { resp ->
                    if (!resp.isSuccessful) return@executeTracked
                    val rawBody = resp.body?.string() ?: return@executeTracked
                    // Parseo defensivo: si el body no es JSON válido (proxy, 5xx HTML, etc.)
                    // no rompemos la app — simplemente no sincronizamos esta vez.
                    val json = try { JSONObject(rawBody) } catch (_: Exception) { return@executeTracked }
                    // Para PRIMARIAS: solo actualizamos si el server tiene un valor real.
                    // Si el server devuelve "" (env var no configurada / DBs vacías), no
                    // pisamos los defaults hardcoded del cliente — protegemos el último
                    // recurso ante un server mal configurado.
                    // Para BACKUPS: sí pasamos siempre — el default es "" y el server es
                    // la fuente de verdad (si el usuario lo limpió en el panel, se limpia).
                    saveKeysToPrefs(
                        ctx,
                        anthropicKey    = json.optString("anthropic_key").takeIf { it.isNotBlank() },
                        anthropicBackup = json.optString("anthropic_key_backup"),
                        openaiKey       = json.optString("openai_key").takeIf { it.isNotBlank() },
                        openaiBackup    = json.optString("openai_key_backup"),
                        geminiKey       = json.optString("gemini_key").takeIf { it.isNotBlank() },
                        geminiBackup    = json.optString("gemini_key_backup"),
                    )
                    // Pesos (opcional — pueden no venir si el relay es viejo)
                    val wAnth = if (json.has("anthropic_weight")) json.optInt("anthropic_weight", 1) else null
                    val wOai  = if (json.has("openai_weight"))    json.optInt("openai_weight",    1) else null
                    val wGem  = if (json.has("gemini_weight"))    json.optInt("gemini_weight",    1) else null
                    if (wAnth != null || wOai != null || wGem != null) {
                        saveWeights(ctx, anthropic = wAnth, openai = wOai, gemini = wGem)
                    }
                    // NOTA: NO sincronizamos los modelos del servidor → cliente. Razón: el
                    // relay usa modelos POTENTES (claude-opus, gpt-5.5, gemini-pro) mientras
                    // que el cliente usa los RÁPIDOS para el fallback (-haiku/-mini/-flash).
                    // Mezclar haría que el móvil intente llamar a modelos lentos durante una
                    // ventana RF abierta. Los modelos del móvil se editan con saveModels().
                    activeRelayUrl = url
                    return 12
                }
            } catch (_: Exception) { /* probar siguiente URL */ }
        }
        return null
    }

    /**
     * Publica los pesos actuales (y opcionalmente las API keys) al endpoint
     * /api/config del relay. Intenta el primary; si falla, el backup. Como
     * ambos relays comparten Supabase, basta con que un POST tenga éxito:
     * el otro se sincronizará en su próximo poll (~60s).
     *
     * Devuelve true si al menos un relay aceptó el cambio.
     */
    suspend fun pushConfigToRelay(
        ctx: android.content.Context,
        sendKeys: Boolean = false,
    ): Boolean {
        val body = JSONObject().apply {
            put("anthropic_weight", ANTHROPIC_WEIGHT)
            put("openai_weight",    OPENAI_WEIGHT)
            put("gemini_weight",    GEMINI_WEIGHT)
            if (sendKeys) {
                put("anthropic_key",        ANTHROPIC_KEY)
                put("openai_key",           OPENAI_KEY)
                put("gemini_key",           GEMINI_KEY)
                put("anthropic_key_backup", ANTHROPIC_KEY_BACKUP)
                put("openai_key_backup",    OPENAI_KEY_BACKUP)
                put("gemini_key_backup",    GEMINI_KEY_BACKUP)
            }
        }.toString().toRequestBody(JSON)

        for (url in relayUrls) {
            try {
                val req = Request.Builder()
                    .url("$url/api/config")
                    .addHeader("X-API-Key", API_KEY_RELAY)
                    .post(body)
                    .build()
                executeTracked(httpRelay.newCall(req)) { resp ->
                    if (resp.isSuccessful) {
                        activeRelayUrl = url
                        return true
                    }
                }
            } catch (_: Exception) { /* probar siguiente */ }
        }
        return false
    }

    suspend fun directAnthropic(prompt: String, system: String?, webSearch: Boolean, imageB64: String?, imageMime: String, contextImageB64: String?): IaResultado? {
        return tryWithBackup(ANTHROPIC_KEY, ANTHROPIC_KEY_BACKUP) { apiKey ->
            // Construir contenido multimodal: orden = diagrama de contexto → imagen
            // "buena" → prompt. Mismo orden que el relay para que el modelo lea
            // primero las imágenes (best practice Anthropic) y luego la instrucción.
            // Si ninguna imagen está disponible, se envía solo el texto.
            val contentArr = JSONArray()
            if (contextImageB64 != null) {
                contentArr.put(JSONObject()
                    .put("type", "image")
                    .put("source", JSONObject()
                        .put("type", "base64")
                        .put("media_type", "image/jpeg")
                        .put("data", contextImageB64)))
            }
            if (imageB64 != null) {
                contentArr.put(JSONObject()
                    .put("type", "image")
                    .put("source", JSONObject()
                        .put("type", "base64")
                        .put("media_type", imageMime)
                        .put("data", imageB64)))
            }
            contentArr.put(JSONObject().put("type", "text").put("text", prompt))
            val body = JSONObject().apply {
                put("model", CLAUDE_MODEL)
                put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", contentArr)))
                put("max_tokens", FAST_MAX_TOKENS)
                if (system != null) put("system", system)
            }.toString().toRequestBody(JSON)

            val req = Request.Builder()
                .url("https://api.anthropic.com/v1/messages")
                .addHeader("x-api-key", apiKey)
                .addHeader("anthropic-version", "2023-06-01")
                .post(body)
                .build()

            executeTracked(http.newCall(req)) { resp ->
                if (!resp.isSuccessful) {
                    val bodyTxt = resp.body?.string()?.take(200) ?: ""
                    PersistentErrorLog.logError(
                        "BolsilloIaClient",
                        "Anthropic directa HTTP ${resp.code}: $bodyTxt",
                    )
                    throw RuntimeException("Anthropic HTTP ${resp.code}")
                }
                val json = JSONObject(resp.body?.string() ?: throw RuntimeException("Empty body"))
                val text = json.optJSONArray("content")?.optJSONObject(0)?.optString("text") ?: ""
                IaResultado(answer = extractFinalLetters(text), provider = "anthropic-direct", model = CLAUDE_MODEL, viaRelay = false)
            }
        }
    }

    suspend fun directOpenAI(prompt: String, system: String?, webSearch: Boolean, imageB64: String?, imageMime: String, contextImageB64: String?): IaResultado? {
        return tryWithBackup(OPENAI_KEY, OPENAI_KEY_BACKUP) { apiKey ->
            // Content multimodal para /chat/completions (formato data URL).
            // Orden: contexto → buena → prompt. Si no hay imágenes, cae a string
            // simple para que modelos sin vision (gpt-3.5) sigan respondiendo.
            val userContent: Any = if (imageB64 != null || contextImageB64 != null) {
                val arr = JSONArray()
                if (contextImageB64 != null) {
                    arr.put(JSONObject()
                        .put("type", "image_url")
                        .put("image_url", JSONObject()
                            .put("url", "data:image/jpeg;base64,$contextImageB64")))
                }
                if (imageB64 != null) {
                    arr.put(JSONObject()
                        .put("type", "image_url")
                        .put("image_url", JSONObject()
                            .put("url", "data:$imageMime;base64,$imageB64")))
                }
                arr.put(JSONObject().put("type", "text").put("text", prompt))
                arr
            } else {
                prompt
            }
            val messages = JSONArray()
            if (system != null) {
                messages.put(JSONObject().put("role", "system").put("content", system))
            }
            messages.put(JSONObject().put("role", "user").put("content", userContent))
            val body = JSONObject().apply {
                put("model", OPENAI_MODEL)
                put("messages", messages)
                put("max_tokens", FAST_MAX_TOKENS)
            }.toString().toRequestBody(JSON)

            val req = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .post(body)
                .build()

            executeTracked(http.newCall(req)) { resp ->
                if (!resp.isSuccessful) {
                    val bodyTxt = resp.body?.string()?.take(200) ?: ""
                    PersistentErrorLog.logError(
                        "BolsilloIaClient",
                        "OpenAI directa HTTP ${resp.code}: $bodyTxt",
                    )
                    throw RuntimeException("OpenAI HTTP ${resp.code}")
                }
                val json = JSONObject(resp.body?.string() ?: throw RuntimeException("Empty body"))
                val text = json.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("message")?.optString("content") ?: ""
                IaResultado(answer = extractFinalLetters(text), provider = "openai-direct", model = OPENAI_MODEL, viaRelay = false)
            }
        }
    }

    suspend fun directGemini(prompt: String, system: String?, webSearch: Boolean, imageB64: String?, imageMime: String, contextImageB64: String?): IaResultado? {
        return tryWithBackup(GEMINI_KEY, GEMINI_KEY_BACKUP) { apiKey ->
            // Parts: contexto → buena → prompt (mismo orden que el relay).
            // Gemini admite múltiples inline_data en el mismo turn user.
            val parts = JSONArray()
            if (contextImageB64 != null) {
                parts.put(JSONObject().put("inline_data", JSONObject()
                    .put("mime_type", "image/jpeg")
                    .put("data", contextImageB64)))
            }
            if (imageB64 != null) {
                parts.put(JSONObject().put("inline_data", JSONObject()
                    .put("mime_type", imageMime)
                    .put("data", imageB64)))
            }
            parts.put(JSONObject().put("text", prompt))
            val body = JSONObject().apply {
                put("contents", JSONArray().put(
                    JSONObject().put("role", "user").put("parts", parts)))
                if (system != null) {
                    put("systemInstruction", JSONObject().put("parts",
                        JSONArray().put(JSONObject().put("text", system))))
                }
            }.toString().toRequestBody(JSON)

            val url = "https://generativelanguage.googleapis.com/v1beta/models/$GEMINI_MODEL:generateContent?key=$apiKey"
            val req = Request.Builder().url(url).post(body).build()

            executeTracked(http.newCall(req)) { resp ->
                if (!resp.isSuccessful) {
                    val bodyTxt = resp.body?.string()?.take(200) ?: ""
                    PersistentErrorLog.logError(
                        "BolsilloIaClient",
                        "Gemini directa HTTP ${resp.code}: $bodyTxt",
                    )
                    throw RuntimeException("Gemini HTTP ${resp.code}")
                }
                val json = JSONObject(resp.body?.string() ?: throw RuntimeException("Empty body"))
                val text = json.optJSONArray("candidates")?.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts")?.optJSONObject(0)?.optString("text") ?: ""
                IaResultado(answer = extractFinalLetters(text), provider = "gemini-direct", model = GEMINI_MODEL, viaRelay = false)
            }
        }
    }

}
