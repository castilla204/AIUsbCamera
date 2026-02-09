package com.example.myapplication

import android.Manifest
import android.app.ActivityManager
import android.content.*
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.*
import android.util.Log
import android.view.*
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.File
import com.google.android.material.slider.Slider

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var mService: IHeadlessUvcService? = null
    private var mIsBound = false
    private lateinit var photoAdapter: PhotoAdapter
    private lateinit var responseAdapter: ResponseAdapter
    private val geminiResponses = mutableListOf<GeminiResponse>()
    private val activityScope = CoroutineScope(Dispatchers.Main + Job())

    private var currentSurface: Surface? = null
    private var lastKnownCamWidth = 0
    private var lastKnownCamHeight = 0
    private var isPreviewFullscreen = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ).let {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) it + Manifest.permission.POST_NOTIFICATIONS else it
    }

    // Estado local para recuperarse de un crash del proceso :uvc.
    // Se actualiza cada vez que el usuario cambia el toggle, para que
    // onServiceDisconnected no tenga que leerlo del proceso muerto.
    private var localSystemEnabled = false
    private var serviceDisconnectedBycrash = false

    // Anti-crash-loop: si el servicio crashea repetidamente al activar (caso típico:
    // setSystemEnabled(true) → applySilence pesado → shell saturado → crash → restart
    // → onServiceConnected re-llama setSystemEnabled(true) → crash → loop visible
    // como botón ACTIVAR/DETENER parpadeando), guardamos el momento de la última
    // recuperación y deshabilitamos el auto-restart si el siguiente crash ocurre en
    // <5 s. Así el usuario ve "ACTIVAR SISTEMA" estable y puede investigar/reintentar
    // manualmente en vez de quedarse en un loop infinito.
    private var lastAutoRecoveryAtMs: Long = 0L

    private fun bindServiceIfNeeded() {
        if (mIsBound) return
        val intent = Intent(this, HeadlessUvcService::class.java)
        bindService(intent, mServiceConnection, BIND_AUTO_CREATE)
    }

    private fun unbindServiceIfNeeded() {
        if (!mIsBound) return
        try {
            mService?.setAppInForeground(false)
            mService?.setPreviewSurface(null)
        } catch (_: Exception) {}
        try {
            unbindService(mServiceConnection)
        } catch (_: Exception) {}
        mIsBound = false
        mService = null
    }

    private val mServiceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            mService = IHeadlessUvcService.Stub.asInterface(service)
            mIsBound = true
            val prefs = getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
            val prefsEnabled = prefs.getBoolean("system_enabled", false)
            localSystemEnabled = try { mService?.isSystemEnabled() ?: prefsEnabled } catch (_: Exception) { prefsEnabled }

            if (serviceDisconnectedBycrash) {
                // Reconexión tras crash: restaurar estado previo automáticamente.
                // debugSinAvion lo leemos del switch (nunca se resetea en la UI);
                // systemEnabled lo leemos de localSystemEnabled (actualizado en el toggle).
                val sinAvion = binding.switchDebugSinAvion.isChecked
                val now = System.currentTimeMillis()
                val tooSoonAfterLastRecovery = lastAutoRecoveryAtMs > 0L &&
                    (now - lastAutoRecoveryAtMs) < 5_000L
                Log.w("MainActivity", "Reconexión post-crash: era=$localSystemEnabled, sinAvion=$sinAvion, tooSoonAfterLastRecovery=$tooSoonAfterLastRecovery")
                serviceDisconnectedBycrash = false
                try { mService?.setDebugSinAvion(sinAvion) } catch (_: Exception) {}
                if (prefsEnabled && !tooSoonAfterLastRecovery) {
                    lastAutoRecoveryAtMs = now
                    try { mService?.setSystemEnabled(true) } catch (_: Exception) {}
                } else if (prefsEnabled && tooSoonAfterLastRecovery) {
                    // Crash-loop detectado. Sincronizamos local con el estado real (false)
                    // para que el botón quede ACTIVAR estable y el usuario reintente a mano.
                    Log.e("MainActivity", "Crash-loop detectado: 2º crash en <5s tras auto-recovery. Auto-restart deshabilitado, esperando al usuario.")
                    localSystemEnabled = false
                    try { prefs.edit().putBoolean("system_enabled", false).commit() } catch (_: Exception) {}
                    try { mService?.setSystemEnabled(false) } catch (_: Exception) {}
                    Toast.makeText(this@MainActivity,
                        "Sistema cayó dos veces seguidas — auto-recovery desactivado, pulsa ACTIVAR a mano para reintentar.",
                        Toast.LENGTH_LONG).show()
                }
            } else {
                Log.i("MainActivity", "Service connected.")
            }

            try {
                mService?.setAppInForeground(true)
            } catch (e: Exception) {
                e.printStackTrace()
            }

            // Solo existe Modo A: forzamos el servicio a "A" en cada reconexión por si
            // arrastraba "C"/"TEST" de versiones previas (la pref se limpia en loadSettingsToUI).
            try { mService?.setCaptureMode("A") } catch (e: Exception) {}

            // Restaurar visualmente el switch SIN MODO AVIÓN desde el estado real del servicio
            // (que a su vez lo cargó de SharedPreferences en loadSettings()). Evita que el
            // switch quede desincronizado tras reabrir la app o tras un crash del servicio.
            // Usamos this@MainActivity dentro de ServiceConnection porque 'this' resuelve al closure.
            val activity = this@MainActivity
            try {
                val sinAvionReal = mService?.isDebugSinAvion() ?: false
                if (binding.switchDebugSinAvion.isChecked != sinAvionReal) {
                    Log.i("MainActivity", "Sincronizando switch SIN AVIÓN: UI=${binding.switchDebugSinAvion.isChecked} → servicio=$sinAvionReal")
                    // Bloquear el listener mientras cambiamos el valor para no escribir en bucle
                    binding.switchDebugSinAvion.setOnCheckedChangeListener(null)
                    binding.switchDebugSinAvion.isChecked = sinAvionReal
                    binding.switchDebugSinAvion.setOnCheckedChangeListener { _, checked ->
                        try {
                            mService?.setDebugSinAvion(checked)
                            val msg = if (checked) "DEBUG ON: sin modo avión (WiFi activo)" else "DEBUG OFF: modo avión normal"
                            Toast.makeText(activity, msg, Toast.LENGTH_SHORT).show()
                        } catch (_: Exception) {}
                    }
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "No se pudo leer isDebugSinAvion: ${e.message}")
            }

            refreshPreviewBufferSize()
            updatePreviewState()

            updateToggleButtonUI()
            try { updateVideoTestButtonUI(mService?.isVideoTestLoopRunning() ?: false) } catch (_: Exception) {}
            loadSettingsToUI()
            val stateName = try { mService?.getCameraStateName() ?: "Idle" } catch (e: Exception) { "Error" }
            updateUIForState(stateName)

            countdownEndAtMs = try { mService?.getInitialDelayEndAtMs() ?: 0L } catch (e: Exception) { 0L }
            countdownHandler.removeCallbacks(countdownTicker)
            countdownHandler.post(countdownTicker)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            // Solo se llama en crash (no en unbind normal).
            // localSystemEnabled ya tiene el estado correcto (actualizado en cada toggle).
            Log.w("MainActivity", "Service DISCONNECTED (crash): sistema era=$localSystemEnabled")
            serviceDisconnectedBycrash = true
            mService = null
            mIsBound = false
            updateToggleButtonUI()

            binding.cameraPreview.visibility = View.GONE
            binding.cameraPreview.postDelayed({
                binding.cameraPreview.visibility = View.VISIBLE
            }, 300)
        }
    }

    private val geminiResponseReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == "GEMINI_RESPONSE") {
                val imgName = intent.getStringExtra("img") ?: "N/A"
                val text = intent.getStringExtra("letters") ?: "X"
                val rawText = intent.getStringExtra("raw_text") ?: "Sin respuesta cruda"

                geminiResponses.add(0, GeminiResponse(imgName, text.replace(",", ", "), rawText))
                responseAdapter.notifyItemInserted(0)
                binding.rvResponses.scrollToPosition(0)

                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                val file = File(dir, imgName)
                if (file.exists()) activityScope.launch(Dispatchers.IO) {
                    // BitmapFactory.decodeFile puede devolver null si el archivo
                    // está corrupto, vacío, o el path es inválido (raro pero
                    // pasa con archivos borrados a mitad). Sin este check,
                    // setImageBitmap(null) lanza NPE en el ImageView interno.
                    val b = try {
                        BitmapFactory.decodeFile(file.absolutePath)
                    } catch (e: Exception) {
                        Log.w("MainActivity", "decodeFile falló para ${file.name}: ${e.message}")
                        null
                    }
                    if (b != null) {
                        withContext(Dispatchers.Main) { binding.previewImageView.setImageBitmap(b) }
                    } else {
                        Log.w("MainActivity", "decodeFile devolvió null para ${file.name} — skip preview")
                    }
                }
            }
        }
    }

    private val cameraStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val state = intent?.getStringExtra("state") ?: "Idle"
            updateUIForState(state)
    
            // Mostrar overlay "CAPTURANDO..." durante la ráfaga
            if (state == "CapturingBurst") {
                binding.overlayCapturando.visibility = View.VISIBLE
                binding.overlayCapturando.text = "CAPTURANDO..."
            } else if (state == "ReadyForCapture") {
                binding.overlayCapturando.visibility = View.GONE
                // Actualizar dimensiones conocidas ANTES del toggle para que surfaceCreated
                // llame a setAspectRatio con los valores correctos en la nueva surface.
                refreshPreviewBufferSize()
                binding.cameraPreview.visibility = View.GONE
                binding.cameraPreview.postDelayed({
                    binding.cameraPreview.visibility = View.VISIBLE
                }, 150)
            }
        }
    }

    private val brazoDetectadoReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            binding.overlayBrazo.visibility = View.VISIBLE
            binding.overlayBrazo.removeCallbacks(hideBrazoOverlay)
            binding.overlayBrazo.postDelayed(hideBrazoOverlay, 5000L)
        }
    }
    private val hideBrazoOverlay = Runnable { binding.overlayBrazo.visibility = View.GONE }

    @Volatile private var countdownEndAtMs: Long = 0L
    private val countdownHandler = Handler(Looper.getMainLooper())
    private val countdownTicker = object : Runnable {
        override fun run() {
            renderCountdown()
            if (countdownEndAtMs > 0L) countdownHandler.postDelayed(this, 500L)
        }
    }

    private val countdownReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val endAt = intent?.getLongExtra("end_at_ms", 0L) ?: 0L
            countdownEndAtMs = endAt
            countdownHandler.removeCallbacks(countdownTicker)
            countdownHandler.post(countdownTicker)
        }
    }

    // Countdown del PRÓXIMO ciclo (próxima foto): pequeño chip en esquina
    // top-right, distinto del overlay grande del initial delay.
    @Volatile private var nextCycleEndAtMs: Long = 0L
    private val nextCycleHandler = Handler(Looper.getMainLooper())
    private val nextCycleTicker = object : Runnable {
        override fun run() {
            renderNextCycle()
            if (nextCycleEndAtMs > 0L) nextCycleHandler.postDelayed(this, 500L)
        }
    }
    private val nextCycleReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val endAt = intent?.getLongExtra("end_at_ms", 0L) ?: 0L
            nextCycleEndAtMs = endAt
            nextCycleHandler.removeCallbacks(nextCycleTicker)
            nextCycleHandler.post(nextCycleTicker)
        }
    }

    private val contextImageReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val hasContext = intent?.getBooleanExtra("has_context", false) ?: false
            binding.tvParte2Banner.visibility = if (hasContext) View.VISIBLE else View.GONE
        }
    }

    private fun renderCountdown() {
        val end = countdownEndAtMs
        if (end <= 0L) {
            binding.tvCountdownOverlay.visibility = View.GONE
            return
        }
        val remainingMs = end - System.currentTimeMillis()
        if (remainingMs <= 0L) {
            countdownEndAtMs = 0L
            binding.tvCountdownOverlay.visibility = View.GONE
            return
        }
        val totalSec = (remainingMs / 1000L).toInt()
        val mm = totalSec / 60
        val ss = totalSec % 60
        binding.tvCountdownOverlay.text = String.format("INICIO EN\n%02d:%02d", mm, ss)
        binding.tvCountdownOverlay.visibility = View.VISIBLE
    }

    private fun renderNextCycle() {
        val end = nextCycleEndAtMs
        if (end <= 0L) {
            binding.tvNextCycleCountdown.visibility = View.GONE
            return
        }
        // CRÍTICO: SystemClock.elapsedRealtime() en vez de System.currentTimeMillis().
        // El servicio broadcastea cycleEndAtMs usando elapsedRealtime (monótono).
        // Si usáramos currentTimeMillis() y el reloj cambia (NTP sync, cambio
        // manual de hora, timezone), el remainingMs saltaría — el usuario vería
        // "se sumó tiempo al temporizador" de repente. elapsedRealtime es inmune
        // a saltos de hora porque sólo avanza con el sistema corriendo.
        val remainingMs = end - SystemClock.elapsedRealtime()
        if (remainingMs <= 0L) {
            nextCycleEndAtMs = 0L
            binding.tvNextCycleCountdown.visibility = View.GONE
            return
        }
        val totalSec = (remainingMs / 1000L).toInt()
        val mm = totalSec / 60
        val ss = totalSec % 60
        binding.tvNextCycleCountdown.text = String.format("📷 %02d:%02d", mm, ss)
        binding.tvNextCycleCountdown.visibility = View.VISIBLE
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Log persistente de errores: tiene que inicializarse ANTES que nada para
        // capturar cualquier crash temprano. Idempotente — si ya está, no-op.
        try { PersistentErrorLog.init(applicationContext) } catch (_: Exception) {}
        // Cargar claves persistidas ANTES de cualquier llamada a IA
        try { BolsilloIaClient.loadKeysFromPrefs(this) } catch (e: Exception) {
            Log.w("MainActivity", "Error cargando claves: ${e.message}")
            PersistentErrorLog.logError("MainActivity", "loadKeysFromPrefs lanzó: ${e.message}", e)
        }
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        localSystemEnabled = getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
            .getBoolean("system_enabled", false)

        if (!allPermissionsGranted()) ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, 10)

        binding.cameraPreview.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                currentSurface = holder.surface
                // Forzar el buffer al tamaño de la cámara antes de conectar el surface.
                // Sin esto, cada recreación de surface (toggle GONE/VISIBLE) deja el buffer
                // al tamaño del View (~1080×607) y copyToSurface recorta la imagen 1280×720.
                if (lastKnownCamWidth > 0 && lastKnownCamHeight > 0) {
                    binding.cameraPreview.setAspectRatio(lastKnownCamWidth, lastKnownCamHeight)
                }
                updatePreviewState()
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                currentSurface = null
                try {
                    mService?.setPreviewSurface(null)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        })

        setupRecyclerViews()
        setupListeners()

        val filterGemini = IntentFilter("GEMINI_RESPONSE")
        val filterCamera = IntentFilter("CAMERA_STATE_CHANGED")
        val filterCountdown = IntentFilter("INITIAL_DELAY_COUNTDOWN")
        val filterNextCycle = IntentFilter("NEXT_CYCLE_COUNTDOWN")
        val filterContext = IntentFilter("CONTEXT_IMAGE_CHANGED")
        val filterBrazo = IntentFilter("BRAZO_DETECTADO")

        ContextCompat.registerReceiver(this, geminiResponseReceiver, filterGemini, ContextCompat.RECEIVER_NOT_EXPORTED)
        ContextCompat.registerReceiver(this, cameraStateReceiver, filterCamera, ContextCompat.RECEIVER_NOT_EXPORTED)
        ContextCompat.registerReceiver(this, countdownReceiver, filterCountdown, ContextCompat.RECEIVER_NOT_EXPORTED)
        ContextCompat.registerReceiver(this, nextCycleReceiver, filterNextCycle, ContextCompat.RECEIVER_NOT_EXPORTED)
        ContextCompat.registerReceiver(this, contextImageReceiver, filterContext, ContextCompat.RECEIVER_NOT_EXPORTED)
        ContextCompat.registerReceiver(this, brazoDetectadoReceiver, filterBrazo, ContextCompat.RECEIVER_NOT_EXPORTED)

        activityScope.launch(Dispatchers.IO) {
            loadSavedPhotos()
            withContext(Dispatchers.Main) { updateToggleButtonUI() }
        }
    }

    private fun getLongSafe(prefs: SharedPreferences, key: String, default: Long): Long {
        return try {
            prefs.getLong(key, default)
        } catch (e: ClassCastException) {
            prefs.getInt(key, default.toInt()).toLong()
        }
    }

    private fun loadSettingsToUI() {
        val prefs = getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)

        // Solo existe Modo A ahora: el "Modo C" y "Modo TEST" se eliminaron de la UI.
        // Forzamos "A" en prefs por si quedó un valor antiguo de versiones anteriores.
        prefs.edit().putString("capture_mode", "A").apply()

        val vibrationIntensity = prefs.getInt("vibration_intensity", 255)
        binding.sliderVibrationIntensity.value = vibrationIntensity.toFloat().coerceIn(binding.sliderVibrationIntensity.valueFrom, binding.sliderVibrationIntensity.valueTo)
        binding.tvVibrationIntensityValue.text = vibrationIntensity.toString()

        val vibrationDuration = getLongSafe(prefs, "vibration_duration", 200L)
        binding.sliderVibrationDuration.value = vibrationDuration.toFloat().coerceIn(binding.sliderVibrationDuration.valueFrom, binding.sliderVibrationDuration.valueTo)
        binding.tvVibrationDurationValue.text = vibrationDuration.toString()

        val intraBeepDelay = getLongSafe(prefs, "intra_beep_delay", 400L)
        binding.sliderIntraDelay.value = intraBeepDelay.toFloat().coerceIn(binding.sliderIntraDelay.valueFrom, binding.sliderIntraDelay.valueTo)
        binding.tvIntraDelayValue.text = intraBeepDelay.toString()

        val interLetterDelay = getLongSafe(prefs, "inter_letter_delay", 800L)
        binding.sliderInterLetterDelay.value = interLetterDelay.toFloat().coerceIn(binding.sliderInterLetterDelay.valueFrom, binding.sliderInterLetterDelay.valueTo)
        binding.tvInterLetterDelayValue.text = interLetterDelay.toString()

        val preBurstVibrationDelayMs = getLongSafe(prefs, "pre_burst_vibration_delay", 0L)
        val preBurstSec = (preBurstVibrationDelayMs / 1000L).toFloat()
        binding.sliderPreBurstVibrationDelay.value = preBurstSec.coerceIn(binding.sliderPreBurstVibrationDelay.valueFrom, binding.sliderPreBurstVibrationDelay.valueTo)
        binding.tvPreBurstVibrationDelayValue.text = preBurstSec.toInt().toString()

        // cycle_duration ya no es editable desde la UI — está fijo a 300 s
        // (CYCLE_SECONDS_BASE en HeadlessUvcService). El slider se eliminó del layout XML.

        val startDelayMin = prefs.getInt("start_delay_minutes", 0)
        binding.sliderStartDelayMinutes.value = startDelayMin.toFloat().coerceIn(binding.sliderStartDelayMinutes.valueFrom, binding.sliderStartDelayMinutes.valueTo)
        binding.tvStartDelayMinutesValue.text = startDelayMin.toString()

        // ── API keys: precargar los EditText con los valores actuales ───────
        try {
            binding.etRelayKey.setText(BolsilloIaClient.API_KEY_RELAY)
            binding.etAnthropicKey.setText(BolsilloIaClient.ANTHROPIC_KEY)
            binding.etAnthropicKeyBackup.setText(BolsilloIaClient.ANTHROPIC_KEY_BACKUP)
            binding.etOpenAiKey.setText(BolsilloIaClient.OPENAI_KEY)
            binding.etOpenAiKeyBackup.setText(BolsilloIaClient.OPENAI_KEY_BACKUP)
            binding.etGeminiKey.setText(BolsilloIaClient.GEMINI_KEY)
            binding.etGeminiKeyBackup.setText(BolsilloIaClient.GEMINI_KEY_BACKUP)
            // Modelos: solo lectura (hardcoded, no editables)
            binding.etClaudeModel.setText(BolsilloIaClient.CLAUDE_MODEL)
            binding.etClaudeModel.isEnabled = false
            binding.etOpenAiModel.setText(BolsilloIaClient.OPENAI_MODEL)
            binding.etOpenAiModel.isEnabled = false
            binding.etGeminiModel.setText(BolsilloIaClient.GEMINI_MODEL)
            binding.etGeminiModel.isEnabled = false
        } catch (e: Exception) {
            Log.w("MainActivity", "No se pudieron precargar las API keys: ${e.message}")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun setupRecyclerViews() {
        photoAdapter = PhotoAdapter(); binding.rvPhotos.layoutManager = LinearLayoutManager(this); binding.rvPhotos.adapter = photoAdapter
        responseAdapter = ResponseAdapter(); binding.rvResponses.layoutManager = LinearLayoutManager(this); binding.rvResponses.adapter = responseAdapter
    }

    private suspend fun loadSavedPhotos() = withContext(Dispatchers.IO) {
        try {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val files = dir?.listFiles { f -> f.extension == "jpg" && (f.name.contains("BURST_") || f.name.contains("STACKED_")) }
                ?.sortedByDescending { it.lastModified() }?.take(10) ?: emptyList()
            withContext(Dispatchers.Main) { photoAdapter.setFiles(files) }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading saved photos", e)
        }
    }

    private fun setupListeners() {
        // Se ha eliminado el Modo Bolsillo por petición del usuario
        // El botón btnPocketMode y el overlay pocketModeOverlay no harán nada (o se pueden ocultar en XML)
        try {
            binding.btnPocketMode.visibility = View.GONE
            binding.pocketModeOverlay.visibility = View.GONE
        } catch (e: Exception) {}

        binding.btnToggleService.setOnClickListener {
            if (!isServiceRunning(HeadlessUvcService::class.java)) {
                startUvcService()
            } else if (mIsBound) {
                try {
                    val now = mService?.isSystemEnabled()
                        ?: getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
                            .getBoolean("system_enabled", false)
                    val next = !now
                    mService?.setSystemEnabled(next)
                    localSystemEnabled = next
                    updateToggleButtonUI()
                    // Ya no reenviamos la Surface aquí; evita reconexiones duplicadas.
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            } else {
                bindServiceIfNeeded()
                Toast.makeText(this, "Conectando al servicio…", Toast.LENGTH_SHORT).show()
            }
        }

        binding.switchDebugSinAvion.setOnCheckedChangeListener { _, checked ->
            try {
                mService?.setDebugSinAvion(checked)
                val msg = if (checked) "DEBUG ON: sin modo avión (WiFi activo)" else "DEBUG OFF: modo avión normal"
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {}
        }

        binding.btnVibrationTest.setOnClickListener {
            try {
                val isRunning = mService?.toggleVibrationTestLoop() ?: false
                if (isRunning) {
                    binding.btnVibrationTest.text = "DETENER TEST VIBRACIÓN"
                    binding.btnVibrationTest.backgroundTintList = ColorStateList.valueOf(0xFFC62828.toInt())
                } else {
                    binding.btnVibrationTest.text = "PROBAR SECUENCIA VIBRACIÓN"
                    binding.btnVibrationTest.backgroundTintList = ColorStateList.valueOf(0xFF1976D2.toInt())
                }
            } catch (e: Exception) {}
        }

        // Botón TEST: dispara una captura inmediata y la manda a la API (Modo A)
        // sin esperar el ciclo de 4 minutos. Útil para validar el flujo completo.
        binding.btnTestCaptureNow.setOnClickListener {
            try {
                val ok = mService?.triggerOneShotCapture() ?: false
                val msg = if (ok) "📸 Captura de prueba lanzada"
                          else    "⚠ No se pudo lanzar (¿USB conectado?)"
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        // BOTÓN TEST CONTINUO DE VIDEO: graba en bucle con la misma pipeline de
        // producción pero sin enviar a IA / sin tocar radios. Cada MP4 se guarda
        // en Movies/UVC_TEST/ para inspección visual. Verifica que el fix del
        // frame gris (stride padding Exynos) está funcionando. Filtra logcat
        // por [video-test] para seguir el progreso.
        binding.btnVideoTestLoop.setOnClickListener {
            try {
                val running = mService?.toggleVideoTestLoop() ?: false
                updateVideoTestButtonUI(running)
                val msg = if (running) "🎬 Test continuo INICIADO — Movies/UVC_TEST/"
                          else          "⏹ Test continuo DETENIDO (o no se pudo iniciar — mira logcat)"
                Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        binding.sliderVibrationIntensity.addOnChangeListener { slider, value, fromUser ->
            binding.tvVibrationIntensityValue.text = value.toInt().toString()
            if (fromUser) {
                // Persistir también desde UI para robustez si el servicio no está enlazado
                // justo en ese instante (evita volver a 255 al reabrir Activity).
                try {
                    getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
                        .edit()
                        .putInt("vibration_intensity", value.toInt())
                        .commit()
                } catch (_: Exception) {}
            }
            if (fromUser && mIsBound) {
                try { mService?.setVibrationIntensity(value.toInt()) } catch (e: Exception) {}
            }
        }

        binding.sliderVibrationIntensity.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) {}
            override fun onStopTrackingTouch(slider: Slider) {
                if (mIsBound) try { mService?.sendHapticFeedback() } catch (e: Exception) {}
            }
        })

        binding.sliderVibrationDuration.addOnChangeListener { _, value, fromUser ->
            binding.tvVibrationDurationValue.text = value.toInt().toString()
            if (fromUser && mIsBound) try { mService?.setBeepDuration(value.toLong()) } catch (e: Exception) {}
        }

        binding.sliderIntraDelay.addOnChangeListener { _, value, fromUser ->
            binding.tvIntraDelayValue.text = value.toInt().toString()
            if (fromUser && mIsBound) try { mService?.setIntraBeepDelay(value.toLong()) } catch (e: Exception) {}
        }

        binding.sliderInterLetterDelay.addOnChangeListener { _, value, fromUser ->
            binding.tvInterLetterDelayValue.text = value.toInt().toString()
            if (fromUser && mIsBound) try { mService?.setInterLetterDelay(value.toLong()) } catch (e: Exception) {}
        }

        binding.sliderPreBurstVibrationDelay.addOnChangeListener { _, value, fromUser ->
            binding.tvPreBurstVibrationDelayValue.text = value.toInt().toString()
            if (fromUser && mIsBound) try { mService?.setPreBurstVibrationDelay(value.toLong() * 1000L) } catch (e: Exception) {}
        }

        // sliderCycleSeconds eliminado del layout — el cycleSec está fijo a 300s
        // (coordinado server-side con stacking + Topaz + OCRs + analyzers + meta).

        binding.sliderStartDelayMinutes.addOnChangeListener { _, value, fromUser ->
            binding.tvStartDelayMinutesValue.text = value.toInt().toString()
            if (fromUser && mIsBound) try { mService?.setStartDelayMinutes(value.toInt()) } catch (e: Exception) {}
        }

        // ── Parámetros de VIDEO ─────────────────────────────────────────────
        // El switch BURST/VIDEO fue retirado: el sistema ahora SIEMPRE graba
        // video (con best frame JPEG fallback automático). El código de BURST
        // sigue presente en HeadlessUvcService por si se reactiva vía AIDL.
        // Forzamos capture_payload="VIDEO" para que cualquier instalación
        // previa con "BURST" en prefs migre automáticamente.
        val prefsUvc = getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
        if (prefsUvc.getString("capture_payload", "VIDEO") != "VIDEO") {
            prefsUvc.edit().putString("capture_payload", "VIDEO").apply()
            if (mIsBound) try { mService?.setCapturePayload("VIDEO") } catch (_: Exception) {}
        }

        binding.sliderVideoDuration.value = prefsUvc.getInt("video_duration_s", 4).toFloat()
        binding.tvVideoDurationValue.text = "${binding.sliderVideoDuration.value.toInt()} s"

        // FPS ya no es configurable: el encoder usa siempre actualCameraFps (el
        // máximo que negocia la cámara). El slider se eliminó.
        binding.sliderVideoDuration.addOnChangeListener { _, value, fromUser ->
            binding.tvVideoDurationValue.text = "${value.toInt()} s"
            if (fromUser) {
                val durationS = value.toInt()
                prefsUvc.edit().putInt("video_duration_s", durationS).apply()
                if (mIsBound) try { mService?.setVideoParams(durationS) } catch (_: Exception) {}
            }
        }

        // ── Ver errores persistentes ──────────────────────────────────────────
        // El log sobrevive al modo avión (cuando ADB/WiFi cae). Muestra los
        // últimos 50 errores en un diálogo scrollable. Long-press = limpiar log.
        binding.btnViewErrors.setOnClickListener { showErrorLogDialog() }
        binding.btnViewErrors.setOnLongClickListener {
            android.app.AlertDialog.Builder(this)
                .setTitle("¿Vaciar log de errores?")
                .setMessage("Se borrarán los ${PersistentErrorLog.size()} errores del buffer y del disco.")
                .setPositiveButton("Vaciar") { _, _ ->
                    PersistentErrorLog.clear()
                    updateErrorButtonCount()
                    Toast.makeText(this, "Log de errores vaciado", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancelar", null)
                .show()
            true
        }
        // Ticker para refrescar el contador "(N)" cada 3s. NO bloquea: solo es lectura
        // de un atómico en RAM. Sobrevive a pausas (Handler de mainLooper).
        startErrorCountTicker()

        // ── Sincronizar keys desde el relay (tira las del panel a este móvil) ──
        binding.btnSyncKeys.setOnClickListener {
            binding.tvSaveKeysStatus.text = "⏳ Descargando del relay..."
            binding.tvSaveKeysStatus.setTextColor(0xFF888888.toInt())
            activityScope.launch(Dispatchers.IO) {
                val n = try { BolsilloIaClient().syncConfigFromRelay(this@MainActivity) }
                        catch (e: Exception) { Log.e("MainActivity", "syncConfig", e); null }
                withContext(Dispatchers.Main) {
                    if (n != null) {
                        loadSettingsToUI()  // refresca los EditText con los nuevos valores
                        binding.tvSaveKeysStatus.text = "✓ Sincronizado desde el relay"
                        binding.tvSaveKeysStatus.setTextColor(0xFF3DDC84.toInt())
                        Toast.makeText(this@MainActivity, "Claves descargadas del relay", Toast.LENGTH_SHORT).show()
                    } else {
                        binding.tvSaveKeysStatus.text = "❌ Relay no responde"
                        binding.tvSaveKeysStatus.setTextColor(0xFFDC3545.toInt())
                    }
                }
            }
        }

        // ── Guardar API keys (con respaldo) ───────────────────────────────
        binding.btnSaveKeys.setOnClickListener {
            try {
                BolsilloIaClient.saveKeysToPrefs(
                    this,
                    relayKey        = binding.etRelayKey.text?.toString(),
                    anthropicKey    = binding.etAnthropicKey.text?.toString(),
                    anthropicBackup = binding.etAnthropicKeyBackup.text?.toString(),
                    openaiKey       = binding.etOpenAiKey.text?.toString(),
                    openaiBackup    = binding.etOpenAiKeyBackup.text?.toString(),
                    geminiKey       = binding.etGeminiKey.text?.toString(),
                    geminiBackup    = binding.etGeminiKeyBackup.text?.toString(),
                )
                val withBackup = listOfNotNull(
                    if (BolsilloIaClient.ANTHROPIC_KEY_BACKUP.isNotBlank()) "Claude" else null,
                    if (BolsilloIaClient.OPENAI_KEY_BACKUP.isNotBlank())    "GPT"    else null,
                    if (BolsilloIaClient.GEMINI_KEY_BACKUP.isNotBlank())    "Gemini" else null,
                )
                val msg = "✓ Guardado · Respaldos: " +
                          (if (withBackup.isEmpty()) "ninguno" else withBackup.joinToString(", "))
                binding.tvSaveKeysStatus.text = msg
                binding.tvSaveKeysStatus.setTextColor(0xFF3DDC84.toInt())
                Toast.makeText(this, "Claves guardadas", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                binding.tvSaveKeysStatus.text = "❌ Error: ${e.message}"
                binding.tvSaveKeysStatus.setTextColor(0xFFDC3545.toInt())
                Log.e("MainActivity", "Error guardando claves", e)
            }
        }
    }

    private fun refreshPreviewBufferSize() {
        try {
            val sz = mService?.getPreviewSize()
            if (sz == null || sz.size < 2) return
            val w = sz[0].takeIf { it > 0 } ?: return
            val h = sz[1].takeIf { it > 0 } ?: return
            if (w == lastKnownCamWidth && h == lastKnownCamHeight) return
            lastKnownCamWidth = w
            lastKnownCamHeight = h
            Log.i("MainActivity", "refreshPreviewBufferSize -> ${w}x${h}")
            // Ajustar el contenedor al ratio real de la cámara para que SurfaceFlinger
            // escale el buffer uniformemente (sin recorte ni distorsión).
            val params = binding.previewArea.layoutParams as ConstraintLayout.LayoutParams
            params.dimensionRatio = "H,$w:$h"
            binding.previewArea.layoutParams = params
            binding.cameraPreview.setAspectRatio(w, h)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updatePreviewState() {
        try {
            if (mIsBound) {
                mService?.setPreviewSurface(currentSurface)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }


    private fun setSettingsEnabled(isEnabled: Boolean) {
        // Nota: btnTestCaptureNow queda intencionalmente FUERA del bloqueo. El servicio
        // ya rechaza el oneshot si el ciclo periódico está en mitad de una captura
        // (devuelve false y mostramos Toast). Así el usuario puede dispararlo incluso
        // con el sistema activado, para verificar el flujo cuando le apetezca.

        // Bloquear/Desbloquear Sliders
        binding.sliderVibrationIntensity.isEnabled = isEnabled
        binding.sliderVibrationDuration.isEnabled = isEnabled
        binding.sliderIntraDelay.isEnabled = isEnabled
        binding.sliderInterLetterDelay.isEnabled = isEnabled
        binding.sliderPreBurstVibrationDelay.isEnabled = isEnabled
        // sliderCycleSeconds eliminado — cycleSec hardcoded a 300s (CYCLE_SECONDS_BASE).
        binding.sliderStartDelayMinutes.isEnabled = isEnabled

        // Bloquear/Desbloquear Botones
        binding.btnVibrationTest.isEnabled = isEnabled
    }

    private fun updateToggleButtonUI() {
        val enabled = try {
            if (mIsBound) {
                mService?.isSystemEnabled() ?: false
            } else {
                getSharedPreferences("UvcAppPrefs", Context.MODE_PRIVATE)
                    .getBoolean("system_enabled", false)
            }
        } catch (e: Exception) {
            false
        }
        binding.btnToggleService.text = if (enabled) "DETENER SISTEMA" else "ACTIVAR SISTEMA"
        binding.btnToggleService.backgroundTintList = ColorStateList.valueOf(
            if (enabled) 0xFFC62828.toInt() else 0xFF2E7D32.toInt()
        )
        
        // Cuando el sistema está ENCENDIDO (enabled == true), queremos DESACTIVAR los controles (isEnabled = false)
        setSettingsEnabled(!enabled)
    }

    /** Refresca etiqueta + color del botón TEST CONTINUO según esté corriendo o no. */
    private fun updateVideoTestButtonUI(running: Boolean) {
        binding.btnVideoTestLoop.text = if (running) {
            "⏹ DETENER TEST CONTINUO DE VIDEO"
        } else {
            "🎬 TEST CONTINUO DE VIDEO (logcat [video-test])"
        }
        binding.btnVideoTestLoop.backgroundTintList = ColorStateList.valueOf(
            if (running) 0xFFC62828.toInt() else 0xFF9C27B0.toInt()
        )
    }

    private fun updateUIForState(state: String) {
        binding.statusTextView.text = "Estado: $state"
        binding.statusOverlay.text = if (state == "ReadyForCapture") "UVC READY" else "UVC: $state"
    }

    private fun startUvcService() {
        val intent = Intent(this, HeadlessUvcService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent) else startService(intent)
        bindServiceIfNeeded()
    }

    private fun stopUvcService() {
        if (mIsBound) {
            try { mService?.setPreviewSurface(null) } catch (e: Exception) {}
            unbindService(mServiceConnection)
            mIsBound = false
        }
        stopService(Intent(this, HeadlessUvcService::class.java))
        updateToggleButtonUI()
    }

    private fun isServiceRunning(sc: Class<*>): Boolean {
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        for (s in am.getRunningServices(Int.MAX_VALUE)) { if (sc.name == s.service.className) return true }
        return false
    }

    override fun onStart() {
        super.onStart()
        bindServiceIfNeeded()
        if (mIsBound) {
            updatePreviewState()
            try { mService?.setAppInForeground(true) } catch (e: Exception) {}
        }
    }

    override fun onStop() {
        super.onStop()
        unbindServiceIfNeeded()
    }

    override fun onDestroy() {
        super.onDestroy()
        unbindServiceIfNeeded()
        unregisterReceiver(geminiResponseReceiver)
        unregisterReceiver(cameraStateReceiver)
        unregisterReceiver(countdownReceiver)
        unregisterReceiver(nextCycleReceiver)
        unregisterReceiver(contextImageReceiver)
        unregisterReceiver(brazoDetectadoReceiver)
        countdownHandler.removeCallbacks(countdownTicker)
        nextCycleHandler.removeCallbacks(nextCycleTicker)
        errorCountHandler.removeCallbacks(errorCountTicker)
        activityScope.cancel()
    }

    // ── Log persistente de errores ──────────────────────────────────────────────
    // Ticker que refresca el contador del botón "VER ERRORES (N)" cada 3s. Es solo
    // lectura de un atómico, no bloquea. El Handler está atado al Looper principal
    // y se cancela en onDestroy junto con activityScope.
    private val errorCountHandler = Handler(Looper.getMainLooper())
    private val errorCountTicker = object : Runnable {
        override fun run() {
            updateErrorButtonCount()
            errorCountHandler.postDelayed(this, 3000L)
        }
    }

    private fun startErrorCountTicker() {
        errorCountHandler.removeCallbacks(errorCountTicker)
        errorCountHandler.post(errorCountTicker)
    }

    private fun updateErrorButtonCount() {
        try {
            val n = PersistentErrorLog.size()
            binding.btnViewErrors.text = "🔴 VER ERRORES ($n)"
            // Cambiar tinte si hay errores acumulados para que llame la atención.
            val tint = if (n > 0) 0xFFDC3545.toInt() else 0xFF6C757D.toInt()
            binding.btnViewErrors.backgroundTintList = ColorStateList.valueOf(tint)
        } catch (_: Exception) {
            // Si el binding no está listo o algo raro, ignoramos — el ticker reintentará.
        }
    }

    /** Diálogo scrollable con los últimos 50 errores. Formato:
     *      [hh:mm:ss] [TAG] mensaje
     *        stack (si lo había)
     */
    private fun showErrorLogDialog() {
        val entries = PersistentErrorLog.getRecent(50)
        val body = if (entries.isEmpty()) {
            "No hay errores registrados.\n\nEl log captura fallos del orquestador, llamadas HTTP, " +
            "comandos root y crashes — sobrevive a modo avión y reinicios. Cuando algo falle " +
            "durante el ciclo con WiFi apagado, aparecerá aquí."
        } else {
            val sb = StringBuilder()
            val fmt = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
            entries.forEachIndexed { i, e ->
                val time = fmt.format(java.util.Date(e.t))
                sb.append("[$time] [${e.tag}]\n")
                sb.append(e.msg).append('\n')
                e.stack?.takeIf { it.isNotBlank() }?.let {
                    sb.append("  ").append(it.replace("\n", "\n  ")).append('\n')
                }
                if (i < entries.size - 1) sb.append("\n────────\n\n")
            }
            sb.toString()
        }
        // TextView dentro de ScrollView para que sea seleccionable + scroll
        val tv = TextView(this).apply {
            text = body
            textSize = 11f
            setTextIsSelectable(true)
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(32, 24, 32, 24)
        }
        val scroll = android.widget.ScrollView(this).apply { addView(tv) }
        android.app.AlertDialog.Builder(this)
            .setTitle("Errores (${entries.size}/${PersistentErrorLog.size()})")
            .setView(scroll)
            .setPositiveButton("Cerrar", null)
            .setNeutralButton("Vaciar") { _, _ ->
                PersistentErrorLog.clear()
                updateErrorButtonCount()
                Toast.makeText(this, "Log vaciado", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    data class GeminiResponse(val fileName: String, val letterSequence: String, val rawText: String)

    inner class ResponseAdapter : RecyclerView.Adapter<ResponseAdapter.ResponseViewHolder>() {
        override fun onCreateViewHolder(p: ViewGroup, vt: Int) = ResponseViewHolder(LayoutInflater.from(p.context).inflate(R.layout.item_response, p, false))

        override fun onBindViewHolder(h: ResponseViewHolder, p: Int) {
            val r = geminiResponses[p]
            h.tvFileName.text = r.fileName
            h.tvLetter.text = r.letterSequence
            h.tvRawText.text = "Respuesta IA:\n${r.rawText}"
        }

        override fun getItemCount() = geminiResponses.size

        inner class ResponseViewHolder(v: View) : RecyclerView.ViewHolder(v) {
            val tvFileName: TextView = v.findViewById(R.id.tvResponseFileName)
            val tvLetter: TextView = v.findViewById(R.id.tvResponseLetter)
            val tvRawText: TextView = v.findViewById(R.id.tvRawResponse)
        }
    }

    inner class PhotoAdapter : RecyclerView.Adapter<PhotoAdapter.PhotoViewHolder>() {
        private var photoFiles = listOf<File>()
        fun setFiles(newFiles: List<File>) { photoFiles = newFiles; notifyDataSetChanged() }
        override fun onCreateViewHolder(p: ViewGroup, vt: Int) = PhotoViewHolder(LayoutInflater.from(p.context).inflate(R.layout.item_photo, p, false))
        override fun onBindViewHolder(h: PhotoViewHolder, p: Int) {
            val f = photoFiles[p]; h.tvName.text = f.name
            h.imageView.tag = f.absolutePath
            activityScope.launch(Dispatchers.IO) {
                val b = BitmapFactory.decodeFile(f.absolutePath, BitmapFactory.Options().apply { inSampleSize = 4 })
                withContext(Dispatchers.Main) {
                    if (h.imageView.tag == f.absolutePath) h.imageView.setImageBitmap(b)
                }
            }
        }
        override fun getItemCount() = photoFiles.size
        inner class PhotoViewHolder(v: View) : RecyclerView.ViewHolder(v) { val imageView: ImageView = v.findViewById(R.id.ivCapturedPhoto); val tvName: TextView = v.findViewById(R.id.tvPhotoName) }
    }
}