// IHeadlessUvcService.aidl
// Interfaz IPC entre el proceso principal (MainActivity) y el proceso :uvc
// que aloja HeadlessUvcService + libuvccamera. La separación permite que un
// SIGABRT de libusb 1.0.19 sólo mate el proceso :uvc, no la UI ni el resto
// de protecciones (modo avión, RF block, ciclo IA, vibración) — el sistema
// rearranca el servicio en silencio sin que el usuario lo note.
package com.example.myapplication;

import android.view.Surface;

interface IHeadlessUvcService {
    void setPreviewSurface(in @nullable Surface surface);
    /** Devuelve [width, height] o int[0] si todavía no hay tamaño negociado. */
    int[] getPreviewSize();
    void sendHapticFeedback();
    void setVibrationIntensity(int amplitude);
    void setCycleDurationSeconds(long seconds);
    void setBeepDuration(long ms);
    void setIntraBeepDelay(long ms);
    void setInterLetterDelay(long ms);
    void setPreBurstVibrationDelay(long ms);
    void setStartDelayMinutes(int minutes);
    long getInitialDelayEndAtMs();
    boolean toggleVibrationTestLoop();
    void setGeminiPrompt(String prompt);
    void setPocketMode(boolean enabled);
    void setToggleAirplaneModeEnabled(boolean enabled);
    void setAppInForeground(boolean isForeground);
    /** Habilita/deshabilita el ciclo. NO mata el servicio ni el preview. */
    void setSystemEnabled(boolean enabled);
    boolean isSystemEnabled();
    /** Lectura síncrona del estado actual de la cámara (nombre de la subclass de CameraState). */
    String getCameraStateName();
    /** Informa al servicio del modo de captura activo ("A", "C" o "TEST"). */
    void setCaptureMode(String modo);
    /**
     * Tipo de payload por ciclo:
     *   "BURST" → foto fija (ráfaga 30 frames → stacking → JPEG, comportamiento legacy)
     *   "VIDEO" → graba MP4 corto y lo manda al relay para OCR + análisis
     */
    void setCapturePayload(String payload);
    /** Devuelve el payload actual ("BURST" o "VIDEO"). */
    String getCapturePayload();
    /** Duración de video fija en servicio: 10s.
     *  Este método se mantiene por compatibilidad pero el valor recibido se ignora. */
    void setVideoParams(int durationSeconds);
    /** Bitrate objetivo del encoder de video en kbps (rango 800-8000).
     *  Sweet spot 4G + OCR: 2500-3500 kbps → ~3-4 MB por 10s en H.265. */
    void setVideoBitrateKbps(int kbps);
    /** Lee el bitrate actual configurado (kbps). */
    int getVideoBitrateKbps();
    /** DEBUG: fuerza avión OFF puntual (para recuperar ADB/WiFi). */
    void forceAirplaneOff();
    /** DEBUG: activa/desactiva modo sin avión — el ciclo funciona igual pero nunca toca las radios. */
    void setDebugSinAvion(boolean enabled);
    /** DEBUG: estado actual de SIN MODO AVIÓN (persistido en SharedPreferences). */
    boolean isDebugSinAvion();
    /**
     * Dispara UNA captura inmediata (Modo A: relay + análisis IA) sin esperar al
     * ciclo periódico. Pensado para probar el flujo de la API sin tener que
     * esperar 4 minutos. Devuelve true si se ha podido encolar la captura
     * (USB conectado y cámara disponible).
     */
    boolean triggerOneShotCapture();
    /**
     * Activa/desactiva el LOOP DE TEST DE VIDEO. Mientras está activo, graba MP4
     * continuamente (misma pipeline de producción: cámara hidden, encoder NV12,
     * mismas resolución/fps/bitrate) pero NO envía a IA, NO toca radios, NO
     * silencia audio. Cada MP4 se guarda en Movies/UVC_TEST/ para inspección.
     * Los frames se logean con tag [video-test]. Pensado para verificar el fix
     * del frame gris (stride padding Exynos) mirando logcat y los .mp4
     * resultantes. Devuelve true si quedó CORRIENDO tras esta llamada.
     */
    boolean toggleVideoTestLoop();
    /** Estado actual del loop de test (true = corriendo). */
    boolean isVideoTestLoopRunning();
}
