package com.example.myapplication

import android.content.Context
import android.telecom.Call
import android.telecom.CallScreeningService
import android.util.Log

/**
 * Bloqueo COMPLETO de llamadas entrantes durante el modo activo.
 *
 * MOTIVACIÓN — Samsung S10e (Exynos 9820) + Android 12:
 *   Cuando entra una llamada, el subsistema audio reconfigura el USB OTG para
 *   el routing del altavoz/auricular → el bus USB queda en estado "zombie" tras
 *   la llamada (libusb -99, release_interface EINVAL). El bug se dispara con que
 *   la llamada llegue a estado RINGING durante ~milisegundos, incluso si la
 *   rechazamos inmediatamente con `service call phone 6`.
 *
 *   Para evitar el bug es necesario que la llamada NUNCA pase a RINGING.
 *
 * SOLUCIÓN — CallScreeningService (API 24+, fortalecido en API 29+):
 *   El framework Android consulta a este service ANTES de hacer sonar el
 *   teléfono. Si respondemos `setDisallowCall=true + setSkipCallLog=true +
 *   setSkipNotification=true`, la llamada se RECHAZA SILENCIOSAMENTE a nivel
 *   telecom-stack sin pasar por RINGING — no toca audio routing, no toca USB
 *   OTG, no llega ni siquiera al PhoneStateListener.
 *
 * REQUISITO — role CALL_SCREENING:
 *   Sólo se invoca este service si la app es el holder del role
 *   `android.app.role.CALL_SCREENING`. Como no podemos pedirlo por UI (Samsung
 *   One UI no expone el diálogo en muchas versiones), lo asignamos vía root:
 *
 *     cmd role add-role-holder android.app.role.CALL_SCREENING com.example.myapplication
 *
 *   Esto lo hace [HeadlessUvcService.setSystemEnabled] al activar el sistema, y
 *   lo libera al desactivar (preserva el holder original Samsung).
 *
 * FLAG — SharedPreferences (NO @Volatile):
 *   Este servicio corre en el PROCESO PRINCIPAL (default) mientras que
 *   HeadlessUvcService corre en `:uvc`. Una variable estática Kotlin se
 *   inicializa POR PROCESO — no se comparte. Por eso usamos SharedPreferences
 *   (file-backed, multi-proceso safe vía MODE_MULTI_PROCESS deprecated pero
 *   funcional, o re-cargando cada lectura). El flag se lee EN CADA llamada
 *   entrante; latencia despreciable (<1ms).
 */
class BlockAllCallScreeningService : CallScreeningService() {

    companion object {
        private const val TAG = "CallScreen"
        private const val PREFS_NAME = "UvcAppPrefs"
        private const val KEY_BLOCKING_ACTIVE = "call_blocking_active"

        /**
         * Activa/desactiva el bloqueo de llamadas. Se invoca desde
         * HeadlessUvcService al activar/desactivar el sistema. Persiste en
         * SharedPreferences para que el proceso del CallScreeningService
         * (distinto del proceso :uvc) lo vea correctamente.
         *
         * Usa `.commit()` (síncrono) en lugar de `.apply()` para garantizar
         * que el valor está en disco ANTES de continuar — sino habría una
         * ventana de race donde la primera llamada vería el valor viejo.
         */
        @JvmStatic
        fun setBlockingActive(ctx: Context, active: Boolean) {
            try {
                ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .edit()
                    .putBoolean(KEY_BLOCKING_ACTIVE, active)
                    .commit()
            } catch (e: Exception) {
                Log.e(TAG, "setBlockingActive falló: ${e.message}")
            }
        }

        @JvmStatic
        fun isBlockingActive(ctx: Context): Boolean = try {
            ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .getBoolean(KEY_BLOCKING_ACTIVE, false)
        } catch (_: Exception) {
            false
        }
    }

    override fun onScreenCall(callDetails: Call.Details) {
        val isIncoming = callDetails.callDirection == Call.Details.DIRECTION_INCOMING
        if (!isIncoming) {
            // Llamadas SALIENTES: nunca las bloqueamos (no aplica el bug del
            // audio routing en outgoing — el USB ya está en estado conocido).
            respondToCall(callDetails, CallResponse.Builder().build())
            return
        }

        // Re-lectura del flag desde disco en CADA llamada. Garantiza que vemos
        // el valor actualizado aunque venga de otro proceso (HeadlessUvcService
        // corre en `:uvc`, este service en proceso default). SharedPreferences
        // está backed por FS y es safe entre procesos.
        val blocking = isBlockingActive(applicationContext)
        if (!blocking) {
            // Sistema desactivado: pasar la llamada normal.
            respondToCall(callDetails, CallResponse.Builder().build())
            return
        }

        // Sistema activo + llamada entrante → bloqueo TOTAL.
        // Banderas:
        //   setDisallowCall(true)     → el framework no permite la llamada
        //   setRejectCall(true)        → además, envía señal de rechazo al caller
        //   setSkipCallLog(true)       → no aparece en historial de llamadas
        //   setSkipNotification(true)  → no aparece notificación "llamada perdida"
        // La combinación garantiza silencio TOTAL (no sonido, no vibración, no UI).
        try {
            val response = CallResponse.Builder()
                .setDisallowCall(true)
                .setRejectCall(true)
                .setSkipCallLog(true)
                .setSkipNotification(true)
                .build()
            respondToCall(callDetails, response)
            Log.i(TAG, "Llamada entrante BLOQUEADA silenciosamente (sistema activo)")
            try {
                PersistentErrorLog.logError(
                    TAG,
                    "Llamada entrante BLOQUEADA por CallScreeningService — sistema activo, pre-RINGING",
                )
            } catch (_: Throwable) {}
        } catch (e: Exception) {
            Log.e(TAG, "Error rechazando llamada: ${e.message}", e)
            // Fallback: si respondToCall lanza, al menos NO permitimos la llamada.
            try {
                respondToCall(
                    callDetails,
                    CallResponse.Builder().setDisallowCall(true).build(),
                )
            } catch (_: Exception) {}
        }
    }
}
