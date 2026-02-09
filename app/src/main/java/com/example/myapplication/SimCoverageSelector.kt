package com.example.myapplication

import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import android.telephony.SubscriptionInfo
import android.telephony.SubscriptionManager
import android.telephony.TelephonyManager
import android.util.Log
import kotlinx.coroutines.delay

/**
 * Selección automática de la SIM de datos con MEJOR cobertura (Opción B).
 *
 * Contexto del problema: el dispositivo vive en MODO AVIÓN (las dos radios
 * apagadas) y sólo abre ventanas breves para enviar/recibir. En modo avión NO
 * hay señal que medir, así que la cobertura sólo se conoce DENTRO de una ventana
 * (radio encendida). La estrategia es:
 *
 *   1. PRE-SELECCIÓN (radio OFF, en avionOff): aplicamos la mejor SIM medida en
 *      la ventana ANTERIOR *antes* de encender la radio. Cuando la radio sube,
 *      los datos enganchan directamente en esa SIM → CERO reattach, ~0 ms extra.
 *
 *   2. MEDICIÓN + CONMUTACIÓN EN VENTANA (radio ON, sólo en el envío): leemos el
 *      RSRP de ambas SIM (gratis, la radio ya está encendida para waitForConnectivity)
 *      y actualizamos la caché para la próxima ventana. Si la cobertura cambió
 *      desde la última medición (te has movido) y la otra SIM es mejor por el
 *      margen de histéresis, conmutamos AHORA y re-esperamos conectividad
 *      (pagamos un reattach puntual, sólo cuando de verdad cambió la cobertura).
 *
 * Lectura de señal: vía API de Android (SubscriptionManager + TelephonyManager),
 * NO parseando texto de dumpsys (cuyo formato cambia entre versiones/OEM). Esto
 * es estable de API 29 a 35 y además nos da el nombre de operadora (Lowi/Movistar)
 * para los logs.
 *
 * Conmutación de la SIM de datos: requiere root y se delega en
 * [NetworkControlManager.setDataSubscription] (comandos shell verificados en Samsung).
 *
 * Este object es SINGLETON a propósito: el orquestador se crea nuevo por cada foto,
 * así que la caché de "mejor SIM" debe vivir aquí para persistir entre fotos.
 */
object SimCoverageSelector {
    private const val TAG = "SimCoverage"

    /** Margen de histéresis (dB): la SIM candidata debe superar a la activa por
     *  al menos este margen para que merezca la pena conmutar. Evita el "ping-pong"
     *  por microfluctuaciones de señal (multitrayecto, interferencias rápidas). */
    private const val HYSTERESIS_DB = 6

    /** dBm tratado como "sin enlace utilizable". */
    private const val DBM_NO_SIGNAL = -140

    /** Mejor subId medido en la última ventana con radio encendida. Se aplica como
     *  pre-selección en la siguiente ventana (con la radio aún apagada). */
    @Volatile
    private var mejorSubIdCache: Int = SubscriptionManager.INVALID_SUBSCRIPTION_ID

    /** Último resumen legible de la decisión (para logs / posible UI). */
    @Volatile
    var ultimoResumen: String = ""
        private set

    /** Lectura de la SIM efectivamente elegida en la última medición (operador + dBm).
     *  Se actualiza en cada `medirYConmutarSiProcede`. El cliente Android lo lee para
     *  inyectarlo en el payload /ask y que el panel del relay vea qué SIM se usó. */
    @Volatile
    var ultimaLectura: Lectura? = null
        private set

    data class Lectura(val subId: Int, val slot: Int, val operador: String, val dbm: Int)
    data class ConmutacionResultado(
        val conmuto: Boolean,
        val verificada: Boolean,
        val subAnterior: Int,
        val subObjetivo: Int,
        val subActualFinal: Int,
    )

    // ---------- API pública ----------

    /**
     * PRE-SELECCIÓN con la radio aún APAGADA (llamar en avionOff, antes de quitar
     * el avión). Aplica la mejor SIM cacheada de la ventana anterior SIN bounce de
     * datos: cuando la radio suba, los datos engancharán directamente en ella.
     * Devuelve true si efectivamente conmutó la preferencia.
     */
    suspend fun preseleccionar(ctx: Context?): Boolean {
        if (ctx == null) return false
        val best = mejorSubIdCache
        if (best == SubscriptionManager.INVALID_SUBSCRIPTION_ID) return false
        val actual = subDatosActual()
        if (best == actual) return false
        Log.i(TAG, "preseleccionar (radio OFF): sub$actual → sub$best")
        return NetworkControlManager.setDataSubscription(best, bounce = false)
    }

    /**
     * MEDICIÓN con la radio ENCENDIDA (llamar tras waitForConnectivity en el envío).
     * Lee la señal de ambas SIM, actualiza la caché para la próxima ventana y, si la
     * SIM de datos actual NO es la mejor (por el margen de histéresis), conmuta AHORA
     * con bounce de datos. Devuelve true si conmutó en caliente (el llamador debe
     * volver a esperar conectividad).
     */
    suspend fun medirYConmutarSiProcede(ctx: Context?): Boolean {
        return medirYConmutarSiProcedeVerificado(ctx).conmuto
    }

    /**
     * Igual que [medirYConmutarSiProcede], pero además valida que el default data
     * sub realmente haya cambiado tras el comando root. En One UI/Samsung esto
     * puede tardar unos instantes o ignorarse silenciosamente.
     */
    suspend fun medirYConmutarSiProcedeVerificado(ctx: Context?): ConmutacionResultado {
        if (ctx == null) {
            return ConmutacionResultado(
                conmuto = false,
                verificada = false,
                subAnterior = SubscriptionManager.INVALID_SUBSCRIPTION_ID,
                subObjetivo = SubscriptionManager.INVALID_SUBSCRIPTION_ID,
                subActualFinal = SubscriptionManager.INVALID_SUBSCRIPTION_ID,
            )
        }

        // Reintento corto: el SIM en standby (no-datos) a veces tarda 1-2 s en
        // reportar señal tras encender la radio. Reintentamos hasta 2 veces (≤1 s).
        var lecturas = leerSenales(ctx)
        var intentos = 0
        while (intentos < 2 && lecturas.size >= 2 && lecturas.any { it.dbm <= DBM_NO_SIGNAL }) {
            delay(500)
            lecturas = leerSenales(ctx)
            intentos++
        }

        if (lecturas.size < 2) {
            // Aunque no haya nada que elegir, registramos la única lectura disponible
            // para que el panel pueda mostrar la SIM activa y su dBm aun con 1 SIM.
            val actualSingle = subDatosActual()
            ultimaLectura = lecturas.firstOrNull { it.subId == actualSingle }
                ?: lecturas.firstOrNull()
            Log.i(TAG, "medir: <2 SIM con señal → nada que elegir (${lecturas.size})")
            return ConmutacionResultado(
                conmuto = false,
                verificada = true,
                subAnterior = actualSingle,
                subObjetivo = actualSingle,
                subActualFinal = actualSingle,
            )
        }

        val actual = subDatosActual()
        val mejor = decidirMejor(lecturas, actual)
        if (mejor == SubscriptionManager.INVALID_SUBSCRIPTION_ID || mejor == actual) {
            return ConmutacionResultado(
                conmuto = false,
                verificada = true,
                subAnterior = actual,
                subObjetivo = actual,
                subActualFinal = actual,
            )  // ya estamos en la mejor → 0 ms extra
        }
        Log.i(TAG, "conmutación EN VENTANA (radio ON, con bounce): sub$actual → sub$mejor")
        val comandoOk = NetworkControlManager.setDataSubscription(mejor, bounce = true)
        if (!comandoOk) {
            return ConmutacionResultado(
                conmuto = false,
                verificada = false,
                subAnterior = actual,
                subObjetivo = mejor,
                subActualFinal = subDatosActual(),
            )
        }

        var finalSub = subDatosActual()
        var ok = finalSub == mejor
        var intentosVerif = 0
        while (!ok && intentosVerif < 4) {
            delay(500)
            finalSub = subDatosActual()
            ok = finalSub == mejor
            intentosVerif++
        }
        if (!ok) {
            Log.w(TAG, "conmutación no verificada tras ${intentosVerif + 1} lecturas: objetivo=sub$mejor, final=sub$finalSub")
        } else {
            Log.i(TAG, "conmutación verificada: sub$actual → sub$finalSub")
        }
        return ConmutacionResultado(
            conmuto = true,
            verificada = ok,
            subAnterior = actual,
            subObjetivo = mejor,
            subActualFinal = finalSub,
        )
    }

    // ---------- Lógica de decisión ----------

    private fun decidirMejor(lecturas: List<Lectura>, actual: Int): Int {
        val mejorMedida = lecturas.maxByOrNull { it.dbm }!!
        val actualLect = lecturas.firstOrNull { it.subId == actual }
        val nueva = when {
            // No sabemos la señal de la SIM actual → confiamos en la mejor medida.
            actualLect == null -> mejorMedida.subId
            // Sólo cambiamos si la candidata supera a la actual por el margen.
            mejorMedida.subId != actual && mejorMedida.dbm >= actualLect.dbm + HYSTERESIS_DB -> mejorMedida.subId
            // Empate técnico → mantenemos la actual (no merece la pena conmutar).
            else -> actual
        }
        mejorSubIdCache = nueva
        ultimaLectura = lecturas.firstOrNull { it.subId == nueva }
        ultimoResumen = lecturas.joinToString(" · ") {
            "${it.operador}(sub${it.subId}/slot${it.slot})=${it.dbm}dBm" +
                (if (it.subId == nueva) " ◀MEJOR" else "")
        }
        Log.i(TAG, "decidir: $ultimoResumen | actual=sub$actual → elegida=sub$nueva")
        return nueva
    }

    // ---------- Acceso a la API de telefonía ----------

    /** subId de la SIM de datos por defecto actual (estático, no depende de la radio). */
    private fun subDatosActual(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
            SubscriptionManager.getDefaultDataSubscriptionId()
        else
            SubscriptionManager.INVALID_SUBSCRIPTION_ID

    @SuppressLint("MissingPermission") // READ_PHONE_STATE se concede vía root (pm grant) en el arranque del servicio
    private fun subsActivas(ctx: Context): List<SubscriptionInfo> {
        val sm = ctx.getSystemService(Context.TELEPHONY_SUBSCRIPTION_SERVICE) as? SubscriptionManager
            ?: return emptyList()
        return try {
            sm.activeSubscriptionInfoList ?: emptyList()
        } catch (e: SecurityException) {
            Log.w(TAG, "Sin READ_PHONE_STATE para activeSubscriptionInfoList: ${e.message}")
            emptyList()
        }
    }

    /** Lee el mejor dBm de cada SIM activa. Requiere la radio ENCENDIDA. */
    @SuppressLint("MissingPermission")
    private fun leerSenales(ctx: Context): List<Lectura> {
        // getCellSignalStrengths() y SignalStrength.getCellSignalStrengths() son API 29.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return emptyList()
        val tmBase = ctx.getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
            ?: return emptyList()

        val out = mutableListOf<Lectura>()
        for (info in subsActivas(ctx)) {
            val subId = info.subscriptionId
            val tm = try {
                tmBase.createForSubscriptionId(subId)
            } catch (e: Exception) {
                Log.w(TAG, "createForSubscriptionId($subId) falló: ${e.message}"); continue
            }
            val ss = try {
                tm.signalStrength  // API 28; null si la radio aún no reportó
            } catch (e: SecurityException) {
                Log.w(TAG, "signalStrength sub$subId SecurityException: ${e.message}"); null
            }
            val dbm = ss?.cellSignalStrengths
                ?.mapNotNull { cs ->
                    val d = cs.dbm
                    if (d != Int.MAX_VALUE && d in DBM_NO_SIGNAL..-1) d else null
                }
                ?.maxOrNull() ?: DBM_NO_SIGNAL
            val operador = info.carrierName?.toString()?.takeIf { it.isNotBlank() }
                ?: "slot${info.simSlotIndex}"
            out += Lectura(subId, info.simSlotIndex, operador, dbm)
        }
        return out
    }
}
