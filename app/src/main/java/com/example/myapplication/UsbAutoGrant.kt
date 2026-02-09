package com.example.myapplication

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.util.Log

/**
 * Auto-concesión de permisos USB sin diálogo y sin notificación.
 *
 * Estrategia en tres capas, todas requieren root y la shell persistente del
 * HeadlessUvcService:
 *
 *   1. Sembrado runtime de [/data/system/users/0/usb_device_manager.xml] y
 *      [/data/system/users/0/usb_permissions.xml] con el par VID/PID del
 *      dispositivo recién atachado. Esto persiste entre reinicios; el módulo
 *      Magisk uvc_autogrant ya hace lo mismo en post-fs-data, pero al ocurrir
 *      un hot-plug DESPUÉS del boot esa siembra puede no incluir el nuevo
 *      VID/PID — esta función la actualiza al vuelo.
 *
 *   2. Tras pedir el permiso vía libuvccamera, se lanza en background un bucle
 *      corto que vigila la aparición de la UsbPermissionActivity y, en cuanto
 *      la detecta, simula la confirmación (TAB-TAB-ENTER por keyevent y, como
 *      respaldo, un input tap en la mitad-bajo derecha que es donde el botón
 *      "Aceptar" / "OK" / "Permitir" suele renderizarse en One UI / AOSP).
 *      Si el sembrado XML ya había concedido el permiso, la actividad nunca
 *      aparece y el bucle termina sin hacer nada.
 *
 *   3. Suprimir las notificaciones residuales de SystemUI / android (canal
 *      USB) — esto se hace una sola vez en HeadlessUvcService.applyRootProtection.
 *
 * Todas las invocaciones se serializan sobre la shell root persistente para
 * evitar spawnear procesos `su` por cada hot-plug.
 */
object UsbAutoGrant {
    private const val TAG = "UsbAutoGrant"

    /**
     * VÍA PRINCIPAL (fiable y 100% headless): concede el permiso USB EN MEMORIA
     * usando la API de sistema `UsbManager.grantPermission(UsbDevice, String)`
     * — @SystemApi, @RequiresPermission(android.permission.MANAGE_USB).
     *
     * Escribe directamente el grant en `UsbUserPermissionManager.mDevicePermissionMap`
     * (lo MISMO que comprueban `hasPermission()` / `requestPermission()`), sin
     * diálogo, sin tocar ficheros, sin depender de la pantalla ni de un reinicio.
     * Por eso es mejor que sembrar XML: esos ficheros solo se releen al boot.
     *
     * Requiere que la app sea PRIVILEGIADA (instalada en /system/priv-app con la
     * whitelist privapp-permissions que concede MANAGE_USB → módulo Magisk
     * `uvc-privapp`). Si la app NO es privilegiada, `grantPermission` lanza
     * SecurityException → devolvemos false y el caller cae al método de respaldo.
     *
     * @return true si tras la llamada `hasPermission(device)` es true.
     */
    fun grantViaApi(usbManager: UsbManager, device: UsbDevice, packageName: String): Boolean {
        return try {
            if (usbManager.hasPermission(device)) return true
            val method = UsbManager::class.java.getMethod(
                "grantPermission", UsbDevice::class.java, String::class.java,
            )
            method.invoke(usbManager, device, packageName)
            val ok = usbManager.hasPermission(device)
            Log.i(TAG, "grantViaApi vid=${device.vendorId} pid=${device.productId} → hasPermission=$ok")
            ok
        } catch (e: Throwable) {
            // SecurityException (sin MANAGE_USB / app no privilegiada),
            // NoSuchMethodException (otra versión de Android) o bloqueo hidden-API
            // → respaldo en el caller.
            Log.w(TAG, "grantViaApi no disponible (¿módulo priv-app sin instalar?): ${e.message}")
            false
        }
    }

    /**
     * Llamar desde onAttach(device) ANTES de invocar usbMonitor.requestPermission().
     * Es seguro y rápido: la shell persistente sólo encola comandos, no bloquea.
     */
    fun grant(
        device: UsbDevice,
        packageName: String,
        runRoot: (cmd: String, tag: String) -> Unit,
    ) {
        val vid = device.vendorId
        val pid = device.productId
        Log.i(TAG, "Sembrando permiso USB para $packageName vid=$vid pid=$pid")

        seedXmlFiles(vid, pid, packageName, runRoot)
        scheduleDialogDismiss(runRoot)
    }

    /**
     * Reescribe los XMLs de permisos USB para incluir nuestra app + el VID/PID
     * pasado. Mantiene cualquier otra entrada existente (de otros paquetes o
     * de otros dispositivos previamente sembrados).
     *
     * Importante: VID/PID se inyectan en DECIMAL, como exige el parser
     * FastXmlSerializer del UsbProfileGroupSettingsManager. Si se mete hex
     * el archivo se descarta entero al siguiente boot.
     *
     * Se cuida la propiedad UNIX (chown system:system, chmod 600) y el
     * contexto SELinux (restorecon) — sin esto, system_server descarta el
     * archivo por considerarlo corrompido.
     */
    private fun seedXmlFiles(
        vid: Int,
        pid: Int,
        packageName: String,
        runRoot: (String, String) -> Unit,
    ) {
        // Script shell que añade la entrada para nuestro paquete sin tocar
        // otras. Si el archivo no existe lo crea con la estructura mínima.
        val script = buildString {
            append("PKG='").append(packageName).append("';")
            append("VID=").append(vid).append(";")
            append("PID=").append(pid).append(";")
            append("for USR in /data/system/users/*/; do ")
            // ----- usb_device_manager.xml -----
            append("F=\"\$USR/usb_device_manager.xml\";")
            append("[ -f \"\$F\" ] || echo '<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"yes\"?><settings></settings>' > \"\$F\";")
            // Si ya hay entrada exacta para este VID/PID y paquete, saltar.
            append("if ! grep -q \"vendor-id=\\\"\$VID\\\" product-id=\\\"\$PID\\\"\" \"\$F\" 2>/dev/null || ! grep -q \"package=\\\"\$PKG\\\"\" \"\$F\" 2>/dev/null; then ")
            // Backup y reescritura
            append("cp \"\$F\" \"\$F.bak\" 2>/dev/null;")
            // Reemplazar </settings> por nuestra entrada + cierre.
            append("awk -v pkg=\"\$PKG\" -v vid=\"\$VID\" -v pid=\"\$PID\" '")
            append("/<\\/settings>/ {")
            append("  print \"    <preference package=\\\"\" pkg \"\\\"><usb-device vendor-id=\\\"\" vid \"\\\" product-id=\\\"\" pid \"\\\" /></preference>\";")
            append("  print \"    <permission package=\\\"\" pkg \"\\\"><usb-device vendor-id=\\\"\" vid \"\\\" product-id=\\\"\" pid \"\\\" /></permission>\";")
            append("  print; next ")
            append("} { print }' \"\$F\" > \"\$F.new\" 2>/dev/null;")
            append("if [ -s \"\$F.new\" ]; then mv \"\$F.new\" \"\$F\"; chown system:system \"\$F\"; chmod 600 \"\$F\"; restorecon \"\$F\" 2>/dev/null; fi;")
            append("fi;")
            // ----- usb_permissions.xml (Android 10+) -----
            append("F=\"\$USR/usb_permissions.xml\";")
            append("if [ -f \"\$F\" ]; then ")
            append("if ! grep -q \"vendor-id=\\\"\$VID\\\" product-id=\\\"\$PID\\\"\" \"\$F\" 2>/dev/null || ! grep -q \"package=\\\"\$PKG\\\"\" \"\$F\" 2>/dev/null; then ")
            append("cp \"\$F\" \"\$F.bak\" 2>/dev/null;")
            append("awk -v pkg=\"\$PKG\" -v vid=\"\$VID\" -v pid=\"\$PID\" '")
            append("/<\\/settings>/ {")
            append("  print \"    <permission package=\\\"\" pkg \"\\\"><usb-device vendor-id=\\\"\" vid \"\\\" product-id=\\\"\" pid \"\\\" /></permission>\";")
            append("  print; next ")
            append("} { print }' \"\$F\" > \"\$F.new\" 2>/dev/null;")
            append("if [ -s \"\$F.new\" ]; then mv \"\$F.new\" \"\$F\"; chown system:system \"\$F\"; chmod 600 \"\$F\"; restorecon \"\$F\" 2>/dev/null; fi;")
            append("fi;")
            append("fi;")
            append("done;")
        }
        runRoot(script, "usb-seed-xml")
    }

    /**
     * Lanza en background un watcher que durante 3 segundos comprueba si la
     * UsbPermissionActivity está visible en la pila de actividades. Si la
     * detecta envía la confirmación.
     *
     * En la mayoría de devices la actividad se llama
     * com.android.systemui/.usb.UsbPermissionActivity; algunos OEMs la han
     * renombrado pero comparten el sufijo "UsbPermissionActivity" en el
     * dump, por eso filtramos por ese substring.
     *
     * Confirmación:
     *   - Primero KEYCODE_DPAD_CENTER (66 = Enter equivalente).
     *   - Luego KEYCODE_TAB + ENTER por si el foco no estaba en el botón.
     *   - Como último recurso un input tap aproximado al centro-bajo de la
     *     pantalla (donde caen los botones de confirmación en One UI/AOSP).
     */
    private fun scheduleDialogDismiss(runRoot: (String, String) -> Unit) {
        val script = buildString {
            append("(")
            // Loop de 15 intentos × 200 ms = 3 s máx
            append("for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do ")
            append("sleep 0.2;")
            // ¿Está la UsbPermissionActivity en pantalla?
            append("if dumpsys activity activities 2>/dev/null | grep -q UsbPermissionActivity; then ")
            // Confirmar con teclas — esto pulsa el botón "Aceptar" sin importar
            // dónde esté en la pantalla (siempre que tenga foco por defecto).
            append("input keyevent 61;") // TAB
            append("input keyevent 61;") // TAB
            append("input keyevent 66;") // ENTER
            append("input keyevent 23;") // DPAD_CENTER (segundo botón)
            // Tap aproximado en mitad-bajo derecha (botón positivo en One UI).
            append("W=\$(wm size 2>/dev/null | grep -o '[0-9]\\+x[0-9]\\+' | head -1 | cut -dx -f1);")
            append("H=\$(wm size 2>/dev/null | grep -o '[0-9]\\+x[0-9]\\+' | head -1 | cut -dx -f2);")
            append("[ -n \"\$W\" ] && [ -n \"\$H\" ] && input tap \$((W*3/4)) \$((H*3/5));")
            append("break;")
            append("fi;")
            append("done")
            append(") &") // background, no bloquea la shell persistente
        }
        runRoot(script, "usb-dialog-dismiss")
    }
}
