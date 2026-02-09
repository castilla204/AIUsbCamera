package com.example.myapplication

import android.hardware.usb.UsbDevice
import de.robv.android.xposed.IXposedHookLoadPackage
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XposedBridge
import de.robv.android.xposed.XposedHelpers
import de.robv.android.xposed.callbacks.XC_LoadPackage

/**
 * Módulo Xposed/LSPosed: auto-concede el permiso USB (cámara UVC) a esta app
 * dentro de system_server, sin diálogo.
 *
 * En Android 12, com.android.server.usb.UsbUserPermissionManager.requestPermission()
 * llama primero a hasPermission(UsbDevice, String packageName, int pid, int uid) y,
 * si devuelve true, concede de inmediato y NO muestra UsbPermissionActivity.
 *
 * Enganchamos hasPermission: cuando el solicitante es nuestro paquete, concedemos
 * el grant real en el mapa en memoria (grantDevicePermission, necesario también
 * para openDevice) y forzamos el resultado a true. Headless, sin pantalla.
 *
 * Ámbito en LSPosed: "Android System" (proceso `android` = system_server).
 */
class UsbPermissionHook : IXposedHookLoadPackage {

    override fun handleLoadPackage(lpparam: XC_LoadPackage.LoadPackageParam) {
        // Solo nos interesa el system_server (paquete "android").
        if (lpparam.packageName != "android") return
        try {
            XposedHelpers.findAndHookMethod(
                "com.android.server.usb.UsbUserPermissionManager",
                lpparam.classLoader,
                "hasPermission",
                UsbDevice::class.java,
                String::class.java,
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                object : XC_MethodHook() {
                    override fun beforeHookedMethod(p: MethodHookParam) {
                        if (p.args[1] == TARGET_PKG) {
                            val device = p.args[0] as UsbDevice
                            val uid = p.args[3] as Int
                            // Concede en el mapa real (para openDevice) + corta el diálogo.
                            runCatching {
                                XposedHelpers.callMethod(
                                    p.thisObject, "grantDevicePermission", device, uid
                                )
                            }
                            p.result = true
                        }
                    }
                }
            )
            XposedBridge.log("UsbPermissionHook: enganchado UsbUserPermissionManager.hasPermission para $TARGET_PKG")
        } catch (t: Throwable) {
            XposedBridge.log("UsbPermissionHook: fallo al enganchar: $t")
        }
    }

    private companion object {
        const val TARGET_PKG = "com.example.myapplication"
    }
}
