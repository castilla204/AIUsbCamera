# UVC Auto-Grant Magisk Module

Pre-seeds `/data/system/users/0/usb_device_manager.xml` (and `usb_permissions.xml` on Android 10+) so `com.example.myapplication` never sees the USB permission dialog when a UVC webcam is plugged in or replugged.

## Install

1. Zip the contents of this folder (NOT the folder itself):
   ```
   cd magisk-module
   zip -r ../uvc_autogrant.zip .
   ```
2. Copy `uvc_autogrant.zip` to the phone.
3. In Magisk app → Modules → Install from storage → pick the zip.
4. Reboot.

## What it does

On every boot, BEFORE `system_server` reads the USB permission file:

- Scans `/sys/bus/usb/devices/` for any device exposing USB Video Class (`bInterfaceClass = 0e`) or IAD MISC (`bDeviceClass = ef`).
- For each one found, writes:
  ```xml
  <preference package="com.example.myapplication">
      <usb-device vendor-id="..." product-id="..." />
  </preference>
  <permission package="com.example.myapplication">
      <usb-device vendor-id="..." product-id="..." />
  </permission>
  ```
- chowns to `system:system`, chmods 600, runs `restorecon`.

If your camera was unplugged at boot, the late `service.sh` will run 30 s after boot to update the file for the *next* boot. So the worst case is: plug camera, reboot once, after that the dialog never appears again.

## Uninstall

Magisk app → Modules → uvc_autogrant → Remove → Reboot.

The module makes a `.bak` of the original XML before the first edit, but a clean uninstall does **not** automatically restore — Android will simply rebuild the file as it sees fit.

## Verify

```
adb shell "cat /data/local/tmp/uvc_autogrant.log"
adb shell "su -c 'cat /data/system/users/0/usb_device_manager.xml'"
```

You should see `<preference package="com.example.myapplication">` and `<permission package="com.example.myapplication">` blocks containing your camera's VID/PID (in decimal).
