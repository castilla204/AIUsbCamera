#!/system/bin/sh
# post-fs-data.sh — corre en CADA arranque, ANTES de que arranque system_server.
#
# Marca com.example.myapplication como "app por defecto" para cualquier camara
# USB de clase video (UVC clase 14 / IAD clase 239). Al estar esta preferencia
# en usb_device_manager.xml cuando system_server la lee al boot, el sistema
# CONCEDE el permiso automaticamente y abre la app al conectar la camara, sin
# ningun dialogo. NO toca /system (por eso no depende del overlay de KernelSU).
#
# Sobrescribimos el fichero entero a proposito: este movil es de uso unico
# (camara), asi garantizamos un estado deterministico en cada arranque.

PKG=com.example.myapplication
F=/data/system/users/0/usb_device_manager.xml

cat > "$F" <<EOF
<?xml version='1.0' encoding='utf-8' standalone='yes' ?>
<settings>
<preference package="$PKG" user="0">
<usb-device class="14" />
</preference>
<preference package="$PKG" user="0">
<usb-device class="239" subclass="2" protocol="1" />
</preference>
</settings>
EOF

# Propiedad y contexto SELinux correctos, o system_server lo descarta por
# considerarlo corrupto.
chown system:system "$F" 2>/dev/null
chmod 600 "$F" 2>/dev/null
restorecon "$F" 2>/dev/null
