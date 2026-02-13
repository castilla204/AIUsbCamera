#!/system/bin/sh
# Watchdog inmortal para com.example.myapplication.
# - Cada 5 s comprueba si el proceso vive.
# - Si vive: re-fija oom_score_adj=-1000 (LMK / phantom-killer / OEM-killers no podrán matarlo).
# - Si no vive: lo arranca con am start-foreground-service + am start (MainActivity).
# Corre con uid=0 (root) lanzado desde service.sh, así sobrevive a cualquier intento de matarlo
# desde userland.

PKG="com.example.myapplication"
SVC="$PKG/.HeadlessUvcService"
ACT="$PKG/.MainActivity"
LOG="/data/local/tmp/uvc_watchdog.log"
INTERVAL=5

echo "[$(date)] watchdog start (pkg=$PKG, interval=${INTERVAL}s)" >>"$LOG"

# Espera a que el sistema esté listo (boot completo + package manager arriba).
while [ "$(getprop sys.boot_completed)" != "1" ]; do
    sleep 2
done
sleep 10

# Aplicar whitelist de Doze, standby bucket, appops y phantom-process killer una vez.
apply_global_protections() {
    dumpsys deviceidle whitelist +"$PKG" >/dev/null 2>&1
    cmd deviceidle whitelist +"$PKG" >/dev/null 2>&1
    am set-standby-bucket "$PKG" active >/dev/null 2>&1
    cmd appops set "$PKG" RUN_IN_BACKGROUND allow >/dev/null 2>&1
    cmd appops set "$PKG" RUN_ANY_IN_BACKGROUND allow >/dev/null 2>&1
    # Android 12+: desactiva el phantom process killer (mata cached procs agresivamente).
    device_config put activity_manager max_phantom_processes 2147483647 >/dev/null 2>&1
    settings put global settings_enable_monitor_phantom_procs false >/dev/null 2>&1
    echo "[$(date)] global protections applied" >>"$LOG"
}

apply_global_protections

LAST_PID=""
LAST_RESTART=0

while true; do
    PID=$(pidof "$PKG" 2>/dev/null | awk '{print $1}')

    if [ -n "$PID" ] && [ -d "/proc/$PID" ]; then
        # Proceso vivo: clavar oom_score_adj y oom_adj a inmortal cada ciclo
        echo -1000 > "/proc/$PID/oom_score_adj" 2>/dev/null
        echo -17   > "/proc/$PID/oom_adj"        2>/dev/null
        # Subir prioridad de CPU
        renice -n -20 -p "$PID" >/dev/null 2>&1

        if [ "$PID" != "$LAST_PID" ]; then
            echo "[$(date)] PID $PID protegido (oom_score_adj=-1000)" >>"$LOG"
            LAST_PID="$PID"
        fi
    else
        NOW=$(date +%s)
        # Anti-bucle: no relanzar más de una vez cada 10 s
        if [ $((NOW - LAST_RESTART)) -ge 10 ]; then
            echo "[$(date)] proceso muerto, resucitando..." >>"$LOG"
            # Foreground service primero
            am start-foreground-service -n "$SVC" >/dev/null 2>&1 || \
                am startservice -n "$SVC" >/dev/null 2>&1
            # Y MainActivity (asegura que la UI esté lista para reconectar la cámara)
            am start -n "$ACT" --activity-clear-top >/dev/null 2>&1
            LAST_RESTART=$NOW
            LAST_PID=""
        fi
    fi

    sleep "$INTERVAL"
done
