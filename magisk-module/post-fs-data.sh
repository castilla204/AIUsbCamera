#!/system/bin/sh
# Runs in post-fs-data, before system_server starts.
# Seeds USB device permissions for com.example.myapplication so no
# permission dialog ever appears for UVC cameras.

PKG="com.example.myapplication"
LOG="/data/local/tmp/uvc_autogrant.log"

echo "[$(date)] uvc_autogrant post-fs-data start" >>"$LOG"

# Scan sysfs for any USB device whose interface advertises bInterfaceClass=0e (UVC video).
# We collect (vid,pid) pairs.
PAIRS=""
for dev in /sys/bus/usb/devices/*; do
    [ -d "$dev" ] || continue
    [ -f "$dev/idVendor" ] || continue
    [ -f "$dev/idProduct" ] || continue
    VID=$(cat "$dev/idVendor" 2>/dev/null)
    PID=$(cat "$dev/idProduct" 2>/dev/null)
    [ -z "$VID" ] && continue
    [ -z "$PID" ] && continue

    # Check any interface for class 0e (video) or 0xef/0x02/0x01 (IAD MISC)
    is_uvc=0
    for intf in "$dev"/*:*/; do
        [ -d "$intf" ] || continue
        [ -f "$intf/bInterfaceClass" ] || continue
        IFC=$(cat "$intf/bInterfaceClass" 2>/dev/null)
        if [ "$IFC" = "0e" ]; then
            is_uvc=1
            break
        fi
    done
    if [ "$is_uvc" = "0" ] && [ -f "$dev/bDeviceClass" ]; then
        DC=$(cat "$dev/bDeviceClass" 2>/dev/null)
        if [ "$DC" = "ef" ]; then
            is_uvc=1
        fi
    fi

    if [ "$is_uvc" = "1" ]; then
        # Convert hex strings to decimal for the XML.
        VID_DEC=$((0x$VID))
        PID_DEC=$((0x$PID))
        PAIRS="$PAIRS $VID_DEC:$PID_DEC"
        echo "[$(date)] found UVC dev vid=$VID pid=$PID" >>"$LOG"
    fi
done

if [ -z "$PAIRS" ]; then
    echo "[$(date)] no UVC device enumerated yet, nothing to seed" >>"$LOG"
    exit 0
fi

seed_user_dir() {
    USER_DIR="$1"
    [ -d "$USER_DIR" ] || return 0

    PERM_FILE="$USER_DIR/usb_device_manager.xml"
    [ -f "$PERM_FILE" ] || return 0

    # Build new entries.
    PREF_ENTRIES=""
    PERM_ENTRIES=""
    for pair in $PAIRS; do
        VID_DEC=${pair%:*}
        PID_DEC=${pair#*:}
        PREF_ENTRIES="$PREF_ENTRIES
        <usb-device vendor-id=\"$VID_DEC\" product-id=\"$PID_DEC\" />"
        PERM_ENTRIES="$PERM_ENTRIES
        <usb-device vendor-id=\"$VID_DEC\" product-id=\"$PID_DEC\" />"
    done

    # Strip any existing entries for our package, then re-insert fresh ones.
    TMP="$PERM_FILE.uvcseed.$$"
    awk -v pkg="$PKG" '
        BEGIN { skip=0 }
        /<preference[ \t]+package="/ { if (index($0,"\""pkg"\"")>0) { skip=1; next } }
        /<permission[ \t]+package="/ { if (index($0,"\""pkg"\"")>0) { skip=1; next } }
        /<\/preference>/ { if (skip) { skip=0; next } }
        /<\/permission>/ { if (skip) { skip=0; next } }
        { if (!skip) print }
    ' "$PERM_FILE" >"$TMP" 2>/dev/null

    if [ ! -s "$TMP" ]; then
        echo "[$(date)] awk produced empty file, aborting for $PERM_FILE" >>"$LOG"
        rm -f "$TMP"
        return 0
    fi

    # Insert before </settings>.
    OUT="$PERM_FILE.uvcseed2.$$"
    awk -v pkg="$PKG" -v prefs="$PREF_ENTRIES" -v perms="$PERM_ENTRIES" '
        /<\/settings>/ {
            print "    <preference package=\"" pkg "\">" prefs
            print "    </preference>"
            print "    <permission package=\"" pkg "\">" perms
            print "    </permission>"
            print
            next
        }
        { print }
    ' "$TMP" >"$OUT"
    rm -f "$TMP"

    if [ -s "$OUT" ]; then
        cp "$PERM_FILE" "$PERM_FILE.bak"
        cat "$OUT" >"$PERM_FILE"
        rm -f "$OUT"
        chown system:system "$PERM_FILE"
        chmod 600 "$PERM_FILE"
        restorecon "$PERM_FILE" 2>/dev/null
        echo "[$(date)] seeded $PERM_FILE" >>"$LOG"
    else
        rm -f "$OUT"
    fi

    # Android 10+ also keeps permissions in usb_permissions.xml. If present, mirror them.
    PERM_FILE2="$USER_DIR/usb_permissions.xml"
    if [ -f "$PERM_FILE2" ]; then
        TMP="$PERM_FILE2.uvcseed.$$"
        awk -v pkg="$PKG" '
            BEGIN { skip=0 }
            /<permission[ \t]+package="/ { if (index($0,"\""pkg"\"")>0) { skip=1; next } }
            /<\/permission>/ { if (skip) { skip=0; next } }
            { if (!skip) print }
        ' "$PERM_FILE2" >"$TMP" 2>/dev/null

        if [ -s "$TMP" ]; then
            OUT="$PERM_FILE2.uvcseed2.$$"
            awk -v pkg="$PKG" -v perms="$PERM_ENTRIES" '
                /<\/settings>/ {
                    print "    <permission package=\"" pkg "\">" perms
                    print "    </permission>"
                    print
                    next
                }
                { print }
            ' "$TMP" >"$OUT"
            rm -f "$TMP"

            if [ -s "$OUT" ]; then
                cp "$PERM_FILE2" "$PERM_FILE2.bak"
                cat "$OUT" >"$PERM_FILE2"
                rm -f "$OUT"
                chown system:system "$PERM_FILE2"
                chmod 600 "$PERM_FILE2"
                restorecon "$PERM_FILE2" 2>/dev/null
                echo "[$(date)] seeded $PERM_FILE2" >>"$LOG"
            else
                rm -f "$OUT"
            fi
        else
            rm -f "$TMP"
        fi
    fi
}

# Seed for every existing user (multi-user devices).
for USER_DIR in /data/system/users/*/; do
    seed_user_dir "$USER_DIR"
done

echo "[$(date)] uvc_autogrant done" >>"$LOG"
exit 0
