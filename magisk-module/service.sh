#!/system/bin/sh
# Late-boot fallback: if the camera was plugged in AFTER post-fs-data
# (very common on phones where USB host enumerates only after boot),
# re-run the seeding script. system_server has already loaded the
# in-memory permission map so this run only updates the on-disk file
# for the NEXT boot — but it also makes the file complete so a single
# reboot resolves any first-time setup.

sleep 30
sh /data/adb/modules/uvc_autogrant/post-fs-data.sh
