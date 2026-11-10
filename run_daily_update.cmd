@echo off
rem Daily 2026 forecast update: refresh polls and race data, run data checks, publish the snapshot.
rem Run by the Windows scheduled task "Election Forecast Daily Update" (7:00 AM through Nov 4, 2026).
rem Output is appended to forecasts\2026\daily_update_output.log; each run is also logged in update_log.csv.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ===== %date% %time% ===== >> "forecasts\2026\daily_update_output.log"
"ep-env\Scripts\python.exe" src\daily_update.py >> "forecasts\2026\daily_update_output.log" 2>&1
