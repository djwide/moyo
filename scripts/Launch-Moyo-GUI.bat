@echo off
REM Launch the moyo PyQt GUI inside WSL (Ubuntu) via WSLg.
REM Double-click this from the Windows Desktop, or use the moyo.lnk shortcut.
setlocal
set DISTRO=Ubuntu
set REPO=/home/david/moyo
set LOG=C:\Users\david\AppData\Local\moyo\launch.log

if not exist "%LOCALAPPDATA%\moyo" mkdir "%LOCALAPPDATA%\moyo"

echo [%date% %time%] Starting moyo GUI via WSL...>> "%LOG%"
wsl.exe -d %DISTRO% --cd %REPO% -- bash scripts/launch-moyo-gui.sh
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
  echo.
  echo moyo GUI failed to start ^(exit %EXITCODE%^).
  echo Check the log:
  echo   %LOG%
  echo   \\wsl$\%DISTRO%\tmp\moyo-gui-launch.log
  echo.
  pause
)
exit /b %EXITCODE%
