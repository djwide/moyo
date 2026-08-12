# Install moyo Desktop launcher + icon on Windows.
# From WSL:
#   /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe `
#     -NoProfile -ExecutionPolicy Bypass -File scripts/create-moyo-desktop-shortcut.ps1

$ErrorActionPreference = "Stop"

$Distro = if ($env:WSL_DISTRO_NAME) { $env:WSL_DISTRO_NAME } else { "Ubuntu" }
$RepoLinux = "/home/david/moyo"
$WslRepo = "\\wsl$\$Distro\home\david\moyo"

$AppDir = Join-Path $env:LOCALAPPDATA "moyo"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null

# Copy launchers + icon onto the Windows filesystem so the shortcut never
# depends on \\wsl$\... being available for IconLocation.
$IconSrc = Join-Path $WslRepo "moyo\gui\assets\MoyoDesktopLogo.ico"
$IconDst = Join-Path $AppDir "MoyoDesktopLogo.ico"
$BatSrc  = Join-Path $WslRepo "scripts\Launch-Moyo-GUI.bat"
$VbsSrc  = Join-Path $WslRepo "scripts\Launch-Moyo-GUI.vbs"
$BatDst  = Join-Path $AppDir "Launch-Moyo-GUI.bat"
$VbsDst  = Join-Path $AppDir "Launch-Moyo-GUI.vbs"

Copy-Item -Force $IconSrc $IconDst
Copy-Item -Force $BatSrc  $BatDst
Copy-Item -Force $VbsSrc  $VbsDst

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "moyo.lnk"

$Wsh = New-Object -ComObject WScript.Shell
$Shortcut = $Wsh.CreateShortcut($ShortcutPath)
# VBS = no console flash; BAT is also on the Desktop as a fallback.
$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"$VbsDst`""
$Shortcut.WorkingDirectory = $AppDir
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Launch the moyo desktop GUI (via WSL / WSLg)"
$Shortcut.IconLocation = "$IconDst,0"
$Shortcut.Save()

# Also drop a visible .bat on the Desktop for debugging.
Copy-Item -Force $BatDst (Join-Path $Desktop "Launch moyo GUI.bat")

Write-Host "Created shortcut : $ShortcutPath"
Write-Host "Icon             : $IconDst"
Write-Host "Silent launcher  : $VbsDst"
Write-Host "Debug bat        : $(Join-Path $Desktop 'Launch moyo GUI.bat')"
Write-Host "Double-click 'moyo' on your Windows Desktop to open the GUI."
