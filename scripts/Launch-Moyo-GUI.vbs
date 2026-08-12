' Silent launcher: starts the moyo GUI in WSL without a console window.
' On failure, opens the launch log.
Option Explicit
Dim sh, fso, distro, repo, logPath, rc
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

distro = "Ubuntu"
repo = "/home/david/moyo"
logPath = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%\moyo")
If Not fso.FolderExists(logPath) Then fso.CreateFolder(logPath)
logPath = logPath & "\launch.log"

rc = sh.Run( _
  "wsl.exe -d " & distro & " --cd " & repo & _
  " -- bash scripts/launch-moyo-gui.sh", _
  0, True)

If rc <> 0 Then
  sh.Popup "moyo GUI failed to start (exit " & rc & ")." & vbCrLf & _
           "Opening launch log.", 8, "moyo", 16
  sh.Run "notepad.exe """ & logPath & """", 1, False
  sh.Run "wsl.exe -d " & distro & " -- cat /tmp/moyo-gui-launch.log", 1, False
End If
