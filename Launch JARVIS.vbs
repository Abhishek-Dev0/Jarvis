' Launch JARVIS.vbs — double-click to start JARVIS in the background, voice
' mode, no console window. Output/errors go to jarvis\data\logs\jarvis.log
' since there's nothing visible to print to. The actual work happens in
' run_jarvis.bat — WScript.Shell.Run has a documented quoting hazard when a
' "cmd /c ..." string embeds more than one quoted segment (here: the python
' path AND the log path), so this just launches one plain quoted .bat path
' and lets the .bat do its own (unambiguous) quoting.
'
' To stop it: say "shutdown" (or "quit"/"exit") near the mic. JARVIS will ask
' for your passphrase by voice — that's the same SecurityGate every risky
' action goes through, see jarvis\security.py. It runs until that succeeds,
' or the process is killed some other way.
'
' Last-resort stop (no verification, use only if the voice path is stuck):
' End Task on "python.exe" in Task Manager. This is a hard kill, not a
' graceful shutdown — nothing is asked, nothing is confirmed.

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\run_jarvis.bat"

Set shell = CreateObject("WScript.Shell")
shell.Run Chr(34) & batPath & Chr(34), 0, False
