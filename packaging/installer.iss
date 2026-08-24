; Inno Setup script for JARVIS.
; Compile with: iscc packaging\installer.iss  (run after packaging\build.ps1's
; PyInstaller step has produced dist\Jarvis\)
;
; Inno Setup itself isn't a pip package -- install it once, free, from
; https://jrsoftware.org/isdl.php

#define MyAppName "Jarvis"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "Abhishek-Dev0"
#define MyAppExeName "Jarvis.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=Jarvis-Setup
SetupIconFile=..\assets\jarvis_cat.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; The whole PyInstaller --onedir output, produced by build.ps1's first step.
Source: "..\dist\Jarvis\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
