; Inno Setup script for JARVIS.
; Compile with: iscc packaging\installer.iss  (run after packaging\build.ps1's
; PyInstaller step has produced dist\Jarvis\)
;
; Inno Setup itself isn't a pip package -- install it once, free, from
; https://jrsoftware.org/isdl.php

#define MyAppName "Jarvis"
#define MyAppVersion "0.2.4"
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

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // Real crash found on a live machine: [Files] above only copies/
    // overwrites matching filenames, it never deletes files an *older*
    // build left behind that the new build doesn't have. PyInstaller's
    // --onedir bundle set changes between builds (different Python
    // version, different dependency set) -- upgrading in place left an
    // old build's python314.dll sitting next to a new build's
    // python311.dll in the same _internal folder, and Python's
    // C-extension ABI check correctly refused to load a .pyd built for
    // the other version, crashing at multiprocessing bootstrap before any
    // app code ran. Deleting the whole bundle before every install (not
    // just overwriting matching names) guarantees a clean, single-version
    // _internal every time. Nothing user-relevant lives under _internal
    // (see jarvis/paths.py) so this is always safe to wipe.
    if DirExists(ExpandConstant('{app}\_internal')) then
      DelTree(ExpandConstant('{app}\_internal'), True, True, True);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // User data (memory, security/admin config, prefs, logs, etc) lives
    // under %LOCALAPPDATA%\Jarvis, not {app} -- see jarvis/paths.py.
    // Inno Setup's default uninstall only removes what it installed under
    // {app}, so without this, uninstalling left every bit of user data
    // behind silently. Requested explicitly: uninstall should mean
    // completely gone, not "gone except a folder nobody's told about."
    if DirExists(ExpandConstant('{localappdata}\Jarvis')) then
      DelTree(ExpandConstant('{localappdata}\Jarvis'), True, True, True);
  end;
end;
