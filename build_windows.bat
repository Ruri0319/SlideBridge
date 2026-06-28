@echo off
setlocal

if "%~1"=="" (
  echo Usage: build_windows.bat ^<python.exe^>
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_windows_tauri.ps1" -PythonExe "%~1"
