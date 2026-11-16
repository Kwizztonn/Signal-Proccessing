@echo off
rem Programi EXE olusturmadan, kurulu sanal ortamla dogrudan calistirir.
rem (Once KUR_VE_EXE_OLUSTUR.bat en az bir kez calistirilmis olmali.)
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Once KUR_VE_EXE_OLUSTUR.bat dosyasini calistirin.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0src\app.py" %*
