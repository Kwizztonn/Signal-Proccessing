@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title UNV2TDMS - Kurulum ve EXE olusturma

echo ================================================================
echo   UNV2TDMS  -  Kurulum + EXE olusturma (tek tik)
echo ================================================================
echo.

rem ---------------------------------------------------------------
rem 1) Python 3.9+ 64-bit bul
rem ---------------------------------------------------------------
set "PY="
call :try_py "py -3"
if not defined PY call :try_py "python"
if not defined PY call :try_path "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY call :try_path "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY call :try_path "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY call :try_path "%ProgramFiles%\Python312\python.exe"
if not defined PY goto :install_python
goto :have_python

:install_python
echo [1/5] Python bulunamadi. Python 3.12 indirilip kuruluyor...
where winget >nul 2>nul
if errorlevel 1 goto :py_download
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :try_path "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto :have_python

:py_download
echo     winget kullanilamadi, python.org'dan indiriliyor...
set "PYINST=%TEMP%\python-3.12.8-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile '%PYINST%'"
if not exist "%PYINST%" goto :py_fail
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_tcltk=1 Include_pip=1
call :try_path "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto :have_python

:py_fail
echo.
echo HATA: Python otomatik kurulamadi.
echo Lutfen https://www.python.org/downloads/ adresinden Python 3.12 64-bit kurun
echo (kurulumda "Add python.exe to PATH" kutusunu isaretleyin) ve bu dosyayi tekrar calistirin.
pause
exit /b 1

:have_python
echo [1/5] Python bulundu: %PY%
%PY% --version

rem ---------------------------------------------------------------
rem 2) Sanal ortam
rem ---------------------------------------------------------------
echo.
echo [2/5] Sanal ortam hazirlaniyor (.venv)...
if exist ".venv\Scripts\python.exe" goto :venv_ok
%PY% -m venv .venv
if errorlevel 1 goto :venv_fail
:venv_ok
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" goto :venv_fail

rem ---------------------------------------------------------------
rem 3) Kutuphaneler
rem ---------------------------------------------------------------
echo.
echo [3/5] Eksik kutuphaneler indiriliyor (numpy, npTDMS, matplotlib, pyinstaller)...
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_fail
"%VPY%" -c "import tkinter, numpy, nptdms, matplotlib; print('    Kutuphaneler tamam.')"
if errorlevel 1 goto :tk_fail

rem ---------------------------------------------------------------
rem 4) Hizli kendi kendini test
rem ---------------------------------------------------------------
echo.
echo [4/5] Donusturucu test ediliyor...
"%VPY%" "%~dp0src\selftest.py"
if errorlevel 1 goto :test_fail

rem ---------------------------------------------------------------
rem 5) EXE
rem ---------------------------------------------------------------
echo.
echo [5/5] EXE olusturuluyor (birkac dakika surebilir)...
"%VPY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name UNV2TDMS ^
  --paths "%~dp0src" --distpath "%~dp0build_out\dist" --workpath "%~dp0build_out\work" ^
  --specpath "%~dp0build_out" ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 ^
  --exclude-module IPython --exclude-module pandas --exclude-module scipy ^
  "%~dp0src\app.py"
if errorlevel 1 goto :build_fail
copy /y "%~dp0build_out\dist\UNV2TDMS.exe" "%~dp0UNV2TDMS.exe" >nul

echo.
echo ================================================================
echo   TAMAM!  UNV2TDMS.exe bu klasorde olusturuldu:
echo   %~dp0UNV2TDMS.exe
echo ================================================================
echo.
start "" explorer /select,"%~dp0UNV2TDMS.exe"
pause
exit /b 0

rem ===============================================================
:try_py
if defined PY exit /b 0
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0

:try_path
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set PY="%~1"
exit /b 0

:venv_fail
echo HATA: Sanal ortam olusturulamadi.
pause
exit /b 1
:pip_fail
echo HATA: Kutuphaneler indirilemedi. Internet baglantisini / proxy ayarlarini kontrol edin.
pause
exit /b 1
:tk_fail
echo HATA: tkinter bulunamadi. Python'u "tcl/tk and IDLE" secenegi ile yeniden kurun.
pause
exit /b 1
:test_fail
echo HATA: Kendi kendini test basarisiz oldu. Yukaridaki mesaji kontrol edin.
pause
exit /b 1
:build_fail
echo HATA: PyInstaller ile EXE olusturulamadi.
pause
exit /b 1
