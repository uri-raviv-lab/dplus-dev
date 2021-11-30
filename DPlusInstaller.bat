@echo off

setlocal ENABLEEXTENSIONS

REM Check to see if the VS 2013 C++ runtime is intalled
reg query HKLM\SOFTWARE\Classes\Installer\Dependencies\{050d4fc8-5d48-4b8f-8972-47c82c46020f} > NUL 2>&1

set haveRedist=%errorlevel%

REM Check to see if VS 2013 is intalled
reg query HKEY_CLASSES_ROOT\VisualStudio.DTE.12.0 > NUL 2>&1

set haveVS=%errorlevel%

set "needToInstall=1"

if NOT %haveRedist%==1 set needToInstall=0
if NOT %haveVS%==1 set needToInstall=0

if %needToInstall%==1 (
	echo Installing VS 2013 C++ runtime...
	goto :installRedist2013	
	)

if %needToInstall%==0 (
	echo VS 2013 C++ or runtime is already installed. Skipping.
)
		
goto :done

:installRedist2013
set tempFile=%temp%\vs2013Redist.exe

bitsadmin /create installVSRuntime
bitsadmin /transfer installVSRuntime https://download.microsoft.com/download/2/E/6/2E61CFA4-993B-4DD4-91DA-3737CD5CD6E3/vcredist_x64.exe %tempFile%
bitsadmin /complete installVSRuntime

%tempFile% /install

goto :done


:done

echo.
echo NOTE for GPU users:
echo If there is a Nvidia GPU installed, CUDA 8 must be installed as well in order for D+ to use it.
echo.
pause


goto :eof