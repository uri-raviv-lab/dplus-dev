@echo off

set be_string=
call :getRevisionString be_string BACKEND
echo %be_string%
set fe_string=
call :getRevisionString fe_string FRONTEND
echo %fe_string%

REM Get the git hash for the current revision
set tmpFilename=%temp%\%random%
git rev-parse --short HEAD  > %tmpFilename%
set /p revStr= < %tmpFilename%
del %tmpFilename%

git rev-list HEAD --count > %tmpFilename%
set /p revCount= < %tmpFilename%
del %tmpFilename%

REM remove the quotes
rem set rev=%rev:"=%

set "newDirName=v%fe_string%_(%be_string%)_(r%revCount%)_%revStr%"

echo %newDirName%

REM Set new and old directories
set newDir="u:\Current D+\%newDirName%\"
set oldDir=.\x64\Release\
set luaDir=.\LuaScripts\
set exampleDir=.\Example Files\

REM svn log -r head:1 --limit 20 > "./x64/Release/changes.txt"
git log -100 > "./x64/Release/changes.txt"

echo %rev%
echo %newDir%

call :copyToDest %oldDir% %newDir%
set newDir=s:\%newDirName%\
call :copyToDest %oldDir% %newDir%
goto :eof

:getRevisionString
REM Get the current revision
rem %1 is the string to return to %2 is FRONTEND or BACKEND
for /f "tokens=2*" %%i in ('findstr "%2_VERSION_MAJOR" .\%2_version.h') do (
	set maj=%%j
	goto foundMaj
	)
:foundMaj

for /f "tokens=2*" %%i in ('findstr "%2_VERSION_MINOR" .\%2_version.h') do (
	set min=%%j
	goto foundMin
	)
:foundMin

for /f "tokens=2*" %%i in ('findstr "%2_VERSION_REVISION" .\%2_version.h') do (
	set rev=%%j
	goto foundRev
	)
:foundRev

for /f "tokens=2*" %%i in ('findstr "%2_VERSION_BUILD" .\%2_version.h') do (
	set bld=%%j
	goto foundBld
	)
:foundBld
set "%1=%maj%.%min%.%rev%.%bld%"
goto :eof

rem %1 is source, %2 is destination
:copyToDest
echo COPYING %1 TO %2
mkdir %2
mkdir %2"D+"
mkdir %2LuaScripts\
mkdir %2"%exampleDir%"
copy %1*.exe %2"D+"
copy %1*.dll %2"D+"
copy %1Latest.* %2"D+"
copy %1xplusmodels.xrn %2"D+"
copy %1changes.txt %2"D+"
copy %luaDir%*.lua %2LuaScripts\
copy %1JSON.lua %2"D+"
copy "%exampleDir%*" %2"%exampleDir%"
copy "S:\Manual\D+Manual 20170119.pdf" %2
copy %1"..\..\Files For Installer\Install.bat" %2
copy %1"..\..\Files For Installer\DPlus.bat" %2
xcopy %1"..\..\Files For Installer\Resources" "%2\D+\Resources\" /Y
copy %1"Suggest Parameters.exe" %2
goto :eof
