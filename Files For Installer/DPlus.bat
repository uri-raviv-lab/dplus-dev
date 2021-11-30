ECHO off
SET mypath=%~dp0
pushd "%mypath:~0,-1%\D+"
start /b DPlus.exe
popd