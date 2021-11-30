@echo off
echo Restarting D+...
timeout /t 5 /nobreak
start "" "%~dp0\\..\\DPlus.exe" --remote