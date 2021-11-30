@echo off
REM Assumes that convert (http://www.imagemagick.org/script/convert.php) is installed and in the PATH
for /R %%r in (*.bmp) do (
	convert "%%r" "%%~dr%%~pr%%~nr.png%"
	del "%%r"
)

