rem call env\Scripts\activate.bat
rem ECHO deleting Backend Files from include
rem @RD /S /Q IncludeFiles\Backend
python setup.py prepare
python setup.py build_ext
rem TODO: add a python command postpare and replace the batch stuff
PUSHD
ECHO copying pyd files
cd build
   for /r %%a in (*.pyd) do (
     COPY "%%a" "%~dp0%%~nxa"
   )
POPD