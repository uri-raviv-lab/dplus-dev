ECHO remove existing files
@RD /S /Q .\dist
@RD /S /Q .\build
ECHO building debuginfo version
set "DPLUS_API_DEBUG=DEBUG"
python setup.py prepare
python setup.py build
python setup.py bdist_wheel
ECHO building release version
ECHO delete build directory
@RD /S /Q .\build
set "DPLUS_API_DEBUG=RELEASE"
python setup.py prepare
python setup.py build
python setup.py bdist_wheel
ECHO copying wheels
for /R .\dist %%f in (*.whl) do copy %%f ..\PythonBackend\PythonBackend\Resources\Wheels
ECHO copying pyd files
for /R .\build %%f in (*.pyd) do copy %%f .\dplus