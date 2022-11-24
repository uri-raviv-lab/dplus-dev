# Build the DPlus wheels and place them in the proper folders for
# further development
Write-Host --ForegroundColor Cyan "Building the D+ Wheels"

# Remove old files
remove-item -recurse -force -erroraction silentlycontinue .\dist
remove-item -recurse -force -erroraction silentlycontinue .\build

Write-Host -ForegroundColor Cyan "Building debug version"
$env:DPLUS_API_DEBUG="DEBUG"
python setup.py prepare
python setup.py build
python setup.py bdist_wheel

Write-Host -ForegroundColor Cyan "Building release version"
remove-item -recurse -force .\build
$env:DPLUS_API_DEBUG="RELEASE"
python setup.py prepare
python setup.py build
python setup.py bdist_wheel

Write-Host -ForegroundColor Cyan "Copying wheels"
Copy-Item dist\*.whl ..\PythonBackend\PythonBackend\Resources\Wheels

Copy-Item build\lib.win-amd64-3.9\dplus\wrappers.cp39-win_amd64.pyd dplus\

# Left over from the old batch file. We're not sure why this is necessary. Also, note it only
# copies the Release files, not the ReleaseWithDebugInfo files.
#ECHO copying pyd files
#for /R .\build %%f in (*.pyd) do copy %%f .\dplus
