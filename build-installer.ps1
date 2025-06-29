# This script builds the installer
# It can be run from a github action. It takes the version from $env:GITHUB_VERSION

# First, set the version
if($env:GITHUB_VERSION -clike 'refs/tags/dplus-v*') {
    $env:DPLUS_VERSION = $env:GITHUB_VERSION.Substring(17)
}
else {
    $env:DPLUS_VERSION = ""
}

Write-Host DPLUS Version is "$env:DPLUS_VERSION"
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "dplus.sln" /p:Configuration=Release /t:Installer
