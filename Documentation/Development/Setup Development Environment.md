# Setting up a D+ Development Environment on Windows
(updated December 6th 2021)

## Prequisites
You need to install the following on your computer, in order to compile and debug D+:

### 1. Visual Studio 2022
Download and install it. You need to include:

* Desktop development with C++ - make sure VC++ v142 and C++/CLI support, MFC and Windows SDK 10.0.18362.0.
* Desktop .NET Development - make sure to include .NET Framework 4.8 Development Tools
* From Individual Modules, add C++ 2019 Redistributables MSMs and Windows Universal CRT SDK.

### 2. CUDA 11.7
Install it, along with the Visual Studio Integration.

### 3. cmake 3.22 or later

### 4. Boost sources and binaries
Download Boost (version 1.77 is used with D+, later versions should work). Download the Windows binaries from here https://boost.teeks99.com/bin/1.77.0/ - choose the right compiler version (msvc-14.2) and choose the 64-bit executable. Download and install it. Then rename the lib64-msvc-14.2 folder to lib64.

You should also set the BOOST_ROOT environment variable to c:\boost\1.77 (or whatever path you choose)

### 5. Python
You must have Python 3.9 64 bit installed

### 6. WIX Toolset - only relevant if you are building an installer for D+
From Visual Studio 2017, open the Tools menu and choose "Extensions and Updates". Search for "wix" and install WiX Toolset Build Tools *and* Wix Toolset Visual 2017 Extension. 
From Visual Studio 2019, open the Extensions menu, choose "Manage Extensions". Search for "wix" and install WiX Toolset Build Tools *and* Wix Toolset Visual 2019 Extension. 

You will need to enable .NET 3.5.1 from the control panel. (Win-S and search Turn Windows Features On and Off). If it is not installed, you will need to install it first (https://www.microsoft.com/en-us/download/details.aspx?id=22).

## HOW TO BUILD
D+ consists of three parts- the backend, the python API, and the frontend (in order of dependence: the backend is independent, the api depends on the backend, the frontend depends on both the backend and the api)

#Building the entire project

1. Build only the Backend in both Release and ReleaseWithDebugInfo
2. In PythonInterface, activate your virtual environment (inside PowerShell) and then run:

	pip install -r requirements.txt

	rebuild-wheels.ps1
3. In the frontend, **rebuild** PythonBackend in both Release and ReleaseWithDebugInfo (otherwise the embedded resources are not updated)
4. Build the frontend

# Building after a change in the backend
Repeat steps 1-3 in the section "Building the entire project"

# Building after a change in the python interface
Repeat steps 2-3 in the section "Building the entire project"

Note that this is for building the python wheels for the dplus program. 

To build the python wheels for uploading to pypi, you will need to run `python setup.py bdist_wheel` for Windows, and the manylinux docker for linux.

# Building after a change in the frontend
Simply build the frontend, no other steps needed. 

# Release vs ReleaseWithDebugInfo
The wheel built with ReleaseWithDebugInfo is *significantly* slower than Release.

## Troubleshooting

### Error "not digitally signed" when running rebuild-wheels.ps1
Inside the PowerShell run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser and when asked press Y(es).
Rerun rebuild-wheels.ps1

### Missing standard headers files (such as stdio.h)
It is possible that the compilation will fail, because standard header files (such as assert.h or stdio.h) can't be found. This is an indication that the wrong Windows SDK is installed. Find out which is the correct version by right clicking on the Backend project and selecting Properties. Use the Visual Studio Installer to install the proper version (you can find it in the Desktop C++ Building Tools section)