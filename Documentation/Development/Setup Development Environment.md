# Setting up a D+ Development Environment on Windows
(updated December 6th 2021)

## Prequisites
You need to install the following on your computer, in order to compile and debug D+:

### 1. Visual Studio 2019
Download and install it. You need to include:

* Desktop development with C++ - make sure VC++ v142 and C++/CLI support, MFC and Windows SDK 10.0.18362.0.
* Desktop .NET Development - make sure to include .NET Framework 4.8 Development Tools
* From Individual Modules, add C++ 2019 Redistributables MSMs and Windows Universal CRT SDK.

### 2. CUDA 11.5
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

## IMPORTANT
D+ consists of three parts- the backend, the python API, and the frontend. 
(We may be making some changes in the future, to consolidate the backend with the python API.)
For now, both the API and the frontend depend on the backend having been built in Release mode.
The frontend depends on the API, but it comes with a prebuild version of it, so you do not need to rebuild the API unless you are using it yourself, or have made changes to it you want reflected in the API. 

## Compiling the Backend
To compile the backend, open the DPlus2017.sln solution, choose x64 as the platform and build. This will build the C++ backend.

### Compiling the Python API
You should also compile the Python dplus API. Do so by creating a Python 3.9 virtual environment. Activate it and

    cd PythonInterface
    pip install -r requirements.txt
    pip install wheel
    python setup.py prepare
    python setup.py bdist_wheel

You will have the wheel in the dist folder.

The dplus-api is also built automatically for Windows and Linux if you push a tag starting with `v` to github.

## Compiling the Frontend
The Frontend project PythonBackend has embedded resources, one of them is the dplus-api wheel. If you have created a new wheel, you should change the embedded resource in this project. Then build the solution.




