# Setting up a D+ Development Environment on Windows
(updated December 2nd 2021)

## Prequisites
You need to install the following on your computer, in order to compile and debug D+:

### 1. Visual Studio 2017
Download and install it. You need to include:

* Desktop development with C++ - make sure VC++ v141 and C++/CLI support, MFC and Windows SDK 10.0.17763.0.
* Desktop .NET Development - make sure to include .NET Framework 4.7.2 Development Tools

### 2. CUDA 11.5
Install it, along with the Visual Studio Integration.

### 3. cmake 3.22 or later

### 4. Boost sources and binaries
Download Boost (version 1.67 is used with D+, later versions should work). Download the sources, unzip into a folder (for instance `c:\boost\1.67`). Then download the binaries from here https://boost.teeks99.com/bin/1.67.0/ - choose the right compiler version (msvc-14.1) and choose the 64-bit executable. Download and install it. Then rename the library folder to lib64.

You should also set the BOOST_ROOT environment variable to c:\boost\1.67 (or whatever path you choose)

### 5. Python
You must have Python 3.9 64 bit installed

### 6. WIX Toolset
From Visual Studio 2017, open the Tools menu and choose "Extensions and Updates". Search for "wix" and install WiX Toolset Build Tools and Wix Toolset Visual 2017 Extension.

## IMPORTANT
At this point, the entire D+ project is in one solution. This is about to change, as you will clearly see - the backend and frontend require two different builds.

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
The Frontend project PythonBackend has embedded resources, one of them is the dplus-api wheel. If you create a new wheel, you should change the embedded resource in this project. Then build the solution.

## Things to do next:
We want to split the solution into two solutions - one for the backend, another for the frontend. Since both use the /Common directory, we may not split DPlus into two different repositories.



