# Running the tests:

## Setup - general
Before running the tests, Dplus should be compiled.
Open the solution and rebuild the `Backend` solution for both `Release` and `Release with debug info` configurations.

You will need to set up your virtual environment.
If the virtual environment does not exist, create one using:
`py -3.9 -m venv env`
> Python version 3.9 is used, because `rebuild-wheels.ps1` looks for the file:  
>`build\lib.win-amd64-3.9\dplus\wrappers.cp39-win_amd64.pyd`.   
>To use a different version, `rebuild-wheels` must be updated.  
>Eventually, `rebuild-wheels` should use a parameter to define the python version. 

Activate the virtual environment:  
`Activate-Virtenv`  
Once the environment is up and active, install requirements:  
`pip install -r requirements.txt`

After setting up the virtual environment, you will also need to build the Cython files:  
`.\rebuild-wheels.ps1`

## Setup for the tests

In PythonInterface/tests/test_settings.py, you can define settings for tests.  
the settings are:  
**tests_dir** - where the tests are located. should probably never be changed.  
**session** - where test result files are stored.  
**USE_GPU** - whether to use GPU for testing or not.  

The tests have some command line arguments you can set. These are primarily for debugging or setting up longer tests on the test server.

## Run the tests
Click on the testing icon in VSCode to discover the tests.  
if there are any errors, they can be viewed in Terminal->Output->Python.  

Now the tests are ready and can be run.
