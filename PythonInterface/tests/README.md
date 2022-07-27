#Running the tests:

## Setup - general

For now: The tests require a working generate.exe in order to run. 
You will need to compile D+ and provide the path to the compiled exe, or have D+ installed on your computer 
(in which case it case the tests can auto-find the path to D+)

The fit tests are currently in a half-working state, because we have not finished the transition away from fit.exe

You will need to set up your virtual environment.
`pip install -r requirements.txt`

After setting up the virtual environment, you will also need to build the Cython files

`python setup.py prepare`
`python setup.py build`

## Setup for the tests

In PythonInterface/tests, you can optionally create a file local_test_settings.py
This will overwrite default settings set in test_settings.py

the settings are:
tests_dir - where the tests are located. should probably never be changed.
exe_directory - where the generate (and fit) executables are located
session - where test result files are stored

The most likely test_setting you may need to change is exe_directory. 
It defaults to {solutionDir}/x64/Release on Windows if that folder exists, and {solutionDir}/bin on linux if that folder exists.
Otherwise it defaults to None.
So if you want to run tests against the Debug compiled versions of the executables, or against an installed version of D+, you can provide a different value

The tests have some command line arguments you can set. These are primarily for debugging or setting up longer tests on the test server.

## Run the tests
You need to create a new virtual environment, from which you will run the tests

```
cd tests
py -3.9 -m venv d:\temp\dplus-test-env --prompt="D+ Test Env"
Activate-Virtenv d:\temp\dplus-test-env
python -m pip install --upgrade pip
pip install numpy pytest
pip install ..
```

Now you can run the test. You must change pytest's importmode to `append`, otherwise it won't find the D+ DLLs.

`pytest unit_tests --import-mode append`  - tests basic classes from the api
`pytest manual_tests  --import-mode append`  - tests all code examples from the manual

`cd reviewer_tests` (from within the tests folder, otherwise cd tests\reviewer_tests if you want to skip straight to this step)
`pytest test_generate_qmin  --import-mode append` - this tests a subset of generate tests, and is therefore faster than the main generate suite
`pytest test_generate  --import-mode append` - tests generate. takes a while
`pytest test_fit  --import-mode append` - tests 7 cases of fit. 

