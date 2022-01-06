import os
from sys import platform

tests_dir = os.path.join(os.path.dirname(__file__))  # pretty sure this never needs to change
_dplus_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
session = os.path.join(os.path.dirname(__file__), "test_session")

exe_directory=None
if platform.startswith("linux"):
    exe_directory = os.path.join(_dplus_dir, "bin")
if platform.startswith("win"):
    exe_directory = os.path.join(_dplus_dir, "x64", "Release")
if exe_directory and not os.path.isdir(exe_directory): #reset to None
    exe_directory=None

try:
    from .local_test_settings import *
except Exception as e:
    print("no local settings imported")

if session:
    os.makedirs(session, exist_ok=True) # make sure session directory exists

