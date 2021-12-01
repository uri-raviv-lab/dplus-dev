import os

tests_dir = os.path.join(os.path.dirname(__file__))  # pretty sure this never needs to change
_dplus_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
exe_directory = os.path.join(_dplus_dir, "bin")  # defaults to linux option
session = os.path.join(os.path.dirname(__file__), "test_session")

try:
    from .local_test_settings import *
except Exception as e:
    print("no local settings imported")

if session:
    os.makedirs(session, exist_ok=True) # make sure session directory exists

