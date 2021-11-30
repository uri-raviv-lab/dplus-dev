The DPLus Python Wrapper
========================
4.3.7
----------------
* Added classes ParameterContainers, Layers, Children to help with input validation
* added function add_layer
* reworked dynamic loading so variables that should be hidden from user are hidden
* fixed an infinity json encoding bug
 
4.3.6
----------------
* added interpolation
* renamed CythonWrapping to CythonGrid
* changed function create_grid to return `index, (q,theta,phi)` instead of `q, theta, phi` 

4.3.5
----------------
* fix installation error that was caused from JacobianSphereGrid

4.3.1
---------------- 
* add the class Signal with represent signal file. All the references to x and y in the other class now passes through Signal class
* add class CJacobianSphereGrid in new module CythonWrapping. 
* change Amplitude class in module Amplitudes to work with CJacobianSphereGrid instead of Grid

4.3.0
----------------
* add support for ampj files instead of amp files
* fix bug in retrieving nested models by ptr, name, or type 

3.2.2
----------------
* better error reporting when loading state

3.2.1
----------------
* additional updates to metadata

3.2.0
----------------
* updated metadata
* fix to web file upload

3.1.9
----------------
small bug fixes

3.1.7
----------------
* handle negative intensity values
* check validity of remote job start

3.1.5
----------------
* bug fixes to ScriptedSymmetry and file uploading

3.1.4
----------------
* bug fixes from the changes to CalculationInput in previous version
* add function get_amps to CalculationResult
* fix accuracy bug by making qmax float64

3.1.3
----------------

* change CalculationInput to inherit from rather than composite State
* combine GenerateInput and FitInput into CalculationInput
* split CalculationResult into GenerateResult and FitResult
* added user-provided header support to Python Amplitudes

3.1.2
----------------

* added property GeneratedPoints to access res_steps (matches D+ UI)
* added module Amplitudes with classes Grid and Amplitude, got rid of Amplitude from FileReaders
* added some support for the Amplitude class to State (add_amplitude)
* added support for adding models to state (add_model)
* bug fixes
* filecontainingmodels can receive a filename as an initialization argument

3.1.0 and 3.1.1
----------------
These updates do not promise backwards compatibility

Readme.md has been brought up-to-date, with up-to-date function and variable names, 
and non public interface functions removed. 

A variety of bugs were fixed along the way


3.0.7
----------------
* Add field res_steps to state file - the length of x vector is calculated according to this field
* Add function "save_to_out_file" to CalculationResult

3.0.7
-----
* Add functionality to Amplitude class (in FileReaders.py) 
* Add function q_indices - return array of [q, theta, phi] for each amplitude item in the amplitude array 
* Add function num_indices - return the numbers of trios [q, theta, phi] in Amplitude file
* Add function complex_amplitude_array - return complex array of amplitudes
* Add property to Amplitude class - step_size


3.0.6
-----

* CalculationResult now has the functions get_amp and get_pdb which return the file location of the pdb/amp of requested model_ptr
* RunningJob now has function get_result that returns a CalculationResult
* Bug fixes to async calculations
* Refactors: In DataModels.py and State.py, Param has been renamed to Parameter and to_dict has been renamed serialize. In 
CalculationInput, load_from_state has been renamed load_from_state_file
* API.py has been removed, the correct file is CalculationRunner.py
* added a class Amplitude in CalculationResult.py that can open amp files and save them.
* added some tests
* support for some added paramters in models, like an optional constant and some pdb features


3.0.5
-----
Made FittingPreferences and DomainPreferences into classes, with validation of properties. This allows for user to be warned if they submit properties that aren't valid.
Reinstated load_from_pdb in GenerateInput

3.0.3
-----
*small fixes to enable webRunner to work again

3.0.2
-----
* Rename load_from_state_file to load_from_state
* Generate a new temporary session directory each time CalculationInput is created.
* Correctly take use_grid from models.

3.0.1
-----
* Add the license back. 

3.0.0
-----
* Bug fixes
* Better fit interface.
* Disabled Python fit.
* Version number 

0.3.5
----
* Encode and decode infinities better.
* Handle generate jobs better
* Fixed to fit

0.3.4
----
* Fixed a minor bug causing Fit to lock if there was a problem with the generate executable.

0.3.3
----
* Added Fitting in Python (just one optimization method)
* Added a Python representation of all the models


0.3.0
----
* Added tests
* Added access to the state and other input properties.
* Better error reporting if files do not exist.
* Allow getting model by name from the input.


0.2.2
----
* Create temporary session directories properly on both Windows and Linux.

0.2.1
----
* Start the generate processes with the proper current working directory.
* Raise an exception in case of generation errors.
* Added a meaningful README

0.2.0
----
* Web interface
* Return Amplitudes
* Put sessions in a temporary directory by default.
* Create a state from a PDB file

0.1.0
----
First version
* Return Amplitudes
* Put sessions in a temporary directory by default.
* Create a state from a PDB file

0.1.0
----
First version
