Lua functions in D+
===================

**Notes**: Optional parameters are denoted by \[brackets\], multiple instances of the same functions are denoted by {curly, braces}

## Public API:

### Main Functionality:

* `dplus.generate([param_tree, save_filename, should_save_amplitudes])` - Generate using the given parameter tree (or `nil` for current paramters), optionally saving the result to a file. `should_save_amplitudes` determines whether D+ should save the grid amplitudes while generating.
* `dplus.fit(data, param_tree, properties)` - Fit a given parameter tree (or `nil` for current parameters) to the given data table, using the fitting properties given in `properties`.

### Model Management:

* `dplus.findmodel(name[, container])` - Finds a (geometric) model by name, optionally with an external container (DLL) filename
* `dplus.findamp(name[, container])` - Finds an amplitude model by name, optionally with an external container (DLL) filename


### Parameter Tree Management:

* `dplus.getparametertree()` - Returns the current parameters as a parameter tree object
* `dplus.setparametertree(param_tree)` - Sets the parameter tree as the current parameters

* `dplus.save(filename)` - Saves parameters to .state file
* `dplus.load(filename)` - Loads parameters from .state file

### File Management:

* `dplus.readdata(filename)` - Reads data from a signal file to a Lua table
* `dplus.writedata(filename, data)` - Writes data from a Lua table to a signal file (TSV format)

### Figures and Plotting:

* `dplus.figure(title, xlabel, ylabel)` - Opens a new figure with a given title, x, and y labels. Returns figure ID for use in `dplus.showgraph*`
* `dplus.showgraph(data, figure_id, color)` - Plots a signal filename (if `data` is a string) or table (if `data` is a Lua table) on a given figure ID, using a specific color (given as a string of the color name). If `figure_id` is `nil`, uses the last opened figure.

### UI and Script Management Functions:

* `dplus.openconsole()` - Opens an external console window (for performance and debugging purposes)
* `dplus.closeconsole()` - Closes the external console window
* `dplus.{msgbox,mbox,message,messagebox}(message)` - Opens a message box (good for user notifications)
* `dplus.{sleep, wait}(time_ms)` - Wait for the given amount of milliseconds


## Private functions (do not publish!)

* `dplus.findmodelfull(name, container)` - Same as `dplus.findmodel`, but always requires two parameters
* `dplus.findampfull(name, container)` - Same as `dplus.findamp`, but always requires two parameters

* `dplus.generatetable(param_tree, save_filename, should_save_amplitudes)` - full API of `dplus.generate` for tables
* `dplus.generatecurrent()` - full API of `dplus.generate` for current parameter tree

* `dplus.figurefull(title, xlabel, ylabel)` - full API of opening a new figure
* `dplus.showgraphtable(figure_id, data, color)` - full API of plotting graphs from a table
* `dplus.showgraphfile(figure_id, filename, color)` - full API of plotting graphs from a file