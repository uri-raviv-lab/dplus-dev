The Examples folder helps new users to learn how to use D+. Each subfolder contains  an example to be loaded and run in D+, and a reference result.

In each example folder you will find one or more ".state" files and one or more ".out" files. 
The ".state" files contains the relevant parameters, and the ".out" files are the exported generated curves for each set of parameters. 

In addition to these files, supporting files for the set of parameters are also included in the folders. There is no need to change these files.

The examples are used as follows:
-	Open D+ 
-	Select "File" menu and click "Import All Parameters"
-	Select the “.state” file in the relevant example subfolder
-	In the "Controls" pane -  Press "Generate" 
	o	default location is at the top right corner, thought location can be changed by the user. If you do not see the "Controls" pane, select "View" menu, and click "Controls". The View menu may add or remove panes.
	o	Note that all panes can be manipulated in D+ main window by drag and drop. If by accident you undock a pane, double click on the title of the pane, and D+ will dock it back.
	o	Default display layout can be reached by selecting "Edit" menu, and click "Default Layout"
-	Once D+ calculation is done, you can save your generated curves (Select "File" menu and click "Export 1D Graph…") and compare it with the reference curve from the example folder, in whatever plotting software you desire. 
You can also compare your calculated result to the reference result using D+:
	o	Your generated curve is in blue, any loaded signal will be in red
	o	To load a signal, Select "File" menu and click "Open Signal…"
	o	In the case of comparing examples with reference curves, the plots should almost overlap (apart from numerical fluctuations when using Monte-Carlo integration). If the generated curve and loaded signals overlap completely, only the calculated (blue) signal will be seen, with a red glow behind it. 
Specific examples notes
-	Sphere (CPU\Fast_CPU or GPU\Fast_GPU)
	o	In the example folder there is a ".dol" file ("5spheres.dol"). This file is a Manual Symmetry input file. If you wish to reproduce the example state by yourself, you will need to load this ".dol" file. Using Manual Symmetry is explained in Tutorial 4 – Using symmetry options in D+.
-	Microtubule (CPU\Slow_CPU or GPU\Slow_GPU)
	o	In the example folder there are two ".out" signals – one is the reference result ("MT_Model.out"), and the other is an actual SAXS curve of MTs ("MT_Signal.out"). The experimental curve and calculated model will not align perfectly!
-	CCl4 (CPU\Fast_CPU or GPU\Fast_GPU)
	o	In the example folder there are two parameters files ("CCl4_Symmetry.state" and "CCl4_PDB.state"), each has its own reference result ("CCl4_Symmetry.out" and " CCl4_PDB.out", respectively). The two results should be identical, as they compute the scattering from the same structure, constructed in two methods (single PDB and PDB with manual symmetry of a PDB).
	o	In the example folder there is a ".dol" file ("Cl4.dol") used for the Cl atoms. This file is a Manual Symmetry input file. If you wish to reproduce the example state by yourself, you will need to load this ".dol" file. Using Manual Symmetry is explained in Tutorial 4 – Using symmetry options in D+.
-	Lysozyme (CPU\Fast_CPU or GPU\Fast_GPU)
	o	In the example folder there are four parameters files:
			Pre_Fit_Constant_and_Scale.state
			Pre_Fit_Outer_Solvent_ED_Constant_and_Scale.state
			Pre_Fit_Outer_Solvent_ED_Constant_and_Scale_From_Amp.state
			Fitted_Outer_Solvent_ED_Constant_and_Scale.state
	o	Each parameters file is intended for a different use:
			Pre_Fit_Constant_and_Scale.state – load this parameter file and press "Fit" in the "Controls" pane. 
			As only the "Domain Scale" and "Domain Constant" fields are marked "Mutable", the fitter will only fit these variables. 
			Compare the result with "Fitted_Scale_and_Constant.out"
			Pre_Fit_Outer_Solvent_ED_Constant_and_Scale.state - load this parameters file and press "Fit" in the "Controls" pane. 
			Here, three parameters are marked "Mutable": "Domain Scale", "Domain Constant" and "Scale" for the "Manual Symmetry". 
			Compare the result with "Fitted_Outer_Solvent_ED_Constant_and_Scale.out"
			Pre_Fit_Outer_Solvent_ED_Constant_and_Scale_From_Amp.state - load this parameters file and press "Fit" in the "Controls" pane. 
			Here, three parameters are marked "Mutable": "Domain Scale", "Domain Constant" and "Scale" for the "1lyz_Hydration_Only.amp" entity.  
			Compare the result with "Fitted_Outer_Solvent_ED_Constant_and_Scale.out"
			Fitted_Outer_Solvent_ED_Constant_and_Scale.state - load this parameters file and press "Generate" in the "Controls" pane.
		Compare the result with "Fitted_Outer_Solvent_ED_Constant_and_Scale.out"
-	Hydration Correction
	o	In the example folder there are two ".out" signals – one is the reference result ("MT_Model.out"), and the other is an actual SAXS curve of MTs ("MT_Signal.out"). The experimental curve and calculated model will not align perfectly!
-	Fits 7-9 (CPU\Fast_CPU or GPU\Fast_GPU)
	o	In each example folder there are two parameters files:
			Initial_State.state
			Fitted_State.state
	o	Each parameters file is intended for a different use:
			Initial_State.state - load this parameters file and press "Fit" in the "Controls" pane. 
			Three parameters are marked "Mutable": "Domain Scale", "Domain Constant" and "Scale" for the "Hydration Only" manual symmetry.  
			Compare the fit result with " Fit.out"
			Fitted_State.state - load this parameters file and press "Generate" in the "Controls" pane.
			Compare the result with " Fit.out"

