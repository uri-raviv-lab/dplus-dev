#pragma once

// This file contains all the non-MainWindow related declarations from MainWindow.h . 
// Putting these declaraions in MainWindow.h caused the designer to crash. Moving them here
// fixes it.

#include "Entity.h"

namespace DPlus {

	enum PanePosition {
		GRAPH2D,
		GRAPH3D,
		SYMMETRY_EDITOR,
		SYMMETRY_VIEWER,
		PREFERENCES,
		CONTROLS,
		PARAMETER_EDITOR,
		FITTINGPREFS,
		SCRIPT_EDITOR,
		COMMAND_WINDOW,
	};


	public ref struct ParameterTreeCLI {
		ParameterTree *pt;
	};


	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Reflection;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace WeifenLuo::WinFormsUI::Docking;
	using namespace LuaInterface;
	using namespace System::Runtime::InteropServices;


	/**
	* Centers the trackbar tb
	* @param tb The trackbar to be centered
	**/
	void centerTrackBar(TrackBar^ tb);


	String ^LuaTableToString(LuaTable ^lt, int curLevel, int maxLevel);
	String ^LuaTableToString(LuaTable ^lt, int curLevel);
	String ^LuaTableToString(LuaTable ^lt);
	Double LuaItemToDouble(Object ^item);
	Boolean LuaItemToBoolean(Object ^item);

	Generic::List<Control^>^ GetAllSubControls(Control^ form);
}
