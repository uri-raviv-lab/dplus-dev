// PopulationGUI.cpp : main project file.

#include "TWindow.h"

using namespace PopulationGUI;

[STAThreadAttribute]
int main(array<System::String ^> ^args)
{
	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false); 

	// Create the main window and run it
	Application::Run(gcnew TWindow());
	return 0;
}
