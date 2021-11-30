// DockingTest.cpp : main project file.

#include "DockingWindow.h"

using namespace DockingTest;

[STAThreadAttribute]
int main(array<System::String ^> ^args)
{
	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false); 

	// Create the main window and run it
	Application::Run(gcnew DockingWindow());
	return 0;
}
