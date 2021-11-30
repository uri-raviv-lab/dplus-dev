#include "SPMainWindow.h"

using namespace System::Windows::Forms;
using namespace SuggestParameters;

[STAThreadAttribute]
int WinMain()
{
	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);

	// Create the main window and run it
	Application::Run(gcnew SPMainWindow());
}
