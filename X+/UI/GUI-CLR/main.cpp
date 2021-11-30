#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <windows.h>
#include <commctrl.h>

#include "OpeningWindow.h"
#include "SignalSeries.h"

using GUICLR::OpeningWindow;

[STAThreadAttribute]
int WINAPI WinMain(      
				   HINSTANCE,
				   HINSTANCE,
				   LPSTR,
				   int
				   ) {
	//Initialize common controls
	InitCommonControls();

	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false); 
	
	//code = InitializeOpeningWindow();

	LPWSTR *szArglist;
	int nArgs;

	szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);
	for(int i = 1; i < nArgs; i++) {
		String ^arg = gcnew String(szArglist[i]);

		if(arg->Equals("signalSeries", System::StringComparison::InvariantCultureIgnoreCase)) {
			Application::Run(gcnew SignalSeries());
	
			LocalFree(szArglist);

			_CrtDumpMemoryLeaks();

			return 0;
		}
	}

	LocalFree(szArglist);

	Application::Run(gcnew OpeningWindow());
	
	_CrtDumpMemoryLeaks();

	return 0;
}
