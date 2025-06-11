// DPlus.cpp : main project file.

#include "MainWindow.h"

//#define CHECK_DLL

#ifdef CHECK_DLL
#include <windows.h>
#endif

using namespace DPlus;
#ifdef CHECK_DLL
using namespace System::IO;

typedef unsigned short MachineType;

/**
 * @file main.cpp
 * @brief Entry point for the DPlus application, responsible for initializing and launching the main GUI window.
 *
 * The main module is responsible for:
 *  - Setting up application-wide visual styles and text rendering options.
 *  - Optionally checking the architecture (32/64-bit) of a required DLL at startup (when CHECK_DLL is defined).
 *  - Creating and running the main application window (MainWindow).
 *
 * Key Concepts:
 *  - Application Initialization: Ensures the application uses modern Windows visual styles and compatible text rendering.
 *  - DLL Architecture Check (optional): Provides a mechanism to verify the bitness of a required DLL before launching the main window.
 *  - MainWindow: The primary user interface window for the DPlus application.
 *
 * Fields:
 *  - (Conditional, CHECK_DLL) String^ winF:
 *      Path to the WeifenLuo.WinFormsUI.Docking.dll file, if found.
 *  - (Conditional, CHECK_DLL) MachineType:
 *      Enum value representing the machine type (architecture) of a DLL.
 *
 * Main Methods:
 *  - int main(array<System::String ^> ^args):
 *      Application entry point. Sets up visual styles, optionally checks DLL architecture, and runs the main window.
 *  - (Conditional, CHECK_DLL) GetDllMachineType(String^ dllPath):
 *      Reads the PE header of a DLL to determine its machine type (architecture).
 *  - (Conditional, CHECK_DLL) UnmanagedDllIs64Bit(String^ dllPath):
 *      Determines if a DLL is 64-bit, 32-bit, or unknown.
 *
 * Threading and Safety:
 *  - All initialization and UI operations are performed on the main thread.
 *  - The application uses the [STAThread] attribute for Windows Forms compatibility.
 *
 * Error Handling:
 *  - Displays message boxes for missing or incompatible DLLs when CHECK_DLL is defined.
 *  - Throws exceptions for invalid PE headers or unknown DLL architectures (when checking DLLs).
 *
 * Dependencies:
 *  - MainWindow.h for the main application window.
 *  - Windows Forms libraries for UI.
 *  - (Conditional, CHECK_DLL) Windows.h and System::IO for DLL inspection.
 *
 * See MainWindow.h for the main window implementation.
 */

MachineType GetDllMachineType(String^ dllPath)
{
	//see http://www.microsoft.com/whdc/system/platform/firmware/PECOFF.mspx
	//offset to PE header is always at 0x3C
	//PE header starts with "PE\0\0" =  0x50 0x45 0x00 0x00
	//followed by 2-byte machine type field (see document above for enum)
	FileStream^ fs = gcnew FileStream(dllPath, FileMode::Open, FileAccess::Read);
	BinaryReader^ br = gcnew BinaryReader(fs);
	fs->Seek(0x3c, SeekOrigin::Begin);
	Int32 peOffset = br->ReadInt32();
	fs->Seek(peOffset, SeekOrigin::Begin);
	UInt32 peHead = br->ReadUInt32();
	if(peHead!=0x00004550) // "PE\0\0", little-endian
		throw gcnew Exception("Can't find PE header");       
	MachineType machineType = (MachineType) br->ReadUInt16();
	br->Close();
	fs->Close();
	return machineType;
}
// returns true if the dll is 64-bit, false if 32-bit, and null if unknown
bool UnmanagedDllIs64Bit(String^ dllPath)
{
	MachineType mt = GetDllMachineType(dllPath);
	switch (mt)
	{
	case IMAGE_FILE_MACHINE_AMD64:
	case IMAGE_FILE_MACHINE_IA64:
		return true;
	case IMAGE_FILE_MACHINE_I386:
		return false;
	default:
		return NULL;
	}
}
#endif

[STAThreadAttribute]
int main(array<System::String ^> ^args)
{
	// Enabling Windows XP visual effects before any controls are created
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false); 

#ifdef CHECK_DLL
	array<System::String^>^ files ;
	//files = gcnew array<System::String^>();
	files = (System::IO::Directory::GetFiles(Application::StartupPath /*+ L"\\dlls" */, L"*.dll"));
	System::String^ winF = nullptr;//= L"WeifenLuo.WinFormsUI.Docking.dll";
	for(int j = 0; j < files->Length; j++) {
		if(files[j]->Contains(L"WeifenLuo.WinFormsUI.Docking.dll")) {
			winF = files[j];
			break;
		}
	}

	if(!winF) {
		MessageBox::Show(L"Could not find WeifenLuo.WinFormsUI.Docking.dll", L"ERROR");
	} else {
		void *hLib;
		hLib = LoadLibrary(L"WeifenLuo.WinFormsUI.Docking.dll");
		if(hLib) {
			if(UnmanagedDllIs64Bit(winF))
				MessageBox::Show(L"64 Bit DLL", L"ERROR");
			else
				MessageBox::Show(L"32 Bit DLL", L"ERROR");
		} else
			MessageBox::Show(L"Cannot open backend DLL!", L"ERROR");
	}
#endif

	// Create the main window and run it
	Application::Run(gcnew MainWindow());

	return 0;
}
