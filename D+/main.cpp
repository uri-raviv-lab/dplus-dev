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
