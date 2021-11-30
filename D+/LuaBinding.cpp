#include "LuaBinding.h"

#include <windows.h>
#include <iostream>
#include <fcntl.h>
#include <io.h>

bool bOpen = false;

void DPlus::Scripting::OpenConsole() {
	static const WORD MAX_CONSOLE_LINES = 5000;
	int hConHandle;
	long lStdHandle;
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	FILE *fp;

	if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
	std::ios::sync_with_stdio();
	// std::cout << "This is a test\n" << std::endl;

	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
	coninfo.dwSize.Y = MAX_CONSOLE_LINES;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);
	bOpen = true;
}

void DPlus::Scripting::CloseConsole() {
	FreeConsole();
	bOpen = false;
}

bool DPlus::Scripting::IsConsoleOpen() {
	return bOpen;
}
