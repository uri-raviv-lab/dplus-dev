#pragma once

namespace DPlus {
	ref class Scripting {
	public:
		static void OpenConsole();
		static void CloseConsole();
		static bool IsConsoleOpen();
	};
}
