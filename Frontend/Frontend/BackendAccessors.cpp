#include "../../Frontend/Frontend/BackendAccessors.h"
#include "LocalComm.h"
#include "CommProtocol.h"
#include <windows.h>

typedef BackendComm *(*clbFunc)(LocalBackendAccessor *lba);

LocalBackendAccessor::LocalBackendAccessor() : _pHLib(NULL), _pBackend(NULL) {
	_pHLib = LoadLibrary(L"xplusbackend.dll");
	if (_pHLib) {
		clbFunc clb = (clbFunc)GetProcAddress((HMODULE)_pHLib, "CreateBackendComm");

		if (clb)
		_pBackend = clb(this); /*comp. err here. can't assign LocalBackend to BackendComm */
		else
			MessageBox(NULL, L"Invalid backend DLL", L"ERROR", MB_ICONERROR);
	}
	else
		MessageBox(NULL, L"Cannot open backend DLL!", L"ERROR", MB_ICONERROR);

}

LocalBackendAccessor::~LocalBackendAccessor() {
	if (_pBackend)
		delete _pBackend;
	if (_pHLib)
		FreeLibrary((HMODULE)_pHLib);
}


BackendComm * LocalBackendAccessor::GetBackend(void)
{
	return _pBackend;
}
