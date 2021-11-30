/*
	A way for the the UI to access a backend so it can pass it to its local frontend
*/
#ifndef _BACKEND_ACCESSORS_H
#define _BACKEND_ACCESSORS_H

#include "../../Common/CommProtocol.h"
class LocalBackend;

class EXPORTED LocalBackendAccessor
{
	public:
		LocalBackendAccessor();
		~LocalBackendAccessor();

		BackendComm *GetBackend();
	protected:
		BackendComm *_pBackend;
		void *_pHLib;
};

#endif