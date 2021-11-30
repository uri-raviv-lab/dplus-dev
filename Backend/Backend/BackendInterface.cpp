#include "../../BackendCommunication/LocalCommunication/LocalComm.h"
#include "BackendInterface.h"

// Can be whatever you like (std::vectors and such) as long as the BackendComm class
// is portable.

#include "DllBasedBackendWrapper.h"
#include "LocalBackend.h"

BackendComm *CreateBackendComm() {
	return new DllBasedBackendWrapper();
}

LocalBackend *CreateLocalBackend() {
	return new LocalBackend();
}
