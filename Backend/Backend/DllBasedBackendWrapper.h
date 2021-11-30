#ifndef DLL_BASED_BACKEND_WRAPPER_H
#define DLL_BASED_BACKEND_WRAPPER_H

#include "BackendWrapper.h"

class EXPORTED_BE DllBasedBackendWrapper : public BackendWrapper
{
public:
	DllBasedBackendWrapper();
	~DllBasedBackendWrapper();

protected:
	virtual const LocalBackendInfo GetBackendInfo(const std::string clientId);

private:
	LocalBackendInfo _info;
};

#endif