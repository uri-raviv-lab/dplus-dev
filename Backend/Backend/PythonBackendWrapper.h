/*
 * This class exposes the Backend functionality to the Python API.
 * 
 * It is a replacement of the CommandLineBackendWrapper, as we will no longer need it to access the backend.
 */

#ifndef PYTHON_BACKEND_WRAPPER_H
#define PYTHON_BACKEND_WRAPPER_H

#include "BackendWrapper.h"
#include <string>

class backend_exception;

class EXPORTED_BE PythonBackendWrapper : public BackendWrapper
{
public:
	PythonBackendWrapper();
	~PythonBackendWrapper();

	void CheckCapabilities(bool checkTdr);
	const BackendWrapper::LocalBackendInfo GetBackendInfo(const std::string clientId) { return _info; }

private:
	static LocalBackendInfo _info;
	static bool _infoInitialized;
	static void InitializeInfo();

	static std::runtime_error ConvertException(const backend_exception& e);
};

#endif
