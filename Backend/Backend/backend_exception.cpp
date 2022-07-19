#include "backend_exception.h"

backend_exception::backend_exception(int error_code, const char *error_message) : _errorCode(error_code), _errorMessage(error_message)
{
	if (_errorMessage == "")
		_errorMessage = g_errorStrings[error_code];
}

int backend_exception::GetErrorCode() const
{
	return _errorCode;
}

std::string backend_exception::GetErrorMessage() const
{
	return _errorMessage;
}


