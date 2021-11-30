#ifndef BACKENDEXC_H
#define BACKENDEXC_H
#include "../../Common/Common.h"
#include <stdexcept>

class EXPORTED_BE backend_exception : public std::exception {
public:
	explicit backend_exception(int error_code, const char *error_message = "");
	virtual ~backend_exception() throw() {}
	int GetErrorCode();
	std::string GetErrorMessage();

private:
	int _errorCode;
	std::string _errorMessage;
};

#endif
