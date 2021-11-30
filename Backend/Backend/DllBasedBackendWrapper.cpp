#include "LocalBackendParameterTree.h"
#include "DllBasedBackendWrapper.h"
#include "LocalBackend.h"
#include "AmplitudeCache.h"

DllBasedBackendWrapper::DllBasedBackendWrapper()
{
	_info.local_backend = new LocalBackend();
	_info.job = _info.local_backend->HandleCreateJob(L"Single job");
	_info.Converter = new LocalBackendParameterTreeConverter(_info.local_backend, _info.job);
	AmplitudeCache::initializeCache(_info.Converter);
}

DllBasedBackendWrapper::~DllBasedBackendWrapper()
{
	if (_info.Converter)
		delete _info.Converter;

	if (_info.local_backend)
	{
		_info.local_backend->HandleDestroyJob(_info.job);
		delete _info.local_backend;
	}
}

const BackendWrapper::LocalBackendInfo DllBasedBackendWrapper::GetBackendInfo(const std::string clientId)
{
	return _info;
}

