#include "LocalBackendParameterTree.h"
#include "DllBasedBackendWrapper.h"
#include "LocalBackend.h"
#include "AmplitudeCache.h"

/**
 * @file DllBasedBackendWrapper.cpp
 * @brief Implements the DllBasedBackendWrapper class, which provides a DLL-based interface to the backend,
 *        managing backend initialization, job lifecycle, and parameter tree conversion for external clients.
 *
 * The DllBasedBackendWrapper module is responsible for:
 *  - Defining the DllBasedBackendWrapper class, which encapsulates the backend logic for use in a DLL context.
 *  - Managing the lifecycle of the local backend, including job creation and destruction.
 *  - Providing access to backend information and parameter tree conversion for client applications.
 *  - Initializing and cleaning up amplitude cache resources for efficient amplitude calculations.
 *  - Ensuring proper memory management and resource cleanup for backend objects.
 *
 * Key Concepts:
 *  - DLL Backend Wrapper: Provides a simplified interface for external clients (e.g., .NET or Python) to interact with the backend.
 *  - Backend Initialization: Handles creation of the LocalBackend, job, and parameter tree converter on construction.
 *  - Resource Management: Ensures all backend resources are properly released on destruction.
 *  - Amplitude Cache: Initializes the amplitude cache for the current job and parameter tree converter.
 *
 * Fields (in BackendWrapper::LocalBackendInfo struct, as used by DllBasedBackendWrapper):
 *  - LocalBackend* local_backend: Pointer to the LocalBackend instance used for all backend operations.
 *  - JobPtr job: Handle to the current backend job, used for model and calculation context.
 *  - LocalBackendParameterTreeConverter* Converter: Pointer to the parameter tree converter for mapping frontend and backend models.
 *
 * Main Methods:
 *  - DllBasedBackendWrapper (constructor): Initializes the backend, creates a job, and sets up the parameter tree converter and amplitude cache.
 *  - ~DllBasedBackendWrapper (destructor): Cleans up the parameter tree converter, destroys the job, and deletes the backend instance.
 *  - GetBackendInfo: Returns the current LocalBackendInfo struct, providing access to backend, job, and converter for clients.
 *
 * Threading and Safety:
 *  - This class is not inherently thread-safe; external synchronization is required if used concurrently.
 *  - Backend resources are managed per instance of the wrapper.
 *
 * Error Handling:
 *  - Ensures all dynamically allocated resources are deleted in the destructor to prevent memory leaks.
 *  - Assumes successful creation of backend components; errors in construction should be handled by the caller.
 *
 * Dependencies:
 *  - LocalBackend for backend operations and job management.
 *  - LocalBackendParameterTreeConverter for parameter tree conversion and model mapping.
 *  - AmplitudeCache for amplitude calculation caching.
 *  - BackendWrapper::LocalBackendInfo struct for encapsulating backend state.
 *
 * See DllBasedBackendWrapper.h for class and method declarations and the definition of LocalBackendInfo.
 */

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

