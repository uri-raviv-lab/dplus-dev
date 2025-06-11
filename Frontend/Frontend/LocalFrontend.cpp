#include "LocalComm.h"
#include "MetadataRepository.h"

#include "LUAConversions.h"
#include "ParamTreeConversions.h"
#include "Conversions.h"

#include "BackendCalls.h"
#include "BackendCallers.h"

#include <iostream>     // std::cout, std::ostream, std::ios
#include <fstream>      // std::filebuf
#include <locale>
#include <codecvt>
#include <Windows.h>


using namespace rapidjson;
using namespace std;

/**
 * @file LocalFrontend.cpp
 * @brief Implements the LocalFrontend class, which provides the frontend logic for managing, executing,
 *        and querying computational jobs, models, and amplitudes in a local (non-remote) context.
 *
 * The LocalFrontend is responsible for:
 *  - Creating, destroying, and managing jobs and their associated models and amplitudes.
 *  - Handling model instantiation, including composite, domain, scripted, symmetry, and file-based models.
 *  - Managing the lifecycle of models and amplitudes, including file associations and resource cleanup.
 *  - Executing fitting and generation jobs, and delegating job execution to the backend via BackendCaller.
 *  - Providing access to job results, graphs, fitting errors, and metadata.
 *  - Handling progress and completion notifications for jobs.
 *  - Querying and returning model, category, and parameter information from loaded containers.
 *  - Supporting file and stream export of amplitude and PDB data.
 *  - Ensuring error handling and job status tracking for all frontend operations.
 *
 * Key Concepts:
 *  - Each job is managed locally and can contain multiple models and amplitudes.
 *  - Models and amplitudes are created, referenced, and destroyed through the frontend interface.
 *  - The frontend supports both synchronous and asynchronous job execution.
 *  - All operations are validated for job and model existence, with detailed error reporting.
 *  - The frontend is designed to be extensible for new model types and data sources.
 *
 * Fields:
 *  - BackendCaller* _backendCaller: Pointer to the backend caller used to communicate with the backend for all operations.
 *  - MetadataRepository* _metadata: Holds all model and parameter metadata loaded from the backend.
 *  - bool _is_valid: Indicates whether the frontend is in a valid state (metadata loaded successfully).
 *  - bool _isJobRunning: Tracks whether a job is currently running.
 *  - bool _checkJobProgress: Internal flag to prevent concurrent progress checks.
 *  - bool _checkJobOvertime: Internal flag to prevent concurrent overtime checks.
 *  - std::map<JobPtr, LocalJob> _jobHandlers: Maps job handles to their associated progress and completion handlers.
 *  - std::map<ModelPtr, std::vector<std::wstring>> _fileMap: Associates models with their corresponding file names.
 *  - std::map<ModelPtr, std::string> _domainHeaders: Stores domain headers for models after generation.
 *  - std::vector<double> _resultGraph: Stores the result graph (e.g., intensity profile) for the last completed job.
 *  - ParameterTree _fitResultTree: Stores the parameter tree resulting from the last fit operation.
 *  - std::string _last_error_message: Stores the last error message returned from the backend.
 *  - ErrorCode _last_error_code: Stores the last error code returned from the backend.
 *  - JobType _localJobType: Tracks the type of the currently running job (fit, generate, etc.).
 *  - ModelPtr _modelPtrCounter: Counter for generating unique model handles.
 *
 * Main Methods:
 *  - LocalFrontend (constructor/destructor): Initializes the frontend, loads metadata, and sets up backend communication.
 *  - IsValid: Checks if the frontend is in a valid state.
 *  - CreateJob / DestroyJob: Manage job lifecycle and handler registration.
 *  - CreateModel / CreateCompositeModel / CreateDomainModel / CreateScriptedModel / CreateFileAmplitude / CreateGeometricAmplitude / CreateSymmetry / CreateScriptedSymmetry: Model instantiation and file association.
 *  - DestroyModel: Safely destroys models and removes file associations.
 *  - Fit / Generate: Initiate fitting and generation jobs, handling result and error management.
 *  - CheckCapabilities: Queries backend for system capabilities.
 *  - Stop / WaitForFinish: Stops or waits for job completion.
 *  - GetLastErrorMessage: Retrieves the last error message for a job.
 *  - GetJobType / GetGraphSize / GetGraph / GetResults: Retrieve job status, results, and output data.
 *  - GetLayerParamNames / GetDisplayParamInfo / GetExtraParamInfo / GetLayerInfo / GetDisplayParams: Query model and parameter metadata.
 *  - GetJobStatus / CheckJobProgress / CheckJobOvertime / MarkCheckJobProgressDone: Job status and progress management.
 *  - HandleProgress / HandleCompletion: Internal handlers for job progress and completion notifications.
 *  - ExportAmplitude / SavePDB: Export amplitude and PDB data to files.
 *  - GetDomainHeader: Retrieves the domain header for a model.
 *  - SetGenerateResults / SetFitResults: Internal methods to update results after job completion.
 *
 * Threading and Safety:
 *  - Job and model operations are managed locally; thread safety is not guaranteed and should be managed externally if needed.
 *  - Internal flags prevent concurrent progress and overtime checks.
 *
 * Error Handling:
 *  - All methods validate input and job/model existence, returning error codes or updating error messages as appropriate.
 *  - Error codes and messages are propagated to the caller for user feedback.
 *
 * See LocalFrontend.h for class and method declarations.
 */

LocalFrontend::LocalFrontend(BackendCaller *caller) {
	_checkJobProgress = false;
	_checkJobOvertime = false;
	_backendCaller = caller;
	std::string called_metadata = GetAllModelMetadata();
	if (called_metadata == "")
	{
		_is_valid = false;
		_metadata = NULL;
	}
	else
	{
		_metadata = new MetadataRepository(called_metadata);
		_is_valid = true;
	}
	_isJobRunning = false;
}

bool LocalFrontend::IsValid() {
	return _is_valid;
}

LocalFrontend::~LocalFrontend() {
	if (_metadata)
		delete _metadata;
}


std::string LocalFrontend::GetAllModelMetadata()
{
	GetAllModelMetadataCall call;
	_backendCaller->CallBackend(call);

	return call.GetMetadata();
}

int LocalFrontend::QueryCategoryCount(const wchar_t *container) {
	return _metadata->QueryCategoryCount(container);
}

int LocalFrontend::QueryModelCount(const wchar_t *container) {
	return _metadata->QueryModelCount(container);
}

ModelCategory LocalFrontend::QueryCategory(const wchar_t *container, int catInd) {
	return _metadata->QueryCategory(container, catInd);
}

ModelInformation LocalFrontend::QueryModel(const wchar_t *container, int index) {
	return _metadata->QueryModel(container, index);
}

JobPtr LocalFrontend::CreateJob(const wchar_t *description, progressFunc progHandler,
	notifyCompletionFunc completionHandler, void *args) {

	JobPtr res = 1; //(jobs no longer stored in _backend)
	LocalJob lJob;
	lJob.progHandler = progHandler;
	lJob.completionHandler = completionHandler;
	lJob.args = args;

	_jobHandlers[res] = lJob;
	return res;
}

void LocalFrontend::DestroyJob(JobPtr job) {
	//do nothing (jobs no longer stored in _backend)
}

ModelPtr LocalFrontend::CreateModel(JobPtr job, const wchar_t *container,
	int modelIndex, EDProfile profile)
{
	return _modelPtrCounter++;
}

ModelPtr LocalFrontend::CreateCompositeModel(JobPtr job)
{
	return _modelPtrCounter++;
}

ModelPtr LocalFrontend::CreateDomainModel(JobPtr job) {
	return _modelPtrCounter++;
}

ModelPtr LocalFrontend::CreateScriptedModel(JobPtr job, const char *script, const char * fileNm, unsigned int len)
{
	ModelPtr newModelPtr = _modelPtrCounter++;

	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	std::wstring wide = converter.from_bytes(fileNm);
	_fileMap[newModelPtr] = std::vector<std::wstring> {wide};

	return newModelPtr++;
}
ModelPtr LocalFrontend::CreateFileAmplitude(JobPtr job, AmpFileType type, const wchar_t **fileNm, int numFiles, bool bCenter)
{
	/*
	ModelPtr newModelPtr = _modelPtrCounter++;
	_fileMap[newModelPtr] = filename;

	return newModelPtr;/**/

	ModelPtr newModelPtr = _modelPtrCounter++;
	std::vector<std::wstring> filenames(numFiles);
	for (int i = 0; i < numFiles; i++)
		filenames[i] = fileNm[i];
	_fileMap[newModelPtr] = filenames;
	return newModelPtr++;
}

void LocalFrontend::AddFileToExistingModel(int modelPtr, const wchar_t *fileNm)
{
	_fileMap[modelPtr].push_back(fileNm);
}

void LocalFrontend::RemoveFileFromExistingModel(int modelPtr, const wchar_t *fileNm)
{
	std::vector<std::wstring>::iterator position = std::find(_fileMap[modelPtr].begin(), _fileMap[modelPtr].end(), fileNm);
	if (position != _fileMap[modelPtr].end())
		_fileMap[modelPtr].erase(position);
}

ModelPtr LocalFrontend::CreateFileAmplitude(JobPtr job, AmpFileType type, const char **buffer, unsigned int *bufferSize, const char **fileNm, unsigned int *fnSize, int numFiles, bool bCenter)
{
	ModelPtr newModelPtr = _modelPtrCounter++;

	std::vector<std::wstring> filenames(numFiles);
	for (int i = 0; i < numFiles; i++)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		filenames[i] = converter.from_bytes(fileNm[i]);
	}
	_fileMap[newModelPtr] = filenames;

	return newModelPtr++;
}

ModelPtr LocalFrontend::CreateGeometricAmplitude(JobPtr job, ModelPtr model)
{
	return _modelPtrCounter++;
}

ModelPtr LocalFrontend::CreateSymmetry(JobPtr job, const wchar_t *container, int symmetryIndex)
{
	return _modelPtrCounter++;
}

ModelPtr LocalFrontend::CreateScriptedSymmetry(JobPtr job, const char *script, const char * fileNm,  unsigned int len)
{
	ModelPtr newModelPtr = _modelPtrCounter++;

	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	std::wstring wide = converter.from_bytes(fileNm);
	_fileMap[newModelPtr] = std::vector<std::wstring> {wide};

	return newModelPtr++;
}

ErrorCode LocalFrontend::DestroyModel(JobPtr job, ModelPtr model, bool bDestroyChildren)
{
	auto modelIterator = _fileMap.find(model);
	if (modelIterator != _fileMap.end())
		_fileMap.erase(modelIterator);
	return OK;
}

ErrorCode LocalFrontend::Fit(JobPtr job, const wchar_t *luaScript, const std::vector<double>& x,
	const std::vector<double>& y, const std::vector<int>& mask, bool useGPU, std::string* message)
{
	ErrorCode	retErr;
	_localJobType = JT_NONE;

	StartFitCall call(luaScript, x, y, mask, _fileMap, useGPU);
	_backendCaller->CallBackend(call);

	//parse out error code from returned json
	retErr = call.GetErrorCode();
	_last_error_code = retErr;
	_last_error_message = call.GetErrorMessage();

	if (OK == retErr)
	{
		_isJobRunning = true;
		_localJobType = JT_FIT;
		return OK;
	}

	return retErr;
}


ErrorCode LocalFrontend::Generate(JobPtr job, const wchar_t *luaScript, const std::vector<double> &x, bool UseGPU)
{
	ErrorCode	retErr;
	_localJobType = JT_NONE;

	StartGenerateCall call(luaScript, x, _fileMap, UseGPU);
	_backendCaller->CallBackend(call);

	retErr = call.GetErrorCode();
	_last_error_code = retErr;
	_last_error_message = call.GetErrorMessage();

	if (OK == retErr)
	{
		_isJobRunning = true;
		_localJobType = JT_GENERATE;
		return OK;
	}

	return retErr;
}

ErrorCode LocalFrontend::CheckCapabilities(bool checkTdr)
{

	// The system should check the capabilities of the server computer just if the backend is local (and not remote)
	LocalBackendCaller* local = dynamic_cast<LocalBackendCaller*> (_backendCaller);
	ManagedBackendCaller* manage = dynamic_cast<ManagedBackendCaller*> (_backendCaller);
	if (local || manage->Python())
	{
		CheckCapabilitiesCall call(checkTdr);
		_backendCaller->CallBackend(call);
		ErrorCode	retErr;
		retErr = call.GetErrorCode();
		_last_error_code = retErr;
		_last_error_message = call.GetErrorMessage();

		return retErr;
	}
	return OK;
}


void LocalFrontend::Stop(JobPtr job)
{
	StopCall call;
	_backendCaller->CallBackend(call);

	_localJobType = JT_NONE;
	_isJobRunning = false;
	HandleCompletion(job, ERROR_STOPPED, "");
}


void LocalFrontend::WaitForFinish(JobPtr job) {

	//simulate the original _backend's 'HandleWaitForFinish' function that used to block until job was finished..
	while (_isJobRunning)
	{
		Sleep(1000);
	}

}

bool LocalFrontend::GetLastErrorMessage(JobPtr job, wchar_t *message, size_t length)
{
	// Assumes ASCII, dangerous?
	std::wstring w_last_message(_last_error_message.begin(), _last_error_message.end());
	wcsncpy(message, w_last_message.c_str(), length);
	return true;

	//last error message not implemented.
	return false;
}

JobType LocalFrontend::GetJobType(JobPtr job) {
	return _localJobType;
}

int LocalFrontend::GetGraphSize(JobPtr job) {
	return int(_resultGraph.size());
}

bool LocalFrontend::GetGraph(JobPtr job, OUT double *yPoints, int nPoints) {

	for (int i = 0; i < nPoints; ++i)
	{
		yPoints[i] = _resultGraph[i];
	}
	return true;
}

ErrorCode LocalFrontend::GetResults(JobPtr job, OUT ParameterTree& tree) {

	//just return the result tree we stored on completion
	tree = _fitResultTree;
	return OK;
}

ErrorCode LocalFrontend::GetLayerParamNames(const wchar_t *container, int index, OUT char **lpNames, int nlp) {
	return _metadata->GetLayerParamNames(container, index, lpNames, nlp);
}

ErrorCode LocalFrontend::GetDisplayParamInfo(const wchar_t *container, int index, OUT char **disp, int nDisp) {
	// DisplayParamInfo is not supported in DPlus
	assert(false);
	return ERROR_BACKEND;
}

ErrorCode LocalFrontend::GetExtraParamInfo(const wchar_t *container, int index, OUT ExtraParam *ep, int nEP) {
	return _metadata->GetExtraParamInfo(container, index, ep, nEP);
}

ErrorCode LocalFrontend::GetLayerInfo(const wchar_t *container, int index, int layerIndex,
	OUT char *layerName, OUT int *applicability,
	OUT double *defaultValues, int nlp) {
	return _metadata->GetLayerInfo(container, index, layerIndex, layerName, applicability, defaultValues, nlp);
}


ErrorCode LocalFrontend::GetDisplayParams(const wchar_t *container, int index, const paramStruct *params, OUT double *disp, int nDisp) {
	assert(false); // Not supported in DPlus
	return ERROR_BACKEND;
}

JobStatus LocalFrontend::GetJobStatus(JobPtr job)
{
	GetJobStatusCall call;
	_backendCaller->CallBackend(call, true);

	return call.GetJobStatus();
}

void  LocalFrontend::CheckJobProgress(JobPtr job)
{

	if (_checkJobProgress)
		return;
	if (!_isJobRunning)
		return;

	_checkJobProgress = true;
	JobStatus jobStatus = GetJobStatus(job);
	_checkJobProgress = false;

	if (jobStatus.isRunning)
	{
		// Job is still running
		HandleProgress(job, jobStatus.progress);
	}
	else
	{
		// Job had been running previous, since _isJobRunning is true
		_isJobRunning = false; //unblocks 'WaitForFinish'
		HandleCompletion(job, jobStatus.code, jobStatus.code_string);
	}
}
bool  LocalFrontend::CheckJobOvertime(JobPtr job, unsigned long long startTime, unsigned long long sec) {

	if (_checkJobOvertime)
		return true;
	if (!_isJobRunning)
		return true;

	_checkJobOvertime = true;
	JobStatus jobStatus = GetJobStatus(job);
	

	if (jobStatus.isRunning && GetJobType(job) == JT_FIT)
	{
		// Job is still running
		unsigned long long current_time = (unsigned long long)time(NULL);
		unsigned long long runtime = current_time - startTime;
		if (runtime > sec) {
			_checkJobOvertime = false;
			return false;
		}
	}
	_checkJobOvertime = false;
	return true;

}
void LocalFrontend::MarkCheckJobProgressDone(JobPtr job)
{
	_checkJobProgress = false;
}

void LocalFrontend::HandleProgress(JobPtr job, double progress) {
	if (job && _jobHandlers.find(job) != _jobHandlers.end()) {
		if (_jobHandlers[job].progHandler)
			_jobHandlers[job].progHandler(_jobHandlers[job].args, progress);
	}
}

void LocalFrontend::HandleCompletion(JobPtr job, int code, const std::string& message) {
	_last_error_message = message;
	ErrorCode e = OK;
	if (job && _jobHandlers.find(job) != _jobHandlers.end())
	{
		if (JT_GENERATE == GetJobType(job))
		{
			e = SetGenerateResults(job);
		}

		if (JT_FIT == GetJobType(job))
		{
			e = SetFitResults(job);
		}
		if (_jobHandlers[job].completionHandler)
		{
			int choose_code = e == OK ? code : e;
			_jobHandlers[job].completionHandler(_jobHandlers[job].args, choose_code);
		}

	}
}

bool LocalFrontend::ExportAmplitude(JobPtr job, ModelPtr model, const wchar_t *filename)
{
	GetAmplitudeCall call(model, filename);
	_backendCaller->CallBackend(call);
	ErrorCode retErr;
	retErr = call.GetErrorCode();
	if (OK == retErr)
	{
		return true;
	}
	return false;
}


bool LocalFrontend::SavePDB(JobPtr job, ModelPtr model, const wchar_t *filename)
{
	GetPDBCall call(model, filename);
	_backendCaller->CallBackend(call);

	ErrorCode retErr;
	retErr = call.GetErrorCode();
	if (OK == retErr)
	{
		return true;
	}
	return false;
}

ErrorCode LocalFrontend::GetDomainHeader(JobPtr job, ModelPtr model, OUT char *header, OUT int &length)
{
	// TODO: Look for model in domainHEaders, if it doesn't exist, return an error code (use domainHeaders.find)
	if (_domainHeaders.find(model) == _domainHeaders.end())
	{
		return ERROR_MODELNOTFOUND;
	}
	strncpy(header, _domainHeaders[model].c_str(), length);
	return OK;
}

ErrorCode LocalFrontend::SetGenerateResults(JobPtr job)
{
	GetGenerateResultsCall call;
	_backendCaller->CallBackend(call);

	ErrorCode	retErr;
	retErr = call.GetErrorCode();

	_last_error_code = retErr;
	_last_error_message = call.GetErrorMessage();
	if (OK != retErr)
		return retErr;

	_domainHeaders = call.GetDomainHeaders();
	_resultGraph = call.GetGraph();
	return retErr;

}

ErrorCode LocalFrontend::SetFitResults(JobPtr job)
{
	GetFitResultsCall call;
	_backendCaller->CallBackend(call);

	ErrorCode	retErr;
	retErr = call.GetErrorCode();
	_last_error_code = retErr;
	_last_error_message = call.GetErrorMessage();

	if (OK != retErr)
		return retErr;

	_fitResultTree = call.GetParameterTree();
	_resultGraph = call.GetGraph();
	return retErr;
}
