#ifndef __LOCAL_COMM_H
#define __LOCAL_COMM_H

#include "../../Common/CommProtocol.h"
#include <map>

#include <fstream>

#include "../../Conversions/JsonWriter.h"
// Forward declaration
class BackendCaller;

// A simple communication static library that uses LoadLibrary and calls the 
// functions locally

class MetadataRepository;

class EXPORTED LocalFrontend : public FrontendComm {
protected:
	BackendCaller *_backendCaller;

	struct LocalJob {
		progressFunc progHandler;
		notifyCompletionFunc completionHandler;
		void *args;
	};
	std::map<JobPtr, LocalJob> _jobHandlers;

	MetadataRepository *_metadata;

private:
	bool _checkJobProgress;
	bool _checkJobOvertime;
	JobType											_localJobType = JT_NONE;
	std::vector<double>								_resultGraph;
	std::map<int, std::string>						_domainHeaders;
	int												_modelPtrCounter = 1; //modelPtr can't be zero
	ParameterTree									_fitResultTree;
	std::map<ModelPtr, std::vector<std::wstring>>	_fileMap; // A map containing models that have filenames (amplitude models)

	int _last_error_code;
	std::string _last_error_message;

	void SetGenerateResults(JobPtr job);
	void SetFitResults(JobPtr job);
	bool _isJobRunning;
	bool _is_valid;

public:
	LocalFrontend(BackendCaller *caller);
	virtual ~LocalFrontend();

	virtual bool IsValid();

	virtual std::string GetAllModelMetadata();

	// Query containers for models
	//////////////////////////////////////////////////////////////////////////
	virtual int QueryCategoryCount(const wchar_t *container);
	virtual int QueryModelCount(const wchar_t *container);
	virtual ModelCategory QueryCategory(const wchar_t *container, int catInd);
	virtual ModelInformation QueryModel(const wchar_t *container, int index);

	// Job Management
	//////////////////////////////////////////////////////////////////////////
	virtual JobPtr CreateJob(const wchar_t *description, progressFunc progHandler = NULL,
		notifyCompletionFunc completionHandler = NULL, void *args = NULL);
	virtual void DestroyJob(JobPtr job);


	// Model Management
	//////////////////////////////////////////////////////////////////////////
	virtual ModelPtr CreateModel(JobPtr job, const wchar_t *container, int modelIndex, EDProfile profile);
	virtual ModelPtr CreateCompositeModel(JobPtr job);
	virtual ModelPtr CreateDomainModel(JobPtr job);
	virtual ModelPtr CreateScriptedModel(JobPtr job, const char *script, const char * fileNm, unsigned int len);
	virtual ModelPtr CreateFileAmplitude(JobPtr job, AmpFileType type, const wchar_t **filename, int numFiles, bool bCenter);
	virtual ModelPtr CreateFileAmplitude(JobPtr job, AmpFileType type, const char **buffer, unsigned int *bufferSize, const char **fileNm, unsigned int *fnSize, int numFiles, bool bCenter);
	virtual ModelPtr CreateGeometricAmplitude(JobPtr job, ModelPtr model);
	virtual ModelPtr CreateSymmetry(JobPtr job, const wchar_t *container, int symmetryIndex);
	virtual ModelPtr CreateScriptedSymmetry(JobPtr job, const char *script, const char * fileNm, unsigned int len);
	virtual ErrorCode DestroyModel(JobPtr job, ModelPtr model, bool bDestroyChildren = false);
	virtual void AddFileToExistingModel(int modelPtr, const wchar_t *fileNm);
	virtual void RemoveFileFromExistingModel(int modelPtr, const wchar_t *fileNm);

	// Actions
	//////////////////////////////////////////////////////////////////////////
	virtual ErrorCode Fit(JobPtr job, const wchar_t *luaScript, const std::vector<double>& x,
		const std::vector<double>& y, const std::vector<int>& mask, bool UseGPU = true,std::string* message = nullptr);

	virtual ErrorCode Generate(JobPtr job, const wchar_t *luaScript, const std::vector<double>& x, bool UseGPU = true);

	virtual ErrorCode CheckCapabilities(bool checkTdr = true);

	virtual void Stop(JobPtr job);
	virtual void WaitForFinish(JobPtr job);

	// Retrieve data from backend
	//////////////////////////////////////////////////////////////////////////
	virtual bool GetLastErrorMessage(JobPtr job, wchar_t *message, size_t length);
	virtual JobType GetJobType(JobPtr job);
	virtual int  GetGraphSize(JobPtr job);
	virtual bool GetGraph(JobPtr job, OUT double *yPoints, int nPoints);
	virtual ErrorCode GetResults(JobPtr job, OUT ParameterTree& tree);
	virtual bool ExportAmplitude(JobPtr job, ModelPtr model, const wchar_t *filename);
	virtual bool SavePDB(JobPtr job, ModelPtr model, const wchar_t *filename);
	virtual void CheckJobProgress(JobPtr job);
	virtual bool CheckJobOvertime(JobPtr job, unsigned long long startTime, unsigned long long sec);
	
	virtual void MarkCheckJobProgressDone(JobPtr job);
	virtual JobStatus GetJobStatus(JobPtr job);

	// Retrieve model data
	//////////////////////////////////////////////////////////////////////////
	virtual ErrorCode GetLayerParamNames(const wchar_t *container, int index, OUT char **lpNames, int nlp);
	virtual ErrorCode GetDisplayParamInfo(const wchar_t *container, int index, OUT char **disp, int nDisp);
	virtual ErrorCode GetExtraParamInfo(const wchar_t *container, int index, OUT ExtraParam *ep, int nEP);
	virtual ErrorCode GetLayerInfo(const wchar_t *container, int index, int layerIndex, OUT char *layerName,
		OUT int *applicability, OUT double *defaultValues,
		int nlp);
	virtual ErrorCode GetDisplayParams(const wchar_t *container, int index, const paramStruct *params, OUT double *disp, int nDisp);

	virtual ErrorCode GetDomainHeader(JobPtr job, ModelPtr model, OUT char *header, OUT int &length);

	// Event handlers
	//////////////////////////////////////////////////////////////////////////
	virtual void HandleProgress(JobPtr job, double progress);
	virtual void HandleCompletion(JobPtr job, int code, const std::string& message);
};


#endif
