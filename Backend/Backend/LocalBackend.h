#ifndef LOCALBACKEND_H
#define LOCALBACKEND_H

#include "../../Common/CommProtocol.h"
#include <map>
#include <stdexcept>
#include <string>
#include "backend_exception.h"
class LocalBackendParameterTreeConverter;

class JsonWriter;

class EXPORTED_BE LocalBackend {

public:
	LocalBackend();
	virtual ~LocalBackend();


	// Get all the model metadata in one big swoop
	//////////////////////////////////////////////////////////////////////////
	// virtual std::string HandleGetAllModelMetadata();

	// Query container handlers
	//////////////////////////////////////////////////////////////////////////
	virtual int HandleQueryCategoryCount(const wchar_t *container);
	virtual int HandleQueryModelCount(const wchar_t *container);
	virtual ModelCategory HandleQueryCategory(const wchar_t *container, int catInd);
	virtual ModelInformation HandleQueryModel(const wchar_t *container, int index);

	// Job Management
	//////////////////////////////////////////////////////////////////////////
	virtual JobPtr HandleCreateJob(const wchar_t *description);
	virtual void HandleDestroyJob(JobPtr job);
	virtual JobStatus GetJobStatus(JobPtr job);

	// Setting Handlers
	//////////////////////////////////////////////////////////////////////////
	virtual ModelPtr HandleCreateModel(JobPtr job, const wchar_t *container, int modelIndex, EDProfile profile);
	virtual ModelPtr HandleCreateCompositeModel(JobPtr job);
	virtual ModelPtr HandleCreateDomainModel(JobPtr job);
	virtual ModelPtr HandleCreateScriptedModel(JobPtr job, const char *script, unsigned int len);
	virtual ModelPtr HandleCreateFileAmplitude(JobPtr job, AmpFileType type, const wchar_t *filename, bool bCenter, const wchar_t *anomfilename = NULL, bool electronPDB = false);
	virtual ModelPtr HandleCreateFileAmplitude(JobPtr job, AmpFileType type, const char *buffer, unsigned int bufferSize, const char *fileNm, unsigned int fnSize, bool bCenter, const char *anomfileNm = NULL, unsigned int anomfnSize = 0, bool electronPDB = false);
	virtual ModelPtr HandleCreateGeometricAmplitude(JobPtr job, ModelPtr model);
	virtual ModelPtr HandleCreateSymmetry(JobPtr job, const wchar_t *container, int symmetryIndex);
	virtual ModelPtr HandleCreateScriptedSymmetry(JobPtr job, const char *script, unsigned int len);
	virtual ErrorCode HandleDestroyModel(JobPtr job, ModelPtr model, bool bDestroyChildren = false);

	// Action Handlers
	//////////////////////////////////////////////////////////////////////////
	virtual ErrorCode HandleFit(JobPtr job, const ParameterTree& tree, const std::vector<double>& x,
		const std::vector<double>& y, const std::vector<int>& mask,
		const FittingProperties& fp);

	virtual ErrorCode HandleGenerate(JobPtr job, const ParameterTree& tree, const std::vector<double>& x,
		const FittingProperties& fp);

	virtual void HandleStop(JobPtr job);

	// Retrieve data from backend
	//////////////////////////////////////////////////////////////////////////
	virtual bool HandleGetLastErrorMessage(JobPtr job, wchar_t *message, size_t length);
	virtual JobType HandleGetJobType(JobPtr job);
	virtual int  HandleGetGraphSize(JobPtr job);
	virtual bool HandleGetGraph(JobPtr job, OUT double *yPoints, int nPoints);
	virtual ErrorCode HandleGetResults(JobPtr job, OUT ParameterTree& tree);
	virtual bool HandleGetFittingErrors(JobPtr job, OUT double *paramErrors,
		OUT double *modelErrors, int nMutParams, int nPoints);
	virtual std::string HandleGetAmplitude(JobPtr job, ModelPtr model);
	virtual std::string HandleGetAmplitude(JobPtr job, ModelPtr model, std::string filename);
	virtual std::string	HandleGetPDB(JobPtr job, ModelPtr stateModel, bool electron=false);

	// Retrieve model data
	//////////////////////////////////////////////////////////////////////////
	virtual ErrorCode HandleGetLayerParamNames(const wchar_t *container, int index, OUT char **lpNames, int nlp);
	virtual ErrorCode HandleGetDisplayParamInfo(const wchar_t *container, int index, OUT char **disp, int nDisp);
	virtual ErrorCode HandleGetExtraParamInfo(const wchar_t *container, int index, OUT ExtraParam *ep, int nEP);
	virtual ErrorCode HandleGetLayerInfo(const wchar_t *container, int index, int layerIndex, OUT char *layerName,
		OUT int *applicability, OUT double *defaultValues,
		int nlp);
	virtual ErrorCode HandleGetDisplayParams(const wchar_t *container, int index, const paramStruct *params, OUT double *disp, int nDisp);
	virtual ErrorCode HandleGetDomainHeader(JobPtr job, ModelPtr model, OUT char *header, int length);

	// Smaller interface methods
	//////////////////////////////////////////////////////////////////////////
	//virtual ErrorCode HandleGenerateJSON(JobPtr job, const std::string stateJSON);
	//virtual std::string HandleGetGenerateResultsJSON(JobPtr ptr);
	//virtual ErrorCode HandleFitJSON(JobPtr job, const std::string stateJSON);
	//virtual std::string HandleGetFittingResultsJSON(JobPtr ptr);

	// Frontend notifications
	//////////////////////////////////////////////////////////////////////////
	void NotifyProgress(JobPtr job, double progress);
	void NotifyCompletion(JobPtr job, bool bSuccess, int code, const std::string &);
};



#endif
