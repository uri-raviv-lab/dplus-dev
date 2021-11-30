#ifndef __LOCALBACKEND_PARAMETERTREE
#define __LOCALBACKEND_PARAMETERTREE

#include <map>
#include <vector>
#include <set>
#include "Amplitude.h"
#include "../../Conversions/ParamTreeConversions.h"
#include "../../BackendCommunication/LocalCommunication/LocalComm.h"

class LocalBackend;

class EXPORTED_BE LocalBackendParameterTreeConverter : public ParameterTreeConverter
{
public:
	LocalBackendParameterTreeConverter(LocalBackend *backend, JobPtr ptr);	
	~LocalBackendParameterTreeConverter() {}

	ModelPtr StateToInternal(ModelPtr stateModel) const;
	ModelPtr InternalToState(ModelPtr internalModel) const;
	ModelPtr StateFromAmp(Amplitude * ampl) const;


	std::vector<ModelPtr> GetStateModels();

	ParameterTree FromStateJSON(const rapidjson::Value &json);  // Convert a state file (in JSON format) to a ParameterTree and create all the models

protected:
	ModelPtr CreateCompositeModel(ModelPtr stateModel);
	ModelPtr CreateDomainModel(ModelPtr stateModel);
	ModelPtr CreateModelFromJSON(const std::string &str, const rapidjson::Value &model, bool bAmp, ModelPtr stateModel);
	ModelPtr GetStateModelPtr(ModelPtr internalModel) { return InternalToState(internalModel); }
	
	
	std::map<ModelPtr, ModelPtr> _stateToInternal;
	std::map<ModelPtr, ModelPtr> _internalToState;

private:
	LocalBackend			*_pBackend;
	JobPtr					_job;
	std::set<ModelPtr>	    _usedModels;

	// Update the state<-->internal maps
	void MapModel(ModelPtr stateModel, ModelPtr internalModel);
	void ClearUnusedModels();

	ModelPtr	ModelFromJsonString(const std::string &str, const rapidjson::Value &model, bool bAmp);
};

#endif
