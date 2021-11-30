#include <rapidjson/document.h>

#include <rapidjson/stringbuffer.h>

#include "ParamTreeConversions.h"

#include <sstream>
#include <stdexcept>

using namespace std;
using namespace rapidjson;


// Default model creation functions all return 0. Derived classes should implement these better
ModelPtr ParameterTreeConverter::CreateCompositeModel(ModelPtr stateModel)
{
	return stateModel;
}

ModelPtr ParameterTreeConverter::CreateDomainModel(ModelPtr stateModel)
{
	return stateModel;
}

ModelPtr ParameterTreeConverter::CreateModelFromJSON(const std::string &str, const rapidjson::Value &model, bool bAmp, ModelPtr stateModel)
{
	return stateModel;
}

ModelPtr ParameterTreeConverter::GetStateModelPtr(ModelPtr internalModel)
{
	return internalModel;
}


/****************************************************************************************************
	Convert from State JSON to a ParameterTree
*****************************************************************************************************/

ParameterTree ParameterTreeConverter::FromStateJSON(const rapidjson::Value &json)
{
	bool bAmplitude = false;
	ParameterTree pt;

	const rapidjson::Value &dom = json.FindMember("Domain")->value;

	std::string type = dom["Geometry"].GetString();

	if (!type.size())
	{
		throw std::invalid_argument("Invalid parameter tree! Root incorrect");
	}

	double domscale = 1.0;
	bool domscalemut = false;

	double domconstant = 0.0;
	bool domconstantmut = false;

	domscale = dom["Scale"].GetDouble();
	domscalemut = dom["ScaleMut"].GetBool();
	
	if (dom.HasMember("Constant")) // For compatibility with existing files that haven't added the constant
	{
		domconstant = dom["Constant"].GetDouble();
		domconstantmut = dom["ConstantMut"].GetBool();
	}
	//before starting to create models, clear the 'inUse' flag from all models


	//create the composite model and add it as root of parameter tree
	int model = dom["ModelPtr"].GetInt();
	ModelPtr compositeModel = CreateCompositeModel(model);
	pt.SetNodeModel(compositeModel);

	ParameterTree *domain = &pt;

	if (type == "Single Geometry") {
		bAmplitude = false;

		Value::ConstMemberIterator mods = dom.FindMember("Models");

		AddModelsToParamTree(domain, mods, bAmplitude);

		// Set scale and population size parameters (default values)
		paramStruct ps;
		ps.nlp = 2 + 1; // THE NUMBER OF PARAMS + NUMBER OF SUBDOMAINS
		ps.layers = 1;
		ps.params.resize(ps.nlp);
		ps.params[0].push_back(Parameter(domscale, domscalemut, true, std::numeric_limits<double>::denorm_min())); // SCALE
		ps.params[1].push_back(Parameter(domconstant, domconstantmut)); // CONSTANT
		ps.params[2].push_back(Parameter(1.0, false)); // AVERAGE POPULATION SIZE

		///the following line was missing in DPlus.. is it a bug?
		pt.SetNodeParameters(ps);
	}
	else if (type  == "Domains")
	{
		bAmplitude = true;

		vector<double> popsizes;
		vector<bool> popsizemut;

		const rapidjson::Value &pops = dom.FindMember("Populations")->value;

		// No populations
		if (pops.Empty()) {

			throw std::invalid_argument("Invalid parameter tree! No populations");
		}

		int i = 0;
		for (auto pop = pops.Begin(); pop != pops.End(); pop++, i++)
		{
			// DomainPreferencesFromJson must be before CreateDomainModel because it fills iterationMethod field
			paramStruct domps = DomainPreferencesFromJson(json);
			//get the ModelPtr from the state file and add to map
			ModelPtr stateModelPtr = (pop->FindMember("ModelPtr"))->value.GetInt();
			ModelPtr model = CreateDomainModel(stateModelPtr);
			
			
			domain = pt.AddSubModel(model, domps);

			//Add models for each population to the tree	
			Value::ConstMemberIterator mods = pop->FindMember("Models");
			AddModelsToParamTree(domain, mods, bAmplitude);


			popsizes.push_back(pop->FindMember("PopulationSize")->value.GetDouble());
			popsizemut.push_back(pop->FindMember("PopulationSizeMut")->value.GetBool());
		}

		// Set scale and population size parameters.
		paramStruct ps;
		ps.nlp = 2 + pops.Size(); // THE NUMBER OF PARAMS + NUMBER OF SUBDOMAINS
		ps.layers = 1;
		ps.params.resize(ps.nlp);
		ps.params[0].push_back(Parameter(domscale, domscalemut, true, std::numeric_limits<double>::denorm_min())); // SCALE
		ps.params[1].push_back(Parameter(domconstant, domconstantmut)); // CONSTANT
		for (size_t i = 0; i < pops.Size(); i++)
			ps.params[i + 1].push_back(Parameter(popsizes[i], popsizemut[i], true, 0.0)); // AVERAGE POPULATION SIZE

		pt.SetNodeParameters(ps);
	}
	else {
		throw std::invalid_argument("Invalid parameter tree! Invalid root type");
	}
	
	return pt;
}

void ParameterTreeConverter::AddModelsToParamTree(ParameterTree *domain, const rapidjson::Value::ConstMemberIterator mods, bool bAmplitude)
{
	if (mods->value.Empty())
	{
		throw std::invalid_argument("didn't find Models in at least one of the populations");
	}

	if (!bAmplitude && mods->value.Size() > 1) {
		throw std::invalid_argument("Invalid parameter tree! Single geometry must be single");
	}

	int numSubmodels = 0;
	for (auto itr = mods->value.Begin(); itr != mods->value.End(); ++itr)
	{
		// Model table in STATE
		paramStruct ps = ModelParamsFromJson(*itr);

		ModelPtr model = 0; // NULL gives a warning in gcc

		string typeStr = (itr->FindMember("Type"))->value.GetString();
		ModelPtr stateModelPtr = (itr->FindMember("ModelPtr"))->value.GetInt();
		model = CreateModelFromJSON(typeStr, *itr, bAmplitude, stateModelPtr);

		//create a domain model for each model
		domain->AddSubModel(model, ps);

		// Now take care of the children
		rapidjson::Value::ConstMemberIterator itrChildren = itr->FindMember("Children");
		if (itrChildren!=itr->MemberEnd())
		{
			ParameterTree *submodel = domain->GetSubModel(numSubmodels);
			AddModelsToParamTree(submodel, itrChildren, bAmplitude);  // Recursively add children
		}
		numSubmodels++;
	}

	if (!bAmplitude) {
		if (domain->GetNumSubModels() == 0) // Add an empty form factor, if required
			domain->AddSubModel(0);
	}
}


paramStruct ParameterTreeConverter::ModelParamsFromJson(const rapidjson::Value &model)
{
	paramStruct ps;

	Value::ConstMemberIterator Parameters = model.FindMember("Parameters");
	assert(Parameters->value.IsArray()); //TODO: return an error here instead of just crashing the bloody program
	ps.layers = Parameters->value.Size();

	if (Parameters->value.Size() > 0)
	{
		ps.nlp = Parameters->value[0].Size();
	}

	Value::ConstMemberIterator ExtraParameters = model.FindMember("ExtraParameters");
	assert(ExtraParameters->value.IsArray());
	ps.nExtraParams = ExtraParameters->value.Size();

	Value::ConstMemberIterator UseGrid = model.FindMember("Use_Grid");
	assert(UseGrid->value.IsBool());
	ps.bSpecificUseGrid = UseGrid->value.GetBool();

	// Create the vectors
	ps.params.resize(ps.nlp);
	ps.extraParams.resize(ps.nExtraParams);
	for (int i = 0; i < ps.nlp; i++)
		ps.params[i].resize(ps.layers);

	Value::ConstMemberIterator Mutables = model.FindMember("Mutables");
	Value::ConstMemberIterator Constraints = model.FindMember("Constraints");
	Value::ConstMemberIterator Sigma = model.FindMember("Sigma");

	for (int layer = 0; layer < ps.layers; layer++)
	{

		//Value::ConstValueIterator layerv = Parameters->value[layer];		//doesn't compile
		Value::ConstValueIterator layerv = Parameters->value.Begin() + layer;
		Value::ConstValueIterator layerm = Mutables->value.Begin() + layer;
		Value::ConstValueIterator layerc = Constraints->value.Begin() + layer;
		Value::ConstValueIterator layersig = Sigma->value.Begin() + layer;

		// Setting Parameters
		for (int i = 0; i < ps.nlp; i++)
		{
			double val = 0.0;
			bool bMutable = false;
			bool bCons = false;
			double consMin = NEGINF, consMax = POSINF;
			int consMinInd = -1, consMaxInd = -1, linkInd = -1;
			double sigma = 0.0;

			//val = Parameters->value[layer][i].GetDouble();
			auto layerv_i = layerv->Begin() + i;
			val = layerv_i->GetDouble();
			bMutable = Mutables->value[layer][i].GetBool();
			sigma = Sigma->value[layer][i].GetDouble();

			if (!layerc->Empty())
			{

				Value::ConstValueIterator layerc_i = layerc->Begin() + i;
				loadConstraints(layerc_i, consMin, consMax, consMinInd, consMaxInd, linkInd, bCons);

			}
			ps.params[i][layer] = Parameter(val, bMutable, bCons, consMin, consMax,
				consMinInd, consMaxInd, linkInd, sigma);
		}
	}

	// Extra parameters
	Value::ConstMemberIterator ExtraMutables = model.FindMember("ExtraMutables");
	assert(ExtraMutables->value.IsArray());
	Value::ConstMemberIterator ExtraConstraints = model.FindMember("ExtraConstraints");
	assert(ExtraConstraints->value.IsArray());
	Value::ConstMemberIterator ExtraSigma = model.FindMember("ExtraSigma");
	assert(ExtraSigma->value.IsArray());

	for (int i = 0; i < ps.nExtraParams; i++) {
		double val = 0.0;
		bool bMutable = false;
		bool bCons = false;
		double consMin = NEGINF, consMax = POSINF;
		int consMinInd = -1, consMaxInd = -1, linkInd = -1;
		double sigma = 0.0;


		val = jsonObjectToDouble(ExtraParameters->value[i]);
		bMutable = ExtraMutables->value[i].GetBool();
		sigma = jsonObjectToDouble(ExtraSigma->value[i]);

		Value::ConstValueIterator ec = ExtraConstraints->value.Begin() + i;
		loadConstraints(ec, consMin, consMax, consMinInd, consMaxInd, linkInd, bCons);

		ps.extraParams[i] = Parameter(val, bMutable, bCons, consMin, consMax,
			consMinInd, consMaxInd, linkInd, sigma);
	}


	ps.x = LoadLocationParams(model, "x");
	ps.y = LoadLocationParams(model, "y");
	ps.z = LoadLocationParams(model, "z");
	ps.alpha = LoadLocationParams(model, "alpha");
	ps.beta = LoadLocationParams(model, "beta");
	ps.gamma = LoadLocationParams(model, "gamma");

	return ps;
}


paramStruct ParameterTreeConverter::DomainPreferencesFromJson(const rapidjson::Value &doc)
{
	paramStruct ps;
	Value::ConstMemberIterator it;

	const rapidjson::Value &domainPrefs = doc.FindMember("DomainPreferences")->value;

	ps.nlp = 7; // THE NUMBER OF PREFERENCES
	ps.layers = 1;
	ps.params.resize(ps.nlp);

	it = domainPrefs.FindMember("OrientationIterations");
	ps.params[0].push_back(Parameter(it->value.GetDouble()));

	it = domainPrefs.FindMember("GridSize");
	ps.params[1].push_back(Parameter(it->value.GetDouble()));

	it = domainPrefs.FindMember("UseGrid");
	ps.params[2].push_back(Parameter(it->value.GetBool() ? 1.0 : 0.0));

	it = domainPrefs.FindMember("Convergence");
	ps.params[3].push_back(Parameter(it->value.GetDouble()));

	it = domainPrefs.FindMember("qMax");
	ps.params[4].push_back(Parameter(it->value.GetDouble()));

	it = domainPrefs.FindMember("OrientationMethod");
	iterationMethod = OAMethodfromCString(it->value.GetString());
	ps.params[5].push_back(Parameter(iterationMethod));

	it = domainPrefs.FindMember("qMin");
	ps.params[6].push_back(Parameter(it->value.GetDouble()));

	return ps;
}


string ParameterTreeConverter::GetScript(const rapidjson::Value &model)
{
	string scr = GetFilename(model);
	ifstream inFile;
	inFile.open(scr);//open the input file
	stringstream strStream;
	strStream << inFile.rdbuf();//read the file

	return strStream.str();//str holds the content of the file
}

string ParameterTreeConverter::GetFilename(const rapidjson::Value &model)
{
	Value::ConstMemberIterator filename = model.FindMember("Filename");
	if (!filename->value.IsString()) {
		throw std::invalid_argument("Invalid file name field");
	}
	return filename->value.GetString();
}

std::string ParameterTreeConverter::GetString(const rapidjson::Value &model, const std::string &propertyName)
{
	Value::ConstMemberIterator key = model.FindMember(propertyName.c_str());
	if (!key->value.IsString()) {
		throw std::invalid_argument("Invalid " + propertyName + " field");
	}
	return key->value.GetString();
}

wchar_t * ParameterTreeConverter::StringToWideCharT(const string &str)
{
	wchar_t* wide_string = new wchar_t[str.length() + 1];
	std::copy(str.begin(), str.end(), wide_string);
	wide_string[str.length()] = 0;

	return wide_string;
}


void ParameterTreeConverter::loadConstraints(const Value::ConstValueIterator constraintItr, double &consMin, double &consMax, int &consMinInd, int &consMaxInd, int &linkInd, bool &bCons)
{
	Value::ConstMemberIterator MinValue = constraintItr->FindMember("MinValue");
	consMin = jsonObjectToDouble(MinValue->value);
	if (consMin != NEGINF)
		bCons = true;

	Value::ConstMemberIterator MaxValue = constraintItr->FindMember("MaxValue");
	consMax = jsonObjectToDouble(MaxValue->value);
	if (consMax != POSINF)
		bCons = true;

	Value::ConstMemberIterator MinIndex = constraintItr->FindMember("MinIndex");
	consMinInd = MinIndex->value.GetInt();
	if (consMinInd != -1)
		bCons = true;

	Value::ConstMemberIterator MaxIndex = constraintItr->FindMember("MaxIndex");
	consMaxInd = MaxIndex->value.GetInt();
	if (consMaxInd != -1)
		bCons = true;

	Value::ConstMemberIterator Link = constraintItr->FindMember("Link");
	linkInd = Link->value.GetInt();
	if (linkInd != -1)
		bCons = true;
}


void ParameterTreeConverter::loadConstraints(const Value::ConstMemberIterator constraintItr, double &consMin, double &consMax, int &consMinInd, int &consMaxInd, int &linkInd, bool &bCons)
{

	Value::ConstMemberIterator MinValue = constraintItr->value.FindMember("MinValue");
	consMin = jsonObjectToDouble(MinValue->value);
	if (consMin != NEGINF)
		bCons = true;

	Value::ConstMemberIterator MaxValue = constraintItr->value.FindMember("MaxValue");
	consMax = jsonObjectToDouble(MaxValue->value);
	if (consMax != POSINF)
		bCons = true;

	Value::ConstMemberIterator MinIndex = constraintItr->value.FindMember("MinIndex");
	consMinInd = MinIndex->value.GetInt();
	if (consMinInd != -1)
		bCons = true;

	Value::ConstMemberIterator MaxIndex = constraintItr->value.FindMember("MaxIndex");
	consMaxInd = MaxIndex->value.GetInt();
	if (consMaxInd != -1)
		bCons = true;

	Value::ConstMemberIterator Link = constraintItr->value.FindMember("Link");
	linkInd = Link->value.GetInt();
	if (linkInd != -1)
		bCons = true;
}


Parameter ParameterTreeConverter::LoadLocationParams(const rapidjson::Value &model, const char *locp)
{
	double val = 0.0;
	bool bMutable = false;
	bool bCons = false;
	double consMin = NEGINF, consMax = POSINF;
	int consMinInd = -1, consMaxInd = -1, linkInd = -1;
	double sigma = 0.0;

	// Location/rotation parameters
	Value::ConstMemberIterator Location = model.FindMember("Location");
	assert(Location != model.MemberEnd());
	Value::ConstMemberIterator LocationMutables = model.FindMember("LocationMutables");
	assert(Location != model.MemberEnd());
	Value::ConstMemberIterator LocationSigma = model.FindMember("LocationSigma");
	assert(LocationSigma != model.MemberEnd());
	Value::ConstMemberIterator LocationConstraints = model.FindMember("LocationConstraints");
	assert(LocationConstraints != model.MemberEnd());

	Value::ConstMemberIterator itr;

	itr = Location->value.FindMember(locp);
	if (itr != model.MemberEnd())
	{
		val = itr->value.GetDouble();
	}
	itr = LocationMutables->value.FindMember(locp);
	if (itr != model.MemberEnd())
	{
		bMutable = itr->value.GetBool();
	}
	itr = LocationSigma->value.FindMember(locp);
	if (itr != model.MemberEnd())
	{
		sigma = itr->value.GetDouble();
	}
	itr = LocationConstraints->value.FindMember(locp);
	//Isn't there a way to send the *value* of the object so we don't need to have two almost identical loadConstraints functions????
	loadConstraints(itr, consMin, consMax, consMinInd, consMaxInd, linkInd, bCons);

	return Parameter(val, bMutable, bCons, consMin, consMax, \
		consMinInd, consMaxInd, linkInd, sigma);
}

//double ParameterTreeConverter::jsonObjectToDouble(Value::ConstMemberIterator val)
double ParameterTreeConverter::jsonObjectToDouble(const rapidjson::Value &val)
{
	double retVal = 0.0;

	if (val.IsString())
	{
		string str = val.GetString();

		if (str == "inf")
			retVal = POSINF;
		else if (str == "-inf")
			retVal = NEGINF;
		else retVal = atof(str.c_str());
	}
	if (val.IsNumber())
	{
		retVal = val.GetDouble();
	}
	return retVal;
}

/****************************************************************************************************
		Convert from a ParameterTree to simple JSON representation and back

  Simple JSON representation is simple, each ParameterTree node looks likes this:

  {
	ModelPtr: modelptr,
	Parameters: [
		{
			value: ...
			isMutable: ...
			...             <-- all the 9 Parameter members
		},
		{ second parameter },
		...
	],
	Submodels: [
		{ first submodel },
		{ second submodel },
		...  Each submodel is another ParameterTree actually
	]
  }
*****************************************************************************************************/


void ParameterTreeConverter::WriteSimpleJSON(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const ParameterTree &pt)
{
	WriteParameterTree(writer, &pt);
}

ParameterTree ParameterTreeConverter::FromSimpleJSON(const rapidjson::Value &json)
{
	return ParameterTreeFromJSON(json);
}


void ParameterTreeConverter::WriteParameterTree(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const ParameterTree *pt)
{
	writer.StartObject();
	writer.Key("ModelPtr");
	writer.Int(GetStateModelPtr(pt->GetNodeModel()));
	writer.Key("Parameters");
	writer.StartArray();
	for (int i = 0; i < pt->GetNumParameters(); i++)
	{
		const Parameter &param = pt->GetParameter(i);
		WriteParameter(writer, param);
	}
	writer.EndArray();
	writer.Key("Submodels");
	writer.StartArray();
	for (int i = 0; i < pt->GetNumSubModels(); i++)
	{
		const ParameterTree *submodel = pt->GetSubModel(i);
		WriteParameterTree(writer, submodel);
	}
	writer.EndArray();
	writer.EndObject();
}

void ParameterTreeConverter::checkForInfinityThenWriteDouble(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, double value)
{
	if (value == INFINITY)
		writer.String("inf");
	else if (value == -INFINITY)
		writer.String("-inf");
	else
		writer.Double(value);
}

void ParameterTreeConverter::WriteParameter(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const Parameter &param)
{
	writer.StartObject();
	writer.Key("Value");
	checkForInfinityThenWriteDouble(writer, param.value);
	writer.Key("isMutable");
	writer.Bool(param.isMutable);
	writer.Key("isConstrained");
	writer.Bool(param.isConstrained);
	writer.Key("consMin");
	checkForInfinityThenWriteDouble(writer, param.consMin);
	writer.Key("consMax");
	checkForInfinityThenWriteDouble(writer, param.consMax);
	writer.Key("consMinIndex");
	writer.Int(param.consMinIndex);
	writer.Key("consMaxIndex");
	writer.Int(param.consMaxIndex);
	writer.Key("linkIndex");
	writer.Int(param.linkIndex);
	writer.Key("sigma");
	checkForInfinityThenWriteDouble(writer, param.sigma);
	writer.EndObject();
}

Parameter ParameterTreeConverter::ParameterFromJSON(const rapidjson::Value &json)
{
	Parameter param;
	param.value = json["Value"].GetDouble();
	param.isMutable = json["isMutable"].GetBool();
	param.isConstrained = json["isConstrained"].GetBool();
	// param.consMin and param.consMax are initialize with -inf and inf. json["consMin"] and json["consMax"] aren't double just if they inf
	param.consMin = json["consMin"].IsDouble() ? json["consMin"].GetDouble() : param.consMin;
	param.consMax = json["consMax"].IsDouble() ? json["consMax"].GetDouble() : param.consMax;

	param.consMinIndex = json["consMinIndex"].GetInt();
	param.consMaxIndex = json["consMaxIndex"].GetInt();
	param.linkIndex = json["linkIndex"].GetInt();
	param.sigma = json["sigma"].GetDouble();

	return param;
}

ParameterTree ParameterTreeConverter::ParameterTreeFromJSON(const rapidjson::Value &json)
{
	ParameterTree pt;

	pt.SetNodeModel(json["ModelPtr"].GetInt());

	const rapidjson::Value &jsonParams = json["Parameters"];
	for (auto it = jsonParams.Begin(); it != jsonParams.End(); it++)
		pt.AddParameter(ParameterFromJSON(*it));

	const rapidjson::Value &jsonSubmodels = json["Submodels"];
	for (auto it = jsonSubmodels.Begin(); it != jsonSubmodels.End(); it++)
		pt.AddSubModel(ParameterTreeFromJSON(*it));

	return pt;
}
