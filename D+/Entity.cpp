#include <windows.h>
#include <GL/gl.h>

#include "Entity.h"
#include "clrfunctionality.h"
#include "GraphPane3D.h" // For PDB rendering

using Aga::Controls::Tree::Node;

using namespace System;
using namespace System::Windows::Forms;
using namespace LuaInterface;

namespace DPlus {

Entity::~Entity() {
	if(params) {
		delete params;
		params = NULL;
	}

	if (xs)
	{
		delete xs;
		xs = nullptr;
	}

	if (ys)
	{
		delete ys;
		ys = nullptr;
	}

	if (zs)
	{
		delete zs;
		zs = nullptr;
	}

	if (atoms)
	{
		delete atoms;
		atoms = nullptr;
	}

	if(BackendModel && frontend && job) {
		for each (Entity ^en in this->Nodes)
			delete en;
		frontend->DestroyModel(job, BackendModel, false);
	}

	if(colorCodedDList) {
		unsigned int tmpdlist = colorCodedDList;
		colorCodedDList = 0;
		glDeleteLists(tmpdlist, 1);
	}

	if(renderDList) {
		unsigned int tmpdlist = renderDList;
		renderDList = 0;
		glDeleteLists(tmpdlist, 1);
	}

	if (hlnf.IsAllocated)
		hlnf.Free();
	if (hlpnf.IsAllocated)
		hlpnf.Free();
	if (hdpvf.IsAllocated)
		hdpvf.Free();
	if (hipaf.IsAllocated)
		hipaf.Free();
}

void Entity::Invalidate(LevelOfDetail lod, bool bInvalidateParents) {

	// Generate the two display lists from the function and the model parameters
	//////////////////////////////////////////////////////////////////////////
	unsigned int tempDList = 0, tempCCDList = 0;
	if(render) {
		tempDList = glGenLists(1);
		glNewList(tempDList, GL_COMPILE);
		render(*params, modelUI->GetEDProfile(), lod, false);
		glEndList();

		tempCCDList = glGenLists(1);
		glNewList(tempCCDList, GL_COMPILE);	
		render(*params, modelUI->GetEDProfile(), lod, true);
		glEndList();
	}
	if(modelfile != nullptr) {
		if(lod == currentLOD)
			return;

		GraphPane3D::SetPDBFile(tempDList, tempCCDList, xs, ys, zs, atoms, lod);
		
		currentLOD = lod;
	}

	// Remove old display lists
	if(colorCodedDList) {
		unsigned int tmpdlist = colorCodedDList;
		colorCodedDList = 0;
		glDeleteLists(tmpdlist, 1);
	}

	if(renderDList) {
		unsigned int tmpdlist = renderDList;
		renderDList = 0;
		glDeleteLists(tmpdlist, 1);
	}

	// Assign the new ones
	renderDList = tempDList;
	colorCodedDList = tempCCDList;

	// Recursively invalidate the parents
	if(bInvalidateParents) {
		if(this->Parent != nullptr && dynamic_cast<Entity ^>(this->Parent) != nullptr)
			dynamic_cast<Entity ^>(this->Parent)->Invalidate(lod);
	}
}

// A slightly different version of LuaItemToDouble, not allowing infinities and NaNs
static Double LuaItemToFiniteDouble(Object ^item) {
	if(item == nullptr)
		return 0.0;
	if(dynamic_cast<Double ^>(item) != nullptr)
		return (Double)item;
	if(dynamic_cast<String ^>(item) != nullptr) {
		String ^str = (String ^)item;
		double db;
		
		if(Double::TryParse(str, db))
			return db;
	}

	return 0.0;
}

static LuaTable ^CreateTable(Lua ^luaState) {
	luaState->NewTable("___TEMP");
	return luaState->GetTable("___TEMP");
}

static LuaTable ^LuaParamValueTreeFromStruct(Lua ^context, const paramStruct& par) {
	LuaTable ^res = CreateTable(context);
	for(int l = 0; l < par.layers; l++) {
		LuaTable ^layerv = CreateTable(context);
		res[l + 1] = layerv;

		// Setting Parameters
		for(int i = 0; i < par.nlp; i++)
			layerv[i + 1] = par.params[i][l].value;
	}

	return res;
}

void Entity::SetParameters(const paramStruct& parConst, LevelOfDetail lod) {
	paramStruct par = parConst;
	// This is a hack to enable backwards compatibility of old state files with PDB nodes
	if ( (type == EntityType::TYPE_PDB || type == EntityType::TYPE_EPDB) && par.nExtraParams == 8 && params->nExtraParams == 10)
	{
		par.extraParams.reserve(par.extraParams.size() + 2);
		auto c1 = params->extraParams[2];
		par.extraParams.insert(par.extraParams.begin() + 2, 1, c1);
		auto solvation_thickness = params->extraParams[5];
		par.extraParams.insert(par.extraParams.begin() + 5, solvation_thickness);
		par.nExtraParams = int(par.extraParams.size());
	}

	int tmpNEP = par.nExtraParams;
	std::vector<Parameter> tmpEP = par.extraParams;

	// This was added in order to allow the addition of the Scale extra parameter
	// when loading a scripted symmetry from within a saved state file.
	if(type == EntityType::TYPE_SYMMETRY && par.nExtraParams < params->nExtraParams)
	{
		tmpEP  = params->extraParams;
		tmpNEP = params->nExtraParams;
	}

	*params = par;

	params->nExtraParams	= tmpNEP;
	params->extraParams		= tmpEP	;

	Invalidate(lod);

	// If a symmetry, generate the underlying points for objects
	////////////////////////////////////////////////////////////
	if(symmrender) {
		std::vector<LocationRotation> locrots = symmrender(par);
		locs->Clear();
		for(int i = 0; i < (int)locrots.size(); i++)
			locs->Add(LocationRotationCLI(locrots[i]));
	}	

	// If a scripted symmetry, generate the underlying points for objects
	/////////////////////////////////////////////////////////////////////
	if(type == EntityType::TYPE_SYMMETRY && modelContext != nullptr) {
		locs->Clear();

		LuaFunction ^populateFunc = modelContext->GetFunction("Populate");
		if(populateFunc == nullptr) {
			MessageBox::Show("Error populating symmetry, missing \"Populate\" function", 
							 "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		// Call the populate function
		cli::array<Object ^> ^retvals = nullptr;
		try {
			retvals = populateFunc->Call(gcnew array<Object ^> {
				LuaParamValueTreeFromStruct(modelContext, par) , par.layers 
			});
		} catch(Exception ^ex) {
			MessageBox::Show("Error populating symmetry, runtime error: " + ex->Message, 
				"ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		// Verify results
		if(retvals == nullptr || retvals->Length < 1) {
			MessageBox::Show("Error populating symmetry, invalid return value", 
				"ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		LuaTable ^locrots = dynamic_cast<LuaTable ^>(retvals[0]);
		if(locrots == nullptr) {
			MessageBox::Show("Error populating symmetry, invalid return value", 
							 "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		// Add results to renderer
		int keycnt = locrots->Length;
		for(int i = 1; i <= keycnt; i++) {
			LuaTable ^locrot = dynamic_cast<LuaTable ^>(locrots[i]);
			if(locrot == nullptr) {
				MessageBox::Show(gcnew String("Error populating symmetry, invalid location at index ") + i, 
					"ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				locs->Clear();
				return;
			}

			LocationRotationCLI lc;
			lc.x     = LuaItemToFiniteDouble(locrot[1]);
			lc.y     = LuaItemToFiniteDouble(locrot[2]);
			lc.z     = LuaItemToFiniteDouble(locrot[3]);
			lc.radAlpha = LuaItemToFiniteDouble(locrot[4]);
			lc.radBeta  = LuaItemToFiniteDouble(locrot[5]);
			lc.radGamma = LuaItemToFiniteDouble(locrot[6]);


			locs->Add(lc);
		}
	}
}

Entity::Entity()
{
	// Default values
	type = EntityType::TYPE_PRIMITIVE;
	BackendModel = NULL;
	frontend = NULL;
	job = NULL;
	model = NULL;
	modelName = "N/A";
	displayName = "";
	i++;
	filename = nullptr;
	modelfile = nullptr;
	anomfilename = nullptr;
	currentLOD = LOD_NONE;
	bCentered = false;
	params = new paramStruct();
	render = NULL;
	symmrender = NULL;
	locs = gcnew System::Collections::Generic::List<LocationRotationCLI>();
	renderDList = colorCodedDList = 0;
	modelContext = nullptr;
	selected = false;
	xs = new std::vector < float > ;
	ys = new std::vector < float > ;
	zs = new std::vector < float > ;
	atoms = new std::vector < u8 > ;
}

void Entity::SetUseGrid(bool bUseG)
{
	if (!params)
		return;
	if (bUseG)
	{
		params->bSpecificUseGrid = bUseG;
		// change the use grid for all the node children
		for (int i = 0; i < Nodes->Count; i++) {
			Entity^ ent = dynamic_cast<Entity^>(Nodes[i]);
			if (ent) {
				ent->SetUseGrid(bUseG); // this will change the value to all the children of this node too;
			}
		}
	}
	else
	{
		params->bSpecificUseGrid = bUseG;
		Entity ^ ent = dynamic_cast<Entity^>(Parent);
		if (ent)
			ent->SetUseGrid(bUseG);
	}
}
bool Entity::IsParentUseGrid()
{
	if (!params)
		return false;
	Entity^ ent = dynamic_cast<Entity^>(this->Parent);
	if (!ent)
		return false; // root, doesn't have a parent
	if (!ent->params)
		return false;
	if (ent->params->bSpecificUseGrid)
		return true;
	return ent->IsParentUseGrid();

}
void Entity::validateConstrains(System::Collections::Generic::List<Entity^>^ invalidVec)
{
	if (!params)
		return ;

	bool hasInvalidParam = false;
	if (params->nExtraParams > 0)
	{

		std::vector<Parameter> tmpEP = params->extraParams;
		for (Parameter p : tmpEP) {
			if (!p.ParamValidateConstraint())
				hasInvalidParam = true;
		}
	}
	for (std::vector<Parameter> paramsVec : params->params) {
		for (Parameter p : paramsVec) {
			if (!p.ParamValidateConstraint())
				hasInvalidParam = true;
		}
	}
	if (!params->x.ParamValidateConstraint() || !params->y.ParamValidateConstraint() || !params->z.ParamValidateConstraint()
		|| !params->alpha.ParamValidateConstraint() || !params->beta.ParamValidateConstraint() || !params->gamma.ParamValidateConstraint())
		hasInvalidParam = true;
	
	if (hasInvalidParam)
		invalidVec->Add(this);

	for (int i = 0; i < Nodes->Count; i++)
	{
		Entity^ ent = dynamic_cast<Entity^>(this->Nodes[i]);
		ent->validateConstrains(invalidVec);
	}
	return ;
}
System::String^ Entity::InvalidParamsString()
{
	System::String^ invalidString = "";
	if (!params)
		invalidString;
	System::String^ ExtraString = "";
	System::String^ ParametersString = "";
	System::String^ XYZABGString = "";
	if (params->nExtraParams > 0)
	{
		std::vector<Parameter> tmpEP = params->extraParams;
		
		for (int i = 0; i < params->nExtraParams; i++)
		{
			if (tmpEP[i].ParamValidateConstraint())
				continue;
			ExtraParam ext = this->modelUI->GetExtraParameter(i);
			ExtraString += String::Format("{0}, ", stringToClr(ext.name));
		}
		//ExtraString = "Extra params: \n" + ExtraString + "\n";
		if (ExtraString->Length > 0)
			ExtraString = "Extra params: " + ExtraString->Remove(ExtraString->Length - 2) + "\n";
		//ExtraString += "\n";
	}
	int layers_num = params->layers;
	int param_size = params->params.size();
	if (param_size == layers_num) {
		for (int i = 0; i < params->params.size(); i++)
		{
			std::vector<Parameter> paramsVec = params->params[i];
			System::String^ paramString = "";
			std::string layer = this->modelUI->GetLayerName(i);
			for (int j = 0; j < paramsVec.size(); j++)
			{
				if (paramsVec[j].ParamValidateConstraint())
					continue;

				std::string paramN = this->modelUI->GetLayerParamName(j);
				paramString += String::Format("{0},", stringToClr(paramN));
			}
			if (paramString->Length > 0)
			{
				paramString = paramString->Remove(paramString->Length - 1);
				ParametersString += String::Format("param layer:{0}, params names: {1} \n", stringToClr(layer), paramString);
			}
		}
	}
	if (param_size < layers_num) // scripted symmetry for example
	{
		for (int i = 0; i < params->params.size(); i++)
		{
			std::vector<Parameter> paramsVec = params->params[i];
			System::String^ paramString = "";
			for (int j = 0; j < paramsVec.size(); j++)
			{
				if (paramsVec[j].ParamValidateConstraint())
					continue;
				std::string paramN = this->modelUI->GetLayerName(j);
				paramString += String::Format("{0},", stringToClr(paramN));
			}
			if (paramString->Length > 0)
			{
				paramString = paramString->Remove(paramString->Length - 1);
				ParametersString += String::Format("params layer:{0} \n", paramString);
			}
		}
	}
	if (param_size > layers_num) // manual symmetry for example - the layers are the params and the params are the layers.... :(
	{
		for (int i = 0; i < layers_num; i++)
		{
			System::String^ paramString = "";
			for (int j = 0; j < param_size; j++)
			{
				Parameter parameter = params->params[j][i];
				if (parameter.ParamValidateConstraint())
					continue;
				std::string paramN = this->modelUI->GetLayerParamName(j);
				paramString += String::Format("{0},", stringToClr(paramN));
			}
			std::string layer = this->modelUI->GetLayerName(i);
			if (paramString->Length > 0) 
			{
				paramString = paramString->Remove(paramString->Length - 1);
				ParametersString += String::Format("param layer:{0}, params names: {1} \n", stringToClr(layer), paramString);
			}
		}
	}

	int num = this->modelUI->GetMaxLayers();
	if (!params->x.ParamValidateConstraint())
		XYZABGString += "X,";
	if (!params->y.ParamValidateConstraint())
		XYZABGString += "Y,";
	if (!params->z.ParamValidateConstraint())
		XYZABGString += "Z,";
	if (!params->alpha.ParamValidateConstraint())
		XYZABGString += "Alpha,";
	if (!params->beta.ParamValidateConstraint())
		XYZABGString += "Beta,";
	if (!params->gamma.ParamValidateConstraint())
		XYZABGString += "Gama,";

	if (XYZABGString->Length > 0)
		XYZABGString = XYZABGString->Remove(XYZABGString->Length - 1);

	invalidString = this->modelName + ": \n" + XYZABGString + "\n" + ParametersString + ExtraString;
	return invalidString;
}

void Entity::FixConstrains()
{
	if (!params)
		return;

	bool hasInvalidParam = false;
	if (params->nExtraParams > 0)
	{

		std::vector<Parameter> tmpEP = params->extraParams;
		for (int i = 0; i < params->extraParams.size(); i++)
		{
			if (!params->extraParams[i].ParamValidateConstraint())
				params->extraParams[i].SwapMinMaxValue();
		}
	}

	for (int i = 0; i < params->params.size(); i++)
	{
		for (int j = 0; j < params->params[i].size(); j++)
		{
			if (!params->params[i][j].ParamValidateConstraint())
				params->params[i][j].SwapMinMaxValue();
		}
	}
	if (!params->x.ParamValidateConstraint()) params->x.SwapMinMaxValue();
	if (!params->y.ParamValidateConstraint()) params->y.SwapMinMaxValue();
	if (!params->z.ParamValidateConstraint()) params->z.SwapMinMaxValue();
	if (!params->alpha.ParamValidateConstraint()) params->alpha.SwapMinMaxValue();
	if (!params->beta.ParamValidateConstraint()) params->beta.SwapMinMaxValue();
	if (!params->gamma.ParamValidateConstraint()) params->gamma.SwapMinMaxValue();

	return;
}
}
