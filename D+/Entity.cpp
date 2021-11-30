#include <windows.h>
#include <GL/gl.h>

#include "Entity.h"

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
	if (type == EntityType::TYPE_PDB && par.nExtraParams == 8 && params->nExtraParams == 10)
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

}
