#include <windows.h> // For COLORREF
#include "MainWindow.h"
#include "clrfunctionality.h"
#include "FrontendExported.h" // For ReadDataFile/WriteDataFile
#include "LuaBinding.h"
#include "GraphPane3D.h"
#include "SymmetryView.h"
#include "PreferencesPane.h"
#include "Controls3D.h"
#include "ScriptPane.h"
#include "FittingPrefsPane.h"
#include "GraphFigure.h"

namespace DPlus {

	void MainWindow::BindLuaFunctions() {
		// Example
		/*
		Class obj = new Class();
		luaState->RegisterFunction("cfunc", obj, typeof(Class).GetMethod("method"));
		*/

		luaState->NewTable("dplus");
		luaState->RegisterFunction("dplus.openconsole", nullptr, DPlus::Scripting::typeid->GetMethod("OpenConsole"));
		luaState->RegisterFunction("dplus.closeconsole", nullptr, DPlus::Scripting::typeid->GetMethod("CloseConsole"));
		luaState->RegisterFunction("dplus.getparametertree", this, MainWindow::typeid->GetMethod("GetParameterTree"));
		luaState->RegisterFunction("dplus.setparametertree", this, MainWindow::typeid->GetMethod("SetParameterTree"));
		luaState->RegisterFunction("dplus.updateparametertree", this, MainWindow::typeid->GetMethod("UpdateParametersLua"));


		luaState->RegisterFunction("dplus.save", this, MainWindow::typeid->GetMethod("SaveParametersToFile"));
		luaState->RegisterFunction("dplus.load", this, MainWindow::typeid->GetMethod("ImportParameterFile"));

		luaState->RegisterFunction("dplus.findmodelfull", this, MainWindow::typeid->GetMethod("LuaFindModel"));
		luaState->RegisterFunction("dplus.findampfull", this, MainWindow::typeid->GetMethod("LuaFindAmplitude"));
		// Register the find model/amplitude functions with 1 or 2 parameters
		luaState->DoString("\
						   					   dplus.findmodel = function (name, container)\
											   					   if container == nil then return dplus.findmodelfull(name, nil);\
																   					   else return dplus.findmodelfull(name, container); end\
																					   					   end\
																										   					   dplus.findamplitude = function (name, container)\
																															   					   if container == nil then return dplus.findampfull(name, nil);\
																																				   					   else return dplus.findampfull(name, container); end\
																																									   					   end");

		luaState->RegisterFunction("dplus.generatetable", this, MainWindow::typeid->GetMethod("LuaGenerateTable"));
		luaState->RegisterFunction("dplus.generatecurrent", this, MainWindow::typeid->GetMethod("LuaGenerateCurrent"));
		// Register the generate function with 0 or 1 parameters
		luaState->DoString(
			"dplus.generate = function (a, b, c)"
			"if a == nil then return dplus.generatecurrent();"
			"else if b == nil then messagebox(\"You must provide a save filename when using dplus.generate() with a specific tree.\"); return nil;"
			"else if c == nil then return dplus.generatetable(a, b, false);"
			"else return dplus.generatetable(a, b, c);"
			" end "
			" end "
			" end "
			" end "
		);

		luaState->RegisterFunction("dplus.fitfull", this, MainWindow::typeid->GetMethod("LuaFit"));
		luaState->RegisterFunction("dplus.fitcurr", this, MainWindow::typeid->GetMethod("LuaFitCurrent"));
		// Register the fit function with 0 to 2 parameters
		luaState->DoString(
						   	"dplus.fit = function (data, ptree, properties)"
							"if (ptree == nil and data == nil)  then return  dplus.fitcurr();"
							"else if (data == nil or ptree == nil or properties==nil) then messagebox(\"You must provide a signal, state and fit properties in order to run fit  .\"); return nil;"
							"else return dplus.fitfull(data, ptree, properties);"	
							" end"
							" end"
							" end");

		luaState->RegisterFunction("dplus.readdata", this, MainWindow::typeid->GetMethod("LuaReadData"));
		luaState->RegisterFunction("dplus.writedata", this, MainWindow::typeid->GetMethod("LuaWriteData"));

		luaState->RegisterFunction("dplus.figurefull", this, MainWindow::typeid->GetMethod("LuaOpenFigure"));
		luaState->DoString("\
						   					   dplus.figure = function (title, xlabel, ylabel)\
											   							return dplus.figurefull(title, xlabel, ylabel);\
																							   end");
		luaState->RegisterFunction("dplus.showgraphtable", this, MainWindow::typeid->GetMethod("LuaShowGraph"));
		luaState->RegisterFunction("dplus.showgraphfile", this, MainWindow::typeid->GetMethod("LuaShowFileGraph"));
		// Register the show graph function with either a file or a table
		luaState->DoString("\
						   					   dplus.showgraph = function (data, fignum, color)\
											   					   if fignum == nil then fignum = -1; end\
																   					   if type(data) == \"string\" then return dplus.showgraphfile(fignum, data, color);\
																					   					   else return dplus.showgraphtable(fignum, data, color); end\
																										   					   end");

		luaState->RegisterFunction("msgbox", this, MainWindow::typeid->GetMethod("LuaMessage"));
		luaState->RegisterFunction("mbox", this, MainWindow::typeid->GetMethod("LuaMessage"));
		luaState->RegisterFunction("message", this, MainWindow::typeid->GetMethod("LuaMessage"));
		luaState->RegisterFunction("messagebox", this, MainWindow::typeid->GetMethod("LuaMessage"));

		//luaState->RegisterFunction("print", nullptr, System::Console::typeid->GetMethod("WriteLine", gcnew array<Type ^> { String::typeid }));

		luaState->RegisterFunction("sleep", nullptr, System::Threading::Thread::typeid->GetMethod("Sleep", gcnew array<Type ^> { Int32::typeid }));
		luaState->RegisterFunction("wait", nullptr, System::Threading::Thread::typeid->GetMethod("Sleep", gcnew array<Type ^> { Int32::typeid }));

		luaState->DoString("\n\
						   	function table.copy(orig)\n\
									local orig_type = type(orig)\n\
											local copy\n\
													if orig_type == 'table' then\n\
																copy = {}\n\
																			for orig_key, orig_value in next, orig, nil do\n\
																							copy[table.copy(orig_key)] = table.copy(orig_value)\n\
																										end\n\
																													setmetatable(copy, table.copy(getmetatable(orig)))\n\
																															else -- number, string, boolean, etc\n\
																																		copy = orig\n\
																																				end\n\
																																						return copy\n\
																																							end");

		luaState->DoString("\n"
			"function PrintKeys(tb)\n"
			"local keyset={}\n"
			"local n=0\n"
			"for k,v in pairs(tb) do\n"
			"n=n+1\n"
			"end\n"
			"print(\"Size: \" .. n ) \n"
			"for k,v in pairs(tb) do\n"
			"print(k)\n"
			"end\n"
			"end\n"
		);
	}


	// This function can't be member of MainWindow because ParametersToLuaTree use it and ParametersToLuaTree can't be member 
	static LuaTable ^ CreateTable(Lua ^ luaState) {
		luaState->NewTable("___TEMP");
		return luaState->GetTable("___TEMP");
	}

	// This function can't be member of MainWindow due to compilation problem (it has a default value)
	static void ParametersToLuaTree(const paramStruct & ps, LuaTable ^ subtbl, Entity ^ ent, Lua ^ luaState, bool bSingleGeometry = false) {
		if (ent != nullptr)
		{
			// Contains "container,INDEX" (or for default models, ",INDEX")
			String ^conttype = "";

			switch (ent->type)
			{
			default:
			case EntityType::TYPE_PRIMITIVE:

				if (ent->modelContext != nullptr) { // If a scripted model or geometry
					if (bSingleGeometry)
						conttype = "Scripted Model";
					else
						conttype = "Scripted Geometry";

					subtbl["Filename"] = ent->filename;
				}
				else {
					if (ent->modelUI->GetContainer(NULL))
						conttype += gcnew String(ent->modelUI->GetContainer(NULL));
					conttype += "," + Int32(ent->modelUI->GetModelInformation().modelIndex).ToString();
				}
				break;

			case EntityType::TYPE_SYMMETRY:
				if (ent->modelContext != nullptr) { // If a scripted symmetry
					conttype = "Scripted Symmetry";
					subtbl["Filename"] = ent->filename;
				}
				else {
					if (ent->modelUI->GetContainer(NULL))
						conttype += gcnew String(ent->modelUI->GetContainer(NULL));
					conttype += "," + Int32(ent->modelUI->GetModelInformation().modelIndex).ToString();
				}
				break;

			case EntityType::TYPE_PDB:
				conttype = "PDB";
				subtbl["Filename"] = ent->filename;
				subtbl["AnomFilename"] = ent->anomfilename;
				subtbl["Centered"] = ent->bCentered;
				break;

			case EntityType::TYPE_EPDB:
				conttype = "EPDB";
				subtbl["Filename"] = ent->filename;
				subtbl["AnomFilename"] = ent->anomfilename;
				subtbl["Centered"] = ent->bCentered;
				break;

			case EntityType::TYPE_AMPGRID:
				conttype = "AMP";
				subtbl["Filename"] = ent->filename;
				subtbl["Centered"] = ent->bCentered;
				break;
			}


			subtbl["Type"] = conttype;
		}

		subtbl["Name"] = ent->displayName;
		subtbl["nlp"] = ps.nlp;
		subtbl["nLayers"] = ps.layers;
		subtbl["nExtraParams"] = ps.nExtraParams;
		subtbl["ModelPtr"] = ent->BackendModel;

		// Layered parameters
		subtbl["Parameters"] = CreateTable(luaState);
		subtbl["Mutables"] = CreateTable(luaState);
		subtbl["Constraints"] = CreateTable(luaState);
		subtbl["Sigma"] = CreateTable(luaState);
		for (int l = 0; l < ps.layers; l++) {
			LuaTable ^layerv = CreateTable(luaState);
			((LuaTable ^)subtbl["Parameters"])[l + 1] = layerv;

			LuaTable ^layerm = CreateTable(luaState);
			((LuaTable ^)subtbl["Mutables"])[l + 1] = layerm;

			LuaTable ^layerc = CreateTable(luaState);
			((LuaTable ^)subtbl["Constraints"])[l + 1] = layerc;

			LuaTable ^layersig = CreateTable(luaState);
			((LuaTable ^)subtbl["Sigma"])[l + 1] = layersig;

			// Setting Parameters
			for (int i = 0; i < ps.nlp; i++) {
				layerv[i + 1] = ps.params[i][l].value;
				layerm[i + 1] = ps.params[i][l].isMutable;
				layersig[i + 1] = ps.params[i][l].sigma;

				// Setting constraints
				{
					LuaTable ^constbl = CreateTable(luaState);

					constbl["MinValue"] = ps.params[i][l].consMin;
					constbl["MaxValue"] = ps.params[i][l].consMax;
					constbl["MinIndex"] = ps.params[i][l].consMinIndex;
					constbl["MaxIndex"] = ps.params[i][l].consMaxIndex;
					constbl["Link"] = ps.params[i][l].linkIndex;

					layerc[i + 1] = constbl;
				}
			}
		}

		// Extra parameters
		subtbl["ExtraParameters"] = CreateTable(luaState);
		subtbl["ExtraMutables"] = CreateTable(luaState);
		subtbl["ExtraConstraints"] = CreateTable(luaState);
		subtbl["ExtraSigma"] = CreateTable(luaState);
		for (int i = 0; i < ps.nExtraParams; i++) {
			((LuaTable ^)subtbl["ExtraParameters"])[i + 1] = ps.extraParams[i].value;
			((LuaTable ^)subtbl["ExtraMutables"])[i + 1] = ps.extraParams[i].isMutable;
			((LuaTable ^)subtbl["ExtraSigma"])[i + 1] = ps.extraParams[i].sigma;

			// Setting constraints
			{
				LuaTable ^constbl = CreateTable(luaState);

				constbl["MinValue"] = ps.extraParams[i].consMin;
				constbl["MaxValue"] = ps.extraParams[i].consMax;
				constbl["MinIndex"] = ps.extraParams[i].consMinIndex;
				constbl["MaxIndex"] = ps.extraParams[i].consMaxIndex;
				constbl["Link"] = ps.extraParams[i].linkIndex;

				((LuaTable ^)subtbl["ExtraConstraints"])[i + 1] = constbl;
			}
		}

		subtbl["Use_Grid"] = ps.bSpecificUseGrid;

		// Location/rotation parameters
		subtbl["Location"] = CreateTable(luaState);
		subtbl["LocationMutables"] = CreateTable(luaState);
		subtbl["LocationConstraints"] = CreateTable(luaState);
		subtbl["LocationSigma"] = CreateTable(luaState);

#define SET_LOCATION_PARAM(LOCP) do {												\
		((LuaTable ^)subtbl["Location"])[#LOCP] = ps.LOCP.value;				    \
		((LuaTable ^)subtbl["LocationMutables"])[#LOCP] = ps.LOCP.isMutable;		\
		((LuaTable ^)subtbl["LocationSigma"])[#LOCP] = ps.LOCP.sigma;				\
																					\
		/* Setting constraints	*/													\
				{																			\
			LuaTable ^constbl = CreateTable(luaState);								\
																					\
			constbl["MinValue"] = ps.LOCP.consMin;									\
			constbl["MaxValue"] = ps.LOCP.consMax;									\
			constbl["MinIndex"] = ps.LOCP.consMinIndex;								\
			constbl["MaxIndex"] = ps.LOCP.consMaxIndex;								\
			constbl["Link"]     = ps.LOCP.linkIndex;								\
																					\
			((LuaTable ^)subtbl["LocationConstraints"])[#LOCP] = constbl;			\
				}																			\
			} while(false);

		SET_LOCATION_PARAM(x);
		SET_LOCATION_PARAM(y);
		SET_LOCATION_PARAM(z);
		SET_LOCATION_PARAM(alpha);
		SET_LOCATION_PARAM(beta);
		SET_LOCATION_PARAM(gamma);

#undef SET_LOCATION_PARAM
	}

	LuaTable ^MainWindow::GetParamTree(Entity ^root, Lua ^luaState) {
		LuaTable ^tbl = CreateTable(luaState);

		Int32 i = 1;
		for each (Entity ^ent in root->Nodes) {
			LuaTable ^subtbl = CreateTable(luaState);
			tbl[i] = subtbl;

			if (ent->Nodes->Count > 0)
				subtbl["Children"] = GetParamTree(ent, luaState);

			ParametersToLuaTree(ent->GetParameters(), subtbl, ent, luaState);

			i++;
		}

		return tbl;
	}

	LuaTable ^MainWindow::GetParameterTree() {
		if (this->InvokeRequired) {
			return (LuaTable ^)this->Invoke(gcnew FuncReturnLuaTable(this, &MainWindow::GetParameterTree));
		}

		luaState->NewTable("dplus.ptree");
		LuaTable ^otbl = luaState->GetTable("dplus.ptree");

		LuaTable ^ptbl = nullptr;

		otbl["ModelPtr"] = compositeModel;
		// Total domain scale and mutability
		otbl["Scale"] = domainScale;
		SymmetryView ^ sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];

		if (sv)
			otbl["ScaleMut"] = sv->scaleMut->Checked;
		else
			otbl["ScaleMut"] = false;

		otbl["Constant"] = domainConstant;
		if (sv)
			otbl["ConstantMut"] = sv->constantMut->Checked;
		else
			otbl["ConstantMut"] = false;




		{
			otbl["Geometry"] = "Domains";
			otbl["Populations"] = CreateTable(luaState);
			ptbl = (LuaTable ^)otbl["Populations"];

			// TODO::CompositeModel (later): Constraints on scale and popsizes???

			for (int i = 0; i < populationTrees->Count; i++)
			{
				ptbl[i + 1] = CreateTable(luaState);
				LuaTable ^poptbl = (LuaTable ^)ptbl[i + 1];
				poptbl["PopulationSize"] = populationSizes[i];
				poptbl["PopulationSizeMut"] = populationSizeMutable[i];
				poptbl["ModelPtr"] = domainModels[i];

				SymmetryView ^sv = dynamic_cast<SymmetryView ^>(PaneList[SYMMETRY_VIEWER]);
				if (sv && sv->populationTabs->TabPages[i]->Tag == "Renamed")
					poptbl["Title"] = sv->populationTabs->TabPages[i]->Text;

				poptbl["Models"] = CreateTable(luaState);
				LuaTable ^tbl = (LuaTable ^)poptbl["Models"];

				Int32 j = 1;
				for each(Entity ^ent in populationTrees[i]->Nodes) {
					LuaTable ^subtbl = CreateTable(luaState);
					tbl[j] = subtbl;

					if (ent->Nodes->Count > 0)
						subtbl["Children"] = GetParamTree(ent, luaState);

					ParametersToLuaTree(ent->GetParameters(), subtbl, ent, luaState);

					j++;
				}
			}

		}

		return otbl;
	}

	delegate void SPT(LuaTable ^ptree);
	void MainWindow::UpdateParametersLua(LuaTable ^ptree) {
		if (this->InvokeRequired) {
			Invoke(gcnew SPT(this, &MainWindow::UpdateParametersLua), gcnew array<Object ^> { ptree });
			return;
		}

		ParameterTreeCLI ^ptref = gcnew ParameterTreeCLI();

		ParameterTree pt = ParamTreeFromTable(ptree, true);

		ptref->pt = &pt;

		UpdateParameters(ptref);
	}

	void MainWindow::SetParameterTree(LuaTable ^ptree) {
		if (this->InvokeRequired) {
			Invoke(gcnew SPT(this, &MainWindow::SetParameterTree), gcnew array<Object ^> { ptree });
			return;
		}

		Controls3D^ c3 = nullptr;
		SymmetryView^ sv = nullptr;
		c3 = dynamic_cast<Controls3D^>(PaneList[CONTROLS]);
		sv = dynamic_cast<SymmetryView^>(PaneList[SYMMETRY_VIEWER]);

		String ^type = (String ^)ptree["Geometry"];
		if (type == nullptr) {
			MessageBox::Show("Invalid parameter tree! Root incorrect");
			return;
		}

		if (!(type->Equals("Domain") || type->Equals("Domains")))
		{
			MessageBox::Show("Invalid parameter tree type!");
			return;
		}

		// TODO: Doesn't this create TWICE AS MANY models in the backend?
		// (since InnerSetParameterTree creates the models as well)
		// SOMEONE SHOULD MANAGE AND DELETE MODELS
		ParameterTree pt = ParamTreeFromTable(ptree);  // Submodels are the populations (1 submodel per population). Creates full tree.
		if (pt.GetNodeModel() == 0)
			return;

		// Set domain scale and mutability
		if (ptree["Scale"] != nullptr)
			domainScale = LuaItemToDouble(ptree["Scale"]);
		else
			domainScale = 1.0;
		if (ptree["ScaleMut"] != nullptr)
			sv->scaleMut->Checked = LuaItemToBoolean(ptree["ScaleMut"]);
		else
			sv->scaleMut->Checked = false;
		sv->scaleBox->Text = "" + domainScale;

		// Set domain scale and mutability
		if (ptree["Constant"] != nullptr)
			domainConstant = LuaItemToDouble(ptree["Constant"]);
		else
			domainConstant = 0.0;
		if (ptree["ConstantMut"] != nullptr)
			sv->constantMut->Checked = LuaItemToBoolean(ptree["ConstantMut"]);
		else
			sv->constantMut->Checked = false;
		sv->constantBox->Text = "" + domainConstant;

		// Re-create entity tree from parameter tree
		// Modify number of populations
		ClearAndSetPopulationCount(pt.GetNumSubModels());

		bool bcompat = false;
		LuaTable ^pops = nullptr;
		if (type->Equals("Domains"))
			pops = (LuaTable ^)ptree["Populations"];
		else if (type->Equals("Domain")) // Backward compatibility - Ignore this case
			bcompat = true;
		System::Collections::Generic::List<Entity^>^ invalidVec = gcnew System::Collections::Generic::List<Entity^>();
		bool constrainsValid = true;
		for (int i = 0; i < pt.GetNumSubModels(); i++) // Loop over populations
		{
			// Get table and pointer to parameter tree
			LuaTable ^tbl = nullptr;

			if (bcompat)
				tbl = ptree;
			else
				tbl = (LuaTable ^)pops[i + 1];  // tbl contains the LuaTable of population i (STATE)

			ParameterTree *domain = pt.GetSubModel(i);  // domain contains the ParameterTree of population i (IN MEMORY)

														// Modify population size and mutability
			populationSizes[i] = LuaItemToDouble(tbl["PopulationSize"]);
			populationSizeMutable[i] = LuaItemToBoolean(tbl["PopulationSizeMut"]);

			// Custom population titles
			if (tbl["Title"] != nullptr)
			{
				sv->populationTabs->TabPages[i]->Text = (String ^)tbl["Title"];
				sv->populationTabs->TabPages[i]->Tag = "Renamed";
			}

			LuaTable ^mods = (LuaTable ^)tbl["Models"];  // Models contains the models of population i (IN THE STATE FILE)
			if (mods != nullptr) {
				int keys = mods->Keys->Count;

				for (int j = 0; j < keys; j++) {
					Entity ^ent = InnerSetParameterTree(domain->GetSubModel(j), (LuaTable ^)mods[j + 1]);  // Set one model
					
					if (ent != nullptr)
					{
						populationTrees[i]->Nodes->Add(ent);
						ent->validateConstrains(invalidVec);
					}
					ClearParameterTree(domain->GetSubModel(j));
				}
			}
		}

		if (invalidVec->Count > 0)
		{

			System::Windows::Forms::DialogResult result;
			result = MessageBox::Show("One of the models constraints upper bound is lower than it's lower bound.\n Do you want the system to swap the values?",
				"Question", MessageBoxButtons::YesNo, MessageBoxIcon::Question);
			if (result == ::DialogResult::Yes)
			{
				for (int i = 0; i < invalidVec->Count; i++)
				{
					invalidVec[i]->FixConstrains();
				}
			}
			else
			{
				System::String^ allInvalidString = "";
				for (int i = 0; i < invalidVec->Count; i++)
				{
					allInvalidString += "\n" + invalidVec[i]->InvalidParamsString();
				}
				result = MessageBox::Show("Invalid params list: \n" + allInvalidString,
					"Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
			}
		}
			 
		sv->tvInvalidate();
	}

	System::String ^ MainWindow::GetLuaScript(void)
	{
		LuaTable ^lua = GetParameterTree();
		return ((PreferencesPane ^)PaneList[PREFERENCES])->SerializePreferences() + "\n" +
			((FittingPrefsPane^)PaneList[FITTINGPREFS])->SerializePreferences() + "\n" +
			((Controls3D ^)PaneList[CONTROLS])->SerializePreferences() + "\n" +
			"Domain = " + LuaTableToString(lua);
	}

	String ^MainWindow::LuaFindModelHelper(FrontendComm *frontend, String ^container, String ^modelName, bool bAmp) {
		std::wstring cppcont;
		const wchar_t *cstr = NULL;

		if (container != nullptr) {
			cppcont = clrToWstring(container);
			cstr = cppcont.c_str();
		}

		// Load available primitives from the container
		int catCount = frontend->QueryCategoryCount(cstr);
		for (int i = 0; i < catCount; i++) {
			ModelCategory cat = frontend->QueryCategory(cstr, i);
			if (cat.type != MT_FORMFACTOR)
				continue;

			for (int j = 0; j < 16; j++) {
				if (cat.models[j] == -1)
					break;

				ModelInformation mi = frontend->QueryModel(cstr, cat.models[j]);

				// If this is not an amplitude and we are looking for one, continue
				if (bAmp && !mi.ffImplemented)
					continue;

				String ^modname = gcnew String(mi.name);
				if (modname->ToLower()->Contains(modelName->ToLower())) {
					if (container == nullptr)
						return "," + Int32(mi.modelIndex).ToString();
					else
						return container + "," + Int32(mi.modelIndex).ToString();
				}
			}
		}

		return nullptr;
	}

	/**
	* LuaFindModel: Finds the first model that matches "modelName" (even partially) inside a container.
	* Example: Finding "cylinder" will result in the Uniform Hollow Cylinder model.
	* @param container The container to look in, or nullptr for all loaded containers.
	* @param modelName The substring of the model's name to look for.
	* @return The Lua string representation of "container,MODEL-ID"
	**/
	String ^MainWindow::LuaFindModel(String ^modelName, String ^container) {
		String ^result = nullptr;

		// Only use the specified container
		if (container != nullptr)
			return LuaFindModelHelper(frontend, container, modelName, false);

		// Try default models first
		result = LuaFindModelHelper(frontend, nullptr, modelName, false);
		if (result != nullptr)
			return result;

		// Then try all the other loaded containers
		for each (String ^cont in loadedContainers) {
			result = LuaFindModelHelper(frontend, cont, modelName, false);
			if (result != nullptr)
				return result;
		}

		// Not found
		return nullptr;
	}

	/**
	* LuaFindModel: Finds the first amplitude that matches "ampmodelName" (even partially) inside a container.
	* Example: Finding "cylinder" will result in the Uniform Hollow Cylinder amplitude model.
	* @param container The container to look in, or nullptr for all loaded containers.
	* @param ampmodelName The substring of the model's name to look for.
	* @return The Lua string representation of "container,MODEL-ID"
	**/
	String ^MainWindow::LuaFindAmplitude(String ^ampmodelName, String ^container) {
		String ^result = nullptr;

		// Only use the specified container
		if (container != nullptr)
			return LuaFindModelHelper(frontend, container, ampmodelName, true);

		// Try default models first
		result = LuaFindModelHelper(frontend, nullptr, ampmodelName, true);
		if (result != nullptr)
			return result;

		// Then try all the other loaded containers
		for each (String ^cont in loadedContainers) {
			result = LuaFindModelHelper(frontend, cont, ampmodelName, true);
			if (result != nullptr)
				return result;
		}

		// Not found
		return nullptr;
	}

	delegate int LOF(String ^title, String ^xLabel, String ^yLabel);
	int MainWindow::LuaOpenFigure(String ^title, String ^xLabel, String ^yLabel) {
		if (this->InvokeRequired) {
			return (Int32)Invoke(gcnew LOF(this, &MainWindow::LuaOpenFigure), gcnew array<Object ^>{ title, xLabel, yLabel });
		}

		lastFigure++;
		GraphFigure ^fig = gcnew GraphFigure(lastFigure);
		openFigures[lastFigure] = fig;

		if (title != nullptr) {
			fig->Text = "Figure " + Int32(lastFigure + 1).ToString() + ": " + title;
			fig->graph->GraphTitle = title;
		}
		else {
			fig->Text = "Figure " + Int32(lastFigure + 1).ToString();
			fig->graph->GraphTitle = "";
		}

		if (xLabel != nullptr)
			fig->graph->XLabel = xLabel;
		if (yLabel != nullptr)
			fig->graph->YLabel = yLabel;

		// Register the form-closing event
		fig->FormClosing += gcnew FormClosingEventHandler(this, &MainWindow::Figure_FormClosing);

		/*System::Threading::Thread ^t = gcnew System::Threading::Thread(gcnew System::Threading::ThreadStart(this, &MainWindow::OpenFigureHelper));
		t->Start();*/
		fig->Owner = this;
		fig->Show();

		return lastFigure;
	}

	delegate void LSG(int fignum, LuaTable ^data, String ^color);
	void MainWindow::LuaShowGraph(int fignum, LuaTable ^data, String ^color) {
		if (this->InvokeRequired) {
			Invoke(gcnew LSG(this, &MainWindow::LuaShowGraph), gcnew array<Object ^>{ fignum, data, color });
			return;
		}

		if (data == nullptr)
			return;

		// Using latest figure
		if (fignum == -1)
			fignum = lastFigure;

		// There are no figures
		if (fignum == -1)
			fignum = LuaOpenFigure(nullptr, nullptr, nullptr);

		std::vector<double> x, y;
		LuaTableToData(data, x, y);
		if (x.size() != y.size() || x.size() == 0)
			return;

		GraphFigure ^fig = (GraphFigure ^)(openFigures[fignum]);
		COLORREF col = RGB(255, 0, 0);

		if (color != nullptr) {
			System::Drawing::Color c = System::Drawing::Color::FromName(color);
			col = RGB(c.R, c.G, c.B);
		}

		fig->graph->Add(col, GraphToolkit::Graph1D::GraphDrawType::DRAW_LINES, vectortoarray(x), vectortoarray(y));

		fig->graph->FitToAllGraphs();
		fig->graph->Invalidate();
	}

	void MainWindow::LuaShowFileGraph(int fignum, String ^filename, String ^color) {
		LuaShowGraph(fignum, LuaReadData(filename), color);
	}

	LuaTable ^MainWindow::LuaGenerateCurrent() {
		return LuaGenerateTable(nullptr, nullptr, false);
	}

	LuaTable ^MainWindow::LuaGenerateTable(LuaTable ^ptree, String ^saveFileName, bool saveAmp) {
		cli::array<ModelPtr> ^modstodel = LuaInnerGenerateTable(ptree);

		frontend->WaitForFinish(job);

		int gSize = frontend->GetGraphSize(job);
		if (gSize == 0) {
			MessageBox::Show("No graph returned");
			return nullptr;
		}

		std::vector<double> graph(gSize);

		if (graph.size() < 1 || !frontend->GetGraph(job, &graph[0], gSize)) {
			MessageBox::Show("ERROR getting graph");
			return nullptr;
		}

		LuaTable ^resData = DataToLuaTable(arraytovector(qvec), graph);

		if (saveFileName != nullptr)
		{
			ModelPtr actualModelPtr = -1;
			if (ptree["Populations"] != nullptr) {
				LuaTable ^ps = (LuaTable ^)(ptree["Populations"]);
				if (ps[1] != nullptr) {
					LuaTable ^p = (LuaTable ^)(ps[1]);
					if (p["Models"] != nullptr) {
						LuaTable ^ms = (LuaTable ^)(p["Models"]);
						if (ms[1] != nullptr) {
							LuaTable ^m = (LuaTable ^)(ms[1]);
							if (m["ModelPtr"] != nullptr) {
								actualModelPtr = ModelPtr(Double(m["ModelPtr"]));
							}
						}
					}
				}
			}


			LuaWriteData(saveFileName + ".dat", resData);
			if (saveAmp && modstodel->Length > 0)
				frontend->ExportAmplitude(job, actualModelPtr, clrToWstring(saveFileName + ".ampj").c_str());
		}

		if (modstodel != nullptr) {
			// Cleanup created models
			int numChildren = modstodel->Length;
			for (int i = 0; i < numChildren; i++)
				frontend->DestroyModel(job, modstodel[i], true);
		}

		return resData;
	}

	LuaTable ^MainWindow::LuaFitCurrent() {
		return LuaFit(nullptr, nullptr, nullptr);
	}
	/**
	* LuaFit: Fits a model to the specified data in script.
	*
	* Called without parameters (or only with the third), performs fitting as if the button
	* was pressed in the UI. If called with the first argument, but not the second,
	* fits to data using the current UI parameter tree. If called with both first arguments,
	* fits from input parameter tree.
	* @param data The given data (.x and .y), or nullptr
	* @param ptree The given parameter tree, or nullptr
	* @param fittingProps The given fitting properties, or nullptr
	*					   Available fitting properties:
	*					   1. Iterations: Number of iterations
	*					   2. TODO
	* @return Resulting data and resulting parameters
	**/
	LuaTable ^MainWindow::LuaFit(LuaTable ^data, LuaTable ^ptree, LuaTable ^fittingProps) {

																						// TODO: FIXXXX

		FittingProperties fp;
		fp.bProgressReport = true;
		fp.liveFitting = false;
		fp.msUpdateInterval = 1000;
		fp.wssrFitting = true;
		//	fp.lossFuncType = LossFunction_Enum::TRIVIAL_LOSS;	// TODO::LOSS Add to GUI and read from there

		// Override fitting properties with the given parameters
		if (fittingProps != nullptr) {
			if (fittingProps["Iterations"] != nullptr)
				fp.fitIterations = Int32((Double)fittingProps["Iterations"]);

			// TODO: Override fp with fittingProps
		}

		if (data == nullptr && ptree != nullptr) {
			MessageBox::Show("ERROR: Data not specified");
			return nullptr;
		}

		std::vector<double> qvecv, ivecv;

		if (data != nullptr) {
			if (data["x"] == nullptr || data["y"] == nullptr) {
				MessageBox::Show("ERROR: Invalid data");
				return nullptr;
			}

			LuaTable ^datax = (LuaTable ^)data["x"];
			LuaTable ^datay = (LuaTable ^)data["y"];
			int numpts = datax->Keys->Count;
			if (numpts != datay->Keys->Count) {
				MessageBox::Show("ERROR: Invalid data (2)");
				return nullptr;
			}

			qvecv.resize(numpts);
			ivecv.resize(numpts);
			for (int i = 0; i < numpts; i++) {
				qvecv[i] = (Double)datax[i + 1];
				ivecv[i] = (Double)datay[i + 1];
			}
		}
		else if (qvec != nullptr) {
			qvecv.resize(qvec->Length);
			ivecv.resize(qvecv.size(), 1.0);

			for (int i = 0; i < qvec->Length; i++) {
				qvecv[i] = qvec[i];
				// TODO: ivecv
			}
		}

		ParameterTree pt;
		if (data != nullptr) {
			if (ptree == nullptr)
				ptree = GetParameterTree();

			pt = ParamTreeFromTable(ptree);

			if (pt.GetNodeModel() == compositeModel)
				statusLabel->Text = "Generating(Script) Domain: 0%";
			else {
				MessageBox::Show("ERROR: Root model must be Domain!");
				return nullptr;
			}

			bIsScriptComputing = true;

			std::vector<int> maskTODO(qvecv.size(), 0); // TODO: Mask
			std::string message;
			ErrorCode err = SaveStateAndFit(maskTODO, message);
			//ErrorCode err = frontend->Fit(job, pt, qvecv, ivecv, maskTODO, fp);
			if (err) {
				bIsScriptComputing = false;
				HandleErr("ERROR initializing fitting", err);
				return nullptr;
			}

			EnableStopButton();
		}
		else /*if(data == nullptr)*/ {
			// Call the usual fit (same as pressing the button)
			Fit();
		}

		frontend->WaitForFinish(job);

		ParameterTree outpt;
		ErrorCode err = frontend->GetResults(job, outpt);
		if (err != OK) {
			HandleErr("ERROR getting fitting results", err);
			return nullptr;
		}
		/*if(ptree != nullptr)
		outPTree = ParamTreeToTable(outpt, ptree);
		else
		outPTree = ParamTreeToTable(outpt, GetParameterTree());*/

		int gSize = frontend->GetGraphSize(job);
		if (gSize == 0) {
			MessageBox::Show("No graph returned");
			return nullptr;
		}

		std::vector<double> graph(gSize);

		if (graph.size() < 1 || !frontend->GetGraph(job, &graph[0], gSize)) {
			MessageBox::Show("ERROR getting graph");
			return nullptr;
		}

		if (ptree != nullptr) {
			// Cleanup created models
			int numChildren = pt.GetNumSubModels();
			for (int i = 0; i < numChildren; i++)
				frontend->DestroyModel(job, pt.GetSubModel(i)->GetNodeModel(), true);
		}

		return DataToLuaTable(qvecv, graph);
	}

	LuaTable ^MainWindow::DataToLuaTable(const std::vector<double>& x, const std::vector<double>& y) {
		LuaTable ^data = CreateTable(luaState);

		LuaTable ^xtbl = CreateTable(luaState);
		LuaTable ^ytbl = CreateTable(luaState);
		data["x"] = xtbl;
		data["y"] = ytbl;
		for (UInt32 i = 0; i < x.size(); i++) {
			xtbl[i + 1] = x[i];
			ytbl[i + 1] = y[i];
		}

		return data;
	}

	void MainWindow::LuaTableToData(LuaTable ^data, std::vector<double>& x, std::vector<double>& y) {
		LuaTable ^xtbl = (LuaTable ^)data["x"];
		LuaTable ^ytbl = (LuaTable ^)data["y"];
		unsigned int sz = xtbl->Values->Count;

		x.resize(sz); y.resize(sz);
		for (UInt32 i = 1; i <= sz; i++) {
			x[i - 1] = (Double)xtbl[i];
			y[i - 1] = (Double)ytbl[i];
		}
	}

	LuaTable ^MainWindow::LuaReadData(String ^filename) {
		std::vector<double> xv, yv;
		ReadDataFile(clrToWstring(filename).c_str(), xv, yv);

		return DataToLuaTable(xv, yv);
	}

	bool MainWindow::LuaWriteData(String ^filename, LuaTable ^data) {
		if (this->InvokeRequired) {
			return (bool)Invoke(gcnew BoolFuncStringLuatable(this, &MainWindow::LuaWriteData), gcnew array<Object ^>{ filename, data });
		}

		std::vector<double> xv, yv;

		LuaTableToData(data, xv, yv);

		std::stringstream ss;
		ss << "# Integration parameters:\n";
		ss << "#\tqmax\t" << xv[xv.size() - 1] << "\n";
		ss << "#\tOrientation Method\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->integrationMethodComboBox->Text) << "\n";
		ss << "#\tOrientation Iterations\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->integIterTextBox->Text) << "\n";
		ss << "#\tConvergence\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->convTextBox->Text) << "\n";
		ss << "# \n";
		// TODO::HEADER Get the domain header and add it to the file.
		char *grry;
		int len = 999999;
		grry = new char[len];
		//std::cout << "frontend " << frontend << "\tjob " << job << "\tdomainModel " << domainModel << "\n";
		// The following line causes the program to crash (r6025 pure virtual function call). --> Help?
		frontend->GetDomainHeader(job, compositeModel, grry, len);	// Will crash if called after the submodels have been deleted!
		ss << grry;
		ss << "# Missing header due to Lua problems... Deal.\n";
		delete grry;

		WriteDataFileWHeader(clrToWstring(filename).c_str(), xv, yv, ss);

		return true;
	}

	void MainWindow::LuaMessage(String ^str) {
		MessageBox::Show(str, "Script Message");
	}



}