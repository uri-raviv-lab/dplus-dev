#include "Symmetry.h"
#include "Grid.h"
#include "../backend_version.h"
#include <boost/lexical_cast.hpp>
#include "md5.h"
#include "GPUHeader.h"

#ifdef _WIN32
#include <windows.h> // For LoadLibrary
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif

#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
};

#include "BackendInterface.h"
#include "declarations.h"
#include "UseGPU.h"


int Symmetry::GetNumSubAmplitudes() {
	return (int)_amps.size();
}

Amplitude * Symmetry::GetSubAmplitude(int index) {
	if(index < 0 || index >= (int)_amps.size())
		return NULL;

	return _amps[index];
}

void Symmetry::SetSubAmplitude(int index, Amplitude *subAmp) {
	if(index < 0 || index >= (int)_amps.size())
		return;

	_amps[index] = subAmp;
}

void Symmetry::AddSubAmplitude(Amplitude *subAmp) {
	_amps.push_back(subAmp);
}

void Symmetry::RemoveSubAmplitude(int index) {
	if(index < 0 || index >= (int)_amps.size())
		return;

	_amps.erase(_amps.begin() + index);
}

void Symmetry::ClearSubAmplitudes() {
	_amps.clear();
}

void Symmetry::OrganizeParameters(const VectorXd& p, int nLayers) {
	_ampParams.resize(_amps.size());
	_ampLayers.resize(_amps.size());

	for(unsigned int i = 0; i < _amps.size(); i++)
	{
		size_t subParams = 0;
		const double *subpd = ParameterTree::GetChildParamVec(p.data(), p.size(), i, subParams, 
															  (ParameterTree::GetChildNumChildren(p.data(), i) > 0));	

		if(ParameterTree::GetChildNumChildren(p.data(), i) > 0) {
			// Since we extracted the whole parameter vector, with info
			_ampLayers[i] = int(ParameterTree::GetChildNLayers(p.data(), i) + 0.1);
		} else {
			// Compute the number of layers
			_ampLayers[i] = int(*subpd++ + 0.1);
			subParams--;
		}
		
		_ampParams[i] = VectorXd::Zero(subParams);
		for(unsigned int j = 0; j < subParams; j++) _ampParams[i][j] = subpd[j];

		LocationRotation locrot;
		ParameterTree::GetChildLocationData(p.data(), i, locrot);
		_amps[i]->SetLocationData(locrot.x, locrot.y, locrot.z, locrot.alpha, locrot.beta, locrot.gamma);

		bool bUseGridLevel;
		ParameterTree::GetChildUseGrid(p.data(), i, bUseGridLevel);
		_amps[i]->SetUseGrid(bUseGridLevel);
	}

}

void Symmetry::PreCalculate(VectorXd& p, int nLayers) {
	Amplitude::PreCalculate(p, nLayers);
	OrganizeParameters(p, nLayers);

	for(unsigned int i = 0; i < _amps.size(); i++)
		_amps[i]->PreCalculate(_ampParams[i], _ampLayers[i]);
}

PDB_READER_ERRS Symmetry::CalculateSubAmplitudeGrids( double qMax, int gridSize, progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/, double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL */ ) {
	// Find the minimal qMax from existing grids
	double effqMax = qMax;
	for(int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if(_amps[i]->GetUseGrid() && _amps[i]->GridIsReadyToBeUsed()) {
			// TODO::Spherical - Check to make sure this is still correct with SphereGrid
			double lqmax = _amps[i]->GetGridStepSize() * double(_amps[i]->GetGridSize()) / 2.0;
			effqMax =  std::min(effqMax, lqmax);
			if ((effqMax / qMax) < (1.0 - 0.001)) {
				_amps[i]->ResetGrid();
				std::cout << "reset grid\n";
			}
			
		}
	}

	// If the grid size has changed or something else, we should recalculate the grid or see if one already exists
	for(int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if(pStop && *pStop)
			return STOPPED;

		if(_amps[i]->GetUseGrid() && _amps[i]->GridIsReadyToBeUsed()) {
			if(_amps[i]->GetGridSize() < gridSize) {
				_amps[i]->ResetGrid();	// TODO:Optimization - see if an existing grid exists as a file
			}
		}
	}

	// Calculate grids that need to be evaluated
	for(int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if(pStop && *pStop)
			return STOPPED;

		// Check to see if amplitude needs to be generated and bUseGrid
		double progSec = (progMax - progMin) / double(GetNumSubAmplitudes());
		if(_amps[i]->GetUseGrid() && !_amps[i]->GridIsReadyToBeUsed()) {
			_amps[i]->calculateGrid(qMax, gridSize, progFunc, progArgs, progMin + double(i) * progSec, 
				progMin + double(i + 1) * progSec, pStop);
		} else if(!_amps[i]->GetUseGrid()) { // TODO::Hybrid
			_amps[i]->calculateGrid(qMax, gridSize, progFunc, progArgs, progMin + double(i) * progSec, 
				progMin + double(i + 1) * progSec, pStop);
		}

	}

	if(pStop && *pStop)
		return STOPPED;
	if(progFunc)
		progFunc(progArgs, progMax);

	return PDB_OK;
}

void Symmetry::calculateGrid( FACC qmax, int sections /*= 150*/,
				progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/,
				double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL*/ ) {
	;
	PDB_READER_ERRS dbg = getError();

	// Hybrid
	if(!bUseGrid) {
		PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qmax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);
		return;
	}

	if(gridStatus == AMP_CACHED) {
		if(PDB_OK == ReadAmplitudeFromCache())
			return;
	}

	PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qmax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);
	
	InitializeGrid(qmax, sections);
	// Fill/refill grid

	std::function< std::complex<FACC>(FACC, FACC, FACC)> bindCalcAmp = [=](FACC qx, FACC qy, FACC qz) {
		return this->calcAmplitude(qx, qy, qz);
	};
	grid->Fill(bindCalcAmp, (void *)progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);

	for (int i = 0; i < _amps.size(); i++)
	{
		if (!_amps[i]->ampiscached())
			_amps[i]->WriteAmplitudeToCache();
	}

	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;

}

void Symmetry::GetSubAmplitudeParams(int index, VectorXd& params, int& nLayers) {
	if(index < 0 || index >= _amps.size())
		return;

	params = _ampParams[index];
	nLayers = _ampLayers[index];
}

bool Symmetry::GetUseGridWithChildren() const {
	for(int i = 0; i < _amps.size(); i++) {
		if(!_amps[i]->GetUseGridWithChildren())
			return false;
	}

	return Amplitude::GetUseGridWithChildren();
}

bool Symmetry::GetUseGridAnyChildren() const {
	if(Amplitude::GetUseGrid())
		return true;

	for (int i = 0; i < _amps.size(); i++) {
		if (_amps[i]->GetUseGridAnyChildren())
			return true;
	}

	return false;
}

void ReplaceAll(std::string& str, const std::string& from, const std::string& to)
{
	if (from.size() == 0)
		return;
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
}

bool Symmetry::SavePDBFile(std::ostream &output) {
	std::vector<std::string> lines;
	std::vector<Eigen::Vector3f> locs;

	std::string st = "REMARK This is the header of the D+ state that was used to create the PDB below.\n";
	GetHeader(0, st);

	ReplaceAll(st, "\n", "\nREMARK ");


	bool res = AssemblePDBFile(lines, locs);

	if(lines.size() != locs.size()) {
		std::cout << "Mismatched sizes" << std::endl;
		return false;
	}

	output << st << "\n";
	for(int i = 0; i < locs.size(); i++) {
		std::string line = lines[i];
		std::string xst, yst, zst;
		char grr = line[54];
		xst.resize(24);
		yst.resize(24);
		zst.resize(24);

		sprintf(&xst[0], "%8f", locs[i].x() * 10. );
		sprintf(&yst[0], "%8f", locs[i].y() * 10. );
		sprintf(&zst[0], "%8f", locs[i].z() * 10. );

		sprintf(&line[30], "%s", xst.substr(0,8).c_str());
		sprintf(&line[38], "%s", yst.substr(0,8).c_str());
		sprintf(&line[46], "%s", zst.substr(0,8).c_str());

		line[54] = grr;

		output << line << std::endl;
	}

	return true;
}

bool Symmetry::AssemblePDBFile( std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs, bool electronPDB) {
	int numSubs = GetNumSubLocations();

	Eigen::Vector3f thisTr(tx, ty, tz);
	Eigen::Matrix3f thisRt = EulerD<float>(ra, rb, rg);

	for(int i = 0; i < numSubs; i++)
	{
		std::vector<std::string> iLines;
		std::vector<Eigen::Vector3f> iLocs;
		
		LocationRotation locRot = GetSubLocation(i);
		
		Eigen::Vector3f tr(locRot.x, locRot.y, locRot.z);
		Eigen::Matrix3f rt = EulerD<float>(locRot.alpha, locRot.beta, locRot.gamma);

		for(int k = 0; k < _amps.size(); k++) {
			std::vector<std::string> subLines;
			std::vector<Eigen::Vector3f> subLocs;

			ISymmetry *symmCast = dynamic_cast<ISymmetry*>(_amps[k]);

			if(symmCast)
			{
				// Collect atoms
				symmCast->AssemblePDBFile(subLines, subLocs);

				// Append atoms
				lines.reserve(lines.size() + subLines.size());
				lines.insert(lines.end(), subLines.begin(), subLines.end());
				iLocs.reserve(iLocs.size() + subLocs.size());
				iLocs.insert(iLocs.end(), subLocs.begin(), subLocs.end());
			} //if symmCast

			if (electronPDB) // Solve this redundancy with inheritance or something !!!
			{
				electronPDBAmplitude* pdbCast = dynamic_cast<electronPDBAmplitude*>(_amps[k]);
				if (pdbCast)
				{
					pdbCast->AssemblePDBFile(subLines, subLocs);

					iLines.reserve(iLines.size() + subLines.size());
					iLines.insert(iLines.end(), subLines.begin(), subLines.end());
					iLocs.reserve(iLocs.size() + subLocs.size());
					iLocs.insert(iLocs.end(), subLocs.begin(), subLocs.end());
				} // if pdbCast
			}
			else
			{
				PDBAmplitude* pdbCast = dynamic_cast<PDBAmplitude*>(_amps[k]);
				if (pdbCast)
				{
					pdbCast->AssemblePDBFile(subLines, subLocs);

					iLines.reserve(iLines.size() + subLines.size());
					iLines.insert(iLines.end(), subLines.begin(), subLines.end());
					iLocs.reserve(iLocs.size() + subLocs.size());
					iLocs.insert(iLocs.end(), subLocs.begin(), subLocs.end());
				} // if pdbCast
			}
			

		} // for k < _amps.size

		// Rotate and translate
		for(int p = 0; p < iLocs.size(); p++) {
			iLocs[p] = (rt * iLocs[p] + tr);
			iLocs[p] = thisRt * (iLocs[p]) + thisTr;
		}

		lines.reserve(lines.size() +  iLines.size());
		lines.insert(lines.end(), iLines.begin(), iLines.end());
		locs.reserve(locs.size() + iLocs.size());
		locs.insert(locs.end(), iLocs.begin(), iLocs.end());

	} // for i < numSubs
	return true;
}

bool Symmetry::GetHasAnomalousScattering()
{
	for (const auto& child : _amps)
		if (child->GetHasAnomalousScattering()) return true;
	return false;
}


//////////////////////////////////////////////////////////////////////////
#define LC ((lua_State *)context)

#ifdef _DEBUG
static std::string stackDump (lua_State *L) {
	int i;
	int top = lua_gettop(L);
	char a[256] = {0};
	std::string res = "";
	for (i = 1; i <= top; i++) {  /* repeat for each level */
		int t = lua_type(L, i);
		switch (t) {

		case LUA_TSTRING:  /* strings */
			sprintf(a, "`%s'", lua_tostring(L, i));
			res += a;
			break;

		case LUA_TBOOLEAN:  /* booleans */
			sprintf(a, lua_toboolean(L, i) ? "true" : "false");
			res += a;
			break;

		case LUA_TNUMBER:  /* numbers */
			sprintf(a, "%g", lua_tonumber(L, i));
			res += a;
			break;

		default:  /* other values */
			sprintf(a, "%s", lua_typename(L, t));
			res += a;
			break;

		}
		res += "  ";  /* put a separator */
	}

	return res;
}
#endif

LuaSymmetry::LuaSymmetry(const std::string& scr, void *luaContext) : script(scr), context(luaContext) {
	if(!context) {
		lua_State *L = luaL_newstate();

		luaL_openlibs(L);		
		luaL_dostring(L, "os = nil"); // Remove access to the system
		luaL_dostring(L, "io = nil"); // Remove access to io
		
		context = L;
		bUsingExternalContext = false;		
	} else {
		context = luaContext;
		bUsingExternalContext = true;
	}

	// Compile the lua code and get the appropriate functions
	if(luaL_dostring(LC, scr.c_str())) {
		// Failed to compile or run code
		status = GENERAL_ERROR;
		gridStatus = AMP_UNINITIALIZED;
		return;
	}

	// Verify and set the function pointers
	if(!VerifyAndObtainData()) {
		status = FILE_ERROR;
		gridStatus = AMP_UNINITIALIZED;
		return;
	}

	Im = std::complex<FACC>(0.0, 1.0);
}

static VectorXd getlocrot(lua_State *L, int index) {
	VectorXd locrot = VectorXd::Zero(6);

	lua_pushnumber(L, index);
	lua_gettable(L, -2);  // get p[row]

	if(!lua_istable(L, -1)) // If it's not a table, return default value
		return locrot;

	// Get location and rotation data	
	for(int i = 0; i < 6; i++) {
		lua_pushnumber(L, i + 1);
		lua_gettable(L, -2);  // get p[row][col]

		if (!lua_isnumber(L, -1))
			continue;

		locrot[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);  // remove number 
	}

	lua_pop(L, 1);  // remove p[row]

	return locrot;	
}

void LuaSymmetry::PreCalculate(VectorXd& p, int nLayers) {
	Symmetry::PreCalculate(p, nLayers);

	dol = MatrixXd();

	int err = 0;

	//std::string stack = stackDump(LC);

	scale = p[p[1]-1];

	// Call the populate function with the arguments
	lua_getglobal(LC, "Populate");        // Populate(
	SetTableFromParamVector(p, nLayers);  // p,
	lua_pushnumber(LC, nLayers);          // nLayers

	//stack = stackDump(LC);

	err = lua_pcall(LC, 2, 1, 0);      // );  // 2 arguments, 1 result
	if(err) {
#ifdef _DEBUG
		std::string errstr = lua_tostring(LC, -1);
#endif
		return;
	}


	// Verify that the result is a table
	if(!lua_istable(LC, -1))
		return;

	//stack = stackDump(LC);

	// Create the resulting DOL matrix
	int rows = luaL_getn(LC, -1);
	dol = MatrixXd::Zero(rows, 6);
	MatrixXd grr = MatrixXd::Zero(1, 6);
	rot.clear();
	rot.reserve(rows);
	trans.clear();
	trans.reserve(rows);
	rotVars.clear();
	rotVars.reserve(rows);
	Vector3d vv, vR;

	translationsPerOrientation.clear();

	for(int i = 0; i < rows; i++) {
#ifdef _DEBUG
		grr = getlocrot(LC, i + 1);
#endif
		dol.row(i) = getlocrot(LC, i + 1);
		rot.push_back(EulerD<double>(Degree(dol.row(i)[3]), Degree(dol.row(i)[4]), Degree(dol.row(i)[5])));
		trans.push_back(Vector3d(dol.row(i)[0], dol.row(i)[1], dol.row(i)[2]));
		rotVars.push_back(Vector3d(dol.row(i)[3], dol.row(i)[4], dol.row(i)[5]));
		translationsPerOrientation[rot[i]].push_back(trans[i]);

	}
}

LuaSymmetry::~LuaSymmetry() {
	if(!bUsingExternalContext)
		lua_close((lua_State *)context); 
}

static int getfield_int(lua_State *L, const char *key) {
	int result;
	lua_pushstring(L, key);
	lua_gettable(L, -2);  // get table[key]
	if (!lua_isnumber(L, -1))
		return -1;
	result = (int)lua_tonumber(L, -1);
	lua_pop(L, 1);  // remove number 
	return result;
}

static bool getfield_boolean(lua_State *L, const char *key) {
	bool result;
	lua_pushstring(L, key);
	lua_gettable(L, -2);  // get table[key] 
	if (!lua_isboolean(L, -1))
		return false;
	result = lua_toboolean(L, -1) ? true : false;
	lua_pop(L, 1);  // remove boolean 
	return result;
}

static double getfield_double(lua_State *L, const char *key) {
	double result;
	lua_pushstring(L, key);
	lua_gettable(L, -2);  // get table[key] 
	if (!lua_isnumber(L, -1))
		return -1.0;
	result = lua_tonumber(L, -1);
	lua_pop(L, 1);  // remove number 
	return result;
}

static std::string getfield_string(lua_State *L, const char *key) {
	std::string result;	
	lua_pushstring(L, key);
	lua_gettable(L, -2);  // get table[key] 
	if (!lua_isstring(L, -1))
		return "(nil)";
	result = lua_tostring(L, -1);
	lua_pop(L, 1);  // remove string 
	return result;
}

static void setfield_double(lua_State *L, int index, double value) {
	lua_pushnumber(L, index);
	lua_pushnumber(L, value);

	lua_rawset(L, -3);    // set table[index] = value
}

bool LuaSymmetry::VerifyAndObtainData() {
	// TODO::Lua: Later, GetExtraParameter and GetDisplayValue 
	// Information (table), Populate (function)
	lua_getglobal(LC, "Information");
	lua_getglobal(LC, "Populate");
	if (!lua_istable(LC, -2))
		status = GENERAL_ERROR;
	if (!lua_isfunction(LC, -1))
		status = GENERAL_ERROR;

	// Pop the function(s)
	lua_pop(LC, 1);

	// Obtain information from information table (the information is obtained inside 
	// the frontend anyway)
	if(getfield_string(LC, "Type").compare("Symmetry"))
		return false;

	nlp = getfield_int(LC, "NLP");
	if(nlp <= 0)
		return false;

	// Pop the table (leave an empty stack)
	lua_pop(LC, 1);

	return true;
}

void LuaSymmetry::SetTableFromParamVector(VectorXd& p, int nLayers) {
	// Get the actual parameters
	size_t sz = 0;
	const double *actualP = ParameterTree::GetNodeParamVec(p.data(), p.size(), sz, false);

	// Store the parameters
	luaParams.resize(nLayers, nlp);

	// Skip nLayers
	actualP++;

	lua_newtable(LC); // Create table

	for(int i = 0; i < nLayers; i++) {
		//std::string stack = stackDump(LC);

		lua_newtable(LC); // Create table[row]

		for(int j = 0; j < nlp; j++) {			
			setfield_double(LC, j + 1, actualP[j * nLayers + i]);
			luaParams(i,j) = actualP[j * nLayers + i];
		}

		//stack = stackDump(LC);

		// Set the parameter table at row i		
		lua_pushnumber(LC, i + 1);
		lua_insert(LC, -2);    // Switch stack locations

		//stack = stackDump(LC);

		lua_rawset(LC, -3);    // set table[i] = table[row]

		//stack = stackDump(LC);		
	}
}

//////////////////////////////////////////////////////////////////////////
// Actual amplitude computation                                         //
//////////////////////////////////////////////////////////////////////////

std::complex<FACC> LuaSymmetry::calcAmplitude(FACC qx, FACC qy, FACC qz) {	
	int rows = dol.rows();
	Vector3d Q(qx,qy,qz);

	std::complex<FACC> result (0.0, 0.0);
	Vector3d qnew;

	for(int i = 0; i < rows; i++) {
		std::complex<FACC> res_i(0.0, 0.0);
		for(unsigned int subAmpNo = 0; subAmpNo < _amps.size(); subAmpNo++) {
			qnew = Vector3d(Q.transpose() * rot[i]);
			res_i += _amps[subAmpNo]->getAmplitude(qnew.x(), qnew.y(), qnew.z());
		}
		result += res_i * exp(Im * (Q.dot(trans[i])));
	}
	return result;
}

void LuaSymmetry::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String("Scripted Symmetry");


	writer.Key("Scale");
	writer.Double(scale);

	writer.Key("Position");
	writer.StartArray();
	writer.Double(tx);
	writer.Double(ty);
	writer.Double(tz);
	writer.EndArray();

	writer.Key("Rotation");
	writer.StartArray();
	writer.Double(ra);
	writer.Double(rb);
	writer.Double(rg);
	writer.EndArray();

	std::string scriptHeader;
	std::stringstream ss;
	ss << this->script;
	while (!ss.eof()) {
		std::string line;
		getline(ss, line);
		scriptHeader.append(line + "\n");
	}
	writer.Key("Script");
	writer.String(scriptHeader.c_str());

	//TODO: Header for parameters (not implemented correctly in original header func) 

	writer.Key("Coordinates");
	writer.StartArray();
	for (int i = 0; i < dol.rows(); i++) {
		writer.StartArray();
		for (int j = 0; j < dol.cols(); j++) {
			writer.Double( dol(i, j));
		}
		writer.EndArray();
	}
	writer.EndArray();

	writer.Key("Used grid");
	writer.Bool(this->bUseGrid);

	writer.Key("SubModels");
	writer.StartArray();
	for (int i = 0; i < _amps.size(); i++) {
		writer.StartObject();
		_amps[i]->GetHeader(depth + 1, writer);
		writer.EndObject();
	}
	writer.EndArray();

}

void LuaSymmetry::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers;
	ampers.resize(depth+1, '#');
	ampers.append(" ");

	std::stringstream ss;

	if(depth == 0) {
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
	}
	
	header.append(ampers + "//////////////////////////////////////\n");

	ss << "Scripted symmetry\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Position (" << tx << "," << ty << "," << tz << ")\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Rotation (" <<  ra << "," << rb << "," << rg << ")\n";
	header.append(ampers + ss.str());
	ss.str("");



	ss << this->script;
	while(!ss.eof()) {
		std::string line;
		getline(ss, line);
		header.append(ampers + line + "\n");	
	}
	ss.str("");

	header.append(ampers + "Parameters:\n");
 	header.append(ampers);
	for(int i = 0; i < nlp; i++) {
		// TODO::Header This doesn't actually call the right function (it's not implemented) and NLP isn't defined as it should be
		header.append("\t" + GetLayerParamName(i, NULL));
	}
	header.append("\n");
	char tempBuf[20];
	for(int i = 0; i < luaParams.rows(); i++) {
		string tmp = GetLayerName(i);	// TODO::Header This doesn't actually call the right function (it's not implemented)

		if (std::string::npos != tmp.find("%d"))
		{
			sprintf(tempBuf, tmp.c_str(), i);
		}

 		for(int j = 0; j < luaParams.cols(); j++) {
			tmp.append("\t" + boost::lexical_cast<std::string>(luaParams(i,j)));
 		}
		tmp.append("\n");
	 	header.append(ampers + tmp);
 	}
	// TODO::Header Add extra parameters if there are any

	header.append(ampers + "Coordinates:\n");
	for(int i = 0; i < dol.rows(); i++) {
		string tmp = "";
		for(int j = 0; j < dol.cols(); j++) {
			ss << "\t" << dol(i,j);
			tmp.append("\t" + boost::lexical_cast<std::string>(dol(i,j)));
		}
		ss << "\n";
		tmp.append("\n");
		header.append(ampers + tmp);
		ss.str("");
	}

	string tmp2 = ampers + "Used grid: " + boost::lexical_cast<std::string>(this->bUseGrid) + "\n";
	header.append(tmp2);

	for(int i = 0; i < _amps.size(); i++) {
		_amps[i]->GetHeader(depth+1, header);
	}
}

std::string LuaSymmetry::Hash() const
{
	std::string str = BACKEND_VERSION "Lua Symmetry: ";
	str += script;
	for (const auto& t : trans)
		str += std::to_string(t.x()) + std::to_string(t.y()) + std::to_string(t.z());
	for (const auto& r : rotVars)
		str += std::to_string(r.x()) + std::to_string(r.y()) + std::to_string(r.z());

	for (const auto& child : _amps)
		str += child->Hash();

	return md5(str);
}

std::string LuaSymmetry::GetName() const {
	return "Lua Symmetry";
}

void LuaSymmetry::calculateGrid( FACC qMax, int sections /*= 150*/, progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/, double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL*/ ) {
	// Copied from ManualSymmetry::calculateGrid
	if(gridStatus == AMP_CACHED) {
		if(PDB_OK == ReadAmplitudeFromCache())
			return;
	}
	//gpuCalcManSym
	
	if(g_useGPUAndAvailable) {
		// double, double
		gpuCalcManSym = (GPUCalculateManSymmetry_t)GPUCalcManSymJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUCalcManSymJacobSphrDF");
	}

	// TODO::Hybrid take into account for kernel
	if(!bUseGrid) {
		Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, progMin, progMax, pStop);
		return;
	}

	if (!bUseGPU || !g_useGPUAndAvailable || (gpuCalcManSym == NULL)) {
		Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, progMin, progMax, pStop);
		return;
	}

	PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qMax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);

	if(res != PDB_OK) {
		status = res;
		return;
	}

	InitializeGrid(qMax, sections);
	memset(grid->GetDataPointer(), 0, grid->GetRealSize());

	int copies = trans.size()+1;	// Plus one for the inner object(s)
	std::vector<double> xx(copies), yy(copies), zz(copies);
	std::vector<double> aa(copies), bb(copies), cc(copies);
	for(int i = 1; i < copies; i++) {
		xx[i] = trans  [i-1](0);
		yy[i] = trans  [i-1](1);
		zz[i] = trans  [i-1](2);
		aa[i] = Radian(Degree(rotVars[i-1](0)));	// Convert to radians
		bb[i] = Radian(Degree(rotVars[i-1](1)));	// Convert to radians
		cc[i] = Radian(Degree(rotVars[i-1](2)));	// Convert to radians
	}
	copies--;	// Minus one so that the inner object isn't counted twice

	long long voxels = grid->GetRealSize() / (sizeof(double) * 2);
	const int thetaDivs = grid->GetDimY(1) - 1;
	const int phiDivs = grid->GetDimZ(1,1);

	int gpuRes = 0;

	for(int i = 0; i < _amps.size() && gpuRes == 0; i++) {
		JacobianSphereGrid *JGrid = (JacobianSphereGrid*)(_amps[i]->GetInternalGridPointer());
#ifdef _DEBUG
		JGrid->DebugMethod();
#endif
		double aa0, bb0, cc0;
		_amps[i]->GetTranslationRotationVariables(xx[0], yy[0], zz[0], aa0, bb0, cc0);
		aa[0] = aa0; bb[0] = bb0; cc[0] = cc0;
		gpuRes = gpuCalcManSym(voxels, thetaDivs, phiDivs, copies, grid->GetStepSize(),
			_amps[i]->GetDataPointer(), grid->GetDataPointer(), JGrid->GetInterpolantPointer(),
			&xx[0], &yy[0], &zz[0], aa.data(), bb.data(), cc.data(), scale, progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
	}
	grid->RunAfterReadingCache(); // CalculateSplines

	if(gpuRes != 0) {
		std::cout << "Error in kernel: " << gpuRes << ". Starting CPU calculations." << std::endl;
		Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
		return;
	}

	for(int i = 0; i < _amps.size(); i++)
		_amps[i]->WriteAmplitudeToCache();

	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;

}

bool LuaSymmetry::Populate(const VectorXd& p, int nLayers) {
	VectorXd tmp = p;
	PreCalculate(tmp, nLayers);
	return true;
}

unsigned int LuaSymmetry::GetNumSubLocations() {
	return (unsigned int)trans.size();
}

LocationRotation LuaSymmetry::GetSubLocation(int posindex) {
	if(posindex < 0 || posindex >= GetNumSubLocations())
		return LocationRotation();

	return LocationRotation(trans[posindex].x(), trans[posindex].y(), trans[posindex].z(),
		Radian(Degree(rotVars[posindex].x())), Radian(Degree(rotVars[posindex].y())), Radian(Degree(rotVars[posindex].z())));
}

bool LuaSymmetry::CalculateGridGPU( GridWorkspace& workspace ) {
	if(!g_useGPUAndAvailable)
		return false;

	bool res = true;

	if (!gpuManSymmetryHybridAmplitude)
		gpuManSymmetryHybridAmplitude = (GPUHybridManSymmetryAmplitude_t)GPUHybrid_ManSymmetryAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_ManSymmetryAmplitudeDLL");
	if (!gpuManSymmetrySemiHybridAmplitude)
		gpuManSymmetrySemiHybridAmplitude = (GPUSemiHybridManSymmetryAmplitude_t)GPUSemiHybrid_ManSymmetryAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUSemiHybrid_ManSymmetryAmplitudeDLL");

	for(int i = 0; i < _amps.size(); i++) {
		IGPUGridCalculable *tst = dynamic_cast<IGPUGridCalculable*>(_amps[i]);

		if(tst && tst->ImplementedHybridGPU())	// Calculate directly on the GPU
		{
			if(!gpuManSymmetryHybridAmplitude)
			{
				std::cout << "GPUHybrid_ManSymmetryAmplitudeDLL not loaded. Skipping model." << std::endl;
				res = false;
				continue;
			}
			workspace.calculator->SetNumChildren(workspace, 1);
			tst->SetModel(workspace.children[0]);
			workspace.children[0].computeStream = workspace.computeStream;
			tst->CalculateGridGPU(workspace.children[0]);

			double3 tr, rt;
			_amps[i]->GetTranslationRotationVariables(tr.x, tr.y, tr.z, rt.x, rt.y, rt.z);
			// TODO:: Deal with rotLoc

			res &= gpuManSymmetryHybridAmplitude(workspace, workspace.children[0], f4(tr), f4(rt));

			workspace.calculator->FreeWorkspace(workspace.children[0]);
		}
		else	// Calculate using CalculateGrid and add to workspace.d_amp
		{
			if(!gpuManSymmetrySemiHybridAmplitude)
			{
				std::cout << "GPUSemiHybrid_ManSymmetryAmplitudeDLL not loaded. Skipping model." << std::endl;
				res = false;
				continue;
			}

			_amps[i]->calculateGrid(workspace.qMax, 2*(workspace.qLayers-4));

			double3 tr, rt;
			_amps[i]->GetTranslationRotationVariables(tr.x, tr.y, tr.z, rt.x, rt.y, rt.z);
			// TODO:: Deal with rotLoc
			gpuManSymmetrySemiHybridAmplitude(workspace, _amps[i]->GetDataPointer(), f4(tr), f4(rt));
		}	//  if else tst
	}	// for i

	workspace.calculator->SetNumChildren(workspace, 0);

	float4 dummy;
	dummy.x = 0.;
	dummy.y = 0.;
	dummy.z = 0.;
	dummy.w = 0.;

	// Calculate splines
	if(gpuManSymmetrySemiHybridAmplitude)
		gpuManSymmetrySemiHybridAmplitude(workspace, NULL, dummy, dummy);

	return res;
}

bool LuaSymmetry::SetModel( GridWorkspace& workspace ) {
	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuHybridSetSymmetry)
		gpuHybridSetSymmetry = (GPUHybridSetSymmetries_t)GPUHybrid_SetSymmetryDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_SetSymmetryDLL");
	if(!gpuHybridSetSymmetry)
		return false;

	std::vector<float4> locs(rotVars.size()), rots(rotVars.size());

	for(int i = 0; i < rotVars.size(); i++) {
		locs[i] = f4(trans[i]);
//		rots[i] = f3(rotVars[i]);
		rots[i].x = Radian(Degree(float(rotVars[i].x())));
		rots[i].y = Radian(Degree(float(rotVars[i].y())));
		rots[i].z = Radian(Degree(float(rotVars[i].z())));
	}

	workspace.scale = scale;

	return gpuHybridSetSymmetry(workspace, locs.data(), rots.data(), int(rots.size()));
}

bool LuaSymmetry::ImplementedHybridGPU() {
	return true;
}

ArrayXcX LuaSymmetry::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
{
	// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
	FACC st = sin(theta);
	FACC ct = cos(theta);
	FACC sp = sin(phi);
	FACC cp = cos(phi);
	const auto Q = relevantQs[0];

	Eigen::Vector3d Qcart(Q * st*cp, Q * st * sp, Q*ct), Qt;
	Eigen::Matrix3d rot;
	Eigen::Vector3d R(tx, ty, tz);
	Qt = (Qcart.transpose() * RotMat) / Q;

	double newTheta = acos(Qt.z());
	double newPhi = atan2(Qt.y(), Qt.x());

	if (newPhi < 0.0)
		newPhi += M_PI * 2.;

	ArrayXcX phases = (
		std::complex<FACC>(0., 1.) *
		(Qt.dot(R) *
		Eigen::Map<const Eigen::ArrayXd>(relevantQs.data(), relevantQs.size()))
		).exp();

	ArrayXcX reses(relevantQs.size());
	reses.setZero();
	if (GetUseGridWithChildren())
	{
		JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

		if (jgrid)
		{
			// Get amplitudes from grid
			return jgrid->getAmplitudesAtPoints(relevantQs, newTheta, newPhi) * phases;
		}
	}

	if (GetUseGridAnyChildren())
	{

		int rows = int(trans.size());

		ArrayXcX results(relevantQs.size());
		results.setZero();
		Vector3d qnew;

		for (auto& orientation : translationsPerOrientation)
		{
			ArrayXcX res_i(relevantQs.size());
			res_i.setZero();

			qnew = Vector3d(Qt.transpose() * orientation.first).normalized();

			newTheta = acos(qnew.z());
			newPhi = atan2(qnew.y(), qnew.x());

			if (newPhi < 0.0)
				newPhi += M_PI * 2.;

			for (auto& subAmp : _amps)
			{
				res_i += subAmp->getAmplitudesAtPoints(relevantQs, newTheta, newPhi);
			}
			ArrayXcX phases(relevantQs.size());
			phases.setZero();

			Eigen::Vector3d qDirection(st * cp, st * sp, ct);
			for (auto& trans : orientation.second)
			{
				std::complex<FACC> tPhase = std::complex<FACC>(0., 1.) * (qDirection.dot(trans));
				for (int j = 0; j < relevantQs.size(); j++)
				{
					phases(j) += exp(relevantQs[j] * tPhase);
				}
			}

			results += res_i * phases;
		}

		return results * scale;

	} // if GetUseGridAnyChildren


	return scale * getAmplitudesAtPointsWithoutGrid(newTheta, newPhi, relevantQs, phases);
}



