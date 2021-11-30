// TODO::INI
/*#pragma once

#include "BackwardsCompatibility.h"
#include <sstream>

using std::string;

////////////////////////////////////////
//////////Internal definitions//////////
////////////////////////////////////////

enum OldModelType { MODEL_ROD, MODEL_SPHERE, MODEL_SLAB,
				 MODEL_ASLAB, MODEL_HELIX, MODEL_RECT, MODEL_CYLINDROID, MODEL_EMULSION, MODEL_DELIX};

typedef struct {
	std::vector<double> r, ed, cs;
	std::vector<char> rMutable, edMutable, csMutable;
	std::vector<double> rmin , rmax, edmin , edmax, extramin, extramax, csmin, csmax;
	std::vector<int> rminInd, rmaxInd, edminInd, edmaxInd, csminInd, csmaxInd;
	std::vector<int> rLinkInd, edLinkInd, csLinkInd;
	bool bConstraints;
} oldParamLayers;

typedef struct {
	oldParamLayers pl;
	std::vector<double> exParams;
	std::vector<bool> mutex;
	int modelType;
	bool b_polydisp;
	double polydispValue;
	int polydispInd;
} oldParamStruct;

// Helper template
template <class T>
std::string StringOf(const T& object)
{
	std::ostringstream os;
	os << object;
	return std::string(os.str());
}

inline void toEnumString(const std::string &s, int en, std::string &result) {
	result = s + StringOf(en);
}

inline std::string toEnumString(const std::string &s, int en) {
	return s + StringOf(en);
}

OldModelType GetModelType(std::string obj) {
	if(obj.compare("Uniform Slab") == 0 ||
		obj.compare("Symmetric Uniform Slabs") == 0 ||
		obj.compare("Symmetric Gaussian Slabs") == 0 ||
		obj.compare("Membrane") == 0)
		return MODEL_SLAB;
	if(	obj.compare("Asymmetric Uniform Slabs") == 0 ||
		obj.compare("Asymmetric Gaussian Slabs") == 0)
		return MODEL_ASLAB;

	if(obj.compare("Uniform Hollow Cylinder") == 0 || 
		obj.compare("Gaussian Hollow Cylinder") == 0 || 
		obj.compare("Rod") == 0)
		return MODEL_ROD;
	if(obj.compare("Uniform Sphere") == 0 || 
		obj.compare("Gaussian Sphere") == 0)
		return MODEL_SPHERE;
	if(obj.compare("Discrete Helix") == 0 ||
		obj.compare("Gaussian Discrete Helix") == 0)
		return MODEL_DELIX;
	if(obj.compare("Helix") == 0)
		return MODEL_HELIX;
	if(obj.compare("Cuboid") == 0)
		return MODEL_RECT;


	return MODEL_ASLAB;
}

bool hasGaussianModel(std::string obj) {
	if(obj.compare("Symmetric Uniform Slabs") == 0 ||
		obj.compare("Asymmetric Uniform Slabs") == 0 ||	// Update as models are added
		obj.compare("Symmetric Gaussian Slabs") == 0 ||	//
		obj.compare("Asymmetric Gaussian Slabs") == 0 ||	//
		obj.compare("Membrane") == 0 ||	//
		obj.compare("Uniform Sphere") == 0 ||	//
		obj.compare("Gaussian Sphere") == 0 ||	//
		obj.compare("Uniform Hollow Cylinder") == 0 || 
		obj.compare("Gaussian Hollow Cylinder") == 0 || 
		obj.compare("Rod") == 0 ||
		//obj.compare("") == 0 ||	//
		//obj.compare("") == 0 ||	//
		false)
		return true;
	return false;

	//switch(GetModelType(obj)) {
	//	case MODEL_ASLAB:
	//	case MODEL_SLAB:
	//	case MODEL_SPHERE:
	//	case MODEL_ROD:
	//	case MODEL_DELIX:
	//		return true;
	//	default:
	//		return false;
	//}
}	//end hasGaussianModel

void ReadMutability(const std::wstring& file, std::string object, oldParamStruct *p, void *ini) {
	int paramNum;

	paramNum = GetIniInt(file, object, "layers", ini);
	
	std::string a;
	for( int i=0; i < paramNum; i++) {
		toEnumString("radmutable", (i + 1), a);
		p->pl.rMutable.push_back(GetIniChar(file,object,a, ini));
		toEnumString("EDMutable", (i + 1), a);
		p->pl.edMutable.push_back(GetIniChar(file,object,a, ini));

		toEnumString("csMutable", (i + 1), a);
		p->pl.csMutable.push_back(GetIniChar(file,object,a, ini));
	}

	// Extra parameters:
	p->mutex.push_back(GetIniChar(file, "ScaleMut", a, ini) == 'Y');
	p->mutex.push_back(GetIniChar(file, "BackgroundMut", a, ini) == 'Y');
	// Model specific extra params:
	if(GetModelType(object) == MODEL_CYLINDROID || GetModelType(object) == MODEL_HELIX ||
		GetModelType(object) == MODEL_ROD)
		p->mutex.push_back(GetIniChar(file, "HeightMut", a, ini) == 'Y');
	if(GetModelType(object) == MODEL_DELIX)
		p->mutex.push_back(GetIniChar(file, "Number_of_Spheres_Mut", a, ini) == 'Y');
	if(GetModelType(object) == MODEL_CYLINDROID)
		p->mutex.push_back(GetIniChar(file, "Short_inner_radiusMut", a, ini) == 'Y');
	if(GetModelType(object) == MODEL_HELIX) {
		p->mutex.push_back(GetIniChar(file, "Helix_RadiusMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "PitchMut", a, ini) == 'Y');
	}
	if(GetModelType(object) == MODEL_DELIX) {
		p->mutex.push_back(GetIniChar(file, "Helix_RadiusMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "PitchMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "Helix_RadiusMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "Water_SpacingMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "Debye_WallerMut", a, ini) == 'Y');
		p->mutex.push_back(GetIniChar(file, "PitchMut", a, ini) == 'Y');
	}

}

void ReadRanges(const std::wstring& file, std::string object, oldParamStruct *p, void *ini) {
	int paramNum;

	paramNum = GetIniInt(file, object, "layers", ini);
	
	std::string a;
	for( int i = 0; i < paramNum; i++) {
		toEnumString("RMinInd", (i + 1), a);
		p->pl.rminInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("RMaxInd", (i + 1), a);
		p->pl.rmaxInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("RLinkInd", (i + 1), a);
		p->pl.rLinkInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("EDMinInd", (i + 1), a);
		p->pl.edminInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("EDMaxInd", (i + 1), a);
		p->pl.edmaxInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("EDLinkInd", (i + 1), a);
		p->pl.edLinkInd.push_back(GetIniInt(file, object, a, ini, -1));
				
		
		toEnumString("RMinNum", (i + 1), a);
		p->pl.rmin.push_back(GetIniDouble(file, object, a, ini));
		toEnumString("RMaxNum", (i + 1), a);
		p->pl.rmax.push_back(GetIniDouble(file, object, a, ini));
		toEnumString("EDMinNum", (i + 1), a);
		p->pl.edmin.push_back(GetIniDouble(file, object, a, ini));
		toEnumString("EDMaxNum", (i + 1), a);
		p->pl.edmax.push_back(GetIniDouble(file, object, a, ini));
		
		toEnumString("CSMinInd", (i + 1), a);
		p->pl.csminInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("CSMaxInd", (i + 1), a);
		p->pl.csmaxInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("CSLinkInd", (i + 1), a);
		p->pl.csLinkInd.push_back(GetIniInt(file, object, a, ini, -1));
		toEnumString("CSMinNum", (i + 1), a);
		p->pl.csmin.push_back(GetIniDouble(file, object, a, ini));
		toEnumString("CSMaxNum", (i + 1), a);
		p->pl.csmax.push_back(GetIniDouble(file, object, a, ini));
	}

	paramNum = GetIniNoOfParameters(file, object, "EXMinNum", ini);

	for(int i = 0; i < paramNum; i++) {	
		toEnumString("EXMinNum", (i + 1), a);
		p->pl.extramin.push_back(GetIniDouble(file, object, a));
		toEnumString("EXMaxNum", (i + 1), a);
		p->pl.extramax.push_back(GetIniDouble(file, object, a));
	}

	if(paramNum < (int)p->exParams.size()) {
		p->pl.extramax.resize(p->exParams.size(), -1.0);
		p->pl.extramin.resize(p->exParams.size(), -1.0);
	}

	p->pl.bConstraints = GetIniChar(file, object, "Constraints") == 'Y';
}

////////////////////////////////////////
/////Functions that can be accessed/////
////////////////////////////////////////

void GetOldIni(const std::wstring &filename, std::string objN, paramStruct *pNew, void* iniP) {
	return;
	//CSimpleIniA *ini = (CSimpleIniA*)iniP;
	//int paramNum;
	//oldParamStruct *p;
	//p = new oldParamStruct();
	//std::string obj = MapObjToStr(objN);

	//paramNum = GetIniInt(filename, obj, "layers", ini);

	//// No such model in the ini file
	//if(!IniHasModelType(filename, obj, ini))
	//	return;

	//if(hasGaussianModel(obj))
	//	p->modelType = GetIniInt(filename, obj, "ModelType_", ini);


	//for(int i = 0; i < paramNum; i++) {
	//	std::string a;
	//	toEnumString("radius", (i + 1), a);
	//	p->pl.r.push_back(GetIniDouble(filename,obj,(a), ini));
	//	toEnumString("elecdensity", (i + 1), a);
	//	p->pl.ed.push_back(GetIniDouble(filename,obj,(a), ini));
	//	toEnumString("crosssection", (i + 1), a);
	//	p->pl.cs.push_back(GetIniDouble(filename,obj,(a), ini));
	//}

	//p->b_polydisp = (GetIniChar(filename, obj, "PolyDispChecked", ini) == 'Y');
	//p->polydispInd = GetIniInt(filename, obj, "PolyDispIndex", ini, -1);
	//double tmpD = GetIniDouble(filename, obj, "PolyDispVal", ini);
	//p->polydispValue = (tmpD < 0.0) ? 0.0 : tmpD;

	//// Get extra parameters
	//p->exParams.push_back(GetIniDouble(filename, obj, "Scale", ini));
	//p->exParams.push_back(GetIniDouble(filename, obj, "Background", ini));

	//if(GetModelType(obj) == MODEL_ROD || 
	//	   GetModelType(obj) == MODEL_CYLINDROID || GetModelType(obj) == MODEL_HELIX)
	//	p->exParams.push_back(GetIniDouble(filename, obj, "Height", ini));

	//if (GetModelType(obj) == MODEL_DELIX) {
	//		p->exParams.push_back(GetIniDouble(filename, obj, "Number_of_Spheres_", ini));
	//		p->exParams.push_back(GetIniDouble(filename, obj, "Helix_Radius", ini));
	//		p->exParams.push_back(GetIniDouble(filename, obj, "Pitch", ini));
	//		p->exParams.push_back(GetIniDouble(filename, obj, "Water_Spacing", ini));
	//		p->exParams.push_back(GetIniDouble(filename, obj, "Debye_Waller", ini));
	//}
	//
	//if(GetModelType(obj) == MODEL_CYLINDROID)
	//	p->exParams.push_back(GetIniDouble(filename, obj, "Short_inner_radius", ini));

	//if(GetModelType(obj) == MODEL_HELIX) {
	//	p->exParams.push_back(GetIniDouble(filename, obj, "Helix_Radius", ini));
	//	p->exParams.push_back(GetIniDouble(filename, obj, "Pitch", ini));
	//}
	//
	//if(GetModelType(obj) == MODEL_EMULSION) {
	//	p->exParams.push_back(GetIniDouble(filename, obj, "I(0)", ini));
	//	p->exParams.push_back(GetIniDouble(filename, obj, "I(max)", ini));
	//	p->exParams.push_back(GetIniDouble(filename, obj, "q(max)", ini));
	//}

	//ReadRanges(filename, obj, p, ini);
	//ReadMutability(filename, obj, p, ini);


	//// Convert old parameters to new parameters
	//// FF Layers
	//int nlp = pNew->model->GetNumLayerParams();
	//pNew->layers = p->pl.r.size();
	//pNew->params.resize(nlp);
	//for(int i = 0; i < nlp; i++)
	//	pNew->params[i].resize(p->pl.r.size());
	//for(int i = 0; i < (int)p->pl.r.size(); i++) {
	//	Parameter par(p->pl.r[i], p->pl.rMutable[i] == 'Y', p->pl.bConstraints,
	//		p->pl.rmin[i], p->pl.rmax[i], p->pl.rminInd[i], p->pl.rmaxInd[i], p->pl.rLinkInd[i]);
	//	pNew->params[0][i] = par;
	//}
	//for(int i = 0; i < (int)p->pl.ed.size(); i++) {
	//	Parameter par(p->pl.ed[i], p->pl.edMutable[i] == 'Y', p->pl.bConstraints,
	//		p->pl.edmin[i], p->pl.edmax[i], p->pl.edminInd[i], p->pl.edmaxInd[i], p->pl.edLinkInd[i]);
	//	pNew->params[1][i] = par;
	//}
	//if(nlp == 3) {
	//	for(int i = 0; i < (int)p->pl.cs.size(); i++) {
	//		Parameter par(p->pl.cs[i], p->pl.csMutable[i] == 'Y', p->pl.bConstraints,
	//			p->pl.csmin[i], p->pl.csmax[i], p->pl.csminInd[i], p->pl.csmaxInd[i], p->pl.csLinkInd[i]);
	//		pNew->params[2][i] = par;
	//	}
	//}
	//
	//// FF extra parameters
	//for(int i = 0; i < (int)p->exParams.size(); i++) {
	//	Parameter par(p->exParams[i], p->mutex[i], (p->pl.extramax[i] > 0.0 && p->pl.extramin[i] > 0.0 ),
	//		p->pl.extramin[i], p->pl.extramax[i], -1, -1, -1);
	//	pNew->extraParams.push_back(par);
	//}

	//if(p)
	//	delete p;
	//p = NULL;

	//// Rename old ini file to "Old_"+filename
	//peakStruct peaks;
	//ReadPeaks(filename, obj, &peaks, ini);
	//phaseStruct ps;
	//int phaseInd;
	//ReadPhases(filename, obj, &ps, &phaseInd, ini);
	//bgStruct BGs;
	//ReadBG(filename, obj, &BGs, ini);

	////Rename
	//
	//std::wstring oldFilename = L"";
	//oldFilename.append(filename);
	//oldFilename.insert(oldFilename.find_last_of(L"\\") + 1, L"Old_");
	//int errCode = _wrename(filename.c_str(),  oldFilename.c_str()); 

	//if(errCode == 0) {
	//	
	//	// Save new version of ini file
	//	WriteParameters(filename, objN, pNew, ini);
	//	WritePeaks(filename, objN, &peaks, ini);
	//	WritePhases(filename, objN, &ps, phaseInd, ini);
	//	WriteBG(filename, objN, &BGs, ini);

	//	//if(_model->HasSpecializedSF()) {
	//	//	graphTable caille;
	//	//	cailleParamStruct cailleP;
	//	//	if(cailleParamListView->Items->Count > 0) {
	//	//		GetCailleFromGUI(&caille, &cailleP);
	//	//		WriteCaille(filename, type, &caille, &cailleP);
	//	//	}
	//	//}
	//	}
}	// end GetOldIni

std::string MapObjToStr(std::string objN) {
	switch(GetModelType(objN))
	{
	case MODEL_ROD:
		return "Rod";
	case MODEL_SPHERE:
		return "Sphere";
	case MODEL_CYLINDROID:
		return "Cylindroid";
	case MODEL_SLAB:
		return "Membrane";
	case MODEL_ASLAB:
		return "Slab";
	case MODEL_RECT:
		return "Rectangular";
	case MODEL_HELIX:
		return "Helix";
	case MODEL_DELIX:
		return "Discrete Helix";
	}
	return "Parameters";
}
*/
