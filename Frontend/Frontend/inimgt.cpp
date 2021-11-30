//#pragma once
#include <sstream>

#include "inimgt.h"
#include "ModelUI.h"
#include "FrontendExported.h"
//#include "inimgtHelper.cpp"
//#include "BackwardsCompatibility.h" // TODO::INI Reinstate!

using std::string;

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

// Basic get/set functions
void GetIniString(const std::wstring& file, const std::string& section, const std::string& key, std::string& result) {
	CSimpleIniA ini(false, false, false);
	GetIniString(file, section, key, result, &ini);
}
void GetIniString(const std::wstring& file, const std::string& section, const std::string& key, std::string& result, void* ini) {
	result.clear();

	if(!ReadableIni(file, section, ini))
		return;

	const char *res =  ((CSimpleIniA*)ini)->GetValue(section.c_str(), key.c_str());
	if(!res)
		result = "N/A";
	else
		result = string(res);
}

void SetIniString (const std::wstring& file, const std::string& object, const std::string& param,
				   const std::string& value) {
	CSimpleIniA ini(false, false, false);
	SetIniString(file, object, param, &ini, value);
}
void SetIniString (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
				   const std::string& value) {
	((CSimpleIniA*)ini)->SetValue(object.c_str(), param.c_str(), value.c_str());
}

char GetIniChar (const std::wstring& file, const std::string& object, const std::string& param) {
	CSimpleIniA ini(false, false, false);
	return GetIniChar(file, object, param, &ini);
}
char GetIniChar (const std::wstring& file, const std::string& object, const std::string& param, void* ini) {
	std::string temp;

	GetIniString(file, object, param, temp, ini);

	if(temp[0] == '1')
		temp[0] = 'Y';
	else if(temp[0] == '0')
		temp[0] = 'N';
	else
		temp[0] = '-';

	return temp[0];
}

void SetIniChar (const std::wstring& file, const std::string& object, const std::string& param,
				 char value) {
	CSimpleIniA ini(false, false, false);
	SetIniChar(file, object, param, &ini, value);
}
void SetIniChar (const std::wstring& file, const std::string& object,const std::string& param, void* ini,
				 char value) {
	char x[2] = {0};

	if(value == 'Y')
		x[0] = '1';
	else if(value == 'N')
		x[0] = '0';
	else
		return;

	SetIniString(file, object, param, ini, x);
}

double GetIniDouble (const std::wstring& file, const std::string& object,const std::string& param) {
	CSimpleIniA ini(false, false, false);
	return GetIniDouble(file, object, param, &ini);
}
double GetIniDouble (const std::wstring& file, const std::string& object,const std::string& param, void* ini) {
	std::string temp;

	GetIniString(file, object, param, temp, ini);

	if(temp.empty())
		temp = "0.0";

	if(temp.find("INF") != std::string::npos || 
	   temp.find("inf") != std::string::npos) {
		if(temp.at(0) == '-')
			return -std::numeric_limits<double>::infinity();
		return std::numeric_limits<double>::infinity();
	}

	if (strcmp(temp.c_str(), "N/A") == 0) return -1.0;
	else return strtod(temp.c_str(), NULL);
}

void SetIniDouble (const std::wstring& file, const std::string& object,const std::string& param,
				   double value, int precision) {
	CSimpleIniA ini(false, false, false);
	SetIniDouble(file, object, param, &ini, value, precision);
}
void SetIniDouble (const std::wstring& file, const std::string& object,const std::string& param, void* ini,
				   double value, int precision) {
	char x[256] = {0};

	if(precision < 0)
		sprintf(x, "N/A");
	else
		sprintf(x, "%.*f", precision, value);

	SetIniString(file, object, param, ini, x);
}

int GetIniNoOfParameters (const std::wstring& file, const std::string& object, const std::string& param, void* ini) {
	std::string temp;

	for(int i = 1; ; i++) {
		GetIniString(file, object, toEnumString(param, i), temp, ini);
		if(temp.at(0) == 'N')
			return i - 1;
	}

	return 0; 
}

int GetIniInt (const std::wstring& file, const std::string& object, const std::string& param) {
	CSimpleIniA ini(false, false, false);
	return GetIniInt(file, object, param, &ini);
}
int GetIniInt (const std::wstring& file, const std::string& object, const std::string& param, void* ini) {
	std::string temp;

	GetIniString(file, object, param, temp, ini);

	if(temp.empty() || strcmp(temp.c_str(), "N/A") == 0)
		temp = "0";

	return atoi(temp.c_str());
}

int GetIniInt(const std::wstring& file, const std::string& object, const std::string& param, void* ini, int defval) {
	std::string temp;

	GetIniString(file, object, param, temp, ini);

	if(temp.empty() || strcmp(temp.c_str(), "N/A") == 0)
		temp = StringOf(defval);

	return atoi(temp.c_str());
}

void SetIniInt (const std::wstring& file, const std::string& object, const std::string& param,
				int value) {
	CSimpleIniA ini(false, false, false);
	SetIniInt(file, object, param, &ini, value);
}
void SetIniInt (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
				int value) {
	char x[64] = {0};

	sprintf(x, "%d", value);

	SetIniString(file, object, param, ini, x);
}

// The real deal

bool ReadParameters(const std::wstring &filename, string obj, paramStruct *p, ModelUI &modelInfo, void* ini) {
	if(!ReadableIni(filename, obj, ini))
		return false;

	if(!IniHasModelType(filename, obj, ini))
		return false;
	// TODO::INI
	//if(GetIniInt(filename, obj, "iniVersion", ini) < INI_VERSION) {
	//	// If none, try and get the old version
	//	GetOldIni(filename, obj, p, ini);
	//	return true;
	//}

	p->layers = GetIniInt(filename, obj, "layers", ini);

	p->params.resize(p->nlp);

	for(int i = 0; i < p->nlp; i++) {
		p->params[i].clear();

		for(int j = 0; j < p->layers; j++) {
			std::string a;
			toEnumString(modelInfo.GetLayerParamName(i), (j + 1), a);

			Parameter param (GetIniDouble(filename, obj, a, ini),
							 GetIniChar(filename, obj, a + "mut", ini) == 'Y',
							 GetIniChar(filename, obj, a + "Cons", ini) == 'Y',
							 GetIniDouble(filename, obj, a + "min", ini),
							 GetIniDouble(filename, obj, a + "max", ini),
							 GetIniInt(filename, obj, a + "minind", ini),
							 GetIniInt(filename, obj, a + "maxind", ini),
							 GetIniInt(filename, obj, a + "linkind", ini),
							 GetIniDouble(filename, obj, a + "sigma", ini));

			p->params[i].push_back(param);

		}
	}

	// Get extra parameters
	p->extraParams.clear();
	for(int i = 0; i < p->nExtraParams; i++) {
		std::string a = modelInfo.GetExtraParameter(i).name;

		bool isinf = GetIniChar(filename, obj, a + "inf", ini) == 'Y';

		Parameter param (isinf ? std::numeric_limits<double>::infinity() :
						 GetIniDouble(filename, obj, a, ini),
						 GetIniChar(filename, obj, a + "mut", ini) == 'Y',
						 GetIniChar(filename, obj, a + "Cons", ini) == 'Y',
						 GetIniDouble(filename, obj, a + "min", ini),
						 GetIniDouble(filename, obj, a + "max", ini),
						 GetIniInt(filename, obj, a + "minind", ini),
						 GetIniInt(filename, obj, a + "maxind", ini),
						 GetIniInt(filename, obj, a + "linkind", ini),
						 GetIniDouble(filename, obj, a + "sigma", ini));

		p->extraParams.push_back(param);
	}

	return true;
}

void WriteParameters(const std::wstring &filename, string obj, paramStruct *p, ModelUI &modelInfo, void* ini) {
	if(!WritableIni(filename, obj, ini))
		return;

	// The integer indicates the version of the ini files. Version 0 is the
	//	pre-abstraction version.
	SetIniInt(filename, obj, "iniVersion", ini, INI_VERSION);
	
	SetIniInt(filename, obj, "layers", ini, p->layers);

	for(int i = 0; i < p->nlp; i++) {
		for(int j = 0; j < p->layers; j++) {
			const Parameter &param = p->params[i][j];
			std::string a;
			toEnumString(modelInfo.GetLayerParamName(i), (j + 1), a);
			

			SetIniDouble(filename, obj, a, ini, param.value);
			SetIniChar(filename, obj, a + "mut", ini, param.isMutable ? 'Y' : 'N');
			SetIniChar(filename, obj, a + "Cons", ini, param.isConstrained ? 'Y' : 'N');
			SetIniDouble(filename, obj, a + "min", ini, param.consMin);
			SetIniDouble(filename, obj, a + "max", ini, param.consMax);
			SetIniInt(filename, obj, a + "minind", ini, param.consMinIndex);
			SetIniInt(filename, obj, a + "maxind", ini, param.consMaxIndex);
			SetIniInt(filename, obj, a + "linkind", ini, param.linkIndex);
			SetIniDouble(filename, obj, a + "sigma", ini, param.sigma);
		}
	}

	// Set extra parameters
	for(int i = 0; i < modelInfo.GetNumExtraParams(); i++) {
		const Parameter& param = p->extraParams[i];
		std::string a = modelInfo.GetExtraParameter(i).name;
		bool canbeinfinite = modelInfo.GetExtraParameter(i).canBeInfinite;

		SetIniDouble(filename, obj, a, ini, param.value, 
					 modelInfo.GetExtraParameter(i).decimalPoints);
		SetIniChar(filename, obj, a + "mut", ini, param.isMutable ? 'Y' : 'N');
		SetIniChar(filename, obj, a + "Cons", ini, param.isConstrained ? 'Y' : 'N');
		SetIniDouble(filename, obj, a + "min", ini, param.consMin);
		SetIniDouble(filename, obj, a + "max", ini, param.consMax);
		SetIniInt(filename, obj, a + "minind", ini, param.consMinIndex);
		SetIniInt(filename, obj, a + "maxind", ini, param.consMaxIndex);
		SetIniInt(filename, obj, a + "linkind", ini, param.linkIndex);
		SetIniChar(filename, obj, a + "inf", ini, 
				  (canbeinfinite && !_finite(param.value)) ? 'Y' : 'N');
		SetIniDouble(filename, obj, a + "sigma", ini, param.sigma);
	}
}

//void ReadPeaks(const std::wstring &filename, std::string obj, peakStruct *peaks, void* ini) {
//	if(!ReadableIni(filename, obj, ini))
//		return;
//
//	//Backwards Compatibility -- Will need to be updated when SF is abstracted
//	if(GetIniInt(filename, obj, "iniVersion", ini) < 1)
//		obj = MapObjToStr(obj);
//
//	
//	int peakNum = GetIniInt(filename, obj, "peaks", ini);
//
//	for(int i = 0; i < peakNum; i++) {
//		std::string a;
//		toEnumString("amplitude", (i + 1), a);
//		peaks->amp.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("fwhm", (i + 1), a);
//		peaks->fwhm.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("center", (i + 1), a);
//		peaks->center.push_back(GetIniDouble(filename, obj, a, ini));
//
//		toEnumString("ampmutable", (i + 1), a);
//		peaks->amp.back().mut = GetIniChar(filename, obj, a, ini);
//		toEnumString("fwhmmutable", (i + 1), a);
//		peaks->fwhm.back().mut = GetIniChar(filename, obj, a, ini);
//		toEnumString("centermutable", (i + 1), a);
//		peaks->center.back().mut = GetIniChar(filename, obj, a, ini);
//	}
//
//	SetPeakType((PeakType)GetIniInt(filename, obj, "peakshape", ini));
//
//}
//
//void WritePeaks(const std::wstring &filename, std::string obj, const peakStruct *peaks, void* ini) {
//	if(!WritableIni(filename, obj, ini))
//		return;
//
//	int peakNum = peaks->amp.size();
//
//	SetIniInt(filename, obj, "peaks", ini, peakNum);
//	
//
//	for(int i = 0; i < peakNum; i++) {
//		std::string a;
//		toEnumString("Amplitude", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, peaks->amp[i].value);
//
//		toEnumString("FWHM", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, peaks->fwhm[i].value);
//
//		toEnumString("Center", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, peaks->center[i].value);
//
//		toEnumString("AmpMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, peaks->amp[i].mut);
//
//		toEnumString("FWHMMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, peaks->fwhm[i].mut);
//		
//		toEnumString("CenterMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, peaks->center[i].mut);
//	}
//
//	SetIniInt(filename, obj, "peakshape", ini, (int)GetPeakType());
//
//}
//
//void WriteBG(const std::wstring &filename, std::string obj, const bgStruct *BGs, void* ini) {
//	if(!WritableIni(filename, obj, ini))
//		return;
//
//	int bgNum = BGs->base.size();
//
//	SetIniInt(filename, obj, "BGFunctions", ini, bgNum);
//	
//	for(int i = 0; i < bgNum; i++) {
//		std::string a;
//		
//		toEnumString("Type", (i + 1), a);
//		SetIniInt(filename, obj, a, ini, (int)BGs->type[i]);
//
//		toEnumString("Base", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->base[i]);
//
//		toEnumString("BGCenter", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->center[i]);
//
//		toEnumString("Decay", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->decay[i]);
//
//		toEnumString("BaseMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, BGs->baseMutable[i]);
//
//		toEnumString("BGCenterMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, BGs->centerMutable[i]);
//
//		toEnumString("DecayMutable", (i + 1), a);
//		SetIniChar(filename, obj, a, ini, BGs->decayMutable[i]);
//
//		toEnumString("BaseMin", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->basemin[i]);
//		toEnumString("BaseMax", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->basemax[i]);
//
//		toEnumString("DecayMin", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->decmin[i]);
//		toEnumString("DecayMax", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->decmax[i]);
//
//		toEnumString("CenterMin", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->centermin[i]);
//		toEnumString("CenterMax", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, BGs->centermax[i]);
//	}
//}
//
//void ReadBG(const std::wstring &filename, std::string obj, bgStruct *BGs, void* ini) {
//	if(!ReadableIni(filename, obj, ini))
//		return;
//
//	//Backwards Compatibility -- Will need to be updated when BG is abstracted
//	if(GetIniInt(filename, obj, "iniVersion", ini) < 1)
//		obj = MapObjToStr(obj);
//
//
//	int bgNum = GetIniInt(filename, obj, "BGFunctions", ini);
//
//	for(int i = 0; i < bgNum; i++) {
//		std::string a;
//
//		toEnumString("Type", (i + 1), a);
//		BGs->type.push_back((BGFuncType)GetIniInt(filename, obj, a, ini));
//		
//		toEnumString("Base", (i + 1), a);
//		BGs->base.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("BGCenter", (i + 1), a);
//		BGs->center.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("Decay", (i + 1), a);
//		BGs->decay.push_back(GetIniDouble(filename, obj, a, ini));
//
//		toEnumString("BaseMutable", (i + 1), a);
//		BGs->baseMutable.push_back(GetIniChar(filename, obj, a, ini));
//		toEnumString("BGCenterMutable", (i + 1), a);
//		BGs->centerMutable.push_back(GetIniChar(filename, obj, a, ini));
//		toEnumString("DecayMutable", (i + 1), a);
//		BGs->decayMutable.push_back(GetIniChar(filename, obj, a, ini));
//
//		toEnumString("BaseMin", (i + 1), a);
//		BGs->basemin.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("BaseMax", (i + 1), a);
//		BGs->basemax.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("DecayMin", (i + 1), a);
//		BGs->decmin.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("DecayMax", (i + 1), a);
//		BGs->decmax.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("CenterMin", (i + 1), a);
//		BGs->centermin.push_back(GetIniDouble(filename, obj, a, ini));
//		toEnumString("CenterMax", (i + 1), a);
//		BGs->centermax.push_back(GetIniDouble(filename, obj, a, ini));
//	}
//}
//
//void WriteCaille(const std::wstring &filename, std::string obj, const graphTable *Cailles, const cailleParamStruct *Caillesdrawing, void* ini) {
//	if(!WritableIni(filename, obj, ini))
//		return;
//
//	int cailleNum = Cailles->x.size();
//
//	SetIniInt(filename, obj, "CaillePeaks", ini, cailleNum);
//	
//	for(int i = 0; i < cailleNum; i++) {
//		std::string a;
//		
//		toEnumString("CailleEta", (i + 1), a);
//		SetIniDouble(filename, obj, a, ini, Cailles->y[i]);
//
//		toEnumString("CailleH", (i + 1), a);
//		SetIniInt(filename, obj, a, ini, (int)(Cailles->x[i]));
//
//	}
//
//	SetIniDouble(filename, obj, "CailleAmpVal", ini, Caillesdrawing->amp.at(0).value);
//	SetIniDouble(filename, obj, "CailleAmpMin", ini, Caillesdrawing->amp.at(0).min);
//	SetIniDouble(filename, obj, "CailleAmpMax", ini, Caillesdrawing->amp.at(0).max);
//	SetIniChar(  filename, obj, "CailleAmpMut", ini, Caillesdrawing->amp.at(0).mut);
//
//	SetIniDouble(filename, obj, "CailleEtaVal", ini, Caillesdrawing->eta.at(0).value);
//	SetIniDouble(filename, obj, "CailleEtaMin", ini, Caillesdrawing->eta.at(0).min);
//	SetIniDouble(filename, obj, "CailleEtaMax", ini, Caillesdrawing->eta.at(0).max);
//	SetIniChar(  filename, obj, "CailleEtaMut", ini, Caillesdrawing->eta.at(0).mut);
//
//	SetIniDouble(filename, obj, "CailleNVal", ini, Caillesdrawing->N.at(0).value);
//	SetIniDouble(filename, obj, "CailleNMin", ini, Caillesdrawing->N.at(0).min);
//	SetIniDouble(filename, obj, "CailleNMax", ini, Caillesdrawing->N.at(0).max);
//	SetIniChar(  filename, obj, "CailleNMut", ini, Caillesdrawing->N.at(0).mut);
//
//	SetIniDouble(filename, obj, "CailleSigmaVal", ini, Caillesdrawing->sig.at(0).value);
//	SetIniDouble(filename, obj, "CailleSigmaMin", ini, Caillesdrawing->sig.at(0).min);
//	SetIniDouble(filename, obj, "CailleSigmaMax", ini, Caillesdrawing->sig.at(0).max);
//	SetIniChar(  filename, obj, "CailleSigmaMut", ini, Caillesdrawing->sig.at(0).mut);
//
//	SetIniDouble(filename, obj, "CailleNDiffVal", ini, Caillesdrawing->N_diffused.at(0).value);
//	SetIniDouble(filename, obj, "CailleNDiffMin", ini, Caillesdrawing->N_diffused.at(0).min);
//	SetIniDouble(filename, obj, "CailleNDiffMax", ini, Caillesdrawing->N_diffused.at(0).max);
//	SetIniChar(  filename, obj, "CailleNDiffMut", ini, Caillesdrawing->N_diffused.at(0).mut);
//
//	
//}
//
//void ReadCaille(const std::wstring &filename, std::string obj, graphTable *Cailles, cailleParamStruct *Caillesdrawing, void* ini) {
//	if(!ReadableIni(filename, obj, ini))
//		return;
//
//	//Backwards Compatibility -- Will need to be updated when SF is abstracted
//	if(GetIniInt(filename, obj, "iniVersion", ini) < 1)
//		obj = MapObjToStr(obj);
//
//
//	int cailleNum = GetIniInt(filename, obj, "CaillePeaks", ini);
//
//	for(int i = 0; i < cailleNum; i++) {
//		std::string a;
//
//		toEnumString("CailleEta", (i + 1), a);
//		Cailles->y.push_back(GetIniDouble(filename, obj, a, ini));
//		
//		toEnumString("CailleH", (i + 1), a);
//		Cailles->x.push_back(GetIniInt(filename, obj, a, ini));
//	}
//	Caillesdrawing->amp.resize(1);
//	Caillesdrawing->eta.resize(1);
//	Caillesdrawing->N.resize(1);
//	Caillesdrawing->sig.resize(1);
//	Caillesdrawing->N_diffused.resize(1);
//
//	Caillesdrawing->amp.at(0).value	= GetIniDouble(filename, obj, "CailleAmpVal", ini);
//	Caillesdrawing->amp.at(0).min	= GetIniDouble(filename, obj, "CailleAmpMin", ini);
//	Caillesdrawing->amp.at(0).max	= GetIniDouble(filename, obj, "CailleAmpMax", ini);
//	Caillesdrawing->amp.at(0).mut	= GetIniChar(  filename, obj, "CailleAmpMut", ini);
//
//	Caillesdrawing->eta.at(0).value	= GetIniDouble(filename, obj, "CailleEtaVal", ini);
//	Caillesdrawing->eta.at(0).min	= GetIniDouble(filename, obj, "CailleEtaMin", ini);
//	Caillesdrawing->eta.at(0).max	= GetIniDouble(filename, obj, "CailleEtaMax", ini);
//	Caillesdrawing->eta.at(0).mut	= GetIniChar(  filename, obj, "CailleEtaMut", ini);
//
//	Caillesdrawing->N.at(0).value	= GetIniDouble(filename, obj, "CailleNVal", ini);
//	Caillesdrawing->N.at(0).min		= GetIniDouble(filename, obj, "CailleNMin", ini);
//	Caillesdrawing->N.at(0).max		= GetIniDouble(filename, obj, "CailleNMax", ini);
//	Caillesdrawing->N.at(0).mut		= GetIniChar(  filename, obj, "CailleNMut", ini);
//	
//	Caillesdrawing->sig.at(0).value	= GetIniDouble(filename, obj, "CailleSigmaVal", ini);
//	Caillesdrawing->sig.at(0).min	= GetIniDouble(filename, obj, "CailleSigmaMin", ini);
//	Caillesdrawing->sig.at(0).max	= GetIniDouble(filename, obj, "CailleSigmaMax", ini);
//	Caillesdrawing->sig.at(0).mut	= GetIniChar(  filename, obj, "CailleSigmaMut", ini);
//
//	Caillesdrawing->N_diffused.at(0).value	= GetIniDouble(filename, obj, "CailleNDiffVal", ini);
//	Caillesdrawing->N_diffused.at(0).min	= GetIniDouble(filename, obj, "CailleNDiffMin", ini);
//	Caillesdrawing->N_diffused.at(0).max	= GetIniDouble(filename, obj, "CailleNDiffMax", ini);
//	Caillesdrawing->N_diffused.at(0).mut	= GetIniChar(  filename, obj, "CailleNDiffMut", ini);
//	
//	if(Caillesdrawing->N_diffused.at(0).value < 0.0) {
//		Caillesdrawing->N_diffused.at(0).value = 0.0;
//		Caillesdrawing->N_diffused.at(0).min = 0.0;
//		Caillesdrawing->N_diffused.at(0).max = 0.0;
//		Caillesdrawing->N_diffused.at(0).mut = 'N';
//	}
//
//}
//
//void WritePhases(const std::wstring &filename, std::string obj, const phaseStruct *ph, int pt, void* ini) {
//	if(!WritableIni(filename, obj, ini))
//		return;
//
//	SetIniInt(filename, obj, "PhaseType", ini, pt);
//
//	SetIniDouble(filename, obj, "Phase_a", ini, ph->a);
//	SetIniDouble(filename, obj, "Phase_b", ini, ph->b);
//	SetIniDouble(filename, obj, "Phase_c", ini, ph->c);
//	SetIniDouble(filename, obj, "Phase alpha", ini, ph->alpha);
//	SetIniDouble(filename, obj, "Phase beta", ini, ph->beta);
//	SetIniDouble(filename, obj, "Phase gamma", ini, ph->gamma);
//
//	SetIniDouble(filename, obj, "Phase amin", ini, ph->amin);
//	SetIniDouble(filename, obj, "Phase amax", ini, ph->amax);
//	SetIniDouble(filename, obj, "Phase bmin", ini, ph->bmin);
//	SetIniDouble(filename, obj, "Phase bmax", ini, ph->bmax);
//	SetIniDouble(filename, obj, "Phase cmin", ini, ph->cmin);
//	SetIniDouble(filename, obj, "Phase cmax", ini, ph->cmax);
//
//	SetIniDouble(filename, obj, "Phase alphmin", ini, ph->alphamin);
//	SetIniDouble(filename, obj, "Phase alphmax", ini, ph->alphamax);
//	SetIniDouble(filename, obj, "Phase betmin", ini, ph->betamin);
//	SetIniDouble(filename, obj, "Phase betmax", ini, ph->betamax);
//	SetIniDouble(filename, obj, "Phase gammmin", ini, ph->gammamin);
//	SetIniDouble(filename, obj, "Phase gammmax", ini, ph->gammamax);
//
//	SetIniChar(filename, obj, "PhaseAMut", ini, ph->aM);
//	SetIniChar(filename, obj, "PhaseBMut", ini, ph->bM);
//	SetIniChar(filename, obj, "PhaseCMut", ini, ph->cM);
//	SetIniChar(filename, obj, "PhaseAlphaMut", ini, ph->alphaM);
//	SetIniChar(filename, obj, "PhaseBetaMut", ini, ph->betaM);
//	SetIniChar(filename, obj, "PhaseGammaMut", ini, ph->gammaM);
//}
//
//void ReadPhases(const std::wstring &filename, std::string obj, phaseStruct *ph, int *pt, void* ini) {
//	if(!ReadableIni(filename, obj, ini))
//		return;
//
//	//Backwards Compatibility
//	if(GetIniInt(filename, obj, "iniVersion", ini) < 1)
//		obj = MapObjToStr(obj);
//
//
//	*pt				 = GetIniInt(filename, obj, "PhaseType", ini);
//
//	ph->a			 = GetIniDouble(filename, obj, "Phase_a", ini);
//	ph->b			 = GetIniDouble(filename, obj, "Phase_b", ini);
//	ph->c			 = GetIniDouble(filename, obj, "Phase_c", ini);
//	ph->alpha		 = GetIniDouble(filename, obj, "Phase alpha", ini);
//	ph->beta		 = GetIniDouble(filename, obj, "Phase beta", ini);
//	ph->gamma		 = GetIniDouble(filename, obj, "Phase gamma", ini);
//
//	ph->amin		 = GetIniDouble(filename, obj, "Phase amin", ini);
//	ph->amax		 = GetIniDouble(filename, obj, "Phase amax", ini);
//	ph->bmin		 = GetIniDouble(filename, obj, "Phase bmin", ini);
//	ph->bmax		 = GetIniDouble(filename, obj, "Phase bmax", ini);
//	ph->cmin		 = GetIniDouble(filename, obj, "Phase cmin", ini);
//	ph->cmax		 = GetIniDouble(filename, obj, "Phase cmax", ini);
//
//	ph->alphamin	 = GetIniDouble(filename, obj, "Phase alphmin", ini);
//	ph->alphamax	 = GetIniDouble(filename, obj, "Phase alphmax", ini);
//	ph->betamin		 = GetIniDouble(filename, obj, "Phase betmin", ini);
//	ph->betamax		 = GetIniDouble(filename, obj, "Phase betmax", ini);
//	ph->gammamin	 = GetIniDouble(filename, obj, "Phase gammmin", ini);
//	ph->gammamax	 = GetIniDouble(filename, obj, "Phase gammmax", ini);
//
//	ph->aM			 = GetIniChar(filename, obj, "PhaseAMut", ini);
//	ph->bM			 = GetIniChar(filename, obj, "PhaseBMut", ini);
//	ph->cM			 = GetIniChar(filename, obj, "PhaseCMut", ini);
//	ph->alphaM		 = GetIniChar(filename, obj, "PhaseAlphaMut", ini);
//	ph->betaM		 = GetIniChar(filename, obj, "PhaseBetaMut", ini);
//	ph->gammaM		 = GetIniChar(filename, obj, "PhaseGammaMut", ini);
//}
//
bool IniHasModelType(const std::wstring& file, const std::string& object, void* ini) {

	if(!(ReadableIni(file, object, ini)))
		return false;

    CSimpleIniA::TNamesDepend sections;
	((CSimpleIniA*)ini)->GetAllSections(sections);

	if(sections.empty())
		return false;

	std::list<CSimpleIniA::Entry>::iterator it;
	for(it = sections.begin(); it != sections.end(); it++) {
		if(strcmp((*it).pItem, object.c_str()) == 0)
			return true;
	}

	return false;
}

bool ReadableIni(const std::wstring& file, const std::string& object, void* ini) {
	if(ini == NULL)
		return false;
	if(((CSimpleIniA*)ini)->IsEmpty()) 
		if(((CSimpleIniA*)ini)->LoadFile(file.c_str()) < 0)
			return false;
	return true;
}

bool WritableIni(const std::wstring& file, const std::string& object, void* ini) {
	if(!ReadableIni(file, object, ini)) {
		// There may not be a file. Make one and try again.
		int errNo = ((CSimpleIniA*)ini)->SaveFile(file.c_str());
		if(((CSimpleIniA*)ini)->LoadFile(file.c_str()) < 0)
			return false;
	}


	return true;
}

// Instantiates ini as a CSimpleIniA
void *NewIniFile() {
	return new CSimpleIniA(false, false, false);
}

// Saves and deletes ini
void SaveAndCloseIniFile(const std::wstring& file, const std::string& object, void* ini) {
	if(ini) {
		if(file.length() > 0 && object.length() > 0)
			if(WritableIni(file, object, ini))
				((CSimpleIniA*)ini)->SaveFile(file.c_str());

		delete ini;
	}
	ini = NULL;
}

// Deletes ini
void CloseIniFile(void* ini) {
	if(ini)
		delete ini;
	ini = NULL;
}
