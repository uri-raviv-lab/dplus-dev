#ifndef __FRONTENDEXPORTED_H
#define __FRONTENDEXPORTED_H

#pragma once
//#include "ModelUI.h"
#include "Common.h"

class ModelUI;

// File management
//////////////////////////////////////////////////////////////////////////

EXPORTED int CheckSizeOfFile(const wchar_t *filename);

EXPORTED void ReadDataFile(const wchar_t *filename,
				           std::vector<double>& x, 
						   std::vector<double>& y);

EXPORTED void Read1DDataFile(const wchar_t *filename,
				             std::vector<double>& x);

EXPORTED void WriteDataFile(const wchar_t *filename, std::vector<double>& x, 
                   std::vector<double>& y);

EXPORTED void WriteDataFileWHeader(const wchar_t *filename, std::vector<double>& x, 
				std::vector<double>& y, std::stringstream& header);

EXPORTED void Write3ColDataFile(const wchar_t *filename, std::vector<double>& x, 
							std::vector<double>& y, std::vector<double>& err);

EXPORTED void Write1DDataFile(const wchar_t *filename, std::vector<double>& x);

EXPORTED void GetDirectory(const wchar_t *file, wchar_t *result, int n = 260);
EXPORTED void GetBasename(const wchar_t *file, wchar_t *result, int n = 260);

/*
EXPORTED std::string GetDirectory(const std::string file);
EXPORTED std::string GetBasename(const std::string file) ;
*/

EXPORTED bool ReadParameters(const std::wstring &filename, std::string obj, paramStruct *p, ModelUI &modelInfo, void* ini);
EXPORTED void WriteParameters(const std::wstring &filename, std::string obj, paramStruct *p, ModelUI &modelInfo, void* ini);

/* TODO::INI
EXPORTED void ReadPeaks(const std::wstring &filename, std::string obj, peakStruct *peaks, void* ini);
EXPORTED void WritePeaks(const std::wstring &filename, std::string obj, const peakStruct *peaks, void* ini);

EXPORTED void WriteBG(const std::wstring &filename, std::string obj, const bgStruct *BGs, void* ini);
EXPORTED void ReadBG(const std::wstring &filename, std::string obj, bgStruct *BGs, void* ini);

EXPORTED void WriteCaille(const std::wstring &filename, std::string obj, const graphTable *Cailles, const cailleParamStruct *Caillesdrawing, void* ini);
EXPORTED void ReadCaille(const std::wstring &filename, std::string obj, graphTable *Cailles, cailleParamStruct *Caillesdrawing, void* ini);

EXPORTED void WritePhases(const std::wstring &filename, std::string obj, const phaseStruct *ph, int pt, void* ini);
EXPORTED void ReadPhases(const std::wstring &filename, std::string obj, phaseStruct *ph, int *pt, void* ini);
*/
EXPORTED bool IniHasModelType(const std::wstring& file, const std::string& object, void* ini);


EXPORTED double GetIniDouble (const std::wstring& file, const std::string& object, const std::string& param, void* ini);
EXPORTED void   SetIniDouble (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
				     double value, int precision = 6);

EXPORTED int  GetIniNoOfParameters (const std::wstring& file, const std::string& object, const std::string& param, void* ini);

EXPORTED int  GetIniInt (const std::wstring& file, const std::string& object, const std::string& param, void* ini);
EXPORTED int  GetIniInt (const std::wstring& file, const std::string& object, const std::string& param, void* ini, int defval);
EXPORTED void SetIniInt (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
			    int value);

EXPORTED char GetIniChar (const std::wstring& file, const std::string& object, const std::string& param, void* ini);
EXPORTED void SetIniChar (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
				 char value);

EXPORTED void GetIniString(const std::wstring& file, const std::string& section, const std::string& key, 
				   std::string& result, void* ini);
EXPORTED void SetIniString (const std::wstring& file, const std::string& object, const std::string& param, void* ini,
				   const std::string& value);
 
EXPORTED bool ReadableIni(const std::wstring& file, const std::string& object, void* ini);
EXPORTED bool WritableIni(const std::wstring& file, const std::string& object, void* ini);

// Instantiates ini as a CSimpleIniA
EXPORTED void *NewIniFile();

/**
 * Saves and closes the ini file and releases *ini.
 *
 * @param file The path and filename to be saved to
 *
 * @param object The section to check to see if the file exists and is readable 
 *
 * @param ini The instantiated pointer with the SimpleIni object to be written and released
**/
EXPORTED void SaveAndCloseIniFile(const std::wstring& file, const std::string& object, void* ini);

/**
 * Closes the ini file and releases *ini.
 *
 * @param ini The instantiated pointer with the SimpleIni object to be released
**/
EXPORTED void CloseIniFile(void* ini);

EXPORTED void AutoBaselineGen(const std::vector<double>& datax, const std::vector<double>& datay, std::vector<double>& bgy);

// Smoothing
//////////////////////////////////////////////////////////////////////////
EXPORTED void smoothVector(int strength, std::vector<double>& data);

EXPORTED std::vector<double> bilateralFilter(std::vector<double> y, std::vector<double> x, double sigD, double sigR);

EXPORTED std::vector <double> MachineResolutionF(const std::vector <double> &q ,const std::vector <double> &orig, double width);

// Correlation coefficients
//////////////////////////////////////////////////////////////////////////
EXPORTED double WSSR(const std::vector<double> &first, const std::vector<double> &second, bool bLogScale = false);

EXPORTED double RSquared(std::vector<double> data, std::vector<double> fit, bool bLogScale = false);

EXPORTED double WSSR_Masked(const std::vector<double> &first, const std::vector<double> &second, const std::vector<bool>& masked,
							bool bLogScale = false);

EXPORTED double RSquared_Masked(std::vector<double> data, std::vector<double> fit, 
								const std::vector<bool>& masked, bool bLogScale = false);


#endif
