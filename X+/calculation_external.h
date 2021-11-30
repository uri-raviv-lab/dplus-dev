#pragma once 

#ifndef __CALCULATION_EXT_H
#define __CALCULATION_EXT_H

#include "Common.h" // For all the definitions

// Local Computations (???.h)
EXPORTED std::vector <double> MachineResolution(const std::vector <double> &q ,const std::vector <double> &orig, double width);

// Linear Fit for Caille
EXPORTED std::pair <double, double> LinearFit(std::vector <double> x, std::vector <double> y);

// Form factor/Structure factor/Background fitting
EXPORTED bool CreateModel(const std::vector<double> ffx, const std::vector<double> ffy, 
						  std::vector<double>& resy, const std::vector<double>& bgy, const std::vector<bool>& mask, paramStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors, int *pStop);

EXPORTED bool CreateModelU(const std::vector<double> ffx, const std::vector<double> ffy, 
						  std::vector<double>& resy, const std::vector<double>& bgy, const std::vector<bool>& mask, paramStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors, plotFunc GraphModify, 
						  int *pStop, progressFunc ProgressReport);

EXPORTED bool GenerateModel(const std::vector<double> x, std::vector<double>& genY,
				 		    paramStruct *p, int *pStop);

EXPORTED bool GenerateModelU(const std::vector<double> x, std::vector<double>& genY, 
							 const std::vector<double>& bgy, paramStruct *p, plotFunc GraphModify, int *pStop,
							 progressFunc ProgressReport);

EXPORTED bool GenerateStructureFactor(const std::vector<double> x, std::vector<double>& y, peakStruct *p);
EXPORTED bool GenerateStructureFactorU(const std::vector<double> x, std::vector<double>& y, const std::vector<double>& bgy, 
									   peakStruct *p, plotFunc GraphModify, int *pStop, progressFunc ProgressReport);

EXPORTED bool FitStructureFactor(const std::vector<double> sfx, const std::vector<double> sfy, 
								 std::vector<double>& my, const std::vector<double>& bgy, const std::vector<bool>& mask, peakStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors);
EXPORTED bool FitStructureFactorU(const std::vector<double> sfx, const std::vector<double> sfy, 
								 std::vector<double>& my, const std::vector<double>& bgy, const std::vector<bool>& mask, peakStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors, plotFunc GraphModify, 
								 int *pStop, progressFunc ProgressReport);

EXPORTED bool FitBackground(const std::vector<double> bgx, const std::vector<double> bgy, 
						    std::vector<double>& resy, const std::vector<double>& signaly, const std::vector<bool>& mask, bgStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors);
EXPORTED bool FitBackgroundU(const std::vector<double> bgx, const std::vector<double> bgy, 
						     std::vector<double>& resy, const std::vector<double>& signaly, const std::vector<bool>& mask, bgStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors, plotFunc GraphModify, 
						     int *pStop, progressFunc ProgressReport);

EXPORTED bool GenerateBackground(const std::vector<double> bgx, std::vector<double>& genY,
				 				 bgStruct *p);
EXPORTED bool GenerateBackgroundU(const std::vector<double> x, std::vector<double>& genY, 
								  const std::vector<double>& signaly, bgStruct *p, plotFunc GraphModify, int *pStop,
								  progressFunc ProgressReport);

// Background/Baseline

EXPORTED void GenerateBGLinesandFormFactor(const wchar_t *datafile,
                                           const wchar_t *baselinefile,
                                           std::vector <double>& bglx,
                                           std::vector <double>& bgly,
                                           std::vector <double>& ffy,bool ang);

EXPORTED void ImportBackground(const wchar_t *filename, 
							   const wchar_t *datafile,
							   const wchar_t *savename,
							   bool bFactor);

EXPORTED void AutoBaselineGen(const std::vector<double>& datax,
							  const std::vector<double>& datay, std::vector<double>& bgy);

// Phases

EXPORTED void FitPhases1D (std::vector <double> peaks,
                           std::vector<std::vector<std::vector<int> > > &indices_loc, double &wssr,
                           double &slope);

EXPORTED bool FitPhases (PhaseType phase , std::vector<double>& peaks, phaseStruct *p, std::vector<double>& paramErrors,
						 std::vector<std::string> &locs);
EXPORTED bool FitPhasesU (PhaseType phase, std::vector<double>& peaks, phaseStruct *p, std::vector<double>& paramErrors,
						  std::vector<std::string> &locs,
						  int *pStop, progressFunc ProgressReport);
EXPORTED std::vector <double> GenPhases (PhaseType phase , phaseStruct *p,
								std::vector<std::string> &locs);

// Configuration and File Management (Configuration.h)

// File management

EXPORTED int CheckSizeOfFile(const wchar_t *filename);

EXPORTED void ReadDataFile(const wchar_t *filename,
						   std::vector<double>& x, 
						   std::vector<double>& y);

EXPORTED void Read1DDataFile(const wchar_t *filename,
							 std::vector<double>& x);

EXPORTED void WriteDataFileWHeader(const wchar_t *filename, vector<double>& x,
				   vector<double>& y, std::stringstream& header);

EXPORTED void WriteDataFile(const wchar_t *filename, std::vector<double>& x, 
							std::vector<double>& y);

EXPORTED void Write3ColDataFile(const wchar_t *filename, std::vector<double>& x, 
							std::vector<double>& y, std::vector<double>& err);

EXPORTED void Write1DDataFile(const wchar_t *filename, std::vector<double>& x);

EXPORTED void GetDirectory(const wchar_t *file, wchar_t *result, int n = 260);
EXPORTED void GetBasename(const wchar_t *file, wchar_t *result, int n = 260);


// INI Management

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
EXPORTED void SetIniString (const std::wstring& file, const std::string& object, const std::string& param,
							void* ini, const std::string& value);

EXPORTED bool ReadParameters(const std::wstring &filename, std::string obj, paramStruct *p, void* ini);
EXPORTED void WriteParameters(const std::wstring &filename, std::string obj, paramStruct *p, void* ini);

EXPORTED void ReadPeaks(const std::wstring &filename, std::string obj, peakStruct *peaks, void* ini);
EXPORTED void WritePeaks(const std::wstring &filename, std::string obj, const peakStruct *peaks, void* ini);

EXPORTED void WriteBG(const std::wstring &filename, std::string obj, const bgStruct *BGs, void* ini);
EXPORTED void ReadBG(const std::wstring &filename, std::string obj, bgStruct *BGs, void* ini);

EXPORTED void WriteCaille(const std::wstring &filename, std::string obj, const graphTable *Cailles, const cailleParamStruct *Caillesdrawing, void* ini);
EXPORTED void ReadCaille(const std::wstring &filename, std::string obj, graphTable *Cailles, cailleParamStruct *Caillesdrawing, void* ini);

EXPORTED void WritePhases(const std::wstring &filename, std::string obj, const phaseStruct *ph, int pt, void* ini);
EXPORTED void ReadPhases(const std::wstring &filename, std::string obj, phaseStruct *ph, int *pt, void* ini);

EXPORTED bool IniHasModelType(const std::wstring& file, const std::string& object, void* ini);

// Instantiates ini as a CSimpleIniA
EXPORTED void *NewIniFile();
// Saves the file and deletes ini
EXPORTED void SaveAndCloseIniFile(const std::wstring& file, const std::string& object, void* ini);
// Deletes ini
EXPORTED void CloseIniFile(void* ini);

#endif
