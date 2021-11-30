#pragma once
#include <string>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include "Amplitude.h"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"

#if defined _WIN32 || defined _WIN64
	#include "Windows.h"
#endif

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif


enum AutoMode {
	NO_AUTO,
	AUTO_AVI,
	AUTO_SF_REC,
	AUTO_SF_REAL,
	AUTO_GRID,
	AUTO_HELP,
	AUTO_ADD_AMPLITUDES,
	AUTO_SUBTRACT_AMPLITUDES,
	AUTO_AVERAGE,
	AUTO_SPHERE,
	AUTO_ALIGN,
	BAD_MODE_SPECIFIED
};

using std::cout;
namespace fs = boost::filesystem;

class CommandHanlderObject {
protected:
	Amplitude *pdb;
	GeometricAmplitude *geo;
	Amplitude *pdb2;
	Amplitude *ampRes, *pamp1, *pamp2;
	Amplitude *pdbPhased;
	std::vector<FACC> Q, res;
	bool bLoadedGrid;
	std::string pdbIn, ampIn, dolIn, saveBaseName;
	int iterOrder;
	AutoMode aMode;
	ATOM_RADIUS_TYPE radType;
	std::vector<std::vector<double> > DockList;

	// List of arguments to be given:
	double Rec_a_arg, Rec_b_arg, Rec_c_arg, Rec_alpha_arg, Rec_beta_arg, Rec_gamma_arg, DW_arg;
	double solvED, qMaxIn, solStep, solRad;
	double atmRadAdd;
	int Rep_a_arg, Rep_b_arg, Rep_c_arg, secs, qsize;
	char outPutSlices;
	char solOnly;
	char fillHoles;
	char discardFiles;

public:
	CommandHanlderObject();
	CommandHanlderObject(AutoMode mode, std::string pdbin, std::string ampin,
		std::string dolin, std::string outFileBaseName, std::string solvEDst,
		std::string order, std::string ain, std::string bin, std::string cin,
		std::string alpahin, std::string betain, std::string gammin,
		std::string repain, std::string repbin, std::string repcin,
		std::string dwin, std::string qmaxin, std::string secin,
		std::string solStepin, std::string solRadin, std::string atmRadDiffin,
		std::string atmRadTypein, char outputSlicesin = 0, char onlySol = 0,
		char fillHoles = 0, char discardAmps = 0, int qresInt = 5000);
	~CommandHanlderObject();

	void writeCrap();

	int DoAutoGrid();
	int DoAutoSF_Rec();
	int DoAutoSF_Real();
	int DoAutoDock();
	int DoAutoAddAmps(std::string amp1, std::string amp2, std::string outFN, std::string scale1 = "", std::string scale2 = "");
	int DoAutoSubtractAmps(std::string amp1, std::string amp2, std::string outFN, std::string scale1 = "", std::string scale2 = "");
	int DoAutoAverage();
	int DoAlign();
	int DoAutoSphere(std::string radSt, std::string rhoSt);

protected:
	void AutoLoadPDB();

	void AutoLoadAmpFile();

	void AutoLoadAmpFile(std::string ampFileName, Amplitude *amp);

	void AutoLoadDOL();

	void GenGridDOL();

	void AutoGenerateNewGrid();

	void AutoSaveNewGridAndPDB();

	void AutoCalculateOrientations();
	
	void AutoCalculateOrientations(Amplitude *ampAv);

	void AutoSave1DFile(string timeStr, string header);

	void AutoSave1DFile(string timeStr, string header, Amplitude *ampP);

	int SF_Param_Check();

	int MathParamCheck(std::string amp1, std::string amp2, std::string outFN);

	void RecToRealSpaceVector(std::vector<double> &av, std::vector<double> &bv, std::vector<double> &cv, 
		std::vector<int> &Rep, double rec_a, double rec_b, double rec_c, double rec_alpha,
		double rec_beta, double rec_gamma, int rep_a, int rep_b, int rep_c);

	// I have no idea why these need to part of the class. If not, I get linking errors
	string GetTimeString(clock_t beg, clock_t en);

	void uBeep(int freq, int dur) {
		// TODO: cross platform?
#if defined _WIN32 || defined _WIN64
		Beep(freq, dur);
#endif
	}

	void WriteDataFileWHeader(const wchar_t *filename, vector<FACC>& x,
		vector<FACC>& y, std::stringstream& header)
	{
		fs::path pth(filename);
		pth = boost::filesystem::system_complete(pth);

		boost::system::error_code er;
		if(!fs::exists(pth.parent_path()) ) {
			if(!fs::create_directories(pth.parent_path(), er) ) {
				std::cout << "Error creating directory: " << pth.parent_path().string() << "\n";
				std::cout << "Error code: " << er << "\n";
			}
		}

		boost::filesystem::wofstream fg(pth);
		if(!fg.is_open()) {
			std::cout << "File " << pth.string() << "  isn't open...\n";
			return;
		}

		fg << header.str() << "\n";

		int size = (x.size() < y.size()) ? x.size() : y.size();
		string st;
		st.resize(20);
		for(int i = 0; i < size; i++) {
			sprintf(&st[0],"%.8g\t%.8g", x.at(i), y.at(i));
			fg << st.c_str() << "\n";
		}

		fg.close();
	}

};

void PrintTime();
