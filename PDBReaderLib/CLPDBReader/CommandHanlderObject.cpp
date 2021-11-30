#include "CommandHanlderObject.h"
#include <math.h>
#include <boost/lexical_cast.hpp>

using namespace boost::filesystem;
namespace fs = boost::filesystem;

bool bdTryParse(std::string st, double &res) {
	char *endp;
	double d = strtod(&st[0], &endp);
	if(endp == &st[0])
		return false;
	res = d;
	return true;
}
bool biTryParse(std::string st, int &res) {
	char *endp;
	int ii = strtol(&st[0], &endp, 0);
	if(endp == &st[0])
		return false;
	res = ii;
	return true;
}

void PrintTime() {
	char buff[100];
	time_t now = time (0);
	strftime (buff, 100, "%Y-%m-%d %H:%M:%S", localtime (&now));
	std::cout << buff;
}

CommandHanlderObject::CommandHanlderObject() {
	Rec_a_arg = Rec_b_arg = Rec_c_arg = Rec_alpha_arg = Rec_beta_arg = Rec_gamma_arg = DW_arg = -1.0;
	Rep_a_arg = Rep_b_arg = Rep_c_arg = -1;
	qsize = 500;
	iterOrder = -1;
	pdb = NULL;
	pdbPhased = NULL;
	discardFiles = 0;
}

CommandHanlderObject::CommandHanlderObject(AutoMode mode, std::string pdbin, std::string ampin,
										   std::string dolin, std::string outFileBaseName, std::string solvEDst,
										   std::string order, std::string ain, std::string bin, std::string cin,
										   std::string alpahin, std::string betain, std::string gammin,
										   std::string repain, std::string repbin, std::string repcin,
										   std::string dwin, std::string qmaxin, std::string secin,
										   std::string solStepin, std::string solRadin, std::string atmRadDiffin,
										   std::string atmRadTypein, char outputSlicesin, char onlySolin,
										   char fillHolesIn, char discardAmps, int qresInt) {
	Rec_a_arg = Rec_b_arg = Rec_c_arg = Rec_alpha_arg = Rec_beta_arg = Rec_gamma_arg = DW_arg = -1.0;
	Rep_a_arg = Rep_b_arg = Rep_c_arg = -1;
	qsize = qresInt;
	fillHoles = fillHolesIn;
	discardFiles = discardAmps;
	iterOrder = -1;
	pdb = NULL;
	pdbPhased = NULL;

	if(pdbin.size() > 0)
		std::cout << "PDB file: " << pdbin << "\n";
	if(ampin.size() > 0)
		std::cout << "Amplitude file: " << ampin << "\n";
	if(dolin.size() > 0)
		std::cout << "DOL file: " << dolin << "\n";
	if(outFileBaseName.size() > 0)
		std::cout << "Output files: " << outFileBaseName << "\n";
	if(order.size() > 0)
		std::cout << "Iterations order: " << order << "\n";
	if(qmaxin.size() > 0)
		std::cout << "q_max: " << qmaxin << "\n";
	if(secin.size() > 0)
		std::cout << "Sections: " << secin << "\n";
	if(solvEDst.size() > 0)
		std::cout << "Solvent ED: " << solvEDst << "\n";
	if(solStepin.size() > 0)
		std::cout << "Solvent step size: " << solStepin << "\n";
	if(solRadin.size() > 0)
		std::cout << "Solvent radius: " << solRadin << "\n";
	if(ain.size() > 0)
		std::cout << "a: " << ain << "\n";
	if(bin.size() > 0)
		std::cout << "b: " << bin << "\n";
	if(cin.size() > 0)
		std::cout << "c: " << cin << "\n";
	if(alpahin.size() > 0)
		std::cout << "alpha: " << alpahin << "\n";
	if(betain.size() > 0)
		std::cout << "beta: " << betain << "\n";
	if(gammin.size() > 0)
		std::cout << "gamma: " << gammin << "\n";
	if(repain.size() > 0)
		std::cout << "a repeats: " << repain << "\n";
	if(repbin.size() > 0)
		std::cout << "b repeats: " << repbin << "\n";
	if(repcin.size() > 0)
		std::cout << "c repeats: " << repcin << "\n";
	if(dwin.size() > 0)
		std::cout << "Debye-Waller factor: " << dwin << "\n";

	aMode = mode;
	pdbIn = pdbin;
	ampIn = ampin;
	dolIn = dolin;
	saveBaseName = outFileBaseName;
	outPutSlices = outputSlicesin;
	this->solOnly = onlySolin;

	if(atmRadTypein.size() > 0) {
		if(boost::iequals(atmRadTypein, "vdw")) {
			radType = RAD_VDW;
		} else if(boost::iequals(atmRadTypein, "calc")) {
			radType = RAD_CALC;
		} else if(boost::iequals(atmRadTypein, "emp")) {
			radType = RAD_EMP;
		} else if(boost::iequals(atmRadTypein, "svrgC")) {
			radType = RAD_Sverg_C;
		} else if(boost::iequals(atmRadTypein, "svrg")) {
			radType = RAD_SVERG_ONLY;
		} else {
			radType = RAD_EMP;
		}

	}
	
	if(!biTryParse(order, iterOrder)) {
		order = -1;
	}

	// List of Arguments to be given:
	if(!bdTryParse(ain, Rec_a_arg)) {
		Rec_a_arg = -1.0;
	}
	if(!bdTryParse(bin, Rec_b_arg)) {
		Rec_b_arg = -1.0;
	}
	if(!bdTryParse(cin, Rec_c_arg)) {
		Rec_c_arg = -1.0;
	}
	if(!bdTryParse(alpahin, Rec_alpha_arg)) {
		Rec_alpha_arg = -1.0;
	}
	if(!bdTryParse(betain, Rec_beta_arg)) {
		Rec_beta_arg = -1.0;
	}
	if(!bdTryParse(gammin, Rec_gamma_arg)) {
		Rec_gamma_arg = -1.0;
	}
	if(!bdTryParse(dwin, DW_arg)) {
		DW_arg = 0.0;
	}

	if(!biTryParse(repain, Rep_a_arg)) {
		Rep_a_arg = -1;
	}
	if(!biTryParse(repbin, Rep_b_arg)) {
		Rep_b_arg = -1;
	}
	if(!biTryParse(repcin, Rep_c_arg)) {
		Rep_c_arg = -1;
	}
	if(!bdTryParse(solvEDst, solvED)) {
		solvED = 0.0;
	}
	if(!bdTryParse(qmaxin, qMaxIn)) {
		qMaxIn = 0.2;
	}
	if(!bdTryParse(solStepin, solStep)) {
		solStep = 0.1;
	}
	if(!bdTryParse(solRadin, solRad)) {
		solRad = 0.0;
	}
	if(!biTryParse(secin, secs)) {
		secs = 2;
	}
	if(!bdTryParse(atmRadDiffin, atmRadAdd)) {
		atmRadAdd = 0.0;
	}
}

CommandHanlderObject::~CommandHanlderObject() {
	delete pdb;
	delete pdb2;
	delete ampRes;
	delete pdbPhased;
	delete geo;
	delete pamp1;
	delete pamp2;
}

int CommandHanlderObject::DoAutoGrid() {
	std::cout << "pdbIn: " << pdbIn << "\n";
	std::cout << "saveBaseName: " << saveBaseName<< "\n";
	std::cout << "iterOrder: " << iterOrder << "\n";

	if(iterOrder < 0) {
		std::cout << "Missing \"-ord\" parameter! Try again.\n";
		std::cout << "Quiting now...\n";
		return 52;
	}
	if(!fs::exists(path(pdbIn))) {
		std::cout << "In \"grid\" mode, the \"-ipdb\" parameter must be supplied.\n";
		return 52;
	}
	if(saveBaseName.size() < 1) {
		std::cout << "In \"grid\" mode, the \"-out\" parameter must be supplied.\n";
		return 52;
	}

	int sec = secs;
	uint64_t iters = uint64_t(pow(10.0, double(iterOrder)));
	clock_t begin, end, c_time;
	string timeStr;
	FACC qmax = qMaxIn, qSmin = 0.001;
	int size = qsize;

	//Load PDB
	AutoLoadPDB();
	PDBAmplitude* cpdb = (PDBAmplitude*)(pdb);
	cpdb->SetOutputSlices(this->outPutSlices == 1, saveBaseName);
	cpdb->SetFillHoles(this->fillHoles == 1);
	cpdb->SetSolventOnlyCalculation(this->solOnly == 1);
	std::stringstream ss1;
	if(pdb->getError() != OK) {
		std::cout << "PDB object\n\t" << "Error: " << pdb->getError();
	}

	if(solvED > 0.0) {
		begin = clock();
		std::cout << "Calculating PDB solvent amplitude...";
		PrintTime(); std::cout << "\n";
		std::cout << "SolvED = " << solvED << "; \n";
		PDB_READER_ERRS er1 = cpdb->CalculateSolventAmp(solStep, solvED, solRad, atmRadAdd);
		if(er1) {
			std::cout << "Error calculating solvent amplitude. Error code " << er1 << "\n";
			std::cout << "Quiting now...\n";
			return 5;
		}
		std::cout << "Calculated PDB solvent amplitude. (" << GetTimeString(begin, clock()) << ")\n";
		uBeep(213,587);
	}
	//calculate grid
	std::cout << "qMax = " << qmax << "; Sections: " << sec << "\n";
	std::cout << "Calculating Grid...";
	PrintTime(); std::cout << "\n";
	begin = clock();
	pdb->calculateGrid(qmax, sec);
	bLoadedGrid = true;
	std::cout << "Calculated amplitude grid. (" << GetTimeString(begin, clock()) << ")\n";
	uBeep(435,250);

	if(pdb->getError()) {
		std::cout << "\nPDB object\n" << "\tError: " << pdb->getError()
					<< " Grid size: " << pdb->GetGridSize() << "\n";
	}

	// Save new grid
	AutoSaveNewGridAndPDB();

	// Orientation average
	if(this->iterOrder > 0) {
		std::cout << "Orientation averaging (10^" << iterOrder << "). ";
		begin = clock();
		this->AutoCalculateOrientations();
		timeStr = GetTimeString(begin, clock());
		ss1.str(std::string());
		ss1 << "# Automated calculation.\n";
		if(solvED > 0.0) {
			ss1 << "# Solvent amplitude subtracted from PDB amplitude:" << solvED << "\n";
			ss1 << "# Solvent step size in nm:" << solStep << "\n";
		}
		AutoSave1DFile(timeStr, ss1.str());
	}

	// Quit
	std::cout << "Quiting now...\n";
	return 0;
}

int CommandHanlderObject::DoAutoSF_Rec() {
	int re = SF_Param_Check();
	if(re != 0) {
		return re;
	}

	std::cout << "ampIn: " << ampIn << "\n";
	std::cout << "SF parameters in the reciprocal space: " << "\n";
	// TODO
	//wtrConsLn("a = " + Double(Rec_a_arg).ToString("0.00"));// + ", b = " + (Rec_b_arg).ToString("0.00") + ", c = " + (Rec_c_arg).ToString("0.00") );
	//wtrConsLn("alpha = " + Rec_alpha_arg + ", beta = " + Rec_beta_arg + ", gamma = " + Rec_gamma_arg);
	//wtrConsLn("Rep a = " + Rep_a_arg + ", Rep c = " + Rep_b_arg + ", Rep c = " + Rep_c_arg + ", DW = " + DW_arg);
	std::cout << "saveBaseName: " << saveBaseName << "\n";
	std::cout << "iterOrder: " << iterOrder << "\n";

	int sec;
	uint64_t iters = uint64_t(pow(10.0, double(iterOrder)));
	clock_t begin, end, c_time;
	string timeStr;
	FACC qmax, qSmin = 0.001;
	int size = qsize;

	begin = clock();
	std::stringstream ss1;

	//Load amp
	pdb = new CPDBReader();

	AutoLoadAmpFile();

	ss1 << "\nPDB object\n" << "\tError: " << pdb->getError()
		<< " Grid size: " << pdb->GetGridSize();
	if(pdb->getError()) {
		std::cout << ss1.str();
	}

	// Run iterations
	std::vector<double> a(3),b(3),c(3);
	std::vector<int> Rep(3);
	RecToRealSpaceVector(a,b,c, Rep, Rec_a_arg , Rec_b_arg, Rec_c_arg, Rec_alpha_arg, Rec_beta_arg, Rec_gamma_arg, Rep_a_arg, Rep_b_arg, Rep_c_arg);		


	std::cout << "Starting iterations...";

	if(bLoadedGrid) {
		qmax = pdb->GetGridStepSize() * double(pdb->GetGridSize()) / 2.0;
		sec = pdb->GetGridSize();
	}
	Q.resize(size);
	res.resize(size);
	for(int i = 0; i < size; i++)
		Q[i] = FACC((qmax - qSmin) / FACC(size - 1) * FACC(i));
	std::cout << " qmax = " << qmax << ";\tqSmin = " << qSmin << "\n";

	begin = clock();
	DomainModel ic;
	std::vector<Amplitude*> aps;
	aps.push_back((Amplitude*)(pdb));
	ic.AddSubAmplitude(aps);

	std::cout << "starting calculation...";
	PrintTime(); std::cout << "\n";

	ic.CalculateVector(Q, res, 1.0e-4, iters, a, b, c, Rep, DW_arg);
	if(pdb->getError()) {
		std::cout << "\nError in calculation. Error code: " << (int)pdb->getError() << "\n";
	}

	timeStr = GetTimeString(begin, clock());
	std::cout << "\nFinished iterations. (" << timeStr << ")\n";
	uBeep(435,500);


	// Save 1D data file
	std::stringstream blank;
	blank << "# AMP file: " << ampIn << "\n";

	AutoSave1DFile(timeStr, blank.str());
 
	// Quit
	std::cout << "Quiting now...\n";
	return 0;
}

int CommandHanlderObject::DoAutoSF_Real() {
	std::cout << "Unimplemented!\n";
	return 12;
}

int CommandHanlderObject::DoAutoDock() {
	std::cout << "pdbIn: " << pdbIn << "\n";
	std::cout << "ampIn: " << ampIn << "\n";
	std::cout << "dolIn: " << dolIn << "\n";
	std::cout << "saveBaseName: " << saveBaseName << "\n";
	std::cout << "iterOrder: " << iterOrder << "\n";

	if(iterOrder < 0) {
		std::cout << "Missing \"-ord\" parameter! Try again.";
		std::cout << "Quiting now...\n";
		return 9;
	}

	int sec;
	uint64_t iters = uint64_t(pow(10.0, double(iterOrder)));
	clock_t begin, end, c_time;
	string timeStr;
	FACC qmax = qMaxIn, qSmin = 0.001;
	int size = qsize;

	AutoLoadPDB();

	//Load/calc amp
	if(ampIn.size() > 0) {
		AutoLoadAmpFile();
	} else {
		begin = clock();
		if(solvED > 0.0) {
			std::cout << "Calculating PDB solvent amplitude...";
			PrintTime(); std::cout << "\n";
			PDB_READER_ERRS er1 = pdb->CalculateSolventAmp(solStep, solvED, solRad, atmRadAdd);
			if(er1 != OK) {
				std::cout << "Error calculating solvent amplitude. Error code " << er1 << ". ";
			} else {
				std::cout << "Calculated PDB solvent amplitude.";
			}
			std::cout << " (" + GetTimeString(begin, clock()) + ")\n";
			uBeep(213,587);
			begin = clock();
		}
		std::cout << "Calculating PDB amplitude...";
		PrintTime(); std::cout << "\n";
		pdb->calculateGrid(qmax, secs);
		std::cout << "Calculated PDB amplitude.";
		std::cout << " (" + GetTimeString(begin, clock()) + ")\n";

		begin = clock();

		if(discardFiles == 0) {
			std::cout << "Writing generated amplitude file...";

			std::stringstream ss;
			ss << "# PDB file: " << pdbIn << "\n";
			ss << "# Program revision: "<< SVN_REV_STR << "\n";
			ss << "# N^3; N = " << pdb->GetGridSize() << "\n";
			ss << "# Calculation time: " << timeStr << "\n";
			ss << "# qMax = " << qmax << "\n";
			ss << "# Grid step size = " << pdb->GetGridStepSize() << "\n";
			ss << "# Generated using command line options.\n";
			if(solvED > 0.0) {
				ss << "# Solvent ED amplitude subtracted: " << solvED << "\n";
				ss << "# Solvent ED step size: " << solStep << "\n";
			}
			string fn = pdbIn;
			fn.erase(fn.find(".pdb"));
			fn.append(string("_").append(boost::lexical_cast<string>(secs)).append(".amp"));
			pdb->WriteAmplitudeToFile(fn, ss);
			std::cout <<"Finished writing amplitude file. (" + GetTimeString(begin, clock()) + ")\n";
		}
	}

	if(pdb->getError()) {
		std::cout << "\nPDB object\n" << "\tError: " << pdb->getError()
			<< " Grid size: " << pdb->GetGridSize();
	}

	// Load DOL
	AutoLoadDOL();

	// Generate new grid
	AutoGenerateNewGrid();

	// Save new grid 
	AutoSaveNewGridAndPDB();

	// Run iterations
	AutoCalculateOrientations();

	// Save 1D data file
	AutoSave1DFile(timeStr, string());

	// Quit
	std::cout << "Quiting now...\n";
	return 0;

}

void CommandHanlderObject::writeCrap() {
	std::cout << "Crap!\n";
}

void CommandHanlderObject::AutoLoadPDB() {
	if(pdbIn.length() == 0)
		return;
	//Load PDB
	if(pdb) {
		delete pdb;
	}
	pdb = new CPDBReader(pdbIn);
	pdb->SetRadiusType(this->radType);
	bLoadedGrid = false;
}

void CommandHanlderObject::AutoLoadAmpFile() {
	if(!pdb) {
		pdb = new CPDBReader();
	}
	AutoLoadAmpFile(ampIn, pdb);
}

void CommandHanlderObject::AutoLoadAmpFile(std::string ampFileName, AmplitudeProducer *amp) {
	clock_t beg = clock();
	std::cout << "Reading amp file...";
	PrintTime(); std::cout << "\n";
	(*amp).ReadAmplitudeFromFile(ampFileName);
	uBeep(429, 750);
	bLoadedGrid = true;
	std::cout << "Loaded amp file. (" << GetTimeString(beg, clock()) << ")\n";

}

void CommandHanlderObject::AutoLoadDOL() {
	// Load DOL
	clock_t begin = clock();
	pdb->ReadDockList(dolIn, DockList);
	std::cout << "Loaded DOL.";
	std::cout << " (" + GetTimeString(begin, clock()) + ")";
}

void CommandHanlderObject::GenGridDOL() {
	//Create an empty grid
	//Loop over the rows of FromList and add the relevant item
	//Save the grid
	//Unload the old grid and make sure the new grid is loaded

	// Make sure grid is loaded
	// Add phase to copy of grid
	if(pdbPhased)
		delete pdbPhased;
	pdbPhased = new CPDBReader(*pdb);

	PrintTime(); std::cout << "\n";

	//Read FromList
	std::vector<std::vector<double> > FormListData, *point;

	FormListData = this->DockList;
	FormListData.resize(6);
	for(int i = 0; i < FormListData.size(); i++) {
		FormListData[i].resize(this->DockList[i + 1].size());
	}

	for (int j = 1 ; j < 4 ; j++) {
		for (int i = 0 ; i < this->DockList[0].size() /*May be without [0]!!*/ ; i++ ) {
			FormListData[j-1][i] = DockList[j][i] * M_PI / 180.0;
		}
	} // read angles

	for (int j = 4 ; j < 7 ; j++) {
		for (int i = 0 ; i < this->DockList[0].size() /*May be without [0]!!*/ ; i++ ) {
			FormListData[j-1][i] = DockList[j][i];
		}
	} // read angles

	//	point = FormListData;
	pdbPhased->BuildFromList(FormListData, *pdb);
	pdbPhased->BuildPDBFromList(FormListData);
	delete pdb;
	pdb = new CPDBReader(*pdbPhased);
	delete(pdbPhased);
}

void CommandHanlderObject::AutoGenerateNewGrid() {
	std::cout << "\nGenerating new grid...";
	PrintTime(); std::cout << "\n";
	clock_t begin = clock();
	GenGridDOL();
	std::cout << "\nGenerated new grid. (" + GetTimeString(begin, clock()) + ")\n";
	uBeep(672, 350);
}

void CommandHanlderObject::AutoSaveNewGridAndPDB() {
	if(discardFiles != 0) {
		return;
	}
	std::cout << "Saving grid...";
	PrintTime(); std::cout << "\n";
	std::stringstream hdr;
	if(ampIn.size() > 1) {
		pdb->ReadAmplitudeHeaderFromFile(ampIn, hdr);
	}
	// Write new amp file
	clock_t begin = clock();
	pdb->WriteAmplitudeToFile(string(saveBaseName).append("_").append(boost::lexical_cast<string>(pdb->GetGridSize())).append(".amp"), hdr);
	//savepdb
	pdb->WritePDBToFile(string(saveBaseName).append(".pdb"), hdr);
	std::cout << "Saved new grid and new PDB files. (" << GetTimeString(begin, clock()) << ")\n";
	uBeep(564, 450);
}

void CommandHanlderObject::AutoCalculateOrientations() {
	AutoCalculateOrientations(*pdb);
}
void CommandHanlderObject::AutoCalculateOrientations(AmplitudeProducer &ampAv) {
	std::cout << "Starting iterations...";
	PrintTime(); std::cout << "\n";

	if(iterOrder < 1) {
		std::cout << "The \"-ord\" parameter must be greater than 0.\n";
		return;
	}

	uint64_t iters = uint64_t(pow(10.0, double(iterOrder)));
	clock_t begin, end, c_time;
	string timeStr;
	FACC qmax = qMaxIn, qSmin = 0.001;
	int size = qsize;
	int sec = secs;

	if(bLoadedGrid) {
		qmax = ampAv.GetGridStepSize() * double(ampAv.GetGridSize()) / 2.0;
		sec = ampAv.GetGridSize();
	}
	Q.resize(size);
	res.resize(size);

	for(int i = 0; i < size; i++)
		Q[i] = FACC((qmax - qSmin) / FACC(size - 1) * FACC(i));
	std::cout << " qmax = " << qmax << ";\tqSmin = " << qSmin << "\n";

	begin = clock();
	IntensityCalculator ic;
	std::vector<AmplitudeProducer*> aps;
	aps.push_back(&ampAv);
	ic.SetAmplitudes(aps);

	std::cout << "Starting calculation iterations...\n";

	ic.CalculateIntensityVector(Q, res, 1.0e-4, iters);
	if(ampAv.getError()) {
		std::cout << "Error in calculation. Error code: " << (int)ampAv.getError() << "\n";
	}
	std::cout << "\nFinished iterations. (" << GetTimeString(begin, clock()) << ")\n";
	uBeep(435,500);
}

void CommandHanlderObject::AutoSave1DFile(string timeStr, string header) {
	if(pdb) {
		AutoSave1DFile(timeStr, header, pdb);
	} else if(geo) {
		AutoSave1DFile(timeStr, header, geo);
	}

}
	
void CommandHanlderObject::AutoSave1DFile(string timeStr, string header, AmplitudeProducer *ampP) {
	std::cout << "Saving 1D data file...";
	AmplitudeProducer *ptr = ampP;

	std::wstring file;
	std::stringstream blank;
	blank << "# PDB file: " << pdbIn << "\n";
	blank << "# Program revision: "<< SVN_REV_STR << "\n";
	blank << "# Orientation calculation time: " << timeStr << "\n" ;
	blank << "# N^3; N = " << ptr->GetGridSize() << "\n";
	blank << "# 10^M Sample points per q; M = " << iterOrder << "\n";
	blank << "# Notes: With 3D linear interpolation.\n";
	blank << "# Notes: Calculated using a * e^{i* Q\\dot R} = a Cos(Q\\dot R) + i Sin(Q\\dot R).\n";
	if(header.size() > 0) {
		blank << "# " << header << "\n";
	}
	std::string st(saveBaseName);
	st.append(".out");
	std::wstring ws;
	ws.assign(st.begin(), st.end());
	WriteDataFileWHeader(ws.c_str(), Q, res, blank);
	std::cout << "Done!\n";
}

string CommandHanlderObject::GetTimeString(clock_t beg, clock_t en) {
	clock_t totu, tot = en - beg;
	std::stringstream res;
	int ms, sec, minut, hr, gh = 0;

	totu = tot;
	hr = int(double(totu) / double(CLOCKS_PER_SEC) / 3600.0);
	totu -= hr * 3600 * CLOCKS_PER_SEC;
	minut = int(double(totu) / double(CLOCKS_PER_SEC) / 60.0);
	totu -= minut * 60 * CLOCKS_PER_SEC;
	sec =  int(double(totu) / double(CLOCKS_PER_SEC));
	totu -= sec * CLOCKS_PER_SEC;

	if(hr > 0) {
		gh++;
		res << " " << hr << " hour" << (hr > 1 ? "s" : "");
	}
	if(minut > 0) {
		gh++;
		res << " " << minut << " minute" << (minut > 1 ? "s" : "");
	}
	if(sec > 0 && gh < 2) {
		gh++;
		res << " " << sec << " second" << (sec > 1 ? "s" : "");
	}
	if(gh < 2) {
		res << " " << totu * double(1000.0 / double(CLOCKS_PER_SEC)) << " ms";
	}

	//if(tot > 10000)
	//	res << (tot / CLOCKS_PER_SEC) << " seconds";
	//else
	//	res << tot * double(1000.0 / double(CLOCKS_PER_SEC)) << " ms";
	return string(res.str().c_str());
}

int CommandHanlderObject::SF_Param_Check() {
	std::stringstream erSt;
	//			Rec_a_arg = Rec_b_arg = Rec_c_arg = Rec_alpha_arg = Rec_beta_arg = Rec_gamma_arg = DW_arg = -1.0;
	//			Rep_a_arg = Rep_b_arg = Rep_c_arg = -1;

	if(!fs::exists(path(ampIn))) {
		erSt << "Missing \"-iamp\" parameter.\n";
	}
	if(Rec_a_arg <= 0.0) {
		erSt << "Missing \"-a\" parameter.\n";
	} 
	std::cout << "Rec_a_arg: " << Rec_a_arg << "\n";
	if(Rec_b_arg <= 0.0) {
		erSt << "Missing \"-b\" parameter.\n";
	}
	if(Rec_c_arg <= 0.0) {
		erSt << "Missing \"-c\" parameter.\n";
	}
	if(Rec_alpha_arg <= 0.0) {
		erSt << "Missing \"-alpha\" parameter.\n";
	}
	if(Rec_beta_arg <= 0.0) {
		erSt << "Missing \"-beta\" parameter.\n";
	}
	if(Rec_gamma_arg <= 0.0) {
		erSt << "Missing \"-gamma\" parameter.\n";
	}
	if(Rep_a_arg <= 0) {
		erSt << "Missing \"-rep_a\" parameter.\n";
	}
	if(Rep_b_arg <= 0) {
		erSt << "Missing \"-rep_b\" parameter.\n";
	}
	if(Rep_c_arg <= 0) {
		erSt << "Missing \"-rep_c\" parameter.\n";
	}
	if(DW_arg <= 0.0) {
		erSt << "Missing \"-dw\" parameter.\n";
	}

	if(erSt.str().size() > 0) {
		std::cout << erSt.str();
		std::cout << "\nQuiting now...\n";
		return 78;
	}
	return 0;
}

void CommandHanlderObject::RecToRealSpaceVector(std::vector<double> &av, std::vector<double> &bv, std::vector<double> &cv, std::vector<int> &Rep, double rec_a, double rec_b, double rec_c, double rec_alpha, double rec_beta, double rec_gamma, int rep_a, int rep_b, int rep_c) {
	// reading parameters from GUI
	double a,b,c,alpha,beta,gamma,sa,sb,sc,ca,cb,cc,cat,Vol;

	Rep[0] = rep_a;
	Rep[1] = rep_b;
	Rep[2] = rep_c;

	// calculating the params in real space and creating vectors
	sa = sin(rec_alpha * M_PI / 180.0);
	sb = sin(rec_beta * M_PI / 180.0);
	sc = sin(rec_gamma * M_PI / 180.0);
	ca = cos(rec_alpha * M_PI / 180.0);
	cb = cos(rec_beta * M_PI / 180.0);
	cc = cos(rec_gamma * M_PI / 180.0);

	// Volume in reciprocal space
	Vol = rec_a * rec_b * rec_c * sqrt(1.0 - ca * ca-cb * cb-cc * cc+2 * ca * cb * cc);

	//Reciprocal magnitudes:
	a = 2.0 * M_PI * rec_b*rec_c*sa/Vol;
	b = 2.0 * M_PI * rec_c*rec_a*sb/Vol;
	c = 2.0 * M_PI * rec_a*rec_b*sc/Vol;
	alpha = acos((ca*cb-cc)/(sa*sb))/ M_PI * 180.0;
	beta = acos((cc*cb-ca)/(sc*sb))/ M_PI * 180.0;
	gamma = acos((ca*cc-cb)/(sa*sc))/ M_PI * 180.0;

	sa = sin(alpha * M_PI / 180.0);
	sb = sin(beta * M_PI / 180.0);
	sc = sin(gamma * M_PI / 180.0);
	ca = cos(alpha * M_PI / 180.0);
	cb = cos(beta * M_PI / 180.0);
	cc = cos(gamma * M_PI / 180.0);

	cat = (ca - cb * cc)/(sb * sc);
	//calculation if Cartesian coordinates
	av[0] = a ;
	av[1] = 0.0;
	av[2] = 0.0;
	bv[0] = b * cc;
	bv[1] = b * sc;
	bv[2] = 0.0;
	cv[0] = c * cb;
	cv[1] = c * sb * cat;
	cv[2] = c * sb * sin(acos(cat));
}

int CommandHanlderObject::DoAutoAddAmps(std::string amp1, std::string amp2, std::string outFN, std::string scale1, std::string scale2) {
	if(MathParamCheck(amp1, amp2, outFN) != 0) {
		return 34;
	}
	float prg1 = 0.0, prg2 = 0.0;
	char *endp;
	clock_t begin = clock();
	std::cout << "Loading amplitude files...";
//#pragma omp parallel sections
	{
//#pragma omp section
		{
			pamp1 = new PureAmplitude();
			pamp1 ->ReadAmplitudeFromFile(amp1);
			if(scale1.length() > 0) {
				double sc = strtod(&scale1[0], &endp);
				std::cout << "\nMultiply: " << sc << "x" << amp1 << "\n";
				(*pamp1) = (*pamp1) * sc;
			}
		}	
//#pragma omp section
		{
			pamp2 = new PureAmplitude();
			pamp2->ReadAmplitudeFromFile(amp2);
			if(scale2.length() > 0) {
				double sc = strtod(&scale2[0], &endp);
				std::cout << "\nMultiply: " << sc << "x" << amp2 << "\n";
				(*pamp2) = (*pamp2) * sc;
			}
		}
// #pragma omp section
// 		{
// 			while(prg1 + prg2 < 2.0) {
// 				printProgressBar(60, prg1 + prg2, 2.0);
// 			}
// 		}
	}
	std::cout << "\nFinished loading amplitude files. " << GetTimeString(begin, clock()) << "\nAdding amplitude files...";
	begin = clock();
	PureAmplitude pureAmp = *pamp1 + *pamp2;
	std::cout << GetTimeString(begin, clock()) << "\n";
	if(discardFiles == 0) {
		std::cout << "Writing sum amplitude file...";
		begin = clock();
		pureAmp.WriteAmplitudeToFile(string(outFN).append("_").append(boost::lexical_cast<string>(pureAmp.GetGridSize())).append(".amp"));
		std::cout << GetTimeString(begin, clock()) << "\n";
	}
	if(iterOrder > 0) {
		bLoadedGrid = true;
		begin = clock();
		AutoCalculateOrientations(pureAmp);
		AutoSave1DFile(GetTimeString(begin, clock()), "Sum of two amplitudes", &pureAmp);
	}
	std::cout << "\nDone.";
	return 0;
}

int CommandHanlderObject::DoAutoSubtractAmps(std::string amp1, std::string amp2, std::string outFN, std::string scale1, std::string scale2) {
	if(MathParamCheck(amp1, amp2, outFN) != 0) {
		return 65;
	}
	float prg1 = 0.0, prg2 = 0.0;
	char *endp;
	clock_t begin = clock();
	std::cout << "Loading amplitude files...";
//#pragma omp parallel sections
	{
// #pragma omp section
		{
			pamp1 = new AmpGridAmplitude(amp1);
			if(scale1.length() > 0) {
				double sc = strtod(&scale1[0], &endp);
				std::cout << "\nMultiply: " << sc << "x" << amp1 << "\n";
				(*pamp1).grid->Scale(sc);
			}

		}	
// #pragma omp section
		{
			pamp2 = new AmpGridAmplitude(amp2);
			if(scale2.length() > 0) {
				double sc = strtod(&scale2[0], &endp);
				std::cout << "\nMultiply: " << sc << "x" << amp2 << "\n";
				(*pamp2) = (*pamp2) * sc;
			}

		}
	}
	std::cout << "\nFinished loading amplitude files. " << GetTimeString(begin, clock()) << "\nSubtracting amplitude files...";
	begin = clock();
	PureAmplitude pureAmp = *pamp1 - *pamp2;
	std::cout << GetTimeString(begin, clock()) << "\n";
	if(discardFiles == 0) {
		std::cout << "Writing difference amplitude file...";
		begin = clock();
		pureAmp.WriteAmplitudeToFile(string(outFN).append("_").append(boost::lexical_cast<string>(pureAmp.GetGridSize())).append(".amp"));
		std::cout << GetTimeString(begin, clock()) << "\n";
	}

	if(iterOrder > 0) {
		bLoadedGrid = true;
		begin = clock();
		AutoCalculateOrientations(pureAmp);
		AutoSave1DFile(GetTimeString(begin, clock()), "Difference of two amplitudes", &pureAmp);
	}
	std::cout << "\nDone.\n";
	return 0;
}

int CommandHanlderObject::DoAutoAverage() {
	// Check to see that all the parameters are correct
	{
		std::stringstream erSt;
		if(!fs::exists(path(this->ampIn))) {
			erSt << "Missing \"-iamp\" parameter.\n";
		}
// 		if(!fs::portable_file_name(this->saveBaseName)) {
// 			erSt << "Invalid filename for \"-out\" parameter. " << this->saveBaseName << "\n";
// 		}
		if(this->iterOrder < 1) {
			erSt << "The \"-ord\" parameter must be included with a value of 1 or more.\n";
		}
		if(erSt.str().size() > 0) {
			std::cout << erSt.str();
			std::cout << "\nQuiting now...\n";
			return 562;
		}
	}	// End of parameter check

	{
		std::string timeStr;
		std::stringstream ss1;
		clock_t begin;

		//AutoLoadAmpFile();
		pdb2 = new CPDBReader();
		pdb2->ReadAmplitudeFromFile(ampIn);

		if(this->iterOrder > 0) {
			std::cout << "Orientation averaging (10^" << iterOrder << "). ";
			begin = clock();
			this->AutoCalculateOrientations(*pdb2);
			timeStr = GetTimeString(begin, clock());
			ss1.str(std::string());
			ss1 << "# Automated calculation.\n";
			if(solvED > 0.0) {
				ss1 << "# Solvent amplitude subtracted from PDB amplitude:" << solvED << "\n";
				ss1 << "# Solvent step size in nm:" << solStep << "\n";
			}
			AutoSave1DFile(timeStr, ss1.str(), (AmplitudeProducer*)pdb2);
		}

		return 0;

	}

	return 236;
}

int CommandHanlderObject::MathParamCheck(std::string amp1, std::string amp2, std::string outFN) {
	std::stringstream erSt;
	if(!fs::exists(path(amp1))) {
		erSt << "Missing \"-iamp1\" parameter.\n";
	}
	if(!fs::exists(path(amp2))) {
		erSt << "Missing \"-iamp2\" parameter.\n";
	}
// 	if(!fs::portable_file_name(outFN)) {
// 		erSt << "Invalid filename for \"-out\" parameter. " << this->saveBaseName << "\n";
// 	}
	if(erSt.str().size() > 0) {
		std::cout << erSt.str();
		std::cout << "\nQuiting now...\n";
		return 7814;
	}
	return 0;
}

int CommandHanlderObject::DoAutoSphere(std::string radSt, std::string rhoSt) {
	char *endptr;
	std::string timeStr;
	std::stringstream ss1;
	clock_t begin;
	geo = new UniformSphere(strtod(&rhoSt[0], &endptr), strtod(&radSt[0], &endptr));
	std::cout << "qMax = " << qMaxIn << "; Sections: " << secs << "\n";

	// Generate grid
	std::cout << "Calculating Grid...";
	geo->SetUseGrid(true);
	PrintTime(); std::cout << "\n";
	begin = clock();
	geo->calculateGrid(qMaxIn, secs);
	std::cout << "Calculated amplitude grid. (" << GetTimeString(begin, clock()) << ")\n";
	uBeep(435,250);

	// Write new amp file
	if(discardFiles == 0) {
		std::cout << "Saving grid...";
		PrintTime(); std::cout << "\n";
		std::stringstream hdr;
		begin = clock();
		geo->WriteAmplitudeToFile(string(saveBaseName).append("_").append(boost::lexical_cast<string>(geo->GetGridSize())).append(".amp"));
		std::cout << "Saved new grid file. (" << GetTimeString(begin, clock()) << ")\n";
		uBeep(564, 450);
	}

	// Orientation average
	if(this->iterOrder > 0) {
		std::cout << "Orientation averaging (10^" << iterOrder << "). ";
		begin = clock();
		this->AutoCalculateOrientations(*geo);
		timeStr = GetTimeString(begin, clock());
		ss1.str(std::string());
		ss1 << "# Automated calculation.\n";
		if(solvED > 0.0) {
			ss1 << "# Solvent amplitude subtracted from PDB amplitude:" << solvED << "\n";
			ss1 << "# Solvent step size in nm:" << solStep << "\n";
		}
		AutoSave1DFile(timeStr, ss1.str());
	}

	// Quit
	std::cout << "Quiting now...\n";
	return 0;
}

int CommandHanlderObject::DoAlign() {
	// Load PDB
	AutoLoadPDB();

	if(!pdb) {
		std::cout << "The \"-ipdb\" parameter must be supplied.";
		return 684;
	}

	// Align to primary axes (saves)
	pdb->AlignPDB();

	return 0;
}


