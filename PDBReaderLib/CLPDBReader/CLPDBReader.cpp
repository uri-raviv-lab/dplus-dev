// CLPDBReader.cpp : Defines the entry point for the console application.
//

#pragma once
#include "Amplitude.h"
#include "CommandHanlderObject.h"

using namespace boost::filesystem;
namespace fs = boost::filesystem;



int main(int argc, char* argv[])
{
// 	cout << "Parameters:\n";
// 	for(int i = 0; i < argc; i++) {
// 		std::cout << argv[i] << "\n";
// 	}
	if(argc < 2) {
		cout << "There should be some help text here.";
		return 0;
	}

	AutoMode mode = BAD_MODE_SPECIFIED;
	string modeS(argv[1]);
	string inPDB, inGr, outName, orderStr, inDOL,
		aSt, bSt, cSt, alphaSt, betaSt, gammaSt, solvED,
		repaSt, repbSt, repcSt, dwSt, qmaxSt, secSt,
		solStepSt, solRadSt;
	string qresSt;
	string amp1, amp2, scale1St, scale2St;
	string rhoSt, radSt;
	string atmRadDiffSt;
	string atmRadTypeSt;
	int order;
	int qresInt = 5000;
	double dbl;
	char outputSlices = 0;
	char solOnly = 0;
	char fillHolesAsAtoms = 0;
	char discardAmps = 0;


	if(boost::iequals(modeS, "dock")) {
		mode = AUTO_AVI;
	} else if(boost::iequals(modeS, "sf_rec")) {
		mode = AUTO_SF_REC;
	} else if(boost::iequals(modeS, "sf_real")) {
		mode = AUTO_SF_REAL;
	} else if(boost::iequals(modeS, "grid")) {
		mode = AUTO_GRID;
	} else if(boost::iequals(modeS, "sphere")) {
		mode = AUTO_SPHERE;
	} else if(boost::iequals(modeS, "subtract")) {
		mode = AUTO_SUBTRACT_AMPLITUDES;
	} else if(boost::iequals(modeS, "add")) {
		mode = AUTO_ADD_AMPLITUDES;
	} else if(boost::iequals(modeS, "orient")) {
		mode = AUTO_AVERAGE;
	} else if(boost::iequals(modeS, "align")) {
		mode = AUTO_ALIGN;
	} else if(boost::iequals(modeS, "help") ||
		boost::iequals(modeS, "--help") ||
		boost::iequals(modeS, "-h") ||
		boost::iequals(modeS, "/?") ||
		boost::iequals(modeS, "/h")
		) {
			mode = AUTO_HELP;
	} else if(boost::iequals(modeS, "--version") ||
		boost::iequals(modeS, "-version") ||
		boost::iequals(modeS, "version") ||
		boost::iequals(modeS, "-v")		
		) {
			std::cout << "Version: " << SVN_REV_STR << "\n";
			return 0;
	}

	if(mode == BAD_MODE_SPECIFIED) {
		cout << "Bad mode specified";
		return 1;
	}

	if(mode == AUTO_HELP) {
		if(argc < 3) {
			std::cout << "A mode must follow the \"help\" option.\n";
		} else {
			modeS = argv[2];
		}
		if(boost::iequals(modeS, "dock")) {
			std::cout << "Dock: Load a PDB, its amplitude and a series of coordinates and rotations "
				<< "to calculate the combined amplitude and scattering profile.\n"
				<< "Required parameters:\n"
				<< "-ipdb: Input PDB file. Must include path, filename and extension.\n"
				<< "AND/OR\n"
				<< "-iamp: Input amplitude file. Must include path, filename and extension."
				<< " Cannot be used in combination with \"-sol\". If not used, must use the "
				<< "\"-sec\" and \"-qmax\" options.\n"
				<< "-idol: Input of DOL file. Must include path, filename and extension. "
				<< "This file contains the coordinates and rotations of all the copies to be calculated.\n"
				<< "-out: Path and file basename of the files to be output (new PDB, amplitude and I(q)).\n"
				<< "\n\tOptional parameters:\n"
				<< "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer"
				<< " (10^N, enter N).\n"
				<< "-sol: A constant (floating point) that represents the electron density of the solvent to"
				<< " be subtracted from the PDB. Cannot be used in combination with \"-iamp\".\n"
				<< "-sec: The number of sections for the grid. Must be used if \"-iamp\" is not used.\n"
				<< "-solStep: The size in nm to which the solvents space is discretized.\n"
				<< "-solRad: The radius of the solvent in nm. Only relevant if used in conjunction with \"-sol\" and \"-solStep.\"\n"
				<< "-qmax: The maximum q-value for the grid. Must be used if \"-iamp\" is not used.\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
			std::cout << "-discardAmp: Do not save generated amplitude and PDB files.\n";
		
		
		} else if(boost::iequals(modeS, "sf_rec")) {
			std::cout << "sf_rec: Loads an amplitude file and calculate the intensity form it assuming it exist in a given lattice.\n"
				<< "Required parameters:\n"
				<< "-iamp: Input amplitude file. Must include path, filename and extension.\n"
				<< "-out: Path and file basename of the files to be output (I(q)).\n"
				<< "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer"
				<<" (10^N, enter N).\n"
				<< "-a: the size of the a vector in the reciprocal space (in nm^-1).\n"
				<< "-b: the size of the b vector in the reciprocal space (in nm^-1).\n"
				<< "-c: the size of the c vector in the reciprocal space (in nm^-1).\n"
				<< "-alpha: the angle between b and c vectors in the reciprocal space.\n"
				<< "-beta: the angle between c and a vectors in the reciprocal space.\n"
				<< "-gamma: the angle between a and b vectors in the reciprocal space.\n"
				<< "-rep_a: the number of repetitions of a in the real space (integer).\n"
				<< "-rep_b: the number of repetitions of b in the real space (integer).\n"
				<< "-rep_c: the number of repetitions of c in the real space (integer).\n"
				<< "-dw: DeBye-Waller factor - the intensity is multiplied by exp(-1/2 q^2 dw) (in nm^2).\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
		
		
		} else if(boost::iequals(modeS, "sf_real")) {
			std::cout << "sf_real: Loads an amplitude file and calculate the intensity form it assuming it exist in a given lattice.\n"
				<< "\n"
				<< "NOT IMPLEMENTED YET!!!\n"
				<< "\n"
				<< "Required parameters:\n"
				<< "-iamp: Input amplitude file. Must include path, filename and extension.\n"
				<< "-out: Path and file basename of the files to be output (I(q)).\n"
				<< "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer"
				<< " (10^N, enter N).\n"
				<< "-a: the size of the a vector in the real space (in nm).\n"
				<< "-b: the size of the b vector in the real space (in nm).\n"
				<< "-c: the size of the c vector in the real space (in nm).\n"
				<< "-alpha: the angle between b and c vectors in the real space.\n"
				<< "-beta: the angle between c and a vectors in the real space.\n"
				<< "-gamma: the angle between a and b vectors in the real space.\n"
				<< "-rep_a: the number of repetitions of a in the real space (integer).\n"
				<< "-rep_b: the number of repetitions of b in the real space (integer).\n"
				<< "-rep_c: the number of repetitions of c in the real space (integer).\n"
				<< "-dw: DeBye-Waller factor - the intensity is multiplied by exp(-1/2 q^2 dw) (in nm^2).\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";


		} else if(boost::iequals(modeS, "grid")) {
			std::cout << "Grid: Load a PDB and calculate its Fourier space and scattering profile.\n";
			std::cout << "Required parameters:\n";
			std::cout << "-ipdb: Input PDB file. Must include path, filename and extension.\n";
			std::cout << "-out: Path and file basename of the files to be output (amplitude and I(q)).\n";
			std::cout << "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer";
			std::cout << " (10^N, enter N).\n";
			std::cout << "-sec: The number of sections for the grid.\n";
			std::cout << "-qmax: The maximum q-value for the grid.\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-sol: A constant (floating point) that represents the electron density of the solvent to";
			std::cout << " be subtracted from the PDB.\n";
			std::cout << "-solStep: The size in nm to which the solvents space is discretized.\n";
			std::cout << "-solRad: The radius of the solvent in nm. Only relevant if used in conjunction "
						<< "with \"-sol\" and \"-solStep.\"\n";
			std::cout << "-atmRad: Choose the radius type of the atoms (vDW, empirical or calculated - vdw, emp, "
						<< "calc, respectively) to be taken when determining the displaced solvent volume and "
						<< "shape. Default is emp. (DEBUG OPTION: svrgC, svrg)\n";
			std::cout << "-output_slices: Dumps slices of the solvent space to files for later examination.\n";
			std::cout << "-solOnly: Calculates the amplitude from the displaced solvent alone.\n";
			std::cout << "-holesAsAtoms: Fill holes in real space as atoms for solvent subtraction.\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
			std::cout << "-discardAmp: Do not save generated amplitude and PDB files.\n";

		
		} else if(boost::iequals(modeS, "add")) {
			std::cout << "Add: Adds the amplitudes of two amplitude files and saves the result.\n";
			std::cout << "-iamp1: Path and filename of the first amplitude file.\n";
			std::cout << "-iamp2: Path and filename of the second amplitude file.\n";
			std::cout << "-out: Path and file basename of the amplitude file to be output.\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-scale1: A scalar by which to scale iamp1.\n";
			std::cout << "-scale2: A scalar by which to scale iamp2.\n";
			std::cout << "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer";
			std::cout << " (10^N, enter N).\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
			std::cout << "-discardAmp: Do not save generated amplitude and PDB files.\n";

		
		} else if(boost::iequals(modeS, "subtract")) {
			std::cout << "Subtract: Subtract the amplitudes of two amplitude files and saves the result.\n";
			std::cout << "(-iamp1) - (-iamp2) = (-out).\n";
			std::cout << "-iamp1: Path and filename of the first amplitude file.\n";
			std::cout << "-iamp2: Path and filename of the second amplitude file.\n";
			std::cout << "-out: Path and file basename of the amplitude file to be output.\n";
			std::cout << "Optional parameters:\n";
			std::cout << "-scale1: A scalar by which to scale iamp1.\n";
			std::cout << "-scale2: A scalar by which to scale iamp2.\n";
			std::cout << "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer";
			std::cout << " (10^N, enter N).\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
			std::cout << "-discardAmp: Do not save generated amplitude and PDB files.\n";

		
		} else if(boost::iequals(modeS, "orient")) {
			std::cout << "Orientation averaging: loads a single amplitude file (for now) and \n";
			std::cout << "-iamp: Input amplitude file. Must include path, filename and extension.\n";
			std::cout << "-out: Path and file basename of the intensity file to be output.\n";
			std::cout << "-ord: The order of Monte Carlo iterations for the spherical averaging. Must be an integer";
			std::cout << " (10^N, enter N).\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";

		
		} else if(boost::iequals(modeS, "align")) {
			std::cout << "PDB Alignment: loads a PDB file, aligns it to its primary axes and saves a new PDB file.\n";
			std::cout << "-ipdb: Input amplitude file. Must include path, filename and extension.\n";


		} else if(boost::iequals(modeS, "sphere")) {
			std::cout << "Sphere: Calculate the grid of a sphere.\n";
			std::cout << "-rad: The spheres radius.\n";
			std::cout << "-rho: Electron density contrast of the sphere.\n";
			std::cout << "-out: Path and file basename of the amplitude file to be output.\n";
			std::cout << "-sec: The number of sections for the grid.\n";
			std::cout << "-qmax: The maximum q-value for the grid.\n";
			std::cout << "\n\tOptional parameters:\n";
			std::cout << "-qres: The number of q values to be calculated between 0 and qmax. Default is 5,000.\n";
			std::cout << "-discardAmp: Do not save generated amplitude and PDB files.\n";

		
		} else {
			std::cout << "Valid modes to get help for are:\n"
				<< "\t\tdock; sf_rec; sf_real; grid; add; subtract; orient; align; sphere.\n"
				<< "\t\tExample: DockingTest.exe help dock\n";
		}

		return 0;
	}


	for(int i = 2; i < argc; i++) {
		if(boost::iequals(argv[i], "-output_slices")) {
			outputSlices = 1;
			continue;
		} else if(boost::iequals(argv[i], "-solOnly")) {
			solOnly = 1;
			continue;
		} else if(boost::iequals(argv[i], "-holesAsAtoms")) {
			fillHolesAsAtoms = 1;
			continue;
		} else if(boost::iequals(argv[i], "-discardAmp")) {
			discardAmps = 1;
			continue;
		}

		if( i + 1 == argc ) {
			std::cout << "Missing parameter " << argv[i];
			return 2;
		}
		if(boost::iequals(argv[i], "-ipdb")) {
			inPDB = argv[i+1];
		} else if(boost::iequals(argv[i], "-idol")) {
			inDOL = argv[i+1];
		} else if(boost::iequals(argv[i], "-iamp")) {
			inGr = argv[i+1];
		} else if(boost::iequals(argv[i], "-out")) {
			outName = argv[i+1];
		} else if(boost::iequals(argv[i], "-ord")) {
			orderStr = argv[i+1];
		} else if(boost::iequals(argv[i], "-qmax")) {
			qmaxSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-qres")) {
			qresSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-sec")) {
			secSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-sol")) {
			solvED = argv[i+1];
		} else if(boost::iequals(argv[i], "-solstep")) {
			solStepSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-solRad")) {
			solRadSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-atmRad")) {
			atmRadTypeSt = argv[i+1];

		// For mathematical manipulations
		} else if(boost::iequals(argv[i], "-iamp1")) {
			amp1 = argv[i+1];
		} else if(boost::iequals(argv[i], "-iamp2")) {
			amp2 = argv[i+1];
		} else if(boost::iequals(argv[i], "-scale1")) {
			scale1St = argv[i+1];
		} else if(boost::iequals(argv[i], "-scale2")) {
			scale2St = argv[i+1];


		// For geometric sphere
		} else if(boost::iequals(argv[i], "-rho")) {
			rhoSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-rad")) {
			radSt = argv[i+1];

		// For debug purposes. Not to be exposed to users
		} else if(boost::iequals(argv[i], "-solRadAdd")) {
			atmRadDiffSt = argv[i+1];


			// For Tom
		} else if(boost::iequals(argv[i], "-a")) {
			aSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-b")) {
			bSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-c")) {
			cSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-alpha")) {
			alphaSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-beta")) {
			betaSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-gamma")) {
			gammaSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-dw")) {
			dwSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-rep_a")) {
			repaSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-rep_b")) {
			repbSt = argv[i+1];
		} else if(boost::iequals(argv[i], "-rep_c")) {
			repcSt = argv[i+1];

		} else {
			cout << "Unknown parameter \"" << argv[i] << "\"";
			return 3;
		}
		i++;
	}

	char *endp;
	if(orderStr.size() > 0) {
		order = strtol(&orderStr[0], &endp, 0);
		if(endp == &orderStr[0]) {
			cout << "The \"-ord\" argument must be followed by an integer! Try again...";
			return 1;
		}
	}
	if(amp1.size() > 0) {
		fs::path pt(amp1);
		if(!fs::exists(pt)) {
			std::cout <<"The \"-iamp1\" argument must be followed by a valid filepath that you have read access to! Try again...";
			return 1;
		}
	}
	if(amp2.size() > 0) {
		fs::path pt(amp2);
		if(!fs::exists(pt)) {
			std::cout <<"The \"-iamp2\" argument must be followed by a valid filepath that you have read access to! Try again...";
			return 1;
		}
	}
	if(inPDB.size() > 0) {
		fs::path pt(inPDB);
		if(!fs::exists(pt)) {
			std::cout <<"The \"-ipdb\" argument must be followed by a valid filepath that you have read access to! Try again...";
			return 1;
		}
	}
	if(inGr.size() > 0) {
		fs::path pt(inGr);
		if(!fs::exists(pt)) {
			std::cout << "The \"-iamp\" argument must be followed by a valid filepath that you have read access to! Try again...";
			return 1;
		}
	}
	if(inDOL.size() > 0) {
		fs::path pt(inDOL);
		if(!fs::exists(pt)) {
			std::cout <<"The \"-idol\" argument must be followed by a valid filepath that you have read access to! Try again...";
			return 1;
		}
	}
	if(rhoSt.size() > 0) {
		dbl = strtod(&rhoSt[0], &endp);
		if(endp == &rhoSt[0]) {
			std::cout << "The \"-rho\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(radSt.size() > 0) {
		dbl = strtod(&radSt[0], &endp);
		if(endp == &radSt[0]) {
			std::cout << "The \"-rad\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(secSt.size() > 0) {
		dbl = strtod(&secSt[0], &endp);
		if(endp == &secSt[0]) {
			std::cout << "The \"-sec\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(qmaxSt.size() > 0) {
		dbl = strtod(&qmaxSt[0], &endp);
		if(endp == &qmaxSt[0]) {
			std::cout << "The \"-qmax\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(solvED.size() > 0) {
		dbl = strtod(&solvED[0], &endp);
		if(endp == &solvED[0]) {
			std::cout << "The \"-sol\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(solStepSt.size() > 0) {
		dbl = strtod(&solStepSt[0], &endp);
		if(endp == &solStepSt[0]) {
			std::cout << "The \"-solStep\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(solRadSt.size() > 0) {
		dbl = strtod(&solRadSt[0], &endp);
		if(endp == &solRadSt[0]) {
			std::cout << "The \"-solRad\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(atmRadDiffSt.size() > 0) {
		dbl = strtod(&atmRadDiffSt[0], &endp);
		if(endp == &atmRadDiffSt[0]) {
			std::cout << "The \"-solRadAdd\" argument must be followed by a float! Try again...";
			return 1;
		}
	}

	if(aSt.size() > 0) {
		dbl = strtod(&aSt[0], &endp);
		if(endp == &aSt[0]) {
			std::cout << "The \"-a\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(bSt.size() > 0) {
		dbl = strtod(&bSt[0], &endp);
		if(endp == &bSt[0]) {
			std::cout << "The \"-b\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(cSt.size() > 0) {
		dbl = strtod(&cSt[0], &endp);
		if(endp == &cSt[0]) {
			std::cout << "The \"-c\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(alphaSt.size() > 0) {
		dbl = strtod(&alphaSt[0], &endp);
		if(endp == &alphaSt[0]) {
			std::cout << "The \"-alpha\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(betaSt.size() > 0) {
		dbl = strtod(&betaSt[0], &endp);
		if(endp == &betaSt[0]) {
			std::cout << "The \"-beta\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(gammaSt.size() > 0) {
		dbl = strtod(&gammaSt[0], &endp);
		if(endp == &gammaSt[0]) {
			std::cout << "The \"-gamma\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(repaSt.size() > 0) {
		dbl = strtol(&repaSt[0], &endp, 0);
		if(endp == &repaSt[0]) {
			std::cout << "The \"-rep_a\" argument must be followed by an integer! Try again...";
			return 1;
		}
	}
	if(repbSt.size() > 0) {
		dbl = strtol(&repbSt[0], &endp, 0);
		if(endp == &repbSt[0]) {
			std::cout << "The \"-rep_b\" argument must be followed by an integer! Try again...";
			return 1;
		}
	}
	if(repcSt.size() > 0) {
		dbl = strtol(&aSt[0], &endp, 0);
		if(endp == &repcSt[0]) {
			std::cout << "The \"-rep_c\" argument must be followed by an integer! Try again...";
			return 1;
		}
	}
	if(dwSt.size() > 0) {
		dbl = strtod(&aSt[0], &endp);
		if(endp == &dwSt[0]) {
			std::cout << "The \"-dw\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(qresSt.size() > 0) {
		qresInt = strtol(&qresSt[0], &endp, 0);
		if(endp == &qresSt[0]) {
			std::cout << "The \"-qres\" argument must be followed by an integer! Try again...";
			return 1;
		}
	}
	if(scale1St.size() > 0) {
		dbl = strtod(&scale1St[0], &endp);
		if(endp == &scale1St[0]) {
			std::cout << "The \"-scale1\" argument must be followed by a float! Try again...";
			return 1;
		}
	}
	if(scale2St.size() > 0) {
		dbl = strtod(&scale2St[0], &endp);
		if(endp == &scale2St[0]) {
			std::cout << "The \"-scale2\" argument must be followed by a float! Try again...";
			return 1;
		}
	}

	if(mode == NO_AUTO) {
		return 2;
	}

	std::cout << "Starting mode\n";
	
	CommandHanlderObject *cho;
	cho = new CommandHanlderObject(mode, inPDB, inGr, inDOL, outName, solvED, orderStr,
		aSt, bSt, cSt, alphaSt, betaSt, gammaSt, repaSt, repbSt, repcSt, dwSt,
		qmaxSt, secSt, solStepSt, solRadSt, atmRadDiffSt, atmRadTypeSt, outputSlices, solOnly, fillHolesAsAtoms, discardAmps, qresInt);

	switch(mode) {
		case AUTO_GRID:
			return cho->DoAutoGrid();
		case AUTO_AVI:
			return cho->DoAutoDock();
		case AUTO_SF_REC:
			return cho->DoAutoSF_Rec();
		case AUTO_SF_REAL:
			return cho->DoAutoSF_Real();
		case AUTO_ADD_AMPLITUDES:
			return cho->DoAutoAddAmps(amp1, amp2, outName, scale1St, scale2St);
		case AUTO_SUBTRACT_AMPLITUDES:
			return cho->DoAutoSubtractAmps(amp1, amp2, outName, scale1St, scale2St);
		case AUTO_AVERAGE:
			return cho->DoAutoAverage();
		case AUTO_SPHERE:
			return cho->DoAutoSphere(radSt, rhoSt);
		case AUTO_ALIGN:
			return cho->DoAlign();
		default:
			cho->writeCrap();
			return 32;
	}

}

