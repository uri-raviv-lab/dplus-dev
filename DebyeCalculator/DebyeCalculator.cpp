#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif

#include "../backend_version.h"

#include "Backend/Amplitude.h"
#include "DefaultModels/Symmetries.h"
#include "PDBReaderLib.h"
#include "Backend/Symmetry.h"
#include <string>
#include <boost/lexical_cast.hpp>
#include "BackendInterface.h"

#include<boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
namespace fs = boost::filesystem;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/*
#include <cuda_runtime_api.h>
void ResetGPU()
{
	cudaDeviceReset();
}
*/

void printProgressBar(int width, float finished, float total) {
	float frac = finished / total;

	int dotz = int((frac * (float)width) + 0.5f);

	// create the "meter"
	int ii=0;
	printf("%3.0f%% [", frac * 100);
	// part  that's full already
	for ( ; ii < dotz; ii++) {
		printf("=");
	}
	// remaining part (spaces)
	for ( ; ii < width ; ii++) {
		printf(" ");
	}
	// and back to line begin - do not forget the fflush to avoid output buffering problems!
	printf("]\r");
	fflush(stdout);
}

void STDCALL OurProgressFunc(void *unused, double progress) {
	printProgressBar(60, (float)progress, 1.0f);
}

static HMODULE GetBackend(const wchar_t *container) {
#ifdef _WIN32
	std::string scon = "xplusmodels.dll";
#else
	std::string scon = "./libxplusmodels.so";
#endif


	HMODULE hMod = NULL;
#ifdef _WIN32
	hMod = LoadLibraryA(scon.c_str());
#else
	hMod = dlopen(scon.c_str(), RTLD_LAZY);
#endif
	return hMod;
}

void PrintTime() {
	char buff[100];
	time_t now = time (0);
	strftime (buff, 100, "%Y-%m-%d %H:%M:%S", localtime (&now));
	std::cout << buff;
}


int main(int argc, char* argv[])
{
	//////////////////////////////////////////////////////////////////////////
	// Command line argument values
	std::string inFilename, saveFilename, anomalousFilename, helpModule;
	bool useGPU, useDW, electron;
	bool printProgress;
	float solventEDensity = 0.0, c1, qMax, qMin;
	int nqVals;
	int kernelVersion;

	//////////////////////////////////////////////////////////////////////////
	// Parse command line
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", po::value<std::string>(&helpModule)->implicit_value(""),
			"Print this help message")
		("PDBFile,i", po::value<std::string>(&inFilename),
			"Input PDB file")
		("Electron,e", po::value<bool>(&electron)->default_value(false),
			"Electron PDB")
		("out,o", po::value<std::string>(&saveFilename)->default_value("Debye.out"),
			"The name of the file to write the results to")
		("gpu,g", po::value<bool>(&useGPU)->implicit_value(true)
			->default_value(false), "Use a GPU to calculate if available")
		("progress,p", po::value<bool>(&printProgress)->implicit_value(true)
			->default_value(true), "Print progressbar")
		("sol,s", po::value<float>(&solventEDensity)->implicit_value(333.f), 
			"The solvent electron density to be subtracted as Gaussians")
		("c1,c", po::value<float>(&c1)->default_value(1.f), 
			"The c1 value used to adjust the excluded volume of the atoms")
		("deb,d", po::value<bool>(&useDW)->implicit_value(true)
			->default_value(false), "Take the B factor into account if available")
		("qMin", po::value<float>(&qMin)->default_value(0.f),
			"The minimum q value from which the intensity will be calculated")
		("qMax,q", po::value<float>(&qMax)->default_value(6.f),
			"The maximum q value to which the intensity will be calculated")
		("qVals,n", po::value<int>(&nqVals)->default_value(512),
			"The number of points to be calculated")
		("kernel,k", po::value<int>(&kernelVersion)->default_value(2),
			"The version of the GPU kernel to use. Not safe. "
			"Don't assume any number works unless you know what you're doing. "
			"Valid options: {1,2,4} Others will fail.")
		;

	//po::positional_options_description p;
	//p.add("PDBFile", 1);
	po::variables_map vm;

	try
	{
		po::store(po::command_line_parser(argc, argv).
			options(desc).run(), vm);
		po::notify(vm);
	}
	catch(std::exception& e)
	{
		std::cout << e.what() << "\n";
		return -11;
	}

	if (vm.count("help")) {
		if (helpModule == "anomalous" || helpModule == "a")
		{
			std::cout
				<< "Enables calculation of an anomalous scattering SAXS signal.\n\n"
				<< "For a given set of atoms that their f' and f'' values are known (usually\nmeasured), "
				<< "the atomic form factors will be calculated as\n\n\tF(q,w) = f(q) + f'(w) + i f''(w)\n\n"
				<< "The input file should be in the format:\n\n"
				<< "\t# Energy    2345.82km\n"
				<< "\t# <f'>\t<f''>\t<atom identifier>\n"
				<< "\t1.2\t2.3\tFe2+\n"
				<< "\t#<f'>\t<f''>\t<atom indices>\n"
				<< "\t1.3\t3.4\t125\t256\n"
				<< "\t#<f'>\t<f''>\t<name of a group thats defined below>\n"
				<< "\t-3e4\t+2.01\t$(someRandomGroupName)\n"
				<< "\t#<name of a group>\t<list of atom indices>\n"
				<< "\t$(someRandomGroupName)\t\t252 472 7344 313\n"
				<< "\n"
				<< "Lines that start with '#' are considered comments and are ignored.\n"
				<< "NOTE: this program does not care what wavelength/energy was used.\n"
				;
			return 0;
		}
		if (helpModule != "")
		{
			std::cout << "The selected option for help does not have a detailed help.\n\n";
		}
		std::cout << "Usage: DebyeCalculator PDBFile [options] \n";
		std::cout << desc;
		return 0;
	}

	if(vm.count("PDBFile"))
	{
		std::cout << "Calculating the Debye scattering of " << inFilename << "." << std::endl;
		std::cout << nqVals << " points between [" << qMin << ", " << qMax << "]" << std::endl;
	}
	else
	{
		std::cout << "\nA PDB file is required as input.\n" << std::endl;
		std::cout << "Usage: DebyeCalculator PDBFile [options] \n";
		std::cout << desc;
		return 0;
	}

	if(vm.count("sol"))
	{
		std::cout << "A solvent electron density of " << solventEDensity << "e/nm^3 will be subtracted." << std::endl;
	}
	if(vm.count("gpu") && useGPU)
		std::cout << "Will attempt to use a GPU" << std::endl;

	if(vm.count("deb") && useDW)
		std::cout << "Using B factors (Debye-Waller)" << std::endl;

	fs::path ipt(inFilename);
	ipt = fs::system_complete(ipt);
	if(!fs::exists(ipt)) {
		std::cout << "No such file " << ipt << std::endl;
		return -3;
	}

	PrintTime(); std::cout << std::endl;

	try
	{
		//////////////////////////////////////////////////////////////////////////
		// Setup Debye calculations
		DebyeCalTester *dct = (anomalousFilename.empty() ? new DebyeCalTester(useGPU, kernelVersion) : new AnomDebyeCalTester(useGPU));

		if (electron)
		{
			if (anomalousFilename.empty())
				dct->pdb = new PDBReader::ElectronPDBReaderOb<F_TYPE>(inFilename, false);
			else
				dct->pdb = new PDBReader::ElectronPDBReaderOb<F_TYPE>(inFilename, false, 0, anomalousFilename);
		}
		else
		{
			if (anomalousFilename.empty())
				dct->pdb = new PDBReader::XRayPDBReaderOb<F_TYPE>(inFilename, false);
			else
				dct->pdb = new PDBReader::XRayPDBReaderOb<F_TYPE>(inFilename, false, 0, anomalousFilename);
		}
		int sz = nqVals;
#ifdef _DEBUG0
		sz = 5;
#endif
		std::vector<double> dqq(sz, 0.0);
		for (int i = 0; i < sz; i++) {
			dqq[i] = qMin + (qMax - qMin) * F_TYPE(i) / F_TYPE(sz - 1);
		}

		dct->SetStop(NULL);
		VectorXd a;
		a.resize(3);
		a[0] = solventEDensity;
		a[1] = useDW ? 1.0 : 0.0;	// Flag to use Debye-Waller
		a(2) = c1;

		//////////////////////////////////////////////////////////////////////////
		// Debugging region
		if (printProgress)
		{
			long long sz = dct->pdb->x.size();
			std::cout << sz << " atoms in pdb file." << std::endl;
			std::cout << "Should be " << sz*sz << " values in the full matrix." << std::endl;
			std::cout << "Should be " << (sz*sz - sz) / 2 + sz << " values in the half matrix." << std::endl;
		}

		//////////////////////////////////////////////////////////////////////////
		// Run calculations
		VectorXd res = dct->CalculateVector(dqq, 0, a, printProgress ? OurProgressFunc : NULL);

		std::cout << std::endl;
		//	ResetGPU();

		//////////////////////////////////////////////////////////////////////////
		// Save results

		fs::path pt(saveFilename);
		pt = fs::system_complete(pt);
		boost::system::error_code er;
		if (!fs::exists(pt.parent_path())) {
			if (!fs::create_directories(pt.parent_path(), er)) {
				std::cout << "Error creating directory: " << pt.string() << "\n";
				std::cout << "Error code: " << er << "\n";
				return -1;
			}
		}
		fs::fstream writeFile;
		writeFile.open(pt, std::ios::out);

		if (writeFile.is_open()) {
			// Header
			writeFile << "# Program revision: " << BACKEND_VERSION << std::endl;
			writeFile << "# Debye scattering of " << inFilename << std::endl;
			writeFile << "# Calculated on ";
			if (useGPU)
				writeFile << "GPU, kernel version " << kernelVersion << std::endl;
			else
				writeFile << "CPU" << std::endl;
			if (solventEDensity > 0.)
				writeFile << "# Subtracted a solvent density of " << solventEDensity << std::endl;
			if (useDW)
				writeFile << "# Used B factors (Debye-Waller)" << std::endl;

			for (int i = 0; i < sz; i++)
				writeFile << dqq[i] << "\t" << res[i] << std::endl;

			writeFile.close();
		}

		delete dct;
	}
	catch (const PDBReader::pdbReader_exception &e)
	{
		std::cout << "PDB exception: " << e.GetErrorMessage() << std::endl;
	}
	catch (const backend_exception& e)
	{
		std::cout << "Backend exception: " << e.what() << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}
}
