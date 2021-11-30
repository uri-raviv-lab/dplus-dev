#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif

#include "../backend_version.h"

#include "Common.h"

#include "Prep.h"

#include <time.h>

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

#include<boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
namespace fs = boost::filesystem;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>

#include "DebyeCalculator.cu"

#include <stdio.h>

void PrintTime() {
	char buff[100];
	time_t now = time (0);
	strftime (buff, 100, "%Y-%m-%d %H:%M:%S", localtime (&now));
	std::cout << buff;
}

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

int main(int argc, char* argv[])
{
	//////////////////////////////////////////////////////////////////////////
	// Command line argument values
	std::string inFilename, saveFilename;
	bool useGPU, useDW;
	float solventEDensity = 0.0, qMax, qMin;
	int nqVals;

	//////////////////////////////////////////////////////////////////////////
	// Parse command line
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Print this help message")
		("PDBFile,i", po::value<std::string>(&inFilename), 
			"Input PDB file")
		("out,o", po::value<std::string>(&saveFilename)->default_value("Debye.out"),
			"The name of the file to write the results to")
		("gpu,g", po::value<bool>(&useGPU)->implicit_value(true)
			->default_value(false), "Use a GPU to calculate if available")
		("sol,s", po::value<float>()->implicit_value(333.f), 
			"The solvent electron density to be subtracted as Gaussians")
		("deb,d", po::value<bool>(&useDW)->implicit_value(true)
			->default_value(false), "Take the B factor into account if available")
		("qMin", po::value<float>(&qMin)->default_value(0.f),
			"The minimum q value from which the intensity will be calculated")
		("qMax,q", po::value<float>(&qMax)->default_value(6.f),
			"The maximum q value to which the intensity will be calculated")
		("qVals,n", po::value<int>(&nqVals)->default_value(512),
			"The number of points to be calculated")
		;

	po::positional_options_description p;
	p.add("PDBFile", 1);
	po::variables_map vm;

	try
	{
		po::store(po::command_line_parser(argc, argv).
			options(desc).positional(p).run(), vm);
		po::notify(vm);
	}
	catch(std::exception& e)
	{
		std::cout << e.what() << "\n";
		return -11;
	}

	if (vm.count("help")) {
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
		std::cout << "A PDB file is required as input." << std::endl;
		return 0;
	}
	if(vm.count("sol"))
	{
		std::cout << "NOT YET IMPLEMENTED:";
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

	// Prepare needed parameters (in Prep.h)
	std::vector<int> atomsPerIon;
	std::vector<float4> loc;
	std::vector<char> atmInd;
	std::vector<u8> ionInd;
	std::vector<float> BFactors;
	std::vector<float> coeffs;
	std::vector<float> atmRad;

	if(!PrepareAllParameters(inFilename,
			atomsPerIon, loc, atmInd, ionInd, BFactors, coeffs, atmRad) )
	{
		std::cout << "Error loading PDB file. Exiting." << std::endl;
		return -3;
	}
	//////////////////////////////////////////////////////////////////////////
	// Setup Debye calculations
//	DebyeCalTester dct(useGPU);
// 	dct.pdb = new PDBReader::PDBReaderOb<F_TYPE>(inFilename, false);
	int sz = nqVals;

	std::vector<double> dqq(sz, 0.0);
	for(int i = 0; i < sz; i++) {
		dqq[i] = qMin +  (qMax - qMin) * double(i) / double(sz - 1);
	}

	//////////////////////////////////////////////////////////////////////////
	// Run calculations
	std::vector<double> res(sz);
	 //= dct.CalculateVector(dqq, 0, a, OurProgressFunc);

	clock_t gpuBeg, gpuEnd;

	gpuBeg = clock();

	int errorCode = GPUCalcDebyeV2(sz, qMin, qMax, res.data(),
	loc.size(), atomsPerIon.data(), loc.data(), ionInd.data(), useDW, 
	BFactors.data(),
	//coeffs.size() / 9,	// This is causing the mismatch
	coeffs.data(),
	solventEDensity > 0.0, false,
	atmInd.data(), atmRad.data(),	
	solventEDensity,
	OurProgressFunc, NULL, 0.0, 1.0, NULL);

	gpuEnd = clock();

	printf("\n\n Timing:\n\tGPU %f seconds",
		double(gpuEnd - gpuBeg)/CLOCKS_PER_SEC);

//	cudaDeviceReset();


	//////////////////////////////////////////////////////////////////////////
	// Save results

	fs::path pt(saveFilename);
	pt = fs::system_complete(pt);
	boost::system::error_code er;
	if(!fs::exists(pt.parent_path()) ) {
		if(!fs::create_directories(pt.parent_path(), er) ) {
			std::cout << "Error creating directory: " << pt.string() << "\n";
			std::cout << "Error code: " << er << "\n";
			return -1;
		}
	}
	fs::fstream writeFile;
	writeFile.open(pt, std::ios::out);

	if(writeFile.is_open()) {
		// Header
		writeFile << "# Program revision: " << BACKEND_VERSION_STR << std::endl;
		writeFile << "# Debye scattering of " << inFilename << std::endl;
		writeFile << "# Calculated on " << (useGPU ? "GPU" : "CPU") << std::endl;
		if(solventEDensity > 0.)
			writeFile << "# Subtracted a solvent density of " << solventEDensity << std::endl;
		if(useDW)
			writeFile << "# Used B factors (Debye-Waller)" << std::endl;

		for(int i = 0; i < sz; i++)
			writeFile << dqq[i] << "\t" << res[i] << std::endl;

		writeFile.close();
	}

}
