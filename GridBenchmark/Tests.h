#ifndef __TESTS_H
#define __TESTS_H
#include "Eigen/Core"

#include <ctime>
#include <iomanip>      // std::setprecision

#include "Backend/Grid.h" // For currently tested grid's name
#include "Backend/Amplitude.h"
#include "DefaultModels/Symmetries.h"
#include "PDBReaderLib.h"
#include "Backend/Symmetry.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif
//#include "mathfuncs.h"
#include <string>
#include <boost/lexical_cast.hpp>

#include<boost/filesystem.hpp>

using boost::lexical_cast;
using std::string;

namespace fs = boost::filesystem;

#define QPOINTS 8000
#define REPETITIONS 5

#ifdef _WIN32
#define PREFIX "C:\\xrff\\branches\\XplusD\\x64\\Release\\"
#define WPREFIX L"C:\\xrff\\branches\\XplusD\\x64\\Release\\"
#else
#define PREFIX "./"
#define WPREFIX L"./"
typedef void *HMODULE;
#endif

using namespace Eigen;

void PrintTime();


class tests
{
public:
	tests();
	~tests();

	bool JacobianIndexTest();

	void sandboxTests();
	
	bool BasicGridTest();
	
	bool SpaceFillingGridTest();

	bool GetEulerAnglesTest();

	bool JacobianManualSymmetryTest();

	bool JacobianSpaceFillingSymmetryTest();

	bool CompareModelResults(Amplitude* amp1, Amplitude* amp2);

	bool JacobianRotationTest();

	bool JacobianGridTest(char pth[]);

	bool DebyeTest();

	bool DebugGridSymmetry(bool useGPU = true);

	bool DebugGrid();

	bool OriginalBenchmark();

	bool GPUIntegrationSpeedTest();

	void TestTemperton();

	bool RunPDBKernel(char pth[], int gridSize = 350, double qMax = 5.0);

	void Playground();

	void Playground2();

	void Playground3();

	void ValidateGrid();

private:

};

#endif // __TESTS_H

