#include "Tests.h"


int main(int argc, char* argv[])
{
	PrintTime();
	printf("\n");

	int testCounter = 0, passedTests = 0;
	tests test1;
	
	test1.ValidateGrid();
	return 0;

	test1.Playground3();
	std::cin >> argc;
	return 0;

	if(true) {
		test1.DebyeTest();
		return 0;
	}

	test1.RunPDBKernel("s:\\Basic Dimer\\D+\\1jff_fill.BL00030001 aligned.pdb", 50, 3.0);

	test1.TestTemperton();

	test1.GPUIntegrationSpeedTest();
	return 0;
//  	test1.BasicGridTest();
//  	std::cout << "BasicGridTest finished" << std::endl << std::endl;
	testCounter++;
	if(test1.JacobianIndexTest()) {
		std::cout << "JacobianIndexTest passed" << std::endl << std::endl; passedTests++;
	} else 
		std::cout << "JacobianIndexTest failed" << std::endl << std::endl;
//	return 0;
	testCounter++;
	if(test1.SpaceFillingGridTest()) {
		std::cout << "SpaceFillingGridTest passed" << std::endl << std::endl; passedTests++;
	} else 
		std::cout << "SpaceFillingGridTest failed" << std::endl << std::endl;
//	return 0;
// 
 	testCounter++;
 	if(test1.JacobianGridTest(argv[0])) {
 		std::cout << "JacobianGridTest passed" << std::endl << std::endl; passedTests++;
 	} else
 		std::cout << "JacobianGridTest failed" << std::endl << std::endl;
 		
	//testCounter++;
	//if(test1.GetEulerAnglesTest()) {
	//	std::cout << "GetEulerAnglesTest passed" << std::endl << std::endl; passedTests++;
	//} else
	//	std::cout << "GetEulerAnglesTest failed" << std::endl << std::endl;

	testCounter++;
	if(test1.JacobianSpaceFillingSymmetryTest()) {
		std::cout << "JacobianSpaceFillingSymmetryTest passed" << std::endl << std::endl; passedTests++;
	} else
		std::cout << "JacobianSpaceFillingSymmetryTest failed" << std::endl << std::endl;

	testCounter++;
	if(test1.JacobianManualSymmetryTest()) {
		std::cout << "JacobianManualSymmetryTest passed" << std::endl << std::endl; passedTests++;
	} else
		std::cout << "JacobianManualSymmetryTest failed" << std::endl << std::endl;

	testCounter++;
	if(test1.JacobianRotationTest()) {
		std::cout << "JacobianRotationTest passed" << std::endl << std::endl; passedTests++;
	} else
		std::cout << "JacobianRotationTest failed" << std::endl << std::endl;

	test1.DebyeTest();	// Doesn't return true/false, just benchmark

	bool useGPU = true;
	if(argc > 1) {
		std::string st(argv[1]);
		if(st.compare("gpu") == 0 || st.compare("cpu") == 0)
			useGPU = (st.compare("gpu") == 0);
	}
	testCounter++;
	if(test1.DebugGridSymmetry(useGPU)) {
		std::cout << "DebugGridSymmetry passed" << std::endl << std::endl; passedTests++;
	} else
		std::cout << "DebugGridSymmetry failed" << std::endl << std::endl;

	testCounter++;
	if(test1.DebugGrid()) {
		std::cout << "DebugGrid passed" << std::endl << std::endl; passedTests++;
	} else
		std::cout << "DebugGrid failed" << std::endl << std::endl;

	test1.OriginalBenchmark();	// Just a benchmark

	std::cout << std::endl << "Passed " << passedTests << " out of " << testCounter << " tests" << std::endl << std::endl;

}

