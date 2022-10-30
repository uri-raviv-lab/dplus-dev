#include "Tests.h"
#include <typeinfo>	// For error catching
#include <random>
// #include "boost/random/uniform_real_distribution.hpp"
// #include "boost/random/uniform_int_distribution.hpp"
// #include "boost/random/random_device.hpp"
// #include "boost/random/mersenne_twister.hpp"
#include "../Backend/Backend/BackendInterface.h"

#ifndef _WIN32
inline void Beep(int freq, int length) {}
#endif


//#include <cuda_runtime_api.h>
// void ResetGPU()
// {
// 	cudaDeviceReset();
// }

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
	hMod = dlopen(scon.c_str(), RTLD_NOW);
#endif
	return hMod;
}

void PrintTime() {
	char buff[100];
	time_t now = time (0);
	strftime (buff, 100, "%Y-%m-%d %H:%M:%S", localtime (&now));
	std::cout << buff;
}


#ifdef _WIN32
std::string GPUFILE = "GPUBackend.dll";
#else
std::string GPUFILE = "./libgpubackend.so";	// I HAVE NO IDEA IF THIS IS RIGHT
#endif


typedef IModel* (*getModelFunc)(int ind);


tests::tests() {

}

tests::~tests() {

}

bool tests::JacobianRotationTest() {
	// START TIME
	clock_t starttm, endtm;
	AmpGridAmplitude* amp1 = NULL;
	AmpGridAmplitude* amp2 = NULL;

	Eigen::Matrix3d rot = EulerD<double>(Radian(0.0), Radian(0.0), Radian(0.0000001));

	starttm = clock();
	//amp1 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ 1JFF 1 Prerotated 100.amp");
	//amp2 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ 1JFF 1 Prerotated 100.amp");
	//amp2 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ 1JFF Nonrotated 100.amp");
	amp1 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ 1JFF Nonrotated 200.amp");
	amp2 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ 1JFF Nonrotated 200.amp");


#ifdef _DEBUG
	((JacobianSphereGrid*)(amp2->grid))->ExportThetaPhiPlane("S:\\Basic Dimer\\Grid Tests\\Plane.amp", 370315);
#endif // _DEBUG

	std::cout << "Loaded both grids. Starting rotation." << std::endl;


	std::cout << "Rotated second grid. Starting comparison." << std::endl;
	int maxSize = amp1->GetGridSize() / 2;
	double stepSz = amp1->GetGridStepSize();

	std::complex<double> sum1(0.0,0.0), sum2(0.0,0.0);
	std::cout << std::setprecision(9);
	long long worstInd = 0, cnt = 0;
	double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
	double wx= 0.0, wy = 0.0, wz = 0.0;

	// 		maxSize = 34;
	// 		for(int r = 33; r < maxSize; r++) {
	for(int r = 1; r < maxSize; r++) {
		int maxt = 4*r+1;
		for(int t = 0; t < maxt; t++) {
			int maxp = 8*r;
			for(int p = 0; p < maxp; p++) {
				std::complex<double> a1, a2;
				double qx, qy, qz;
				double th = M_PI * double(t) / double(maxt - 1);
				double ph = M_PI * double(2*p) / double(maxp);
				double st = sin(th);
				qx = r * stepSz * st * cos(ph);
				qy = r * stepSz * st * sin(ph);
				qz = r * stepSz * cos(th);
				Eigen::Vector3d qv(qx,qy,qz);
				qv = rot.transpose() * qv;

				a1 = amp1->getAmplitude(qx,qy,qz);
				a2 = amp2->getAmplitude(qv(0), qv(1), qv(2));
				//a2 = amp2->getAmplitude(qx,qy,qz);

				sum1 += a1;
				sum2 += a2;

				meanDiff += fabs(a1.real() - a2.real());
				meanDiff += fabs(a1.imag() - a2.imag());

				// Real comparison
				double v = (closeToZero(a2.real()) ? 0.0 : (1.0 - a1.real() / a2.real()));
				if(fabs(a1.real() - a2.real()) > maxLinDiff && !closeToZero(a1.real()))
					maxLinDiff = (fabs(a1.real() - a2.real()));
				if(fabs(v) > maxDiff) {
					maxDiff = fabs(v);
					worstInd = cnt;
				}

				// Imag comparison
				v = (closeToZero(a2.imag()) ? 0.0 : (1.0 - a1.imag() / a2.imag()));
				if(fabs(a1.imag() - a2.imag()) > maxLinDiff && !closeToZero(a1.imag()))
					maxLinDiff = (fabs(a1.imag() - a2.imag()));
				if(fabs(v) > maxDiff) {
					maxDiff = fabs(v);
					worstInd = cnt;
				}

				if(worstInd == cnt) {
					wx = qx;
					wy = qy;
					wz = qz;
				}
				cnt++;
			}	// for p
		}	// for t
		printProgressBar(60, float(r), float(maxSize));
	} // for r

	endtm = clock();

	OurProgressFunc(NULL, 1.0);

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds! (" CURGRIDNAME ")\n", timeItTook);

	Eigen::Vector3d qv(wx,wy,wz);
	qv = rot * qv;

	std::cout << "Mean error: " << meanDiff / double(2*cnt) << "\n";
	std::cout << "Max (1-a/b) error: " << maxDiff << "\t-log: " << -log10(maxDiff) << "\n";
	std::cout << "Max linear (a-b) error: " << (maxLinDiff) << "\n";
	std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";
	std::cout << "Worst: " << worstInd << "\t" << amp1->getAmplitude(wx,wy,wz) << "\t" 
		<< amp2->getAmplitude(wx,wy,wz) << "\t" << amp2->getAmplitude(qv(0),qv(1),qv(2)) << "\n";
	std::cout << "\t(" << wx << ", " << wy << ", " << wz << ")\n\t" << qv.transpose() << "\n";

	std::complex<double> goodAmp = amp2->getAmplitude(wx,wy,wz);
	std::complex<double> badAmp  = amp2->getAmplitude(qv(0),qv(1),qv(2));

	// Cleanup
	delete amp1;
	delete amp2;

	return true;
}

bool tests::JacobianGridTest(char pth[]) {
	bool pass = true;
	std::cout << "*** Compare the amplitude of 1JFF using the Jacobian grid calculated on the CPU/GPU ***" << std::endl;

	// START TIME
	clock_t starttm, endtm;

	starttm = clock();
	ElectronPDBAmplitude *amp = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);

	amp->calculateGrid(5.0, 50, OurProgressFunc, NULL);

	endtm = clock();

	OurProgressFunc(NULL, 1.0);

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to generate grid! (" CURGRIDNAME ")\n", timeItTook);

	bool bRanGPU;
	fs::path runPath(pth);
	runPath = fs::system_complete(runPath);
	bRanGPU = fs::exists(runPath.parent_path() / fs::path(GPUFILE));

	starttm = clock();
	if(bRanGPU) {
		amp->WriteAmplitudeToFile(L"S:\\Basic Dimer\\Grid Tests\\D+ GPU 1JFF 50 GB.amp");
	} else {
		amp->WriteAmplitudeToFile(L"S:\\Basic Dimer\\Grid Tests\\D+ CPU 1JFF 50 GB.amp");
	}
	endtm = clock();
	timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to write file! (" CURGRIDNAME ")\n", timeItTook);

	AmpGridAmplitude* amp2 = NULL;

	if(bRanGPU) {
		amp2 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ CPU 1JFF 50 GB.amp");
	} else {
		amp2 = new AmpGridAmplitude("S:\\Basic Dimer\\Grid Tests\\D+ GPU 1JFF 50 GB.amp");
	}

	{
		double *dat1 = amp->GetDataPointer();
		double *dat2 = amp2->GetDataPointer();
		double sum1 = 0.0, sum2 = 0.0;
		u64 len;
		len = amp->GetDataPointerSize() / sizeof(double);
		std::cout << std::setprecision(9);
		u64 worstInd = 0;
		if(amp2 && len == amp2->GetDataPointerSize() / sizeof(double)) {
			double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
			for(u64 i = 0; i < 40/*len/10*/; i++) {
				// Coarse sum of all elements
				sum1 += dat1[i];
				sum2 += dat2[i];
				// Average difference
				meanDiff += fabs(dat1[i] - dat2[i]);
				double v = (closeToZero(dat2[i]) ? 0.0 : (1.0 - dat1[i] / dat2[i]));
				if(fabs(dat1[i] - dat2[i]) > maxLinDiff && !closeToZero(dat2[i]))
					maxLinDiff = (fabs(dat1[i] - dat2[i]));
				if(v > maxDiff) {
					maxDiff = v;
					worstInd = i;
				}
				//i++;
			}
			std::cout << "Mean error: " << meanDiff / double(len) << "\n";
			std::cout << "Max (1-a/b) error: " << maxDiff << "\t-log: " << -log10(maxDiff) << "\n";
			std::cout << "Max linear (a-b) error: " << (maxLinDiff) << "\n";
			std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";
			std::cout << "Worst: " << worstInd << "\t" << dat1[worstInd] << "\t" << dat2[worstInd] << "\n";

			endtm = clock();

			timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
			printf("\nTook %f seconds to compare files!\n", timeItTook);

			if(-log10(maxDiff) < 6.0) {
				std::cout << "BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!!\n";
				Beep(600, 450);
				pass = false;
			}

		} else {
			std::cout << "Different length data\n";
			std::cout << "Generated length: " << len << "\n";
			std::cout << "File length: " << amp2->GetDataPointerSize() / sizeof(double) << "\n";
			pass = false;
		}
	}

	delete amp;
	delete amp2;

#ifdef _DEBUG
	//std::cin >> argc;
#endif // _DEBUG

	return pass;
}

bool tests::DebyeTest() {
	//	S:\Basic Dimer
	electronDebyeCalTester dct(false);
	//	dct.pdb = new PDBReader::PDBReaderOb<F_TYPE>("C:\\DesktopStorage\\TUB PDBs\\CF4\\cf4.pdb", false);
	dct.pdb = new PDBReader::ElectronPDBReaderOb<F_TYPE>("S:\\Basic Dimer\\1JFF_Aligned q7- in vacuo_pdb.pdb", false);
	int sz = 2*512;
#ifdef _DEBUG
	sz = 5;
#endif
	F_TYPE qMa = 7.0;
	std::vector<double> dqq(sz, 0.0);
	for(int i = 0; i < sz; i++) {
		dqq[i] = qMa * F_TYPE(i) / F_TYPE(sz - 1);
	}

	dct.SetStop(NULL);
    VectorXd bla;
	dct.CalculateVector(dqq, 0, bla, OurProgressFunc, NULL);

#ifdef _DEBUG
	//std::cin >> argc;
#endif

	return true;
}

bool tests::DebugGridSymmetry( bool useGPU /*= true*/ ) {
	bool pass = true;
	clock_t bg, nd;
	double tit = 0.0;
	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	IModel *model = getModel(2);
	if(dynamic_cast<Geometry *>(model))
		dynamic_cast<Geometry *>(model)->SetEDProfile(EDProfile());
	//dynamic_cast<Geometry *>(model)->


	ISymmetry *isymm = (ISymmetry *)getModel(25);
	Symmetry *symm = dynamic_cast<Symmetry *>(isymm);

	GridSymmetry *gs = dynamic_cast<GridSymmetry *>(symm);
	GeometricAmplitude *geo = new GeometricAmplitude(dynamic_cast<FFModel *>(model));
	gs->AddSubAmplitude(geo);
	gs->SetUseGrid(true);
	gs->SetLocationData(0.0, 0.0, 0.0, Radian(), Radian(), Radian());
	VectorXd pr;
	pr.resize(33);
	pr(0) = 1.0;
	pr(1) = 19.0;
	pr(2) = 0.0;
	pr(3) = 0.0;
	pr(4) = 0.0;
	pr(5) = 0.0;
	pr(6) = 0.0;
	pr(7) = 0.0;
	pr(8) = 3.0;
	pr(9) = 14.5;	// Spacing
	pr(10) = 14.5;
	pr(11) = 14.5;
	pr(12) = 120.0;	// Angles
	pr(13) = 120.0;
	pr(14) = 90.0;
	pr(15) = 2.0;	// Repeats
	pr(16) = 2.0;
	pr(17) = 2.0;
	pr(18) = 1.0;	// What's this??
	pr(19) = 0.0;
	pr(20) = 0.0;
	pr(21) = 0.0;
	pr(22) = 0.0;
	pr(23) = 0.0;
	pr(24) = 0.0;
	pr(25) = 0.0;
	pr(26) = 2.0;
	pr(27) = 0.0;
	pr(28) = 7.0;	// Sphere radius
	pr(29) = 333.0;
	pr(30) = 400.0;
	pr(31) = 1.0;	// Sphere scale
	pr(32) = 0.0;	// Sphere background
	gs->OrganizeParameters(pr, 3);
	gs->PreCalculate(pr, 3);

	double qm = 5.0;
	int secs = 500;
	//	useGPU = false;

	bg = clock();
	if(useGPU)
		gs->calculateGrid(qm, secs);
	else
		gs->Symmetry::calculateGrid(qm, secs);
	nd = clock();

	tit =  ((double)(nd - bg)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to generate grid! (" CURGRIDNAME ")\n", tit);
	std::wstring outName = ((useGPU) ? (L"S:\\SYmmetry\\gpu_" WCURGRIDNAME  L".amp") : (L"S:\\SYmmetry\\cpu_" WCURGRIDNAME L".amp"));
	gs->WriteAmplitudeToFile(outName);

	AmpGridAmplitude* amp3 = NULL;
	amp3 = new AmpGridAmplitude("S:\\SYmmetry\\cpu_"  CURGRIDNAME  ".amp");
	{
		bg = clock();
		double *dat1 = gs->GetDataPointer();
		double *dat2 = amp3->GetDataPointer();
		double sum1 = 0.0, sum2 = 0.0;
		u64 len;
		len = gs->GetDataPointerSize() / sizeof(double);
		std::cout << std::setprecision(9);
		if(amp3 && len == (amp3->GetDataPointerSize() / sizeof(double))) {
			u64 mInd = 0, mlInd = 0;
			double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
			std::cout << "First voxel:\n\t" << dat1[0] << "\t" << dat2[0] << "\n\t"  << dat1[1] << "\t" << dat2[1] << "\n";
			for(u64 i = 0; i < len; i++) {
				if(dat1[i] != dat1[i])
					printf("Bad number at index %llu, %f\n", i, dat1[i]);
				sum1 += dat1[i];
				sum2 += dat2[i];
				meanDiff += fabs(dat1[i] - dat2[i]);
				double v = (closeToZero(dat2[i]) ? 0.0 : (1.0 - dat1[i] / dat2[i]));
				if(fabs(dat1[i] - dat2[i]) > maxLinDiff) {
					maxLinDiff = fabs(dat1[i] - dat2[i]);
					mInd = i;
				}
				if(v > maxDiff) {
					maxDiff = v;
					mlInd = i;
				}
			}
			std::cout << "Zero threshold: " << 10000.0 * std::numeric_limits<double>::epsilon() << "\n";
			std::cout << "Mean error: " << meanDiff / double(len) << "\n";
			std::cout << "Max (1-a/b) error: " << (maxDiff) << " @ "<< mlInd << "\t" << lexical_cast<string>(dat1[mlInd]) << " / " << lexical_cast<string>(dat2[mlInd]) << "\n";
			std::cout << "Max log(1-a/b) error: " << -log10(maxDiff) << "\n";
			std::cout << "Max linear (a-b) error: " << (maxLinDiff) << " @ "<< mInd << "\t" << lexical_cast<string>(dat1[mInd]) << " - " << lexical_cast<string>(dat2[mInd]) <<"\n";
			std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

			nd = clock();

			tit =  ((double)(nd - bg)) / CLOCKS_PER_SEC;
			printf("\nTook %f seconds to compare files!\n", tit);

			if(fabs(maxDiff) != 0.0 && -log10(maxDiff) < 6.0) {
				std::cout << "BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!!\n";
				Beep(600, 450);
				Beep(500, 450);
				pass = false;
			}

		} else {
			std::cout << "Different length data\n";
			std::cout << "Generated length: " << len << "\n";
			std::cout << "File length: " << amp3->GetDataPointerSize() / sizeof(double) << "\n";
			pass = false;
		}
		Beep(600, 450);
	}
#ifdef _DEBUG
	//std::cin >> argc;
#endif
	return pass;
}

bool tests::DebugGrid() {
	bool pass = true;
	ElectronPDBAmplitude *pdb = NULL;
	ElectronPDBAmplitude *pdbG = NULL;

	// Generate grid mode
	/*if((argc == 2 && !strcmp(argv[1], "-g")))*/ {
#ifndef _DEBUG
		//pdb = new PDBAmplitude("1JFF_Aligned q4- in vacuo_CL_pdb.pdb", true);
		//delete pdb;
		pdb = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);
		//pdb = new PDBAmplitude("S:\\Roi\\one pentamer deltaC center align to Z.pdb", false);
#else
		//pdb = new PDBAmplitude("C:\\Delete\\CF4.pdb", false);	
		//pdb = new PDBAmplitude("C:\\Delete\\1JFF_Aligned q4- in vacuo_CL_pdb.pdb", true);

		//pdb = new PDBAmplitude("s:\\Roi\\one pentamer deltaC center align to Z.pdb", true);
		pdb = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);


#endif

		clock_t starttm, endtm;
		double timeItTook;
		AmpGridAmplitude* amp2 = NULL;
		OurProgressFunc(NULL, 0.0);

#pragma omp parallel sections
		{
#pragma omp parallel section
			{
#ifndef USE_SPHERE_GRID
				//amp2 = new AmpGridAmplitude("s:\\Basic Dimer\\1JFF_Aligned q4- in vacuo_D+_Centered_350.amp");
				//amp2 = new AmpGridAmplitude("s:\\Basic Dimer\\1JFF_Aligned q4- SVG_solvent_D+_40.amp");
				//amp2 = new AmpGridAmplitude("S:\\Roi\\Solvation tests\\D+ CPU.amp");
				amp2 = new AmpGridAmplitude("S:\\Roi\\Solvation tests\\D+ CPU Artificial layout.amp");

#endif // USE_SPHERE_GRID
			}
#pragma omp parallel section
			{
				starttm = clock();

				//pdb->calculateGrid(4.0, 350, OurProgressFunc);
				pdb->calculateGrid(4.0, 200, OurProgressFunc);
				//pdb->calculateGrid(4.0, 350, OurProgressFunc);

				endtm = clock();

				OurProgressFunc(NULL, 1.0);

				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to generate grid! (" CURGRIDNAME ")\n", timeItTook);


				return pass;


				starttm = clock();
				pdbG = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);
				pdbG->calculateGrid(4.0, 350, OurProgressFunc);
				endtm = clock();
				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\n%x\n%x\n", pdb, pdbG);
				printf("\nTook %f seconds to generate second amplitude! (" CURGRIDNAME ")\n", timeItTook);

				starttm = clock();
				pdb->WriteAmplitudeToFile(L"S:\\Roi\\Solvation tests\\D+ GPU Artificial layout.amp");
				endtm = clock();
				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to write file! (SSD) (" CURGRIDNAME ")\n", timeItTook);

				starttm = clock();
				pdb->WriteAmplitudeToFile(L"C:\\Delete\\D+ GPU Artificial layout.amp");
				endtm = clock();
				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to write file! (HDD) (" CURGRIDNAME ")\n", timeItTook);

				/*
				starttm = clock();
				pdb->Rotate(0.49163, 4.04158);
				endtm = clock();
				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to rotate the grid (%d)! (" CURGRIDNAME ")\n", timeItTook, pdb->GetGridSize());

				starttm = clock();
				pdb->Rotate(-0.49163, -4.04158);
				endtm = clock();
				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to rotate the grid back (%d)! (" CURGRIDNAME ")\n", timeItTook, pdb->GetGridSize());
				*/


			}
		}
		{
			double *dat1 = pdb->GetDataPointer();
			double *dat2 = pdbG->GetDataPointer();
			double sum1 = 0.0, sum2 = 0.0;
			u64 len;
			len = pdb->GetDataPointerSize() / sizeof(double);
			std::cout << std::setprecision(9);
			u64 worstInd = 0;
			if(pdbG && len == pdbG->GetDataPointerSize() / sizeof(double)) {
				double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
				FILE *fp2 = fopen("c:\\delete\\Diff-1JFF__ODD.txt", "wb");
				FILE *fp3 = fopen("c:\\delete\\Diff-1JFF__EVEN.txt", "wb");
				fprintf(fp2, "#Generated\tFile\n\n", timeItTook);
				fprintf(fp3, "#Generated\tFile\n\n", timeItTook);
				u64 hf = 2044900;
				for(u64 i = 0; i < len; i++) {
					if((i >= hf) && (i < hf + 1024) && (i%2 == 1))
						fprintf(fp2, "%f,\t%f,\t\t%f\t%f\n", dat1[i], dat2[i], (dat1[i]-dat2[i]), (1.0-(dat1[i]/dat2[i])));
					if((i >= hf) && (i < hf + 1024) && (i%2 == 0))
						fprintf(fp3, "%f,\t%f,\t\t%f\t%f\n", dat1[i], dat2[i], (dat1[i]-dat2[i]), (1.0-(dat1[i]/dat2[i])));

					sum1 += dat1[i];
					sum2 += dat2[i];
					meanDiff += fabs(dat1[i] - dat2[i]);
					double v = (closeToZero(dat2[i]) ? 0.0 : (1.0 - dat1[i] / dat2[i]));
					if(fabs(dat1[i] - dat2[i]) > maxLinDiff && !closeToZero(dat2[i]))
						maxLinDiff = (1.0 - (dat1[i] / dat2[i]));
					if(v > maxDiff) {
						maxDiff = v;
					}
					//i++;
				}
				fclose(fp2);
				fclose(fp3);
				std::cout << "Mean error: " << meanDiff / double(len) << "\n";
				std::cout << "Max (1-a/b) error: " << -log10(maxDiff) << "\n";
				std::cout << "Max linear (a-b) error: " << -log10(maxLinDiff) << "\n";
				std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

				endtm = clock();

				timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
				printf("\nTook %f seconds to compare files!\n", timeItTook);

				if(-log10(maxDiff) < 6.0) {
					std::cout << "BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!!\n";
					Beep(600, 450);
					pass = false;
				}

			} else {
				std::cout << "Different length data\n";
				std::cout << "Generated length: " << len << "\n";
				std::cout << "File length: " << pdbG->GetDataPointerSize() / sizeof(double) << "\n";
				pass = false;
			}
		}
		return -354;
		starttm = clock();
		double *dat1 = pdb->GetDataPointer();
		double *dat2 = amp2->GetDataPointer();
		double sum1 = 0.0, sum2 = 0.0;
		u64 len;
		len = pdb->GetDataPointerSize() / sizeof(double);
		std::cout << std::setprecision(9);
		if(amp2 && len == amp2->GetDataPointerSize() / sizeof(double)) {
			double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
			FILE *fp2 = fopen("c:\\delete\\Diff-1JFF.txt", "wb");
			fprintf(fp2, "#Generated\tFile\n\n", timeItTook);
			for(u64 i = 0; i < len; i++) {
				//i++;
				if(i < 1024)
					// 					std::cout << dat1[i] << "\t" << dat2[i] << "\n";
						fprintf(fp2, "%f,\t%f,\n", dat1[i], dat2[i]);

				sum1 += dat1[i];
				sum2 += dat2[i];
				meanDiff += fabs(dat1[i] - dat2[i]);
				double v = (closeToZero(dat2[i]) ? 0.0 : (1.0 - dat1[i] / dat2[i]));
				if(fabs(dat1[i] - dat2[i]) > maxLinDiff && !closeToZero(dat2[i]))
					maxLinDiff = (1.0 - (dat1[i] / dat2[i]));
				if(v > maxDiff) {
					maxDiff = v;
				}
				//i++;
			}
			fclose(fp2);
			std::cout << "Mean error: " << meanDiff / double(len) << "\n";
			std::cout << "Max (1-a/b) error: " << -log10(maxDiff) << "\n";
			std::cout << "Max linear (a-b) error: " << -log10(maxLinDiff) << "\n";
			std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

			endtm = clock();

			timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
			printf("\nTook %f seconds to compare files!\n", timeItTook);

			if(-log10(maxDiff) < 6.0) {
				std::cout << "BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!!\n";
				Beep(600, 450);
				Beep(700, 450);
				Beep(500, 450);
				Beep(800, 1500);
				Beep(300, 1500);
				pass = false;
			}

		} else {
			std::cout << "Different length data\n";
			std::cout << "Generated length: " << len << "\n";
			std::cout << "File length: " << amp2->GetDataPointerSize() / sizeof(double) << "\n";
			pass = false;
		}
		Beep(600, 450);

		// 		pdb->SetOutputSlices(false, "c:\\Delete\\Slices3\\");
		// 		starttm = clock();
		// 		pdb->CalculateSolventAmp(0.02, 333.0, 355.0, 0.14, 0.0);
		// 		endtm = clock();
		// 		timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
		// 
		// 		std::cout << "\nTook " << timeItTook << " seconds to prepare the solvent for calculation.\n" ;

#ifdef _DEBUG
		std::cin >> timeItTook;
#endif
		return pass;
	}
}

bool tests::OriginalBenchmark() {
	bool pass = true;
	// START TIME
	clock_t starttm, endtm;

	starttm = clock();
	//AmpGridAmplitude *amp = new AmpGridAmplitude("S:\\Basic Dimer\\1JFF_Aligned q7- in vacuo_350.amp");
	Amplitude *amp = new AmpGridAmplitude(PREFIX "blah_" CURGRIDNAME ".amp");
	if(amp->getError() != PDB_OK) {
		printf("ERROR while creating amp grid\n");
		return 2;
	}

	endtm = clock();

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to read file! (" CURGRIDNAME ")\n", timeItTook);


	DomainModel *dom = new DomainModel();
	int stopper = 0;
	dom->AddSubAmplitude(amp);
	double qmax = amp->GetGridStepSize() * double(amp->GetGridSize()) / 2.0;

	// Create q
	std::vector<double> q (QPOINTS);
	printf("qmax %f\n", qmax);
	for(int i = 0; i < QPOINTS; i++)
		q[i] = ((double)i / (double)QPOINTS) * qmax;

	// Create p
	VectorXd p (22);
	p[0] = 1;
	p[1] = 14;
	p[2] = 0;
	p[3] = 0;
	p[4] = 0;
	p[5] = 0;
	p[6] = 0;
	p[7] = 0;
	p[8] = 1;
	p[9] = 10000;
	p[10] = amp->GetGridSize();
	p[11] = 1;
	p[12] = 0.0001;
	p[13] = qmax;
	p[14] = 0;
	p[15] = 0;
	p[16] = 0;
	p[17] = 0;
	p[18] = 0;
	p[19] = 0;
	p[20] = 0;
	p[21] = 0;

	VectorXd res;

	dom->SetStop(&stopper);

	printProgressBar(60, 0, REPETITIONS);
	starttm = clock();

	for(int j = 0; j < REPETITIONS; j++) {
		res = dom->CalculateVector(q, 0, p);
		printProgressBar(60, float(j+1), (float)REPETITIONS);
	}

	endtm = clock();
	// END TIME

	timeItTook =  ((double)(endtm - starttm) / (double)REPETITIONS) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds!\n", timeItTook);

	char a[256] = {0};
	sprintf(a, PREFIX "result_" CURGRIDNAME "_%d.out", (int)time(NULL));
	FILE *fp = fopen(a, "wb");
	if(fp) {
		fprintf(fp, "#Time: %f\n", timeItTook);
		for(int i = 0; i < QPOINTS; i++) 
			fprintf(fp, "%f\t%f\n", q[i], res[i]);
		fclose(fp);
	}

	delete amp;
	delete dom;

	return pass;
}

bool tests::JacobianManualSymmetryTest() {
	std::cout << "****** Comparison of a Manual Symmetry calculated on the CPU/GPU (Jacobian) ******" << std::endl;

	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	bool pass = true;

	int grdSz = 350;
	//grdSz = 50;

	double qRange = 5.0;
#ifdef _DEBUG
	grdSz = 20;
#endif
	int manSymEnum = 26;
	int nLayers = 0;
	IModel *model = getModel(manSymEnum);
	ManualSymmetry *manSym;
	try {
		manSym = (ManualSymmetry*)(model);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model != NULL) {
			std::cout << "Cast of IModel to ManualSymmetry unsuccessful." << std::endl;
		}
		delete model;
		return false;
	}
	VectorXd p(86);
	Amplitude *subAmp;
#define PDB_SUBAMP
#ifdef PDB_SUBAMP
	std::cout << "Creating PDBAmplitude..." << std::endl;
	subAmp = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);

	p(0) = 1.0000000000000000;	p(1) = 70.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
	p(8) = 10.000000000000000;	p(9) = 4.1147514540000003;	p(10) = -8.343278307000000;	p(11) = 9.4867636760000007;
	p(12) = 6.2863478629999996;	p(13) = -14.70873943000000;	p(14) = -10.24404243000000;	p(15) = 3.1547733070000001;
	p(16) = -2.540077494999999;	p(17) = -2.532337184999999;	p(18) = -8.857757503000000;	p(19) = -2.8901054689999999;
	p(20) = -6.399830001999999;	p(21) = 10.783613819999999;	p(22) = 4.8431385230000004;	p(23) = 5.9663745459999999;
	p(24) = 0.2191073520000000;	p(25) = -13.23222537999999;	p(26) = -5.659040147999999;	p(27) = 3.5963329289999999;
	p(28) = 7.6038280949999999;	p(29) = -13.99933552000000;	p(30) = -5.225255552000000;	p(31) = 8.2114413210000006;
	p(32) = -13.91640138000000;	p(33) = 1.9941857080000001;	p(34) = 1.8844872770000001;	p(35) = 6.1106792270000003;
	p(36) = -10.07376061000000;	p(37) = -9.227227838999999;	p(38) = 9.6978777990000005;	p(39) = 178.89649370000001;
	p(40) = 291.13294170000000;	p(41) = 210.82573429999999;	p(42) = 101.65634040000000;	p(43) = 242.24427009999999;
	p(44) = 66.006096260000007;	p(45) = 337.66925479999998;	p(46) = 116.79019780000000;	p(47) = 57.420019519999997;
	p(48) = 358.45665520000000;	p(49) = 275.46750960000003;	p(50) = 32.451171670000001;	p(51) = 32.854133590000004;
	p(52) = 271.56833840000002;	p(53) = 267.74819489999999;	p(54) = 259.75106240000002;	p(55) = 295.28973170000000;
	p(56) = 29.593768319999999;	p(57) = 318.33142320000002;	p(58) = 309.19729999999998;	p(59) = 220.08061369999999;
	p(60) = 356.94084290000001;	p(61) = 321.37907100000001;	p(62) = 1.0152247010000000;	p(63) = 58.687721160000002;
	p(64) = 195.28404040000001;	p(65) = 169.47030889999999;	p(66) = 333.98224920000001;	p(67) = 79.809767919999999;
	p(68) = 252.23137190000000;	p(69) = 1.0000000000000000;	p(70) = 0.0000000000000000;
	// Inner translation
	p(71) = 20.000000000000000;	p(72) = 4.0000000000000000;	p(73) = 7.5000000000000000;
	// Inner rotation
	p(74) = 9.0000000000000000;	p(75) = 23.000000000000000;	p(76) = 76.000000000000000;	
	p(77) = 0.0000000000000000;	p(78) = 1.0000000000000000;	p(79) = 0.0000000000000000;
	p(80) = 0.0500000000000000;	p(81) = 0.1400000000000000;	p(82) = 0.0000000000000000;	p(83) = 0.00000000000000000;
	p(84) = 0.0000000000000000;	p(85) = 0.0000000000000000;
	nLayers = 10;

	p.segment(71,3) = (Vector3d::Random(3) + Vector3d::Constant(1.0)) * 4.0;
	p.segment(74,3) = (Vector3d::Random(3) + Vector3d::Constant(1.0)) * 180.0;
	//p.segment(9,60) = VectorXd::Zero(60);

	// Single object
	if(false) {
   		p.resize(32);
   		p(0) = 1.0000000000000000;	p(1) = 16.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
   		p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
   		p(8) = 1.0000000000000000;	p(9) = 11.1100000000000001;	p(10) = 22.200000000000000;	p(11) = 30.000000000000000;
   		p(12) = 44.400000000000004;	p(13) = 55.000000000000000;	p(14) = 66.000000000000000;	p(15) = 1.0000000000000000;
   		p(16) = 0.0000000000000000;	
 		// inner translation
 		p(17) = 5.500000000000000;	p(18) = 3.30000000000000000;	p(19) = 4.60000000000000000;
 		// inner rotation
   		p(20) = 30.000000000000000;	p(21) = 50.000000000000000;	p(22) = 70.000000000000000;	p(23) = 0.00000000000000000;
   		p(24) = 1.0000000000000000;	p(25) = 0.0000000000000000;	p(26) = 0.0500000000000000;	p(27) = 0.14000000000000001;
   		p(28) = 0.0000000000000000;	p(29) = 0.0000000000000000;	p(30) = 0.0000000000000000;	p(31) = 0.00000000000000000;
 
 		nLayers = 1;
//  		int blSize = 6;
//   		int begPos = 17;
//     	p.segment(begPos, blSize) = VectorXd::Zero(blSize);
	}
#else // PDB_SUBAMP
#ifdef SOME_OTHER_SUBAMP
#else // SOME_OTHER_SUBAMP
	IModel *imodel = getModel(2);
	if(dynamic_cast<Geometry *>(imodel))
		dynamic_cast<Geometry *>(imodel)->SetEDProfile(EDProfile());
	subAmp = new GeometricAmplitude(dynamic_cast<FFModel *>(imodel));
	//delete imodel;

	p.resize(30);

	p(0) = 1.0000000000000000;	p(1) = 16.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
	p(8) = 1.0000000000000000;	p(9) = 1.1000000000000001;	p(10) = 2.2000000000000002;	p(11) = 3.2999999999999998;
	p(12) = 4.4000000000000004;	p(13) = 5.5000000000000000;	p(14) = 6.5999999999999996;	p(15) = 1.0000000000000000;
	p(16) = 0.0000000000000000;	p(17) = 0.0000000000000000;	p(18) = 0.0000000000000000;	p(19) = 0.00000000000000000;
	p(20) = 0.0000000000000000;	p(21) = 0.0000000000000000;	p(22) = 0.0000000000000000;	p(23) = 2.0000000000000000;
	p(24) = 0.0000000000000000;	p(25) = 1.0000000000000000;	p(26) = 333.00000000000000;	p(27) = 400.00000000000000;
	p(28) = 1.0000000000000000;	p(29) = 5.0000000000000000;

	p.segment(9,3) = VectorXd::Zero(3);	// Translation
	p.segment(12,3) = VectorXd::Zero(3);	// Rotation
	nLayers = 1;


#endif // SOME_OTHER_SUBAMP
#endif // PDB_SUBAMP

	manSym->AddSubAmplitude(subAmp);

	manSym->SetUseGrid(true);
	manSym->SetUseGPU(true);
	manSym->PreCalculate(p, nLayers);

	std::cout << "Calculate ManualSymmetry (GPU)..." << std::endl;
	manSym->calculateGrid(qRange, grdSz);

	IModel *model2 = getModel(manSymEnum);
	ManualSymmetry *manSym2;
	try {
		manSym2 = (ManualSymmetry*)(model2);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model2 != NULL) {
			std::cout << "Cast of IModel to ManualSymmetry unsuccessful." << std::endl;
		}
		delete model2;
		delete model;
		return false;
	}

	manSym2->AddSubAmplitude(subAmp);
	manSym2->SetUseGrid(true);
	manSym2->SetUseGPU(false);
	manSym2->PreCalculate(p, nLayers);


	clock_t starttm, endtm;
	starttm = clock();
	std::cout << "Calculate ManualSymmetry (CPU)..." << std::endl;
	manSym2->calculateGrid(qRange, grdSz);
	endtm = clock();

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to calculate the Manual Symmetry on the CPU!\n", timeItTook);
	
	pass = CompareModelResults(manSym, manSym2);

	std::cout << "JacobianManualSymmetryTest done." << std::endl;

	delete manSym;
	delete manSym2;
	delete subAmp;

	return pass;
}

bool tests::BasicGridTest() {
	std::cout << "****** Compare the amplitude accuracy with and without using a grid ******" << std::endl;

	int grdSz = 350;
#ifdef _DEBUG
	grdSz = 300;
#endif
	double qRange = 5.0;
	long long maxIters = 1000000;
	
	ElectronPDBAmplitude *pdbAmp   = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);
	ElectronPDBAmplitude *pdbAmpGr = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);

	pdbAmp->SetUseGrid(false);
	pdbAmpGr->SetUseGrid(true);

	std::cout << "Calculating the basic PDB grid..." << std::endl;
	pdbAmpGr->calculateGrid(qRange, grdSz);

	std::cout << "Comparing amplitude with and without grid..." << std::endl;

	std::mt19937 rng1, rng2;
	rng1.seed(static_cast<unsigned int>(std::time(0)));
	rng2.seed(static_cast<unsigned int>(2*std::time(0)));

	std::uniform_real_distribution<double> qRan(0.0, qRange);
	std::uniform_real_distribution<double> u2(0.0, 2.0);

	std::complex<double> sum1(0.0,0.0), sum2(0.0,0.0);
	double wq = 0.0, wth = 0.0, wph = 0.0;
	double wlq = 0.0, wlth = 0.0, wlph = 0.0;
	double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
	for(long long i = 0; i < maxIters; i++) {
		double q  = qRan(rng1);
		double th = acos(u2(rng2) - 1.0);
		double ph = u2(rng2) * M_PI;

		double st = sin(th);

		double x = q * st * cos(ph);
		double y = q * st * sin(ph);
		double z = q * cos(th);

		std::complex<double> dir = pdbAmp->getAmplitude(x,y,z);
		std::complex<double> grd = pdbAmpGr->getAmplitude(x,y,z);

		std::complex<double> diff = (dir - grd);
		double nDiff = std::abs(diff);
		
		meanDiff += nDiff;

		sum1 += dir;
		sum2 += grd;

		if(maxLinDiff < nDiff) {
			maxLinDiff = nDiff;
			wlq = x; wlth = y; wlph = z;
		}
		double lDif = fabs(1.0 - std::abs(grd) / std::abs(dir));
		if(maxDiff < lDif) {
			maxDiff = lDif;
			wq = x; wth = y; wph = z;
		}
	}

	std::cout << "Mean error: " << meanDiff / double(maxIters) << "\n";
	std::cout << "("<<wq <<", "<<wth <<", "<<wph <<"): "<< "Max (1-a/b) error: " << maxDiff 
				<< "\t-log: " << -log10(maxDiff);// <<
	std::cout << pdbAmp->getAmplitude(wq,wth,wph) << " vs. " << pdbAmpGr->getAmplitude(wq,wth,wph)
				<< "\n";
	std::cout << "("<<wlq<<", "<<wlth<<", "<<wlph<<"): "<< "Max linear (a-b) error: " << (maxLinDiff) 
				<< pdbAmp->getAmplitude(wlq,wlth,wlph) << " vs. " << pdbAmpGr->getAmplitude(wlq,wlth,wlph)
				<< "\n";
	std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

	return true;
}

bool tests::CompareModelResults( Amplitude* amp1, Amplitude* amp2 ) {
	clock_t starttm, endtm;
	double timeItTook;

	starttm = clock();
	std::cout << "Comparing CPU and GPU results" << std::endl;
	double *dat1 = amp1->GetDataPointer();
	double *dat2 = amp2->GetDataPointer();
	double sum1 = 0.0, sum2 = 0.0;
	u64 len;
	len = amp1->GetDataPointerSize() / sizeof(double);
	std::cout << std::setprecision(9);
	u64 worstInd = 0;
	if(len == amp2->GetDataPointerSize() / sizeof(double)) {
		double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
		for(u64 i = 0; i < len; i++) {
			sum1 += dat1[i];
			sum2 += dat2[i];
			meanDiff += fabs(dat1[i] - dat2[i]);
			double v = (closeToZero(dat2[i]) ? 0.0 : fabs(1.0 - dat1[i] / dat2[i]));
			if(v > maxDiff && !closeToZero(dat2[i])
				&& fabs(dat1[i] - dat2[i]) > 2.0e-4
				) {
					maxDiff = v;
					worstInd = i;
			}
			if(fabs(dat1[i] - dat2[i]) > maxLinDiff) {
				maxLinDiff = fabs(dat1[i] - dat2[i]);
			}
			//i++;
		}
		std::cout << "Mean error: " << meanDiff / double(len) << "\n";
		std::cout << "Max (1-a/b) error: " << maxDiff  << "\t-log: "<< -log10(maxDiff) << "\n";
		std::cout << "\t worst values: " << dat1[worstInd] << " " << dat2[worstInd] << "\t" << -log10(fabs(dat2[worstInd])) << std::endl;
		std::cout << "Max linear (a-b) error: " << maxLinDiff << "\n";
		std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

		endtm = clock();

		timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
		printf("\nTook %f seconds to compare methods!\n", timeItTook);

		if(-log10(maxDiff) < 5.0) {
			std::cout << "BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!! BAD!!\n";
			Beep(600, 450);
			return false;
		}

	} else {
		std::cout << "Different length data\n";
		return false;
	}

	return true;
}

bool tests::JacobianSpaceFillingSymmetryTest() {
	std::cout << "****** Comparison of a Space Filling Symmetry calculated on the CPU/GPU (Jacobian) ******" << std::endl;

	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	bool pass = true;

	int grdSz = 350;

	double qRange = 5.0;
#ifdef _DEBUG
	grdSz = 20;
#endif
	int spcFillSymEnum = 25;
	int nLayers = 0;
	IModel *model = getModel(spcFillSymEnum);
	GridSymmetry *grdSym;
	try {
		grdSym = (GridSymmetry*)(model);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model != NULL) {
			std::cout << "Cast of IModel to GridSymmetry unsuccessful." << std::endl;
		}
		delete model;
		return false;
	}
	VectorXd p(35);
	Amplitude *subAmp;
#define PDB_SUBAMP
#ifdef PDB_SUBAMP
	std::cout << "Creating PDBAmplitude..." << std::endl;
	subAmp = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);

	p(0) = 1.0000000000000000;	p(1) = 19.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
	p(8) = 3.0000000000000000;	
	// Length of the unit vectors
	p(9) = 8.8000000000000007;	p(10) = 9.9000000000000004;	p(11) = 10.119999999999999;
	// Angles of the unit vector
	p(12) = 60.000000000000000;	p(13) = 73.000000000000000;	p(14) = 110.000000000000000;
//	p(12) = 90.000000000000000;	p(13) = 90.000000000000000;	p(14) = 90.000000000000000;
	// Repeats in each dimension
	p(15) = 5.0000000000000000;	p(16) = 2.0000000000000000;	p(17) = 4.0000000000000000;
//	p(15) = 1.0000000000000000;	p(16) = 1.0000000000000000;	p(17) = 1.0000000000000000;
	p(18) = 1.0000000000000000;	p(19) = 0.00000000000000000;
	// Translation of the inner object (pdb)
	p(20) = 1.1000000000000001;	p(21) = 2.2000000000000002;	p(22) = 3.2999999999999998;
	// Rotation of the inner object (pdb)
	p(23) = 49.000000000000000;	p(24) = 72.000000000000000;	p(25) = 134.00000000000000;
	p(26) = 0.0000000000000000;	p(27) = 1.0000000000000000;	p(28) = 0.00000000000000000;
	p(29) = 0.0500000000000000;	p(30) = 0.1400000000000000;	p(31) = 0.00000000000000000;
	p(32) = 0.0000000000000000;	p(33) = 0.0000000000000000;	p(34) = 0.00000000000000000;

 	int blSize = 3;
 	int begPos = 20;
 	//p.segment(begPos, blSize) = VectorXd::Zero(blSize);

	nLayers = 3;

#endif // PDB_SUBAMP


	grdSym->AddSubAmplitude(subAmp);

	grdSym->SetUseGrid(true);
	grdSym->SetUseGPU(true);
	grdSym->PreCalculate(p, nLayers);

	std::cout << "Calculate SpaceFillingSymmetry (GPU)..." << std::endl;
	grdSym->calculateGrid(qRange, grdSz);

	IModel *model2 = getModel(spcFillSymEnum);
	GridSymmetry *grdSym2;
	try {
		grdSym2 = (GridSymmetry *)(model2);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model2 != NULL) {
			std::cout << "Cast of IModel to GridSymmetry unsuccessful." << std::endl;
		}
		delete model2;
		delete grdSym;	grdSym	= NULL;
		return false;
	}

	grdSym2->AddSubAmplitude(subAmp);
	grdSym2->SetUseGrid(true);
	grdSym2->SetUseGPU(false);
	grdSym2->PreCalculate(p, nLayers);


	clock_t starttm, endtm;
	starttm = clock();
	std::cout << "Calculate SpaceFillingSymmetry (CPU)..." << std::endl;
	grdSym2->calculateGrid(qRange, grdSz);
	endtm = clock();

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to calculate the Space Filling Symmetry on the CPU!\n", timeItTook);

	pass = CompareModelResults(grdSym, grdSym2);

	delete grdSym;	grdSym	= NULL;
	delete grdSym2;	grdSym2	= NULL;
	delete subAmp;	subAmp	= NULL;

	return pass;
}

float MatDiff(const Matrix3f& a, const Matrix3f& b)
{
	float diff = 0.0f;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			diff += fabs(a(i,j) - b(i,j));

	return diff;
}

bool tests::GetEulerAnglesTest() {
	bool result = true;
	float maxDiff = 0.0f;
	for(int alpha = 0; alpha < 360; alpha++)
	{
		for(int beta = 0; beta < 360; beta++)
		{
			for(int gamma = 0; gamma < 360; gamma++)
			{
				Radian rx = Radian(Degree(alpha));
				Radian ry = Radian(Degree(beta ));
				Radian rz = Radian(Degree(gamma));
				Matrix3f origMat = EulerD<float>(rx, ry, rz);
				Radian resx, resy, resz;
				GetEulerAngles(Eigen::Quaternionf(origMat), resx, resy, resz);
				Matrix3f newMat = EulerD<float>(resx, resy, resz);				
				float diff = MatDiff(origMat, newMat);
				if(diff > maxDiff)
					maxDiff = diff;
				if(diff > 0.001)
				{
					printf("DIFFERENCE IN ANGLE %3d,%3d,%3d: %f\n", alpha,beta,gamma,diff);
					result = false;
				}

			}
		}
	}

	printf("Maximal difference: %f\n", maxDiff);

	return result;
}

/*
*/
bool tests::SpaceFillingGridTest() {
	std::cout << "****** Compare the space filling amplitude accuracy with and without using a grid ******" << std::endl;

	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	bool pass = true;

	int grdSz = 350;

	double qRange = 5.0;
#ifdef _DEBUG
	grdSz = 20;
#endif
	int spcFillSymEnum = 25;
	int nLayers = 0;
	IModel *model = getModel(spcFillSymEnum);
	GridSymmetry *grdSym;
	try {
		grdSym = (GridSymmetry*)(model);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model != NULL) {
			std::cout << "Cast of IModel to GridSymmetry unsuccessful." << std::endl;
		}
		delete model;
		return false;
	}
	VectorXd p(35);
	Amplitude *subAmp;
#define PDB_SUBAMP
#ifdef PDB_SUBAMP
	std::cout << "Creating PDBAmplitude..." << std::endl;
	subAmp = new ElectronPDBAmplitude("S:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb", true);

	p(0) = 1.0000000000000000;	p(1) = 19.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
	p(8) = 3.0000000000000000;	
	// Length of the unit vectors
	p(9) = 2.5000000000000000;	p(10) = 2.5000000000000000;	p(11) = 8.6999999999999993;
	// Angles of the unit vector
	p(12) = 90.000000000000000;	p(13) = 90.000000000000000;	p(14) = 90.000000000000000;
	// Repeats in each dimension
	p(15) = 1.0000000000000000;	p(16) = 1.0000000000000000;	p(17) = 3.000000000000000;
	p(18) = 1.0000000000000000;	p(19) = 0.00000000000000000;
	// Translation of the inner object (pdb)
	p(20) = 0.0000000000000000;	p(21) = 0.0000000000000000;	p(22) = 0.0000000000000000;
	// Rotation of the inner object (pdb)
	p(23) = 0.0000000000000000;	p(24) = 0.0000000000000000;	p(25) = 0.0000000000000000;
	p(26) = 0.0000000000000000;	p(27) = 1.0000000000000000;
	p(28) = 0.0000000000000000;	p(29) = 0.0500000000000000;	p(30) = 0.1400000000000000;	p(31) = 0.00000000000000000;
	p(32) = 0.0000000000000000;	p(33) = 0.0000000000000000;	p(34) = 0.0000000000000000;

	int blSize = 3;
	int begPos = 20;
	//p.segment(begPos, blSize) = VectorXd::Zero(blSize);

	nLayers = 3;

#endif // PDB_SUBAMP


	grdSym->AddSubAmplitude(subAmp);

	grdSym->SetUseGrid(true);
	grdSym->SetUseGPU(true);
	grdSym->PreCalculate(p, nLayers);

	std::cout << "Calculate SpaceFillingSymmetry (GPU)..." << std::endl;
	grdSym->calculateGrid(qRange, grdSz);

	IModel *model2 = getModel(spcFillSymEnum);
	GridSymmetry *grdSym2;
	try {
		grdSym2 = (GridSymmetry *)(model2);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model2 != NULL) {
			std::cout << "Cast of IModel to GridSymmetry unsuccessful." << std::endl;
		}
		delete model2;
		delete grdSym;	grdSym	= NULL;
		return false;
	}

	subAmp->SetUseGrid(false);
	subAmp->SetUseGPU(false);
	grdSym2->AddSubAmplitude(subAmp);
	grdSym2->SetUseGrid(false);
	grdSym2->SetUseGPU(false);
	grdSym2->PreCalculate(p, nLayers);

	std::cout << "Comparing amplitude with and without grid..." << std::endl;
	long long maxIters = 1000000;

	std::mt19937 rng1, rng2;
	rng1.seed(static_cast<unsigned int>(std::time(0)));
	rng2.seed(static_cast<unsigned int>(2*std::time(0)));

	std::uniform_real_distribution<double> qRan(0.0, qRange);
	std::uniform_real_distribution<double> u2(0.0, 2.0);

	std::complex<double> sum1(0.0,0.0), sum2(0.0,0.0);
	double wq = 0.0, wth = 0.0, wph = 0.0;
	double wlq = 0.0, wlth = 0.0, wlph = 0.0;
	double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
	for(long long i = 0; i < maxIters; i++) {
		double q  = qRan(rng1);
		double th = acos(u2(rng2) - 1.0);
		double ph = u2(rng2) * M_PI;

		double st = sin(th);

		double x = q * st * cos(ph);
		double y = q * st * sin(ph);
		double z = q * cos(th);

		std::complex<double> dir = grdSym2->getAmplitude(x,y,z);
		std::complex<double> grd = grdSym->getAmplitude(x,y,z);

		std::complex<double> diff = (dir - grd);
		double nDiff = std::abs(diff);

		meanDiff += nDiff;

		sum1 += dir;
		sum2 += grd;

		if(maxLinDiff < nDiff) {
			maxLinDiff = nDiff;
			wlq = x; wlth = y; wlph = z;
		}
		double lDif = fabs(1.0 - std::abs(grd) / std::abs(dir));
		if(maxDiff < lDif && nDiff > 0.0001) {
			maxDiff = lDif;
			wq = x; wth = y; wph = z;
		}
	}

	std::cout << "Mean error: " << meanDiff / double(maxIters) << "\n";
	std::cout << "("<<wq <<", "<<wth <<", "<<wph <<"): "<< "Max (1-a/b) error: " << maxDiff 
		<< "\t-log: " << -log10(maxDiff);// <<
	std::cout << grdSym2->getAmplitude(wq,wth,wph) << " vs. " << grdSym->getAmplitude(wq,wth,wph)
		<< "\n";
	std::cout << "("<<wlq<<", "<<wlth<<", "<<wlph<<"): "<< "Max linear (a-b) error: " << (maxLinDiff) 
		<< grdSym2->getAmplitude(wlq,wlth,wlph) << " vs. " << grdSym->getAmplitude(wlq,wlth,wlph)
		<< "\n";
	std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";

	return true;
}

void tests::sandboxTests() {
	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	bool pass = true;

	int grdSz = 350;

	double qRange = 5.0;
#ifdef _DEBUG
//	grdSz = 20;
#endif
	int spcFillSymEnum = 25;
	const int geometryEnum = 0;	// 0 cyl, 2 sphere
	int nLayers = 0;
	IModel *model = getModel(spcFillSymEnum);
	GridSymmetry *grdSym;
	try {
		grdSym = (GridSymmetry*)(model);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(model != NULL) {
			std::cout << "Cast of IModel to GridSymmetry unsuccessful." << std::endl;
		}
		delete model;
		return;
	}
	VectorXd p(6);
#define SPHERE_SUBAMP
#ifdef SPHERE_SUBAMP
	IModel *smodel = getModel(geometryEnum);
	GeometricAmplitude *geo;
	try {
		geo = (GeometricAmplitude*)(smodel);	// For some reason, dynamic_cast doesn't work...
	} catch (const std::bad_cast& e){
		std::cerr << e.what() << std::endl;

		if(smodel != NULL) {
			std::cout << "Cast of IModel to GeometricAmplitude unsuccessful." << std::endl;
		}
		delete smodel;
		return;
	}
	geo = new GeometricAmplitude(dynamic_cast<FFModel *>(smodel));
	
// 	p(0) = 1.0000000000000000;	p(1) = 19.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
// 	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
// 	p(8) = 3.0000000000000000;	p(9) = 2.5000000000000000;	p(10) = 2.5000000000000000;	p(11) = 4.0 /*Repeat Distance*/;
// 	p(12) = 90.000000000000000;	p(13) = 90.000000000000000;	p(14) = 90.000000000000000;	p(15) = 1.0000000000000000;
// 	p(16) = 1.0000000000000000;	p(17) = 60.0 /* Repeats */;	p(18) = 1.0000000000000000;	p(19) = 0.00000000000000000;
// 	p(20) = 0.0000000000000000;	p(21) = 0.0000000000000000;	p(22) = 0.000000000000000;	p(23) = 0.00000000000000000;
// 	p(24) = 0.0000000000000000;	p(25) = 0.0000000000000000;	p(26) = 2.0000000000000000;	p(27) = 0.00000000000000000;
// 	p(28) = 2.0000000000000000;	p(29) = 333.00000000000000;	p(30) = 400.00000000000000;	p(31) = 1.0; /*Scale*/
// 	p(32) = 0.0 /*Background*/;
	nLayers = 3;


	p(0) = 0.00000000000000000;	p(1) = 2.0000000000000000;	p(2) = 333.00000000000000;
	p(3) = 400.00000000000000;	p(4) = 1.0; /*Scale*/		p(5) = 0.0 /*Background*/;
	nLayers = 2;

	p.resize(7);
	p(0) = 0.00000000000000000;	p(1) = 2.3000000000000000;	p(2) = 333.00000000000000;
	p(3) = 400.00000000000000;	p(4) = 1.0; /*Scale*/		p(5) = 0.0 /*Background*/;
	p(6) = 240.0 /*Length*/;


	int blSize = 3;
	int begPos = 20;
	//p.segment(begPos, blSize) = VectorXd::Zero(blSize);


#endif // PDB_SUBAMP

	//geo->SetUseGPU(true);
	geo->SetUseGrid(true);
// 	grdSym->AddSubAmplitude(geo);
// 
// 	grdSym->SetUseGrid(true);
// 	grdSym->SetUseGPU(false);
// 	grdSym->PreCalculate(p, nLayers);

	geo->PreCalculate(p, nLayers);
	std::cout << "Calculate Geometric (" << geo->GetName() << ") Amplitude (CPU)..." << std::endl;
	geo->calculateGrid(qRange, grdSz);

	std::cout << "Checking values..." << std::endl;

	double deltaQ = 2.0 * qRange / grdSz;
	VectorXd ranTh, ranPh;
	for(int i = 1; i < 2*grdSz; i++) {
		double qq = double(i) * deltaQ / 2.0;
		int len = 10 * i;
		ranTh = (VectorXd::Random(len/2) + VectorXd::Constant(len/2, 1.0)) * M_PI_2;
		ranPh = (VectorXd::Random(len) + VectorXd::Constant(len, 1.0)) * M_PI;
		double snt, cst, snp, csp;
		std::complex<FACC> res, dif, pres = geo->getAmplitude(qq, 0., 0.);
		FACC difD;
		pres = geo->getAmplitude(0., qq, 0.);
		pres = geo->getAmplitude(0., 0., qq);
		for(int ti = 0; ti < len / 2; ti++) {
			snt = sin(ranTh[ti]);	cst = cos(ranTh[ti]);
			pres = geo->getAmplitude(qq*snt, 0., qq*cst);
			for(int ri = 0; ri < len; ri++) {
				snp = sin(ranPh[ri]);	csp = cos(ranPh[ri]);
				res = geo->getAmplitude(qq*snt*csp, qq*snt*snp, qq*cst);
				assert(res.imag() == 0.0);
				dif = res - pres;
				difD = std::abs(dif);
				if(fabs(1. - std::abs(pres) / std::abs(res)) > 0.0000001 ) {
					std::cout << "Sir, we have issues..." << std::endl;
				}
			}
		}

	}
}

int phiDivisions = 6, thetaDivisions = 3;
typedef uint32_t U32;
typedef uint64_t U64;

long long botInd(long long i) {
	if (i == 0)
		return 0;
	i--;
	return 1 + ( (i*(i+1)*(3+thetaDivisions+2*i*thetaDivisions)*phiDivisions ) / 6) /*((4*i*(i+1)*(7+8*i)) / 3)*/;
}
long long laySz(long long i) {
	if(i == 0)
		return 1;
	return (thetaDivisions*i+1)*(phiDivisions*i);
}

U32 icbrt64(U64 x) {
	int s;
	U32 y;
	U64 b;

	y = 0;
	for (s = 63; s >= 0; s -= 3) {
		y += y;
		b = 3*y*((U64) y + 1) + 1;
		if ((x >> s) >= b) {
			x -= b << s;
			y++;
		}
	}
	return y;
}
#ifndef M_2PI
#define M_2PI 6.28318530717958647692528676656
#endif

void IndicesFromRadians(const u16 ri, const double theta, const double phi,
						long long &tI, long long &pI, long long &base, double &tTh, double &tPh ) {
	// Determine the first cell using ri
	int qi = ri - 1;
	base = 1 + ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;

	// Determine the lowest neighbors coordinates within the plane
	int phiPoints = phiDivisions * ri;
	int thePoints = thetaDivisions * ri;
	double edge = M_2PI / double(phiPoints);

	tI = (theta / M_PI) * double(thePoints);
	pI = (phi  / M_2PI) * double(phiPoints);

	// The value [0, 1] representing the location ratio between the two points
	tTh = (theta / edge) - tI; //fmod(theta, edge) / edge;
	tPh = (phi   / edge) - pI; //fmod(phi, edge) / edge;

	if(fabs(tTh) < 1.0e-10)
	tTh = 0.0;
	if(fabs(tPh) < 1.0e-10)
	tPh = 0.0;
	assert(tTh >= 0.0);
	assert(tPh >= 0.0);
	assert(tTh <= 1.0000000001);
	assert(tPh <= 1.0000000001);

	//pI = (pI == phiPoints) ? 0 : pI;
	if(pI == phiPoints) {
	assert(tPh <= 0.000001);
	pI = 0;
	}
}

long long IndexFromIndices( long long qi, long long ti, long long pi ) {
	if(qi == 0)
		return 0;
	qi--;
	long long base = ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;
	return (base + ti * phiDivisions * (qi+1) + pi + 1);	// The +1 is for the origin
}
void IndicesFromIndex( long long index, long long &qi, long long &ti, long long &pi ) {
	// Check the origin
	if(index == 0) {
		qi = 0;
		ti = 0;
		pi = 0;
		return;
	}

	long long bot, rem;
	// Find the q-radius
	// The numbers here assume thetaDivisions == 4 and phiDivisions == 8	--> no longer
	long long lqi = icbrt64((3*index)/(thetaDivisions*phiDivisions));
	bot = (lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	if(index > bot )
		lqi++;
	lqi--;
	bot = (lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	lqi++;
	qi = lqi;
	rem = index - bot - 1;
	// Find the theta and phi radii
	ti = rem / (phiDivisions*qi);
	pi = rem % (phiDivisions*qi);

}
template<typename T>
T cb(T t) {return t*t*t;}
bool tests::JacobianIndexTest() {
	int gridSize = 350;
	bool pass = true;
	long long prevBot = 0, qind = 0;
	long long voxels = botInd(gridSize+1);
	long long cnt = 1, qq, tt, pp;
	for(int qi = 1; qi < gridSize; qi++) {
		for(int ti = 0; ti < thetaDivisions*qi+1; ti++) {
			for(int pi = 0; pi < phiDivisions*qi; pi++) {
				IndicesFromIndex(cnt, qq, tt, pp);
				long long testInd = IndexFromIndices(qq,tt,pp);
				if(qi != qq)
					pass = false;
				if(ti != tt)
					pass = false;
				if(pi != pp)
					pass = false;
				if(testInd != cnt)
					pass = false;
				cnt++;
			}
		}
	}

	return pass;
}

bool tests::GPUIntegrationSpeedTest() {
	// Set up parameters for integration
	// START TIME
	clock_t starttm, endtm;

	starttm = clock();
	//AmpGridAmplitude *amp = new AmpGridAmplitude("S:\\Basic Dimer\\1JFF_Aligned q7- in vacuo_350.amp");
	Amplitude *amp = new AmpGridAmplitude("S:\\Basic Dimer\\1JFF OA 350 q5.amp");
	if(amp->getError() != PDB_OK) {
		printf("ERROR while creating amp grid\n");
		return 2;
	}

	endtm = clock();

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to read file! (" CURGRIDNAME ")\n", timeItTook);


	DomainModel *dom = new DomainModel();
	int stopper = 0;
	int qpoints = 800;
	dom->AddSubAmplitude(amp);
	double qmax = amp->GetGridStepSize() * double(amp->GetGridSize()) / 2.0;

	// Create p
	VectorXd p (24);
	p(0) = 1.0000000000000000;	p(1) = 15.000000000000000;	p(2) = 0.00000000000000000;	p(3) = 0.00000000000000000;
	p(4) = 0.00000000000000000;	p(5) = 0.00000000000000000;	p(6) = 0.00000000000000000;	p(7) = 0.00000000000000000;
	p(8) = 1.0000000000000000;	
	// Iterations				Grid Size
	p(9) = 1000000.0000000000;	p(10) = 350.00000000000000;	p(11) = 1.0000000000000000;
	// Convergence
	p(12) = 0.001000000000000;	
	// qMax
	p(13) = qmax;
	p(14) = 0.0000000000000000;	p(15) = 0.00000000000000000;
	p(16) = 0.0000000000000000;	p(17) = 0.0000000000000000;	p(18) = 0.0000000000000000;	p(19) = 0.00000000000000000;
	p(20) = 0.0000000000000000;	p(21) = 0.0000000000000000;	p(22) = 0.0000000000000000;	p(23) = 1.0000000000000000;

	// Create q
	std::vector<double> q (qpoints+1);
	printf("qmax %f\n", qmax);
	for(int i = 0; i <= qpoints; i++)
		q[i] = ((double)i / (double)qpoints) * qmax;



	VectorXd res;

	dom->SetStop(&stopper);

	// Run OA

	starttm = clock();

	res = dom->CalculateVector(q, 0, p);

	endtm = clock();
	// END TIME

	timeItTook =  (double)(endtm - starttm) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds!\n", timeItTook);

	delete amp;
	delete dom;
	
//	ResetGPU();
	return true;
}

#include "../Backend/Backend/PeriodicSplineSolver.h"

void tests::TestTemperton() {
//	typedef std::complex<double> TT;
	typedef double TT;
	static const TT arr[] = {0.7071067811865475, 1.000000000000000, 0.7071067811865475, 0., -0.7071067811865475, -1.000000000000000, -0.7071067811865475, 0.};
//	static const TT arr[] = {0.7818314824680298, 0.9749279121818236, 0.4338837391175581, -0.4338837391175581, -0.9749279121818236, -0.7818314824680298, 0.};
// 	static const TT arr[] = {0.40673664307580020775, 0.74314482547739423501, 0.95105651629515357212, 0.99452189536827333692,
// 		0.86602540378443864676, 0.58778525229247312917, 0.20791169081775933710, -0.20791169081775933710, -0.58778525229247312917,
// 		-0.86602540378443864676, -0.99452189536827333692, -0.95105651629515357212, -0.74314482547739423501, -0.40673664307580020775, 0.};
// 	static const TT arr[] = {TT(-707.157, 449.041), TT(-749.197, -195.138), TT(79.3401, -331.19), TT(621.087, 936.675), TT(-789.723, 326.005), TT(-328.311, 724.698), TT(-917.5, 619.531), TT(412.739, 88.131), TT(-921.357, 633.626), TT(960.234, 76.2822), TT(329.126, 558.693), TT(479.204, -366.688), TT(640.817, 144.252), TT(956.499, 51.8553), TT(724.557, -207.162), TT(-709.916, -427.926), TT(-207.209, 36.133), TT(-426.77, 107.215), TT(-919.501, 536.447), TT(955.812, -932.662), TT(403.533, -485.247), TT(-517.652, 152.048), TT(661.409, 714.779), TT(-828.299, -798.494), TT(343.497, 768.923)};

	std::vector<TT> snDat(arr, arr + sizeof(arr) / sizeof(arr[0]));

  	snDat.resize(15);
  	for(int i = 0; i < snDat.size(); i++) {
  		snDat[i] = sin(M_PI * 2. * (i+1) / log(double(snDat.size())));
  	}
	int largestMatrixDim = snDat.size();

	Eigen::ArrayXd cPrime(largestMatrixDim);
	cPrime[0] = 0.25;
	for(int i = 1; i < largestMatrixDim; i++) {
		cPrime[i] = 1.0 / (4.0 - cPrime[i-1]);
	}

	std::vector<TT> altInterpolantCoeffs(snDat.size());

	TempertonEvenlySpacedPeriodicSplineSolver(snDat.data(),
		altInterpolantCoeffs.data(),
		snDat.size());

	Eigen::Matrix<TT,Eigen::Dynamic,1> oldRes = EvenlySpacedPeriodicSplineSolver(snDat.data(), snDat.size());
	
	printf("\nCurrent method:\n");
	std::cout.precision(16);
	std::cout << std::scientific;
	for(int i = 0; i < oldRes.size(); i++) {
		std::cout << oldRes[i] << std::endl;
	}

	char iii[1024];
	scanf("%c", iii);

}

bool tests::RunPDBKernel( char pth[], int gridSize /*= 350*/, double qMax /*= 5.0*/ ) {
	// START TIME
	clock_t starttm, endtm;

	starttm = clock();
	ElectronPDBAmplitude *amp = new ElectronPDBAmplitude(std::string(pth), false);
	ElectronPDBAmplitude *ampNoGrid = new ElectronPDBAmplitude(std::string(pth), false);
	
	ampNoGrid->SetUseGPU(false);
	ampNoGrid->SetUseGrid(true);

	amp->SetUseGPU(true);

	amp->calculateGrid(qMax, gridSize, OurProgressFunc);

	endtm = clock();

	OurProgressFunc(NULL, 1.0);

	double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
	printf("\nTook %f seconds to generate grid! (" CURGRIDNAME ")\n", timeItTook);

	std::cout << "Calculating on the CPU..." << std::endl;
	ampNoGrid->calculateGrid(qMax, gridSize, OurProgressFunc);

	// Reset GPU
//	ResetGPU();
	bool pass = CompareModelResults(amp, ampNoGrid);

/*
	std::cout << "Comparing amplitude with and without grid..." << std::endl;
	long long maxIters = 1000000;

	std::mt19937 rng1,rng2;
	rng1.seed(static_cast<unsigned int>(std::time(0)));
	rng2.seed(static_cast<unsigned int>(2*std::time(0)));

	std::uniform_real_distribution<double> qRan(0.0, qMax);
	std::uniform_real_distribution<double> u2(0.0, 2.0);

	std::complex<double> sum1(0.0,0.0), sum2(0.0,0.0);
	double wq = 0.0, wth = 0.0, wph = 0.0;
	double wlq = 0.0, wlth = 0.0, wlph = 0.0;
	double maxDiff = 0.0, maxLinDiff = 0.0, meanDiff = 0.0;
	for(long long i = 0; i < maxIters; i++) {
		double q  = qRan(rng1);
		double th = acos(u2(rng2) - 1.0);
		double ph = u2(rng2) * M_PI;

		double st = sin(th);

		double x = q * st * cos(ph);
		double y = q * st * sin(ph);
		double z = q * cos(th);

		std::complex<double> dir = ampNoGrid->getAmplitude(x,y,z);
		std::complex<double> grd = amp->getAmplitude(x,y,z);

		std::complex<double> diff = (dir - grd);
		double nDiff = std::abs(diff);

		meanDiff += nDiff;

		sum1 += dir;
		sum2 += grd;

		if(maxLinDiff < nDiff) {
			maxLinDiff = nDiff;
			wlq = x; wlth = y; wlph = z;
		}
		double lDif = fabs(1.0 - std::abs(grd) / std::abs(dir));
		if(maxDiff < lDif && nDiff > 0.0001) {
			maxDiff = lDif;
			wq = x; wth = y; wph = z;
		}
	}

	std::cout << "Mean error: " << meanDiff / double(maxIters) << "\n";
	std::cout << "("<<wq <<", "<<wth <<", "<<wph <<"): "<< "Max (1-a/b) error: " << maxDiff 
		<< "\t-log: " << -log10(maxDiff);// <<
	std::cout << ampNoGrid->getAmplitude(wq,wth,wph) << " vs. " << amp->getAmplitude(wq,wth,wph)
		<< "\n";
	std::cout << "("<<wlq<<", "<<wlth<<", "<<wlph<<"): "<< "Max linear (a-b) error: " << (maxLinDiff) 
		<< ampNoGrid->getAmplitude(wlq,wlth,wlph) << " vs. " << amp->getAmplitude(wlq,wlth,wlph)
		<< "\n";
	std::cout << "Sum1: " << sum1 << "\tSum2: " << sum2 << "\n";
*/
	delete amp;
	delete ampNoGrid;

	return pass;
}

void tests::Playground() {
	std::string pdbf("s:\\Basic Dimer\\1JFF_Aligned outer in positive X direction.pdb");
//	pdbf  = "C:\\xrff\\branches\\XplusD\\CF4.pdb";
//	pdbf  = "s:\\c60.pdb";
// 	pdbf  = "s:\\Carbon copy.pdb";
// 	pdbf  = "s:\\Carbon.pdb";
	
	ElectronPDBAmplitude *ampNoGrid = new ElectronPDBAmplitude(pdbf, false);

	ampNoGrid->SetUseGPU(false);
	ampNoGrid->SetUseGrid(false);
	Eigen::Vector3d qV(-0.302930, 0.215716, 1.266531);
	std::cout << "CPU: I(0,0,0) = "<< ampNoGrid->getAmplitude(0.,0.,0.) << std::endl;
	std::complex<double> res1 = ampNoGrid->getAmplitude(qV[0], qV[1], qV[2]);
	std::cout << "CPU: I(" << qV.transpose() << ") = "<< res1 << std::endl;
	
	

	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	Amplitude *amp = new ElectronPDBAmplitude(pdbf, false);
	amp->SetUseGrid(true);

	ISymmetry *isymm = (ISymmetry *)getModel(26);
	Symmetry *symm = dynamic_cast<Symmetry *>(isymm);
	symm->AddSubAmplitude(amp);

	DomainModel *dom = new DomainModel();
	int stopper = 0;
	dom->AddSubAmplitude(symm);
	double qmax = 6.;
	int q_points = 800;
	double grds =  100;
	double iters = 1024.* 8 * 8;
	double eps = 0.001;

	std::vector<double> q (q_points);
	printf("qmax %f\n", qmax);
	for(int i = 0; i < q_points; i++)
		q[i] = ((double)i / (double)q_points) * qmax;

	VectorXd p(50);
	p[0]	= 1.0000;	p[1]	= 16.000;	p[2]	= 0.00000;	p[3]	= 0.00000;	p[4]	= 0.00000;
	p[5]	= 0.00000;	p[6]	= 0.00000;	p[7]	= 0.00000;	p[8]	= 1.0000;	p[9]	= 1.0000;
	p[10]	= iters;	p[11]	= grds;		p[12]	= 1.0000;	p[13]	= eps;		p[14]	= qmax;
	p[15]	= 0.00000;	p[16]	= 1.0000;	p[17]	= 17.000;	p[18]	= 0.00000;	p[19]	= 0.00000;
	p[20]	= 0.00000;	p[21]	= 0.00000;	p[22]	= 0.00000;	p[23]	= 0.00000;	p[24]	= 0.00000;
	p[25]	= 1.0000;	p[26]	= 0.00000;	p[27]	= 0.00000;	p[28]	= 0.00000;	p[29]	= 0.00000;
	p[30]	= 0.00000;	p[31]	= 0.00000;	p[32]	= 1.0000;	p[33]	= 0.00000;	p[34]	= 0.00000;
	p[35]	= 0.00000;	p[36]	= 0.00000;	p[37]	= 0.00000;	p[38]	= 0.00000;	p[39]	= 0.00000;
	p[40]	= 1.0000;	p[41]	= 0.00000;	p[42]	= 1.0000;	p[43]	= 0.00000;	p[44]	= 0.050003;
	p[45]	= 0.14001;	p[46]	= 0.00000;	p[47]	= 0.00000;	p[48]	= 0.00000;	p[49]	= 0.00000;

	VectorXd res;

	dom->SetStop(NULL);
	//for(int i = 0; i < 100; i++)
	res = dom->CalculateVector(q, 0, p);

	std::cout << "Done." << std::endl;

	char a[256] = {0};
	sprintf(a, "c:\\delete\\result_%d.out", (int)time(NULL));
	FILE *fp = fopen(a, "wb");
	if(fp) {
		fprintf(fp, "#q\tIntensity\n");
		for(int i = 0; i < q_points; i++) 
			fprintf(fp, "%f\t%f\n", q[i], res[i]);
		fclose(fp);
	}

//	ResetGPU();

#define PRINT_AMPLITUDE(qq) \
	do {																				\
		res1 = ampNoGrid->getAmplitude(qq[0], qq[1], qq[2]);							\
		std::cout << "CPU: A(" << qV.transpose() << ") = "<< res1 << std::endl;			\
	} while(false)

	qV = Vector3d(0.03, 0.03, 0.03);
	PRINT_AMPLITUDE(qV);

	qV = Vector3d(qmax / (grds / 2.), 0.0, 0.0);
	PRINT_AMPLITUDE(qV);
	qV = Vector3d(0.0, qmax / (grds / 2.), 0.0);
	PRINT_AMPLITUDE(qV);
	qV = Vector3d( 0.0, 0.0, qmax / (grds / 2.));
	PRINT_AMPLITUDE(qV);

	qV = Vector3d(0.0, 0.0, 0.0);
	PRINT_AMPLITUDE(qV);

	qV = Vector3d(0.0, 0.0, 17.0);
	PRINT_AMPLITUDE(qV);
	qV = Vector3d(0.0, 17.0, 0.0);
	PRINT_AMPLITUDE(qV);
	qV = Vector3d(17.0, 0.0, 0.0);
	PRINT_AMPLITUDE(qV);
	qV = Vector3d(9.81495, 9.81495, 9.81495);
	PRINT_AMPLITUDE(qV);

}

void tests::Playground2() {
	HMODULE hMod = GetBackend(L"g");
	getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");

	IModel *model = getModel(2);
	if(dynamic_cast<Geometry *>(model))
		dynamic_cast<Geometry *>(model)->SetEDProfile(EDProfile());

	Amplitude *amp = new GeometricAmplitude(dynamic_cast<FFModel *>(model));
	amp->SetUseGrid(true);

	ISymmetry *isymm = (ISymmetry *)getModel(26);
	Symmetry *symm = dynamic_cast<Symmetry *>(isymm);
	symm->AddSubAmplitude(amp);

	DomainModel *dom = new DomainModel();
	int stopper = 0;
	dom->AddSubAmplitude(symm);
	double qmax = 6.0;
	int q_points = 800;
	double grds =  100.;

	std::vector<double> q (q_points);
	printf("qmax %f\n", qmax);
	for(int i = 0; i < q_points; i++)
		q[i] = ((double)i / (double)q_points) * qmax;

	VectorXd p(48);

	p[0]	= 1.;	p[1]	= 16.;	p[2]	= 0.0;	p[3]	= 0.0;	p[4]	= 0.0;	p[5]	= 0.0;
	p[6]	= 0.0;	p[7]	= 0.0;	p[8]	= 1.;	p[9]	= 1.;	p[10]	= 10000.000;
	p[11]	= grds;	p[12]	= 1.;	p[13]	= 0.001;	p[14]	= qmax;	p[15]	= 0.0;
	p[16]	= 1.;	p[17]	= 17.;	p[18]	= 0.0;	p[19]	= 0.0;	p[20]	= 0.0;	p[21]	= 0.0;
	p[22]	= 0.0;	p[23]	= 0.0;	p[24]	= 0.0;	p[25]	= 1.;	p[26]	= 0.0;	p[27]	= 0.0;
	p[28]	= 0.0;	p[29]	= 0.0;	p[30]	= 0.0;	p[31]	= 0.0;	p[32]	= 1.;	p[33]	= 0.0;
	p[34]	= 0.0;	p[35]	= 0.0;	p[36]	= 0.0;	p[37]	= 0.0;	p[38]	= 0.0;	p[39]	= 0.0;
	p[40]	= 1.;	p[41]	= 2.;	p[42]	= 0.0;	p[43]	= 1.;	p[44]	= 333.;	p[45]	= 400.;
	p[46]	= 1.;	p[47]	= 0.;


	VectorXd res;

	dom->SetStop(NULL);
	res = dom->CalculateVector(q, 0, p);

	std::cout << "Done." << std::endl;

	char a[256] = {0};
	sprintf(a, "c:\\delete\\result_%d.out", (int)time(NULL));
	FILE *fp = fopen(a, "wb");
	if(fp) {
		fprintf(fp, "#q\tIntensity\n");
		for(int i = 0; i < q_points; i++) 
			fprintf(fp, "%f\t%f\n", q[i], res[i]);
		fclose(fp);
	}

//	ResetGPU();
}

void tests::Playground3() {

	std::string h_file, g_file;
	h_file = "C:\\Delete\\1JFFHybridAmp_250.amp";
	g_file = "C:\\Delete\\1JFF 250 GRID.amp";

	h_file = "C:\\Delete\\HybridAmp_Angs.amp";
	g_file = "C:\\Delete\\AnglesGrid_250.amp";	

	h_file = "C:\\Delete\\C60_250_Hybrid.amp";
	g_file = "C:\\Delete\\C60_250_Grid.amp";	

	AmpGridAmplitude* hybridAmp = NULL;
	AmpGridAmplitude* gridAmp = NULL;

 	hybridAmp = new AmpGridAmplitude(h_file);
 	gridAmp   = new AmpGridAmplitude(g_file);
	

	bool pass = true;
	pass = CompareModelResults(hybridAmp, gridAmp);
	std::cout << "Playground3 " << (pass ? "passed" : "failed") << std::endl;

	delete hybridAmp;
	delete gridAmp;
}

void tests::ValidateGrid() {
	std::string amp_file;
	amp_file = "C:\\Delete\\1234 Pitch 250.amp";

	Amplitude *amp = new AmpGridAmplitude(amp_file);
	amp->SetUseGrid(true);

	DomainModel *dom = new DomainModel();
	int stopper = 0;
	dom->AddSubAmplitude(amp);
	double qmax = amp->GetGridStepSize() * double(amp->GetGridSize()) / 2.0;

	// Create q
	std::vector<double> q (QPOINTS);
	printf("qmax %f\n", qmax);
	for(int i = 0; i < QPOINTS; i++)
		q[i] = ((double)i / (double)QPOINTS) * qmax;

	// Create p
	VectorXd p (22);
	p[0] = 1;
	p[1] = 14;
	p[2] = 0;
	p[3] = 0;
	p[4] = 0;
	p[5] = 0;
	p[6] = 0;
	p[7] = 0;
	p[8] = 1;
	p[9] = 10000;
	p[10] = amp->GetGridSize();
	p[11] = 1;
	p[12] = 0.001;
	p[13] = qmax;
	p[14] = 0;
	p[15] = 0;
	p[16] = 0;
	p[17] = 0;
	p[18] = 0;
	p[19] = 0;
	p[20] = 0;
	p[21] = 0;

	VectorXd res;

	dom->SetStop(&stopper);

	return;

	res = dom->CalculateVector(q, 0, p);

}
