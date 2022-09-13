#include <cstdio>
#include <ctime>
#include "Backend/Amplitude.h"

#define QPOINTS 800
#define REPETITIONS 1

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


int main(int argc, char **argv) {
  clock_t starttm, endtm;

  if(argc < 2) {
    printf("USAGE: pdbgen <PDB file>\n");
    return 1;
  }

  starttm = clock();
  Amplitude *amp = new electronPDBAmplitude(argv[1], true);
  if(amp->getError() != PDB_OK) {
    printf("ERROR loading PDB\n");
    return 2;
  }
  endtm = clock();
  
  double timeItTook =  ((double)(endtm - starttm)) / CLOCKS_PER_SEC;
  printf("\nTook %f seconds to read file! (Direct)\n", timeItTook);
  
  
  DomainModel *dom = new DomainModel();
  int stopper = 0;
  dom->AddSubAmplitude(amp);
  double qmax = 5.0;//amp->GetGridStepSize() * double(amp->GetGridSize()) / 2.0;
  
  // Create q
  std::vector<double> q (QPOINTS);
  printf("qmax %f\n", qmax);
  for(int i = 0; i < QPOINTS; i++)
    q[i] = ((double)(i+1) / (double)QPOINTS) * qmax;

  int orientationIterations = 10000;
  
  // Create the parameter tree
  ParameterTree pt;

  // Domain params
  paramStruct ps;

  ps.nExtraParams = 6;
  ps.extraParams.push_back(orientationIterations); // oIters
  ps.extraParams.push_back(0);      // GridSize
  ps.extraParams.push_back(0);      // DefUseGrid
  ps.extraParams.push_back(1e-6);   // eps
  ps.extraParams.push_back(q[QPOINTS - 1]); // qmax
  //ps.extraParams.push_back(OA_DIRECT_GPU);      // orientationmethod

  pt.SetNodeParameters(ps);

  // PDB params
  paramStruct pdbps;
  pdbps.nExtraParams = 8;
  pdbps.extraParams.push_back(1); // scale
  pdbps.extraParams.push_back(333); // solvent
  pdbps.extraParams.push_back(1); // voxelstep
  pdbps.extraParams.push_back(0); // solventrad
  pdbps.extraParams.push_back(333); // outersolvent
  pdbps.extraParams.push_back(0); // fill holes
  pdbps.extraParams.push_back(0); // solvent only
  pdbps.extraParams.push_back(1); // radius type (VDW)
 

  pt.AddSubModel(1, pdbps);


  // Create the parameter vector from the tree
  VectorXd p = VectorXd::Zero(pt.ToParamVector(NULL));
  pt.ToParamVector(p.data());
  
  std::vector<double> res (QPOINTS);
  
  dom->SetStop(&stopper);
        
  printProgressBar(60, 0, REPETITIONS);
  starttm = clock();

  dom->PreCalculate(p, 0);
  
  for(int j = 0; j < REPETITIONS; j++) {
      dom->CalculateIntensityVector<double>(q, res, 1e-6, orientationIterations);
    printProgressBar(60, float(j+1), (float)REPETITIONS);
  }
  
  endtm = clock();
  // END TIME
  
  timeItTook =  ((double)(endtm - starttm) / (double)REPETITIONS) / CLOCKS_PER_SEC;
  printf("\nTook %f seconds!\n", timeItTook);
  
  char a[256] = {0};
  sprintf(a, "result_direct_%d.out", (int)time(NULL));
  FILE *fp = fopen(a, "wb");
  if(fp) {
    fprintf(fp, "#Time: %f\n", timeItTook);
    for(int i = 0; i < QPOINTS; i++) 
      fprintf(fp, "%f\t%f\n", q[i], res[i]);
    fclose(fp);
  }
  
  delete amp;
  delete dom;
  
  return 0;
}
