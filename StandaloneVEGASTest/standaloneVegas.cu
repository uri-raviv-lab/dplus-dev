
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <assert.h>
#include <stdio.h>

#include "CommonCUDA.cuh"
#include "../Backend/GPU/HybridOA.cu"

#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/random_device.hpp"
#include "boost/random/mersenne_twister.hpp"

#include <time.h>

template <typename T>
inline T sq(T x) {return x*x;}

double f(double x, double sig) {
	return (1. / (1.77245 * sig)) * exp(-sq(x - 0.5) / sq(sig)); 
}

double f2(double x, double y, double sig) {
	return (1. / (1.77245 * sig)) * exp(-(sq(x - 0.5) + sq(y - 0.5)) / sq(sig)); 
}

int main()
{

	boost::random::mt19937 rng;
	rng.seed(std::time(0));
	boost::random::uniform_real_distribution<double> rnd(0.0, 1.0);

    cudaError_t err;	

	//////////////////////////////////////////////////////////////
	// 2D Gaussian check
	//////////////////////////////////////////////////////////////
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
#define AXIS_LENGTH 50
#define NUM_PER_BIN 3
		std::vector<double> xAxis(AXIS_LENGTH+1), yAxis(AXIS_LENGTH+1);
		for(int i = 0; i <= AXIS_LENGTH; i++)
			xAxis[i] = yAxis[i] = double(i)/AXIS_LENGTH;

		double evals[AXIS_LENGTH*AXIS_LENGTH*NUM_PER_BIN] = {};
		int nEvals = AXIS_LENGTH*AXIS_LENGTH*NUM_PER_BIN;

		int ind = 0;
		for(int i = 0; i < AXIS_LENGTH; i++)
		{
			for(int j = 0; j < AXIS_LENGTH; j++)
			{
				for(int k = 0; k < NUM_PER_BIN; k++)
				{
					evals[ind++] = f2(rnd(rng) * (xAxis[i+1] - xAxis[i]) + xAxis[i],
										rnd(rng) * (yAxis[j+1] - yAxis[j]) + yAxis[j],
										0.1);
				} // for k
			} // for j
		} // for i

		double *d_zVals, *d_yVarMult, *d_xIn;
		CHKERR(cudaMalloc(&d_zVals, sizeof(double) * nEvals));
		CHKERR(cudaMalloc(&d_yVarMult, sizeof(double) * AXIS_LENGTH*AXIS_LENGTH) );
		CHKERR(cudaMemset(d_yVarMult, 0, sizeof(double) * (AXIS_LENGTH*AXIS_LENGTH) ) );
		CHKERR(cudaMemcpy(d_zVals, evals, sizeof(double) * nEvals, cudaMemcpyHostToDevice));
		


		printf("Reducing to variance...\n");
		cudaEventRecord(start);
		HybridVEGASBinReduceKernel<double, AXIS_LENGTH*AXIS_LENGTH, AXIS_LENGTH, AXIS_LENGTH, NUM_PER_BIN >
			<<< AXIS_LENGTH, AXIS_LENGTH >>> 
					(d_zVals, 1, d_yVarMult);
		CHKERR(cudaPeekAtLastError());
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		CHKERR(cudaDeviceSynchronize());
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("Timing: %fms\n", milliseconds);

		std::vector<double> bothAxes;
		bothAxes = xAxis;
		bothAxes.insert(bothAxes.end(), yAxis.begin(), yAxis.end());

		CHKERR(cudaMalloc(&d_xIn, sizeof(double) * bothAxes.size() ) );

		CHKERR(cudaMemcpy(d_xIn, bothAxes.data(), sizeof(double) * bothAxes.size(), cudaMemcpyHostToDevice));

		printf("Resizing grid...\n");
		HybridVEGASResizeGridKernel<double, AXIS_LENGTH, AXIS_LENGTH><<<1, AXIS_LENGTH+AXIS_LENGTH>>>(d_xIn, d_yVarMult);
		CHKERR(cudaPeekAtLastError());

		CHKERR(cudaDeviceSynchronize());

		PrintFloatDeviceMemory<<<1, AXIS_LENGTH*2+2>>>(d_xIn, AXIS_LENGTH*2+2);
		CHKERR(cudaDeviceSynchronize());

		err = cudaDeviceReset();
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		//return 0;

	}

	double h_y[8*2] = {3.19052e-10, 0.000147302, 0.0023471, 0.0954126, 0.338462,
		0.0140886, 4.86118e-6, 5.10224e-8, 2.74999e-8, 0.000207896, 0.00169844,
		0.339626, 0.352198, 0.00966761, 1.1233e-6, 1.39751e-9};

	double h_yTr[2*8] = {3.19052e-10, 2.74999e-8, 0.000147302, 0.000207896, 0.0023471, 
		0.00169844, 0.0954126, 0.339626, 0.338462, 0.352198, 0.0140886, 
		0.00966761, 4.86118e-6, 1.1233e-6, 5.10224e-8, 1.39751e-9};

	double h_yMult[4*6*3] = {1.66637e-6, 1.43918e-8, 6.81462e-10,  8.87249e-11,
		0.000614743,  1.48262e-6, 0.000292074, 0.00024081, 0.761679, 0.50192, 
		0.644401, 0.188934, 0.538838, 0.502899, 0.552224, 0.237454, 0.000216681, 
		0.00120336, 0.000173382, 0.0018148, 1.04224e-6, 2.77592e-8,  8.63271e-10, 
		2.12739e-7, 1.77424e-7, 5.1547e-9,  2.69473e-7,  1.13499e-7, 0.00248996, 
		0.160138, 0.00132338, 0.00194593, 0.444898, 1.43095, 2.60759, 3.11565, 
		1.63221, 2.31581, 0.119756, 0.686863, 0.00212357, 0.00128154, 0.0336825, 
		0.00659889,  1.50496e-9, 0.0000128149, 4.98657e-8,  1.24493e-9, 0.0000205698,
		5.67445e-9, 3.45591e-8,  4.634e-6, 0.00192677, 0.0432597, 0.000617312, 
		0.000122203, 3.9202, 2.45311, 1.0151, 1.49228, 0.819577, 3.93284, 3.89103,
		4.41455, 0.0621697, 0.00453033, 0.0253788, 0.000342527, 0.0000101814,
		2.37963e-10, 1.33344e-9, 1.17271e-6};
	double h_xMult[6+1+4+1] = {0., 0.166667, 0.333333, 0.5, 0.666667, 0.833333, 1., 0., 0.25, 0.5, 0.75, 1.};
	double h_y2DVarExpected[3*6] = {1.38007e-12, 1.27397e-7, 0.122496, 0.0440462,
		1.27639e-6,  4.80258e-13,  2.46945e-14, 0.0125169, 2.87721, 1.9077, 0.000471388,
		8.18867e-11,  1.90184e-10, 0.000898809, 3.28425, 5.42632, 0.00159625,  4.85322e-11};

	static const double h_x[9] = {0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.};
	static const double h_yVar[8] = {7.38798e-16,3.67163e-9,4.20757e-7,0.0596401,0.00018868,0.0000195454,1.39718e-11,2.46263e-15};
	static const double h_desiredRes[9] = {0,0.29213,0.343625,0.395081,0.446477,0.497872,0.549258,0.600644,0};
	double gpuRes[9];

	double *d_xIn, *d_xOut, *d_yVariance, *d_yMult, *d_xMult, *d_yVarMult;
	
	CHKERR(cudaMalloc(&d_xIn, sizeof(double) * (9*2*2) ) );
	CHKERR(cudaMalloc(&d_xOut, sizeof(double) * (9*2) ) );
	CHKERR(cudaMalloc(&d_yVariance, sizeof(double) * 64*(8*2) ) );

	CHKERR(cudaMalloc(&d_xMult, sizeof(double) * (6+1+4+1) ) );
	CHKERR(cudaMalloc(&d_yMult, sizeof(double) * (4*6*3) ) );

	CHKERR(cudaMalloc(&d_yVarMult, sizeof(double) * (6*3) ) );

	//////////////////////////////////////////////////////////////
	// Check the Bin reduction to variance of a constant
	//////////////////////////////////////////////////////////////
	printf("Checking bin reduction to variance of a constant\n");
	CHKERR(cudaMemset(d_yVarMult, 0, sizeof(double) * (6*3) ) );
	std::vector<double> h_yconst(18, 4.430465416048259e6);
		
	CHKERR(cudaMemcpy(d_yMult, h_yconst.data(), sizeof(double) * (18) , cudaMemcpyHostToDevice));
	HybridVEGASBinReduceKernel<double, 18, 6, 1, 3 > <<< 1, 6*1 >>> 
				(d_yMult, 1, d_yVarMult);
	cudaDeviceSynchronize();

	std::vector<double> h_y2DVar(18);
	CHKERR(cudaMemcpy(h_y2DVar.data(), d_yVarMult, sizeof(double) * 6*3, cudaMemcpyDeviceToHost));

	printf("\t1D:\n");
	for(int i = 0; i < 6; i++)
		printf("[%d] %g\t\%g\n", i, h_y2DVar[i], 0.);
	CHKERR(cudaMemset(d_yVarMult, 0, sizeof(double) * (6*3) ) );

	HybridVEGASBinReduceKernel<double, 18, 2, 3, 3 > <<< 1, 6*1 >>> 
				(d_yMult, 1, d_yVarMult);
	cudaDeviceSynchronize();

	CHKERR(cudaMemcpy(h_y2DVar.data(), d_yVarMult, sizeof(double) * 6*3, cudaMemcpyDeviceToHost));

	printf("\t2D:\n");
	for(int i = 0; i < 6; i++)
		printf("[%d] %g\t\%g\n", i, h_y2DVar[i], 0.);

	// This will probably always fail, unless I think of something else:
	
	CHKERR(cudaMemcpy(d_xIn, h_x, sizeof(double) * 9, cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(d_yVariance, h_y2DVar.data(), sizeof(double) * 9, cudaMemcpyHostToDevice));

	HybridVEGASResizeGridKernel<double, 8, 1><<<1, 8>>>(d_xIn, d_yVariance);

	cudaDeviceSynchronize();

	CHKERR(cudaMemcpy(gpuRes, d_xIn, sizeof(double) * 9, cudaMemcpyDeviceToHost));

	printf("\tResized grid from a constant:\n");
	for(int i = 0; i < 9; i++)
		printf("[%d] %f\t%f\n", i, gpuRes[i], 0.);


	//////////////////////////////////////////////////////////////
	// Check the 2D grid resizing
	//////////////////////////////////////////////////////////////

	double h_cylVar[24] = {565467469891252.37, 5834516102981999., 23807915887364976.,
		66727050988628440.0, 183902348473291900.000, 410721967609943100.00,
		682303537778947070.0, 1433094173247223000.000, 34553410168380704.00,
		13997065430057304.0, 75969008405339184.000, 202196205200423550.00,
		281014536279643360.0, 290646621493995010.000, 49140538643511712.00,
		43641525299078736.0, 114613440798441600.000, 361241152949682500.00,
		217855306460563940.0, 244973761551346620.000, 32283153775785228.00,
		105067449836324030.0, 314301463747559810.000, 425462337518138430.000};

	std::vector<double> h_th8ph16(26);
	for(int i = 0; i < 9; i++)
		h_th8ph16[i] = double(i) / 8.;
	for(int i = 0; i < 17; i++)
		h_th8ph16[9+i] = double(i) / 16.;

	CHKERR(cudaMemcpy(d_yMult, h_th8ph16.data(), sizeof(double) * (h_th8ph16.size()) , cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(d_yVariance, h_cylVar, sizeof(double) * 24, cudaMemcpyHostToDevice));

	HybridVEGASResizeGridKernel<double, 8, 16><<<1, 8+16>>>(/*d_xIn*/d_yMult, d_yVariance);

	cudaDeviceSynchronize();

	CHKERR(cudaMemcpy(h_th8ph16.data(), d_yMult, sizeof(double) * 26, cudaMemcpyDeviceToHost));
	
	printf("\tResized grid from cylinder variance:\n");

	for(int i = 0; i < 26; i++)
		printf("[%d] %f\n", i, h_th8ph16[i]);


	//////////////////////////////////////////////////////////////
	// Check the 2D Bin reduction to variance
	//////////////////////////////////////////////////////////////
	printf("Checking 2D bin reduction to variance\n");
	CHKERR(cudaMemcpy(d_xMult, h_xMult, sizeof(double) * (6+1+4+1), cudaMemcpyHostToDevice));
	CHKERR(cudaMemset(d_yVarMult, 0, sizeof(double) * (6*3) ) );
	CHKERR(cudaMemcpy(d_yMult, h_yMult, sizeof(double) * (4*6*3) , cudaMemcpyHostToDevice));
	HybridVEGASBinReduceKernel<double, 4*6*3, 6, 3, 4 > <<< 1, 6*3 >>> 
				(d_yMult, 1, d_yVarMult);
	cudaDeviceSynchronize();

	h_y2DVar.resize(3*6);

	CHKERR(cudaMemcpy(h_y2DVar.data(), d_yVarMult, sizeof(double) * 6*3, cudaMemcpyDeviceToHost));

	for(int i = 0; i < 6*3; i++)
		printf("[%d] %g\t\%g\n", i, h_y2DVar[i], h_y2DVarExpected[i]);



	//////////////////////////////////////////////////////////////
	// Check the Bin reduction to variance
	//////////////////////////////////////////////////////////////
	printf("Checking bin reduction to variance\n");
	CHKERR(cudaMemcpy(d_xIn, h_y, sizeof(double) * 16, cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(d_xIn+16, h_y, sizeof(double) * 16, cudaMemcpyHostToDevice));
	HybridVEGASBinReduceKernel<double, 2*2*8, 8, 2, 2 > <<< 1, 8*2 >>> 
				(d_xIn, 1, d_yVariance);
	cudaDeviceSynchronize();


	std::vector<double> h_yVarVec(2*8);
	CHKERR(cudaMemcpy(h_yVarVec.data(), d_yVariance, sizeof(double) * 8*2, cudaMemcpyDeviceToHost));

	for(int i = 0; i < 8; i++)
		printf("[%d] %g\t\%g\t%g\n", i, h_yVarVec[i], h_yVar[i], h_yVarVec[i+8]);



	//////////////////////////////////////////////////////////////
	// Check the grid resizing
	//////////////////////////////////////////////////////////////
	printf("Checking grid resizing\n");

	CHKERR(cudaMemcpy(d_xIn, h_x, sizeof(double) * 9, cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(d_yVariance, h_yVar, sizeof(double) * 9, cudaMemcpyHostToDevice));

	HybridVEGASResizeGridKernel<double, 8, 4><<<1, 8>>>(d_xIn, d_yVariance);

	cudaDeviceSynchronize();

	CHKERR(cudaMemcpy(gpuRes, d_xIn, sizeof(double) * 9, cudaMemcpyDeviceToHost));

	for(int i = 0; i < 9; i++)
		printf("[%d] %f\t%f\n", i, gpuRes[i], h_desiredRes[i]);

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

