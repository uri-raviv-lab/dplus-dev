#include <cstdio>
#include <cstdlib>

#include "GPUDirectCalc.cuh"
#include "CommonCUDA.cuh"
#include <cuda_runtime.h>

#include <vector_functions.h>

__host__ __device__ double2 operator+(const double2& lhs, const double2& rhs) {
    return make_double2(lhs.x + rhs.x, lhs.y + rhs.y);
}

// For integration
#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0 ) // Turn off all warnings (bypasses #pragma warning(default : X )
#endif
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif

GPUDirectCalculator::GPUDirectCalculator() {}

bool GPUDirectCalculator::Initialize(int gpuID, const float2 *angles, size_t numAngles,
									 const float *qPoints, size_t numQ,
									 size_t maxNumAtoms, size_t maxNumCoeffs,
									 size_t maxTranslations, Workspace& res)
{
    cudaError_t err = cudaSuccess;
    
	res.parent = this;

    res.numAngles = numAngles;
    res.numQ = numQ;


	CHKERR(cudaSetDevice(gpuID));

	cudaStream_t stream;

	CHKERR(cudaStreamCreate(&stream));

	res.gpuID = gpuID;
	res.stream = stream;

    ////////////////////////////////
    // Allocate memory
    
	CHKERR(cudaGetLastError());

	CHKERR(cudaMallocPitch(&res.d_angles, &res.anglePitch, numAngles * sizeof(float2),
                           numQ));
	CHKERR(cudaMallocPitch(&res.d_work, &res.workPitch, numAngles * sizeof(double2),
						   numQ));
	CHKERR(cudaMallocPitch(&res.d_amp, &res.ampPitch, numAngles * sizeof(double2),
						   numQ));
	CHKERR(cudaMalloc(&res.d_transamp, numQ * numAngles * sizeof(double)));
    CHKERR(cudaMalloc(&res.d_qPoints, numQ * sizeof(float)));
	CHKERR(cudaMalloc(&res.d_intensity, numQ * sizeof(double)));
	CHKERR(cudaMalloc(&res.d_intensityIndices, numQ * sizeof(int)));	

	res.numQ = numQ;
	res.numAngles = numAngles;

	CHKERR(cudaMalloc(&res.d_translations, maxTranslations * sizeof(float4)));
	CHKERR(cudaMallocHost(&res.h_translations, maxTranslations * sizeof(float4)));
	res.maxTranslations = maxTranslations;

    // PDB Memory
	if(maxNumAtoms > 0)
	{
		CHKERR(cudaMalloc(&res.d_atomLocs, maxNumAtoms * sizeof(float4)));
		CHKERR(cudaMalloc(&res.d_rotAtomLocs, maxNumAtoms * sizeof(float4)));
		CHKERR(cudaMalloc(&res.d_affs, numQ * maxNumCoeffs * sizeof(float)));

		CHKERR(cudaMallocHost(&res.h_atomsPerIon, maxNumCoeffs * sizeof(unsigned int)));
	}
	
	res.maxNumAtoms = maxNumAtoms;
	res.maxNumCoeffs = maxNumCoeffs;
	
    ////////////////////////////////////
    // Copy required memory to device

    CHKERR(cudaMemcpyAsync(res.d_qPoints, qPoints, numQ * sizeof(float),
                           cudaMemcpyHostToDevice, stream));
    CHKERR(cudaMemcpy2DAsync(res.d_angles, res.anglePitch, angles,
                             numAngles * sizeof(float2),
                             numAngles * sizeof(float2), numQ,
                             cudaMemcpyHostToDevice, stream));
    CHKERR(cudaMemset(res.d_amp, 0, res.ampPitch * numQ));


	// No stream support for now (no gain, no pain)
	cudaStreamSynchronize(stream);

    return true;
}

bool GPUDirectCalculator::FreeWorkspace(Workspace& res)
{
	cudaError_t err = cudaSuccess;

	CHKERR(cudaGetLastError());

	CHKERR(cudaFree(res.d_angles));
	CHKERR(cudaFree(res.d_work));
	CHKERR(cudaFree(res.d_amp));
	CHKERR(cudaFree(res.d_transamp));
    CHKERR(cudaFree(res.d_qPoints));
	CHKERR(cudaFree(res.d_intensity));
	CHKERR(cudaFree(res.d_intensityIndices));
	CHKERR(cudaFree(res.d_translations));
	CHKERR(cudaFreeHost(res.h_translations));

    // PDB Memory
	if(res.maxNumAtoms > 0)
	{
		CHKERR(cudaFree(res.d_atomLocs));
		CHKERR(cudaFree(res.d_rotAtomLocs));
		CHKERR(cudaFree(res.d_affs));

		CHKERR(cudaFreeHost(res.h_atomsPerIon));
	}

	CHKERR(cudaStreamDestroy((cudaStream_t)res.stream));

	return (err == cudaSuccess);
}


__global__ void TranslateWorkspaceKernel(const double2 *inAmp, const float2 *angles,
									  const float *qPoints,
									  const float4* trans,
									  unsigned int numAngles, unsigned int numQ,
									  unsigned int ampPitch, unsigned int numTrans,
									  double2 *outAmp)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int i;

	if(x >= numAngles || y >= numQ)
		return;

#ifdef FINISH_IMPLEMENTING_USE_OF_THE_SHARED_MEMORY_NOT_JUST_SETTING
	// The first 128 translations of the workspace
	__shared__ float4 sTrans[BLOCK_WIDTH * 4];

	// Load 4 translations each run
	if((threadIdx.x + 1) * 4 < numTrans)
	{
		sTrans[threadIdx.x * 4]     = trans[threadIdx.x * 4];
		sTrans[threadIdx.x * 4 + 1] = trans[threadIdx.x * 4 + 1];
		sTrans[threadIdx.x * 4 + 2] = trans[threadIdx.x * 4 + 2];
		sTrans[threadIdx.x * 4 + 3] = trans[threadIdx.x * 4 + 3];
	}
	else if(threadIdx.x * 4 < numTrans) // Remainder
	{
		unsigned int rem = numTrans % 4;
		for(i = 0; i < rem; i++)
			sTrans[threadIdx.x * 4 + i] = trans[threadIdx.x * 4 + i];
	}
#endif

	float q = qPoints[y];

	float2 thetaPhi = angles[x];
	

	// Spherical to cartesian coordinates
	float qx, qy, qz;

	float sinth, costh, sinphi, cosphi;
	sincosf(thetaPhi.x, &sinth, &costh);
	sincosf(thetaPhi.y, &sinphi, &cosphi);

	qx = q * sinth * cosphi;
	qy = q * sinth * sinphi;
	qz = q * costh;
	
	// END of coordinate conversion

	// Shared mem version
	unsigned int loop = min(numTrans, 128);

	double2 factor = make_double2(0.0, 0.0);
	float sn, cs, phase;

	__syncthreads();
	/*
	for(i = 0; i < loop; i++)
	{
		//if(x == 0 && y == 0)
		//	printf("DEV %d: %f, %f, %f\n", i, sTrans[i].x, sTrans[i].y, sTrans[i].z);

		phase = qx * sTrans[i].x + qy * sTrans[i].y + qz * sTrans[i].z;			
		sincosf(phase, &sn, &cs);

		factor.x += cs;
		factor.y += sn;
	}

	// The rest of the memory
	for(i = BLOCK_WIDTH * 4; i < numTrans; i++)
	*/
	for(i = 0; i < numTrans; i++)
	{
		phase = qx * trans[i].x + qy * trans[i].y + qz * trans[i].z;			
		sincosf(phase, &sn, &cs);

		factor.x += cs;
		factor.y += sn;
	}

	double2 tmp = *((double2*)((char*)inAmp + y * ampPitch) + x);
	double2 *pElement = ((double2*)((char*)outAmp + y * ampPitch) + x);
	
	// Complex multiplication: (a+bi) * (c+di) = (ac - bd) + (ad + bc)i
	pElement->x += tmp.x * factor.x - tmp.y * factor.y;
	pElement->y += tmp.x * factor.y + tmp.y * factor.x;
}

__global__ void AddToMatrix(double2 *dst, size_t dstPitch, const double2 *src, size_t srcPitch,
							size_t width, size_t height)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 

	if(x >= width || y >= height)
		return;

	double2 tmp = *((double2*)((char*)src + y * srcPitch) + x);
	double2 *pElement = ((double2*)((char*)dst + y * dstPitch) + x);
	
	*pElement = *pElement + tmp;
}

bool GPUDirectCalculator::TranslateWorkspace(Workspace& workspace, float3 *translations, unsigned int numTrans)
{
	cudaError_t err = cudaSuccess;

	if(!translations || numTrans == 0)
		return false;

	// Set device
	CHKERR(cudaSetDevice(workspace.gpuID));

	// Translate from d_work to d_amp
	//printf("WTF %d\n", numTrans);

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = BLOCK_HEIGHT;
	dimGrid.x = ComputeGridSize(workspace.numAngles, BLOCK_WIDTH);
	dimGrid.y = ComputeGridSize(workspace.numQ, BLOCK_HEIGHT);

	// Optimize for the special case of 1 translation of 0,0,0
	if(numTrans == 1 && fabs(translations[0].x) < 1e-10 && fabs(translations[0].y) < 1e-10 && fabs(translations[0].z) < 1e-10)
	{
		TRACE_KERNEL("AddToMatrix");
		AddToMatrix<<<dimGrid, dimBlock, 0, (cudaStream_t)workspace.stream>>>
			(workspace.d_amp, workspace.ampPitch, workspace.d_work, workspace.workPitch,
			 workspace.numAngles, workspace.numQ);
		
		return true;
	}

	// Convert float3 to float4
	for(int i = 0; i < numTrans; i++)
		workspace.h_translations[i] = make_float4(translations[i].x, translations[i].y, translations[i].z, 0.0f);
	CHKERR(cudaMemcpyAsync(workspace.d_translations, workspace.h_translations, 
						   numTrans * sizeof(float4), cudaMemcpyHostToDevice, (cudaStream_t)workspace.stream));
	
	TRACE_KERNEL("TranslateWorkSpaceKernel");
	TranslateWorkspaceKernel<<<dimGrid, dimBlock, 0, 
							   (cudaStream_t)workspace.stream>>>(
							workspace.d_work, workspace.d_angles,
							workspace.d_qPoints,
							workspace.d_translations,
							workspace.numAngles, workspace.numQ,
							workspace.ampPitch, numTrans,
							workspace.d_amp);
	return true;
}

// Transforms amplitude (F$) to (|F|^2 * sin(theta)) for orientation
// average
// NOTE that the data already arrives as F$ * sqrt(sin(theta))
__global__ void TransformAmplitude(const double2 *amps, double factor, 
								   unsigned int numAngles, unsigned int numQ,
								   unsigned int ampPitch, double *res)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 

	if(x >= numAngles || y >= numQ)
		return;

	double2 elem = *((double2*)((char*)amps + y * ampPitch) + x);	

	// |F|^2 * sin(theta) / 2pi^2
	res[y * numAngles + x] = ((elem.x * elem.x + elem.y * elem.y)) * factor;
}

// convert a linear index to a row index
// TAKEN FROM the sum_rows example of Thrust:
// https://github.com/thrust/thrust/blob/master/examples/sum_rows.cu
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
	T C; // number of columns

	__host__ __device__
		linear_index_to_row_index(T C) : C(C) {}

	__host__ __device__
		T operator()(T i)
	{
		return i / C;
	}
};

bool GPUDirectCalculator::ComputeIntensity(Workspace *workspaces, unsigned int numWorkspaces, double *data)
{
	cudaError_t err = cudaSuccess;

	if(numWorkspaces == 0)
		return true;

	if(!workspaces || !data)
		return false;

	Workspace &master = workspaces[0];
	int computeGPU = master.gpuID;
	size_t numAngles = master.numAngles, numQ = master.numQ, ampPitch = master.ampPitch;
	cudaStream_t masterStream = (cudaStream_t)master.stream;	

	// First, sync
	CHKERR(cudaSetDevice(computeGPU));
	cudaStreamSynchronize(masterStream);

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = BLOCK_HEIGHT;
	dimGrid.x = ((numAngles % BLOCK_WIDTH == 0) ? (numAngles / BLOCK_WIDTH) : 
		(numAngles / BLOCK_WIDTH + 1));
	dimGrid.y = ((numQ % BLOCK_HEIGHT == 0) ? (numQ / BLOCK_HEIGHT) : 
		(numQ / BLOCK_HEIGHT + 1));

	// Sum everything to the first workspace and run on first workspace
	for(size_t p = 1; p < numWorkspaces; p++)
	{
		Workspace &work = workspaces[p];
		
		// Cannot run on a stream due to thrust
		cudaStreamSynchronize((cudaStream_t)work.stream);

		// Assuming same pitch for all devices/streams
		cudaMemcpyPeer(master.d_work, computeGPU, work.d_amp, work.gpuID, master.ampPitch * numQ);
		
		TRACE_KERNEL("AddToMatrix");
		AddToMatrix<<<dimGrid, dimBlock>>>
			(master.d_amp, master.ampPitch, master.d_work, master.workPitch,
			 master.numAngles, master.numQ);

	}
	
    // Perform the transformation to an array of floats
	TRACE_KERNEL("TransformAmplitude");
	TransformAmplitude<<<dimGrid, dimBlock>>>(master.d_amp, 
											  //1.0 / sumTheta,
											  //(M_PI / (2.0 * numAngles)),
											  1.0 / numAngles,
											  numAngles, numQ,
											  ampPitch, master.d_transamp);
	
    CHKERR(cudaGetLastError());

	// OK

	// Perform sum of all rows (taken from sum_rows example, see above link)
#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0) // Disabling 4305 (name/decoration too long)
#endif
	thrust::device_ptr<double> dptr_tamp (master.d_transamp);
	thrust::device_ptr<double> dptr_intensity (master.d_intensity);
	thrust::device_ptr<int> dptr_intind (master.d_intensityIndices);
	thrust::reduce_by_key
		(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(numAngles)),
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(numAngles)) + (numQ*numAngles),
		dptr_tamp,
		dptr_intind,
		dptr_intensity,
		thrust::equal_to<int>(),
		thrust::plus<double>());
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif

	// Copy back to data
	CHKERR(cudaMemcpy(data, master.d_intensity, numQ * sizeof(double), cudaMemcpyDeviceToHost));
	
    return true;
}

GPUDirectCalculator::~GPUDirectCalculator() {
	// For profiling reasons
	cudaDeviceReset();
}


