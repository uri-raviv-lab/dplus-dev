#include <cstdio>
#include <cstdlib>

#include "PDBGenDirect-rcosbCaching.cuh"

#include <cuda_runtime.h>

// For integration
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

#define M_PI 3.1415926535897932384626433832795

#define CHKERR(expr) do {\
    err = expr;\
    if(err != cudaSuccess) {                                            \
        printf("ERROR in file %s, line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err);\
        return false;\
    }                \
} while(0)


__constant__ float dev_a1[208];
__constant__ float dev_b1[208];
__constant__ float dev_a2[208];
__constant__ float dev_b2[208];
__constant__ float dev_a3[208];
__constant__ float dev_b3[208];
__constant__ float dev_a4[208];
__constant__ float dev_b4[208];
__constant__ float dev_c [208];

GPUDirectCalculator::GPUDirectCalculator() : d_angles(NULL), d_qPoints(NULL),
			_numAngles(0), _numQ(0), _anglePitch(0), d_atomLocs(NULL),
			d_affs(NULL), _numAtoms(0), _maxNumAtoms(0), h_atomsPerIon(NULL),
			_numCoeffs(0), _maxNumCoeffs(0), d_amp(NULL), d_transamp(NULL),
			d_intensity(NULL), d_intensityIndices(NULL), d_rcosb(NULL) {}

bool GPUDirectCalculator::Initialize(const float2 *angles, size_t numAngles,
									 const float *qPoints, size_t numQ,
									 size_t maxNumAtoms, size_t maxNumCoeffs)
{
    cudaError_t err = cudaSuccess;
    
    _maxNumAtoms = maxNumAtoms;
    _maxNumCoeffs = maxNumCoeffs;

    _numAngles = numAngles;
    _numQ = numQ;

    CHKERR(cudaDeviceSynchronize());
    
    ////////////////////////////////
    // Allocate memory
    
	CHKERR(cudaGetLastError());

    if(d_angles)
        cudaFree(d_angles);
    CHKERR(cudaMallocPitch(&d_angles, &_anglePitch, numAngles * sizeof(float2),
                           numQ));
    if(d_amp)
        cudaFree(d_amp);
	CHKERR(cudaMallocPitch(&d_amp, &_ampPitch, numAngles * sizeof(double2),
						   numQ));
	if(d_rcosb)
        cudaFree(d_rcosb);
	CHKERR(cudaMallocPitch(&d_rcosb, &_rcosbPitch, numAngles * sizeof(float),
						   maxNumAtoms));
	if(d_transamp)
		cudaFree(d_transamp);
	CHKERR(cudaMalloc(&d_transamp, numQ * numAngles * sizeof(double)));

    if(d_qPoints)
        cudaFree(d_qPoints);
    CHKERR(cudaMalloc(&d_qPoints, numQ * sizeof(float)));

	if(d_intensity)
		cudaFree(d_intensity);
	CHKERR(cudaMalloc(&d_intensity, numQ * sizeof(double)));

	if(d_intensityIndices)
		cudaFree(d_intensityIndices);
	CHKERR(cudaMalloc(&d_intensityIndices, numQ * sizeof(int)));

    if(d_atomLocs)
        cudaFree(d_atomLocs);
    CHKERR(cudaMalloc(&d_atomLocs, maxNumAtoms * sizeof(float4)));

    if(d_affs)
        cudaFree(d_affs);
    CHKERR(cudaMalloc(&d_affs, maxNumCoeffs * sizeof(float)));

	if(h_atomsPerIon)
        cudaFree(h_atomsPerIon);
    CHKERR(cudaMallocHost(&h_atomsPerIon, maxNumCoeffs * sizeof(unsigned int)));
	
    ////////////////////////////////////
    // Copy required memory to device

    CHKERR(cudaMemsetAsync(d_amp, 0, _ampPitch * _numQ));

    CHKERR(cudaMemcpyAsync(d_qPoints, qPoints, numQ * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHKERR(cudaMemcpy2DAsync(d_angles, _anglePitch, angles,
                             numAngles * sizeof(float2),
                             numAngles * sizeof(float2), numQ,
                             cudaMemcpyHostToDevice));
    
    return true;
}

__global__ void AtomicFormFactorKernel(const float *qPoints, unsigned int numQ,
									   float *affs, unsigned int numCoeffs)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
	if(idx >= numQ)
		return;

	float q = qPoints[idx];

	// The number is (4pi)^2         
	float sqq = (q * q / 100.0f) / (157.913670417429737901351855998f);

	for(int i = 0; i < numCoeffs; ++i)
	{
		affs[i * numQ + idx] = dev_a1[i] * expf(-dev_b1[i] * sqq) + dev_a2[i] * expf(-dev_b2[i] * sqq)
							 + dev_a3[i] * expf(-dev_b3[i] * sqq) + dev_a4[i] * expf(-dev_b4[i] * sqq) + dev_c[i];
	}
}

bool GPUDirectCalculator::SetPDB(const float4 *atomLocs,
                                 const unsigned char *ionInd,
                                 size_t numAtoms, const float *coeffs,
								 const unsigned int *atomsPerIon,
                                 size_t numCoeffs)
{
    cudaError_t err = cudaSuccess;
    
    _numAtoms = numAtoms;
    _numCoeffs = numCoeffs;

	CHKERR(cudaGetLastError());
    
    // Copy the PDB memory to the device
    CHKERR(cudaMemcpyAsync(d_atomLocs, atomLocs, numAtoms * sizeof(float4),
                           cudaMemcpyHostToDevice));


	CHKERR(cudaMemcpy(h_atomsPerIon, atomsPerIon, numCoeffs * sizeof(unsigned int),
                      cudaMemcpyHostToHost));

    if(coeffs)
    {
		size_t offset = 0;
		CHKERR(cudaMemcpyToSymbol(dev_a1, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_b1, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_a2, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_b2, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_a3, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_b3, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_a4, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_b4, coeffs + offset, numCoeffs * sizeof(float)));
		offset += numCoeffs;
		CHKERR(cudaMemcpyToSymbol(dev_c,  coeffs + offset, numCoeffs * sizeof(float)));

		// Compute atomic form factors
		dim3 dimGrid, dimBlock;
		dimBlock.x = BLOCK_WIDTH;
		dimBlock.y = 1;
		dimGrid.x = ((_numQ % BLOCK_WIDTH == 0) ? (_numQ / BLOCK_WIDTH) : 
						(_numQ / BLOCK_WIDTH + 1));
		dimGrid.y = 1;

		AtomicFormFactorKernel<<<dimGrid, dimBlock>>>(d_qPoints, _numQ, d_affs, _numCoeffs);

		CHKERR(cudaGetLastError());
    }
    
    return true;
}

bool GPUDirectCalculator::ClearAmplitude()
{
	cudaError_t err = cudaSuccess;
	CHKERR(cudaMemsetAsync(d_amp, 0, _ampPitch * _numQ));
	return true;
}

__global__ void ComputeCoefficients(const float2 *angles, const float4 *atomLocs, 
									float *rcosb, float3 translation, float3 rotation,
									unsigned int numAngles, unsigned int numAtoms,
									unsigned int rcosbPitch)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 

	if(x >= numAngles || y >= numAtoms)
		return;

	//__shared__ float4 s_atomLocs[BLOCK_HEIGHT];

	float2 thetaPhi = angles[x];
	float4 atomLoc  = atomLocs[y];

	// Spherical to cartesian coordinates
	float qx, qy, qz;

	float sinth, costh, sinphi, cosphi;
	sincosf(thetaPhi.x, &sinth, &costh);
	sincosf(thetaPhi.y, &sinphi, &cosphi);

	qx = sinth * cosphi;
	qy = sinth * sinphi;
	qz = costh;	
	// END of coordinate conversion

	float* rcosbElement = (float*)((char*)rcosb + y * rcosbPitch) + x;

	// TODO: Translate and rotate atoms

	// Computes the dot product of two atoms (|Q||R|cos b), sans the q
	*rcosbElement = qx * atomLoc.x + qy * atomLoc.y + qz * atomLoc.z;
}

__global__ void AppendAmplitudeKernel(const float2 *angles,
									  const float *gAffs,
									  unsigned char ionInd, 
									  unsigned int numAtoms, 
									  const float *qPoints,
									  const float *rcosb, unsigned int rcosbPitch,
									  unsigned int numAngles, unsigned int numQ,
									  unsigned int ampPitch, double2 *outAmp)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 

	if(x >= numAngles || y >= numQ)
		return;

	__shared__ float affs[BLOCK_HEIGHT];

	float q = qPoints[y];

	// Load shared memory
	if(threadIdx.x == 0)
	{
		affs[threadIdx.y] = gAffs[ionInd * numQ + threadIdx.y];
	}

	float sqrtsinth = sqrtf(sinf(angles[x].x));

	float2 res = make_float2(0.0f, 0.0f);
		
	float sn, cs;

	syncthreads();
	
	for(int i = 0; i < numAtoms; i++)
	{
		float* rcosbElement = (float*)((char*)rcosb + i * rcosbPitch) + x;

		sincosf(q * (*rcosbElement), &sn, &cs);

		res.x += cs;
		res.y += sn;			
	}				

	

	double2* pElement = (double2*)((char*)outAmp + y * ampPitch) + x;

	sqrtsinth *= affs[threadIdx.y];
	pElement->x += res.x * sqrtsinth;
	pElement->y += res.y * sqrtsinth;
}

bool GPUDirectCalculator::AppendAmplitude(float3 translation, float3 rotation)
{
	cudaError_t err = cudaSuccess;

	dim3 dimGrid, dimCoeffGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = BLOCK_HEIGHT;
	dimGrid.x = ((_numAngles % BLOCK_WIDTH == 0) ? (_numAngles / BLOCK_WIDTH) : 
					(_numAngles / BLOCK_WIDTH + 1));
	dimGrid.y = ((_numQ % BLOCK_HEIGHT == 0) ? (_numQ / BLOCK_HEIGHT) : 
					(_numQ / BLOCK_HEIGHT + 1));
	dimCoeffGrid.x = ((_numAngles % BLOCK_WIDTH == 0) ? (_numAngles / BLOCK_WIDTH) : 
					(_numAngles / BLOCK_WIDTH + 1));

	unsigned int blockStart = 0;
	for(int i = 0; i < _numCoeffs; i++)
	{
		if(h_atomsPerIon[i] == 0)
			continue;

		dimCoeffGrid.y = ((h_atomsPerIon[i] % BLOCK_HEIGHT == 0) ? (h_atomsPerIon[i] / BLOCK_HEIGHT) : 
						  (h_atomsPerIon[i] / BLOCK_HEIGHT + 1));

		ComputeCoefficients<<<dimCoeffGrid, dimBlock>>>(d_angles, d_atomLocs + blockStart, d_rcosb,
														translation, rotation,
														_numAngles, h_atomsPerIon[i], _rcosbPitch);

		AppendAmplitudeKernel<<<dimGrid, dimBlock>>>(d_angles, d_affs, i,
													 h_atomsPerIon[i],
													 d_qPoints,
													 d_rcosb, _rcosbPitch,
													 _numAngles, _numQ,
													 _ampPitch, d_amp);
		blockStart += h_atomsPerIon[i];
	}

	CHKERR(cudaGetLastError());

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

bool GPUDirectCalculator::ComputeIntensity(double *data)
{
	cudaError_t err = cudaSuccess;

	if(!data)
		return false;

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = BLOCK_HEIGHT;
	dimGrid.x = ((_numAngles % BLOCK_WIDTH == 0) ? (_numAngles / BLOCK_WIDTH) : 
		(_numAngles / BLOCK_WIDTH + 1));
	dimGrid.y = ((_numQ % BLOCK_HEIGHT == 0) ? (_numQ / BLOCK_HEIGHT) : 
		(_numQ / BLOCK_HEIGHT + 1));

	printf("%dx%d --> %dx%d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

	/*
	float2 *hAngles;
	cudaMallocHost(&hAngles, sizeof(float2) * _numAngles);
	CHKERR(cudaStreamSynchronize(0));
	cudaMemcpy(hAngles, d_angles, sizeof(float2) * _numAngles, cudaMemcpyDeviceToHost);
	double sumTheta = 0.0;
	for(int i = 0; i < _numAngles; i++)
		sumTheta += sin(hAngles[i].x);
	*/

    // Perform the transformation to an array of floats
	TransformAmplitude<<<dimGrid, dimBlock>>>(d_amp, 
											  //1.0 / sumTheta,
											  (M_PI / (2.0 * _numAngles)),
											  _numAngles, _numQ,
											  _ampPitch, d_transamp);
	
    CHKERR(cudaGetLastError());

	// OK

	// Perform sum of all rows (taken from sum_rows example, see above link)
	thrust::device_ptr<double> dptr_tamp (d_transamp);
	thrust::device_ptr<double> dptr_intensity (d_intensity);
	thrust::device_ptr<int> dptr_intind (d_intensityIndices);
	thrust::reduce_by_key
		(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(_numAngles)),
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(_numAngles)) + (_numQ*_numAngles),
		dptr_tamp,
		dptr_intind,
		dptr_intensity,
		thrust::equal_to<int>(),
		thrust::plus<double>());
	
	// Copy back to data
	CHKERR(cudaMemcpyAsync(data, d_intensity, _numQ * sizeof(double), cudaMemcpyDeviceToHost));

	// Wait until the data arrives
	CHKERR(cudaStreamSynchronize(0));
    return true;
}

GPUDirectCalculator::~GPUDirectCalculator() {
	// For profiling reasons
	cudaDeviceReset();
}


