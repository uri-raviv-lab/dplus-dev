
#include "CalculateJacobianSplines.cuh"

namespace CudaJacobianSplines {
	__device__ __constant__ double _GCPRIME_ [16];

	template int GPUCalculateSplinesJacobSphrOnCardTempl(int, int, int, 
		double2 *,  double2 *, const cudaStream_t&, const cudaStream_t&);
	/* Disable single precision
	template int GPUCalculateSplinesJacobSphrOnCardTempl(int, int, int, 
		float2 *,  float2 *, const cudaStream_t&, const cudaStream_t&);

	template int GPUCalculateSplinesJacobSphrOnCardTempl(int, int, int, 
		double2 *,  float2*, const cudaStream_t&, const cudaStream_t&);
	*/

	template <typename indType, typename divType, typename outType>
	void __device__ __host__ __forceinline__ IndicesForSpline(
		indType m, divType thetaDivs, outType *n, outType *t)
	{
		if(m == 0) {
			*n = 0;
			*t = 0;
			return;
		}
		double frc = (sqrt(4. + 4 * thetaDivs + 8 * (m-1) * thetaDivs + thetaDivs * thetaDivs) - thetaDivs - 2) / (2 * thetaDivs) + 0.000001;
		*n = outType(frc);
		*t = outType( m - ( *n*(2 + (1 + *n)*thetaDivs) ) / 2 ) - 1;
		(*n)++;
	}

	///////////////////////////////////////////////////////////////////////////////
	// Complex type operation overloads
	template <typename REAL, typename COMPLEX>
	COMPLEX __device__ __host__ __forceinline__ operator/(const COMPLEX c, const REAL r) {
		COMPLEX res;
		res.x = c.x / r;
		res.y = c.y / r;
		return res;
	}

	template <typename REAL, typename COMPLEX>
	COMPLEX __device__ __host__ __forceinline__ operator*(const REAL r, const COMPLEX c) {
		COMPLEX res;
		res.x = c.x * r;
		res.y = c.y * r;
		return res;
	}

	template <typename COMPLEX>
	COMPLEX __device__ __host__ __forceinline__ operator+(const COMPLEX l, const COMPLEX r) {
		COMPLEX res;
		res.x = l.x + r.x;
		res.y = l.y + r.y;
		return res;
	}

	template <typename COMPLEX1, typename COMPLEX2>
	COMPLEX1 __device__ __host__ __forceinline__ operator-(const COMPLEX1 l, const COMPLEX2 r) {
		COMPLEX1 res;
		res.x = l.x - r.x;
		res.y = l.y - r.y;
		return res;
	}

	template <typename T1, typename T2>
	void __device__ __host__ __forceinline__ ass(T1& to, const T2 from) {
		to.x = from.x;
		to.y = from.y;
	}

/*
	__host__ __device__ __forceinline__ operator float2() const {
		double2 res;
		res.x = r.x;
		res.y = r.y;
		return res;
	}
*/
/*
	struct __device_builtin__ __builtin_align__(16) double2
	{
		double x, y;
		__host__ __device__ __forceinline__ const double2& operator=(const float2 f) {
			x = f.x;
			y = f.y;
			return *this;
		}
	};
*/
//__host__ __device__ inline const float2& float2::operator=(const double2 a)  { x = (float)a.x;   y = (float)a.y;     return *this; }
/*
	struct double2 {
		__host__ __device__ __forceinline__ const double2& operator=(const float2 f) {
			x = f.x;
			y = f.y;
			return *this;
		}
	}
*/

	// Both templates must be either Complex or Real, not a mix
	template <typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
	void __global__ CalculateSplinesJacobian(const int offset, 
											const int thetaDivs, const int phiDivs,
											DCF_FLOAT_TYPE *ds, RES_FLOAT_TYPE *inAmp,
											RES_FLOAT_TYPE *d, RES_FLOAT_TYPE *dPrime,
											int dBlockSize,
											long long totalPoints)
	{
		typedef typename std::conditional<sizeof(RES_FLOAT_TYPE) == 2*sizeof(double), double, float>::type fType;
		typedef typename std::conditional<sizeof(DCF_FLOAT_TYPE) == 2*sizeof(double), double, float>::type cofType;

		int scratchPos = (blockIdx.x * blockDim.x + threadIdx.x);
		long long id = scratchPos + offset;
		scratchPos *= dBlockSize;
		// Figure out what the parameters are going to be, or if we're going to determine it all in the function
	#ifndef ADDITIONAL_ALLOC
		if(id >= totalPoints)
			return;
	#endif
        
		int layer, thetaInd;
		IndicesForSpline(id, thetaDivs, &layer, &thetaInd);
		int matSize = phiDivs * layer;
		long long pos = IndexFromIndices(layer, thetaInd, 0, thetaDivs, phiDivs);

		for(int i = 0; i < matSize; i++) {
			dPrime[scratchPos + i] = fType(3.0) * (inAmp[pos + (i+1)%matSize] - inAmp[pos + (matSize+i-1)%matSize]);
		}

		const fType sigma = 1.0717967697244908259 / (3.7128129211020366964 * (1. - pow(-0.26794919243112270647, matSize)));
		RES_FLOAT_TYPE x0 = sigma * (1.0 + pow(-0.26794919243112270647, matSize)) * dPrime[scratchPos];
		int m = (matSize+1) / 2;

		for(int i = 1; i < m; i++) {
			x0 = x0 + sigma * (pow(-0.26794919243112270647, i) + pow(-0.26794919243112270647, matSize-(i))) * (dPrime[scratchPos + i] + dPrime[scratchPos + matSize-i]);
		}
		if(matSize % 2 == 0) {
			x0 = x0 + sigma * (pow(-0.26794919243112270647, m) + pow(-0.26794919243112270647, matSize-m)) * dPrime[scratchPos + m];
		}
		
		dPrime[scratchPos + 1] = dPrime[scratchPos + 1] - x0;
		dPrime[scratchPos + matSize - 1] = dPrime[scratchPos + matSize - 1] - x0;

		// Copy data so that it isn't written over
		for(int i = scratchPos; i < scratchPos + matSize; i++)
			d[i] = dPrime[i];

		ass(ds[pos], x0);

		// Tridiagonal matrix algorithm, the c' are precalculated
		dPrime[scratchPos + 1] = dPrime[scratchPos + 1] / 4.0;
		for(int i = 2; i < matSize; i++) {
			dPrime[scratchPos + i] = (d[scratchPos + i] - dPrime[scratchPos + i-1]) / 
				fType(4. - _GCPRIME_[min(i-2, 15)]);
		}

		// Back substitution for the coefficients
		ass(ds[pos + matSize-1], dPrime[scratchPos + matSize-1]);
		for(int i = matSize - 2; i > 0; i--) {
			ass(ds[pos + i],
				(dPrime[scratchPos + i] - (cofType(_GCPRIME_[min(i-1, 15)]) * ds[pos + i+1])));
		}

	}	// End kernel

	template <typename divType>
	bool testIndices(divType thetaDivs) {
		bool pass = true;
		int maxPoint = thetaDivs * 300 + 1;
		int i = 0;
		int testr, testt;

		for(int r = 1; r < 300; r++) {
			for(int t = 0; t < r * thetaDivs + 1; t++) {
				i++;
				IndicesForSpline(i, thetaDivs, &testr, &testt);
				if(testr != r || testt != t) {
					pass = false;
					printf("%d: (%d,%d) vs (%d, %d)\n", i, testr, testt, r, t);
				}
			}
		}
		if(pass) {
			printf("IndicesForSpline works\n");
		}
		return pass;
	}

	
	// A template for calculating the cubic splines of a Jacobian Grid on a CUDA device.
	// The parameter pointers are device pointers that are preallocated by the calling
	// function.
	template <typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
	int GPUCalculateSplinesJacobSphrOnCardTempl(int thDivs, int phDivs, int outerLayerIndex,
									RES_FLOAT_TYPE *dInAmpData,  DCF_FLOAT_TYPE *dOutD, const cudaStream_t& memStream, const cudaStream_t& comStream)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
	
		// TODO: Confirm I didn't write crap here.
		const int splineDivs = 1024;
		int numSplines = (outerLayerIndex * (2 + thDivs*(1+outerLayerIndex) ) ) / 2;
		int numKernelLaunches = max(numSplines / splineDivs, 1);
		const int maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
		const int N = ((numSplines/numKernelLaunches) / maxThreadsPerBlock);
		if(numKernelLaunches * splineDivs != numSplines)
			numKernelLaunches++;

		dim3 grid(N, 1, 1);
		dim3 threads(maxThreadsPerBlock, 1, 1);

		cudaError err = cudaSuccess;

		const double cPrime[16] = {0.25000000000000000000,0.266666666666666666666,0.26785714285714285714,0.26794258373205741627,0.26794871794871794872,0.26794915836482308485,0.26794918998527245950,0.26794919225551855963,0.26794919241851489598,0.26794919243021750641,0.26794919243105771603,0.26794919243111804037,0.26794919243112237146,0.26794919243112268242,0.26794919243112270475,0.26794919243112270635};
		CHKERR(cudaMemcpyToSymbol(_GCPRIME_, cPrime, 16 * sizeof(double)));
		// For some reason, trying this Async causes:
		// Unhandled exception at 0x1103e34e in DPlus.exe: 0xC0000005: Access violation reading location 0x00000003.
		//CHKERR(cudaMemcpyToSymbolAsync(_GCPRIME_, cPrime, 16 * sizeof(double), 0, cudaMemcpyHostToDevice, *memStream));

		////////////////////////////
		// Device arrays
		RES_FLOAT_TYPE *dD, *dDPrime;

		////////////////////////////////////////////////////////////////
		// Mallocs

		// Determine the amount of memory that will need to be allocated for scratch space
		const int maxMem = (outerLayerIndex*phDivs) * N * maxThreadsPerBlock;
		CHKERR(cudaMalloc(&dD,		sizeof(RES_FLOAT_TYPE) * maxMem));
		CHKERR(cudaMalloc(&dDPrime,	sizeof(RES_FLOAT_TYPE) * maxMem));

		////////////////////////////////////////////////////////////////
		// Kernel launches

		int offset = 1;
		//CHKERR(cudaStreamSynchronize(*memStream));
		for(int i = 0; i < numKernelLaunches; i++) {
			TRACE_KERNEL("CalculateSplineJacobian");
			CalculateSplinesJacobian<RES_FLOAT_TYPE, DCF_FLOAT_TYPE><<<grid, threads, 0, /**comStream*/0>>>
				(offset, thDivs, phDivs, dOutD, dInAmpData, dD, dDPrime,
				(outerLayerIndex*phDivs), min(offset + N*maxThreadsPerBlock, numSplines));
			offset += N*maxThreadsPerBlock;
			CHKERR(cudaStreamSynchronize(/*comStream*/0));
		}
	
		cudaFree(dD);
		cudaFree(dDPrime);

		return err;
	}
}
