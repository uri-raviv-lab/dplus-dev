#ifndef __COMMONPDB_CUH
#define __COMMONPDB_CUH

#include <cuda_runtime.h>

/*****************************************************************************
Valid bitwise modes:
0x00 --> Solvent only
0x01 --> Not only solvent
0x02 --> Fraser (dummy atom) style solvent subtraction used
0x04 --> Use voxel based solvent. NOT RELEVANT FOR THIS KERNEL (AtomicFormFactorKernel)
0x08 --> Used voxel based solvation layer. NOT RELEVANT FOR THIS KERNEL (AtomicFormFactorKernel)
0x16 --> Use Debye-Waller factors when calculating atomic positions.  NOT RELEVANT FOR THIS KERNEL (AtomicFormFactorKernel)

*****************************************************************************/
template<int mode, bool bTranspose>
__global__ void AtomicFormFactorKernel(const float qMin, const float qStepSize, int numQLayers,
									   float *gCoef, float *affs, int numUIons,
									   float* atmRad, float solvED, bool electronPDB=false)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;

	float q = (qMin + idx * qStepSize);

	// The number is (4pi)^2         
	float sqq = (q * q / 100.0f) / (157.913670417429737901351855998f);

// 	__shared__ float sCoef[9 * 32];
// 
// 	if(threadIdx.x < 9)
// 		sCoef[threadIdx.x] = gCoef[];
// 	syncthreads();

	if(idx > numQLayers)
		return;

	for(int i = 0; i < numUIons; ++i)
	{
		float res = 0.0;
		if(mode & 0x01) {
			res = 
				gCoef[i+numUIons*0] * expf(-gCoef[i+numUIons*1] * sqq) + 
				gCoef[i+numUIons*2] * expf(-gCoef[i+numUIons*3] * sqq) + 
				gCoef[i+numUIons*4] * expf(-gCoef[i+numUIons*5] * sqq) + 
				gCoef[i+numUIons*6] * expf(-gCoef[i+numUIons*7] * sqq) ;
				
				if (electronPDB)
				{
					res += gCoef[i+numUIons*8] * expf(-gCoef[i+numUIons*9] * sqq);
				}
				else
				{
					res += gCoef[i+numUIons*8];
				}
		}
		// dummy atom solvent here
		if(mode	& 0x02) {
			const float rad = atmRad[i];
#ifdef USE_FRASER
			res += -5.5683279968317084528/*4.1887902047863909846 /*4\pi/3*/ * rad * rad * rad * exp( -(rad * rad * (q*q) / 4.) ) * solvED;
#else
			res += -4.1887902047863909846 * rad * rad * rad * exp(-(0.20678349696647 * rad * rad * q * q)) * solvED;
#endif
		}
		if(bTranspose)
		{
			affs[idx * numUIons + i] = res;
		}
		else
		{
			affs[i * numQLayers + idx] = res;
		}
	}
}


#endif	// __COMMONPDB_CUH
