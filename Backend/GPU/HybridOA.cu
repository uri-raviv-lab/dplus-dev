#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "GPUHybridCalc.cuh"
#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"

#include <cuda_runtime.h>

#include <time.h>

#include <curand_kernel.h>

#include "assert.h"

template<typename fType, typename interpCFType, typename cfType, int avePoints>
__device__ __forceinline__ void AddAmplitudeAtPoint(const cfType* inAmp,
									  const interpCFType* ds, const int tDiv, const int pDiv,
									  float qStepSize,
									  const float4* __restrict__ rotations, const int numRots,
									  const float4* __restrict__ translations, const int* __restrict__ numTrans,
									  const int qInd,	// The index of the lower plane
									  const int m,	// The number of q-values between the planes (lower, upper]
									  const float* __restrict__ qs,	// The list of m q-values that are being averaged
									  const double2 random,
									  cfType *reses /*a [avePoints][m] matrix where m is the number of q-values between the two planes*/
									  )
{
	//called by both vegas and monte carlo
	float s1, c1, s2, c2, s3, c3;


	sincos(random.x, &s2, &c2);	// theta [0,\pi]
	sincos(random.y, &s1, &c1);	// phi [0,2\pi)
/*
	sincosf(M_PI / 4.0, &s2, &c2);	// theta [0,\pi]
	sincosf(M_PI / 4.5, &s1, &c1);	// phi [0,2\pi)
	*/
	float3 qVec;
	qVec.x = c1 * s2;
	qVec.y = s1 * s2;
	qVec.z =      c2;

	int cumTrans = 0;

	for(int rr = 0; rr < numRots; rr++)
	{
		// Rotate original qVector
		float4 rt = rotations[rr];
		//float4 rt = __ldg(&rotations[rr]);
		sincos(rt.x, &s1, &c1);
		sincos(rt.y, &s2, &c2);
		sincos(rt.z, &s3, &c3);
		float scaleR = rt.w;

#define TESTVEC qVec
		double3 rotVec;

		// Inverse of the above - needed to match the Debye model (see /Test files/r2622)
		rotVec.x =  (c2*c3)	* qVec.x + (c1*s3+c3*s1*s2)	* qVec.y + (s1*s3-c1*c3*s2) * qVec.z;
		rotVec.y = -(c2*s3)	* qVec.x + (c1*c3-s1*s2*s3)	* qVec.y + (c3*s1+c1*s2*s3) * qVec.z;
		rotVec.z =  (s2   )	* qVec.x - (c2*s1         )	* qVec.y + (c1*c2         ) * qVec.z;

		// Convert to polar
		if(fabs(rotVec.z) > 1.) rotVec.z = (rotVec.z > 0.) ? 1. : -1.;
		double newTheta	= acos (rotVec.z /*/ q*/);
		double newPhi	= atan2(rotVec.y, rotVec.x);


		if(newPhi < 0.0)
 			newPhi += M_2PI;
		cfType amps[4];

#pragma unroll 4
		for(int i = - 1; i <= 2; i++)
		{
			long long lqi = (long long)(i + qInd - 1);	// For the base
			long long bot = (lqi * pDiv * (lqi + 1) * (3 + tDiv + 2 * tDiv * lqi)) / 6;
			bot++;

			lqi++;	// The actual layer
			switch (lqi)
			{
			case -1:
			{
				bot = 1; // Very important as the equation for bot assumes that the layer is positive
				double newThetaTag = M_PI - newTheta;
				if (newThetaTag < 0)
					newThetaTag = 0.;
				if (newThetaTag > M_PI)
					newThetaTag = M_PI;

				double newPhiTag = newPhi + M_PI;
				if (newPhiTag < 0)
					newPhiTag = M_2PI + (newPhiTag - int(newPhiTag / M_2PI)* M_2PI);
				if (newPhiTag >= M_2PI)
					newPhiTag = newPhiTag - int(newPhiTag / M_2PI)*M_2PI;
				amps[1 + i] = GetAmpAtPointInPlaneJacob<fType, interpCFType, cfType>(
					-lqi, newThetaTag, newPhiTag, tDiv, pDiv, (fType*)(inAmp + bot), ds + bot);
				break;
			}
			case 0:
				amps[1 + i] = inAmp[0];
				break;
			default:
				amps[1 + i] = GetAmpAtPointInPlaneJacob<fType, interpCFType, cfType>(
					lqi, newTheta, newPhi, tDiv, pDiv, (fType*)(inAmp + bot), ds + bot);
				break;
			}

		} // for i -1 to 2

		// Get interpolation coefficients
		interpCFType d1, d2;
		FourPointEvenlySpacedSpline<cfType, interpCFType>(amps[0], amps[1], amps[2], amps[3], &d1, &d2);

		// Save results
		for(int j = 0; j < m; j++)
		{
			fType t = (qs[j] - qInd * qStepSize) / qStepSize;	// t[m] can be in constant/shared memory
			cfType tmpAmp;
			tmpAmp.x = amps[1].x + d1.x * t +
			  (3.0 * (amps[2].x - amps[1].x) - 2.0 * d1.x - d2.x) * (t*t) + 
			  (2.0 * (amps[1].x - amps[2].x) + d1.x + d2.x) * (t*t*t);
			tmpAmp.y = amps[1].y + d1.y * t +
			  (3.0 * (amps[2].y - amps[1].y) - 2.0 * d1.y - d2.y) * (t*t) + 
			  (2.0 * (amps[1].y - amps[2].y) + d1.y + d2.y) * (t*t*t);

			// Sum over the translations
			const float4 *tran = translations + cumTrans;
			double sumCs = 0., sumSn = 0.;
			for(int tr = 0; tr < numTrans[rr]; tr++)
			{
				float4 ttr = tran[tr];
				double qd = qs[j] * (ttr.x * TESTVEC.x +
									ttr.y * TESTVEC.y +
									ttr.z * TESTVEC.z);
				double sn, cs;
				sincos(qd, &sn, &cs);
				sumSn += sn;
				sumCs += cs;
			} // for tr

			// Multiply the result and multiply by the scale
			cfType tmp = tmpAmp;
			tmpAmp.x = (tmp.x * sumCs - tmp.y * sumSn) * scaleR;
 			tmpAmp.y = (tmp.y * sumCs + tmp.x * sumSn) * scaleR;

			tmp = reses[blockIdx.x * blockDim.x + threadIdx.x + j * avePoints];
			tmp.x = tmp.x + tmpAmp.x;
			tmp.y = tmp.y + tmpAmp.y;
			reses[blockIdx.x * blockDim.x + threadIdx.x + j * avePoints] = tmp;
		} // for j

		cumTrans += numTrans[rr];
	} // for num rotations

} // AddAmplitudeAtPoint

template<typename fType, typename cfType, int avePoints>
__global__ void ConvertAmplitudeToIntensity(
	cfType *resA /*a [avePoints][m] matrix where m is the number of q-values between the two planes representing amplitude*/,
	fType  *resI /*a [avePoints][m] matrix where m is the number of q-values between the two planes representing Intensity*/,
	const int m	// The number of q-values between the planes (lower, upper]
	)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= avePoints ||
		y >= m)
		return;
	
	cfType ampl = resA[x * m + y];
	// Convert to intensity
	resI[x * m + y] = ampl.x * ampl.x + ampl.y * ampl.y;
}

// IMPORTANT! Assumes dimSize is less than or equal to the block size
template<typename fType, int dimSize>
__device__ __inline__ void ResizeVegasPoints
	(
	fType *s_sumSq,
	fType *s_boundaries, ///< The boundaries of the previous iteration. Of length dimSize.
	fType *g_boundaries ///< The boundaries of the next iteration. Of length dimSize.
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	typedef cub::BlockReduce<fType, dimSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	fType totalVar = BlockReduce(temp_storage).Sum((idx < dimSize ? s_sumSq[idx] : fType(0)), dimSize);
	__shared__ fType allRes[1];
	
	if(idx == 0)
		allRes[0] = totalVar;
	__syncthreads();
	totalVar = allRes[0];
	if(totalVar == 0.)	// No variance, don't resize the bins
		return;

	__shared__ fType blurredVar[dimSize];

	if (idx < dimSize)
	{
		// 	if(idx == 0 || idx == dimSize-1)
		// 		blurredVar[idx] = ((s_sumSq[idx] + s_sumSq[idx + (idx == 0 ? 1 : -1) ]) / 2.);
		if (idx == 0) {
			blurredVar[idx] = ((s_sumSq[idx] + s_sumSq[idx + 1]) / 2.);
			//		printf(" (%f + %f ) / 2 == %f || %f\n", s_sumSq[idx], s_sumSq[idx + 1], (s_sumSq[idx] + s_sumSq[idx + 1]) / 2., blurredVar[idx]);
		}
		else if (idx == dimSize - 1)
			blurredVar[idx] = ((s_sumSq[idx] + s_sumSq[idx - 1]) / 2.);
		else
			blurredVar[idx] = ((s_sumSq[idx - 1] + s_sumSq[idx] + s_sumSq[idx + 1]) / 3.);

		blurredVar[idx] = allRes[0] / blurredVar[idx];

		// From here, blurredVar is the weights
		blurredVar[idx] = pow(
			(blurredVar[idx] - 1.) / (blurredVar[idx] * log(blurredVar[idx])),
			1.5 // Damping
			);
		if (blurredVar[idx] != blurredVar[idx])
			blurredVar[idx] = 0.;
	}
	__syncthreads();
	fType totalWgt = BlockReduce(temp_storage).Sum((idx < dimSize ? blurredVar[idx] : fType(0) ), dimSize);
	
	if(idx == 0)
		allRes[0] = totalWgt;
	__syncthreads();
	totalWgt = allRes[0];

/*
	__syncthreads();
	if(idx == 0)
	{
		for(int i = 0; i < dimSize; i++)
			printf("weight[%d] %f\n", i, blurredVar[i]);
		printf("total weight %f\n", totalWgt);
	}
*/
	__shared__ fType s_newBoundaries[dimSize];

	// This is very serial and I don't know how to parallelize, but who cares. It's short.
	if(idx == 0)
	{
		fType xold, xnew = 0., dWgt = 0., points = totalWgt / dimSize;
		int j = 0;
		for(int i = 0; i < dimSize; i++)
		{
			dWgt += blurredVar[i];	// REMINDER: blurredVar[i] is weight[i]
			xold = xnew;
			xnew = s_boundaries[i+1];
			for(; dWgt > points; j++)
			{
				dWgt -= points;
				s_newBoundaries[j] = xnew - dWgt * (xnew - xold) / blurredVar[i];
			} // for j
		} // for i
	} // if idx == 0

	__syncthreads();
	if(idx < dimSize-1)
		g_boundaries[1+idx] = s_newBoundaries[idx];
}

// A total of thetaBins + phiBins threads should be used
template<typename fType, int thetaBins, int phiBins>
__global__ void HybridVEGASResizeGridKernel(
	fType *boundariesThetaPhi,	///< The boundaries for each iteration (first Theta, then Phi)
	fType *thetaPhiBinResults ///< Array of variances [sum(dif^2)] (first the theta bins, then the phi bins)
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ fType s_oldBoundaries[2 + thetaBins + phiBins];
	__shared__ fType s_thetaPhiBinResults[(thetaBins + phiBins)];

	s_oldBoundaries[idx] = boundariesThetaPhi[idx];
	s_thetaPhiBinResults[idx] = thetaPhiBinResults[idx]; // Load the variances
/*
	s_thetaPhiBinResults[(thetaBins + phiBins)+idx] = thetaPhiBinResults[(thetaBins + phiBins)+idx];
*/
	if(idx == 0)
		s_oldBoundaries[thetaBins + phiBins] = boundariesThetaPhi[thetaBins + phiBins];
	if(idx == thetaBins + phiBins - 1)
		s_oldBoundaries[thetaBins + phiBins + 1] = boundariesThetaPhi[thetaBins + phiBins + 1];
	__syncthreads();

	ResizeVegasPoints<fType, thetaBins>(s_thetaPhiBinResults, s_oldBoundaries, boundariesThetaPhi);
	if(phiBins > 2)
		ResizeVegasPoints<fType, phiBins>(s_thetaPhiBinResults + thetaBins, s_oldBoundaries + 1 + thetaBins, boundariesThetaPhi + 1 + thetaBins);

}

// A total of thetaBins X phiBins threads should be used
template<typename fType, int avePoints,  int thetaBins, int phiBins, int pointsPerBin>
__global__ void HybridVEGASBinReduceToIKernel(
	const fType *resI,	///< The input 2D matrix (avePoints x m) where avePoints is divided into theta and phi bins
	const fType *boundariesThetaPhi,	///< The boundaries for each iteration (first Theta, then Phi)
	const int m,		///< The number of q-values between the planes (lower, upper]
	fType *integrationResults ///< The results of the integration (surprise). Of length m
	)
{
	//assert(avePoints == thetaBins * phiBins * pointsPerBin);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	__shared__ fType s_boundaries[2 + thetaBins + phiBins];
	if(idx < 2 + thetaBins + phiBins)
		s_boundaries[idx] = boundariesThetaPhi[idx];
	// For the case where thetaBins or phiBins is 1, the total number of threads will be less 
	// than (2 + thetaBins + phiBins). In this case, we need to load the remaining 3 boundaries
	if(thetaBins < 2 || phiBins < 2)
	{
		// Make the first thread do all the loads; who knows, maybe the other dimensions
		// only has two bins. We hope this is optimized out anyway.
		int maxD = max(thetaBins, phiBins);
		if(idx == 0)
		{
			for(int i = 0; i < maxD; i++)
				s_boundaries[thetaBins + phiBins - 1 + i] = boundariesThetaPhi[thetaBins + phiBins - 1 + i];
		}
	}

	__syncthreads();


	if(idx >= thetaBins * phiBins)
		return;

	fType qSums;
	typedef cub::BlockReduce<fType, thetaBins * phiBins> BlockReduce;

	int thInd = (idx / phiBins);
	int phInd = (idx % phiBins);

	// The box is NOT in theta-phi space!
	fType binVolume = (s_boundaries[thInd+1] - s_boundaries[thInd]) * 
		(s_boundaries[thetaBins + 1 + phInd + 1] - s_boundaries[thetaBins + 1 + phInd]);

#ifdef _DEBUG55
	if(idx == 0)
	{
		printf("Bin volume = %f\n", binVolume);
	}
#endif

	for(int i = 0; i < m; i++)
	{
		qSums = 0.;
		for(int p = 0; p < pointsPerBin; p++)
		{
			qSums += resI[idx*pointsPerBin + p + i*avePoints];
#ifdef _DEBUG33
	if(idx == 0)
	{
		printf("Sum = %f\tI = %f\n", qSums, resI[idx*pointsPerBin + p + i*avePoints]);
	}
#endif
		}

		qSums *= binVolume / pointsPerBin;

		__shared__ typename BlockReduce::TempStorage temp_storage;
		fType totalIntegration = BlockReduce(temp_storage).Sum(qSums);
#ifdef _DEBUG33
	if(idx == 0)
	{
		printf("totalIntegration = %f\n", totalIntegration);
	}
#endif

 		if(idx == 0)
			atomicAdd(&integrationResults[i], totalIntegration);
#ifdef _DEBUG33
	if(idx == 0)
	{
		printf("integrationResults[i] = %f\n", integrationResults[i]);
	}
#endif
	}

}


// A total of thetaBins X phiBins threads should be used
template<typename fType, int avePoints,  int thetaBins, int phiBins, int pointsPerBin>
__global__ void HybridVEGASBinReduceKernel(
	const fType *resI,	///< The input 2D matrix (avePoints x m) where avePoints is divided into theta and phi bins
	const fType *boundariesThetaPhi,	///< The boundaries for each iteration (first Theta, then Phi)
	const int m,		///< The number of q-values between the planes (lower, upper]
	fType *thetaPhiBinResults ///< Array of summed results (first the theta bins, then the phi bins)
	)
{
	//assert(avePoints == thetaBins * phiBins * pointsPerBin);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ fType s_boundaries[2 + thetaBins + phiBins];
	if(idx < 2 + thetaBins + phiBins)
		s_boundaries[idx] = boundariesThetaPhi[idx];
	// For the case where thetaBins or phiBins is 1, the total number of threads will be less 
	// than (2 + thetaBins + phiBins). In this case, we need to load the remaining 3 boundaries
	if(thetaBins < 2 || phiBins < 2)
	{
		// Make the first thread do all the loads; who knows, maybe the other dimensions
		// only has two bins. We hope this is optimized out anyway.
		int maxD = max(thetaBins, phiBins);
		if(idx == 0)
		{
			for(int i = 0; i < maxD; i++)
				s_boundaries[thetaBins + phiBins - 1 + i] = boundariesThetaPhi[thetaBins + phiBins - 1 + i];
		}
	}

	__syncthreads();

	if(idx >= thetaBins * phiBins)
		return;

	int thInd = (idx / phiBins);
	int phInd = (idx % phiBins);

	// The box is NOT in theta-phi space!
	fType binVolume = (s_boundaries[thInd+1] - s_boundaries[thInd]) * 
		(s_boundaries[thetaBins + 1 + phInd + 1] - s_boundaries[thetaBins + 1 + phInd]);

	fType sumOverQs = 0., runningMean = 0., runningSumSq = 0., diff;
	for(int p = 0; p < pointsPerBin; p++)
	{
		sumOverQs = 0.;
		for(int i = 0; i < m; i++)
		{
			sumOverQs += resI[idx*pointsPerBin + p + i*avePoints];
		}
		diff = (binVolume*sumOverQs) - runningMean;
		runningMean += diff / fType(p+1);
		runningSumSq += diff * diff * fType(p) / fType(p+1);
	}

	atomicAdd(&thetaPhiBinResults[thInd], runningSumSq * pointsPerBin);
	atomicAdd(&thetaPhiBinResults[thetaBins + phInd], runningSumSq * pointsPerBin);
	
}

template<typename fType, typename interpCFType, typename cfType, int avePoints,  int thetaBins, int phiBins, int pointsPerBin>
__global__ void HybridVEGASMCOAJacobianKernel(
	const fType * __restrict__ boundariesThetaPhi,	// The boundaries for each iteration (first Theta, then Phi)
	const fType *const* __restrict__ grids, const interpCFType *const* __restrict__ ds,
	int numgrids, const int tDiv, const int pDiv, float qStepSize, 
	const float4 *const* __restrict__ rotations, const int * __restrict__ numRots,
	const float4 *const* __restrict__ translations, const int *const* __restrict__ numTrans,
	const int qInd,	// The index of the lower plane
	const int m,	// The number of q-values between the planes (lower, upper]
	const float * __restrict__ qs,	// The list of m q-values that are being averaged
	const double2 * __restrict__ randoms,	// Values [0,1) that should represent the value
	cfType *resA /*a [avePoints][m] matrix where m is the number of q-values between the two planes representing amplitude*/
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= avePoints)
		return;

	assert(avePoints == thetaBins * phiBins * pointsPerBin);

	int thInd, phInd;
	int binIdx = idx / pointsPerBin;
	thInd = binIdx / phiBins;
	phInd = binIdx % phiBins;

	double2 point;
	fType thtMin = boundariesThetaPhi[thInd  ];
	fType thtMax = boundariesThetaPhi[thInd+1];
	fType phiMin = boundariesThetaPhi[thetaBins + 1 + phInd  ];
	fType phiMax = boundariesThetaPhi[thetaBins + 1 + phInd+1];
	// Reversed the order to make the range [0, pi) instead of (0,pi]
	point = make_double2(acos(1. - 2. * (randoms[idx].x * (thtMax - thtMin) + thtMin)), //t.x
						(randoms[idx].y * (phiMax - phiMin) + phiMin) * 2. * M_PI       //t.y
						);
	// Calculate amplitude
	for(int i = 0; i < numgrids; i++)
	{
 		AddAmplitudeAtPoint<fType, interpCFType, cfType, avePoints>
 			((cfType*)(grids[i]), ds[i], tDiv, pDiv, qStepSize, rotations[i], numRots[i],
 				translations[i], numTrans[i], qInd, m, qs, point, resA);
	}

#ifdef _DEBUG000
	if(threadIdx.x == 0)
	{
		for(int i = 0; i < m; i++)
			printf("F(%f, %f) resA[%d, %d] = {%f, %f}\n", point.x, point.y, idx, i, resA[idx+i].x, resA[idx+i].y);
	}
#endif
}


template<typename fType, typename interpCFType, typename cfType, int avePoints>
__global__ void HybridMCOAJacobianKernel(const fType * const* __restrict__ grids,
									  const interpCFType * const* __restrict__ ds, int numgrids,
									  const int tDiv, const int pDiv, float qStepSize,
									  const float4 * const* __restrict__ rotations, const int* __restrict__ numRots,
									  const float4 * const* __restrict__ translations, const int * const* __restrict__ numTrans,
									  const int qInd,	// The index of the lower plane
									  const int m,	// The number of q-values between the planes (lower, upper]
									  const float* __restrict__ qs,	// The list of m q-values that are being averaged
									  const double2* randoms,
									  cfType *resA /*a [avePoints][m] matrix where m is the number of q-values between the two planes representing amplitude*/
									  )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= avePoints)
		return;

#ifdef _DEBUG0

	if(!(blockIdx.x * blockDim.x + threadIdx.x == 0 &&
			qs[0] == 0.))
			return;

	printf("numgrids = %d\n", numgrids);

	for(int k = 0; k < numgrids; k++) 
	{
		printf("translation[%d] = %p\n",
			k, translations[k]);
	}
	for(int k = 0; k < 3; k++) 
	{
		float4* tr = translations[0];
		printf("translation[%d] %p = [%f, %f, %f]\n",
			k, tr+k, (tr+k)->x, (tr+k)->y, (tr+k)->z);
	}


	for(int i = 0; i < numgrids; i++)
	{
		float4* rotationsi = rotations[i];
		int numRotsi = numRots[i];
		int* numTransi = numTrans[i];
 		float4* translationsi = translations[i];
		int cumTrans = 0;

		for(int rr = 0; rr < numRots[i]; rr++)
		{
			// Rotate original qVector
			float4 rt = rotationsi[rr];
			float4 *tran = translationsi + cumTrans;

			for(int tr = 0; tr < numTransi[rr]; tr++)
			{
				printf(
					"[i, rr, tr -> %p] = [%d, %d, %d]\t{%f, %f, %f}   {%f, %f, %f}"
					"\n",
					tran+tr, i, rr, tr,
					rt.x, rt.y, rt.z,
					tran[tr].x, tran[tr].y, tran[tr].z
					);
			}

			cumTrans += numTransi[rr];
		}
	}
	return;
#endif

	// Calculate amplitude
	for(int i = 0; i < numgrids; i++)
	{
 		AddAmplitudeAtPoint<fType, interpCFType, cfType, avePoints>
 			((cfType*)(grids[i]), ds[i], tDiv, pDiv, qStepSize, rotations[i], numRots[i],
 				translations[i], numTrans[i], qInd, m, qs, randoms[idx], resA);
	}

}

template<typename inType>
__global__ void UniformRandomToSphericalPoints(inType* in, int maxNums)
{
	long long id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= maxNums)
		return;

	inType tmp = in[id];

	// Theta
	tmp.x = acos(tmp.x * 2. - 1.);
	// Phi
	tmp.y = tmp.y * 2. * M_PI;

	in[id] = tmp;
}

template<typename inType>
__global__ void ValidateGridWorkspaceGrid(inType *in, long long voxels)
{
	long long pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= voxels)
		return;

	if(
		in[pos].x != in[pos].x || 
		in[pos].y != in[pos].y
		)
	{
		printf("BAD! data[%lld] = [%f, %f]\n", pos, in[pos].x, in[pos].y);
	}

}
