#ifndef __COMMONJACOBGPUMETHODS__
#define __COMMONJACOBGPUMETHODS__

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/functional.h>

#include "CommonCUDA.cuh"

#ifndef M_2PI
	#define M_2PI 6.28318530717958647692528676656
#endif

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

template <typename cfType, typename resCFType>
__device__ __forceinline__ resCFType FourPointEvenlySpacedSpline1(cfType p0, cfType p1, cfType p2, cfType p3) {
	typedef typename std::conditional<sizeof(cfType) == 2*sizeof(double), double, float>::type fType;
	resCFType d1;
	d1.x = (-fType(7. / 15.)*(p1.x - p0.x) + fType(14. / 15.)*(p2.x - p0.x) - fType(4. / 15.)*(p3.x - p1.x) + fType(2. / 15.)*(p3.x - p2.x));
	d1.y = (-fType(7. / 15.)*(p1.y - p0.y) + fType(14. / 15.)*(p2.y - p0.y) - fType(4. / 15.)*(p3.y - p1.y) + fType(2. / 15.)*(p3.y - p2.y));
	return d1;
}

template <typename cfType, typename resCFType>
__device__ __forceinline__ resCFType FourPointEvenlySpacedSpline2(cfType p0, cfType p1, cfType p2, cfType p3) {
	typedef typename std::conditional<sizeof(cfType) == 2*sizeof(double), double, float>::type fType;
	resCFType d2;
	d2.x = (fType(2. / 15.)*(p1.x - p0.x) - fType(4. / 15.)*(p2.x - p0.x) + fType(14. / 15.)*(p3.x - p1.x) - fType(7. / 15.)*(p3.x - p2.x));
	d2.y = (fType(2. / 15.)*(p1.y - p0.y) - fType(4. / 15.)*(p2.y - p0.y) + fType(14. / 15.)*(p3.y - p1.y) - fType(7. / 15.)*(p3.y - p2.y));
	return d2;
}

template <typename cfType, typename resCFType>
__device__ __forceinline__ void FourPointEvenlySpacedSpline(cfType p0, cfType p1, cfType p2, cfType p3, resCFType *d1, resCFType *d2) {
	typedef typename std::conditional<sizeof(cfType) == 2*sizeof(double), double, float>::type fType;
	
	*d1 = FourPointEvenlySpacedSpline1<cfType, resCFType>(p0, p1, p2, p3);
	*d2 = FourPointEvenlySpacedSpline2<cfType, resCFType>(p0, p1, p2, p3);
//	d1->x = (-fType(7./15.)*p0.x - fType( 3./15.)*p1.x + fType(12./15.)*p2.x - fType(2./15.)*p3.x);
//	d1->y = (-fType(7./15.)*p0.y - fType( 3./15.)*p1.y + fType(12./15.)*p2.y - fType(2./15.)*p3.y);
//	d2->x = ( fType(2./15.)*p0.x - fType(12./15.)*p1.x + fType( 3./15.)*p2.x + fType(7./15.)*p3.x);
//	d2->y = ( fType(2./15.)*p0.y - fType(12./15.)*p1.y + fType( 3./15.)*p2.y + fType(7./15.)*p3.y);
}


/**
  * NOTES:
  * data must point to the begining of the plane.
  */
template <typename fType, typename interpCFType, typename resCFType>
__device__ resCFType GetAmpAtPointInPlaneJacob(
										  int shellIndex, fType theta, fType phi,
										  int /*char?*/ thetaDivs, int /*char?*/ phiDivs,
										  const fType* __restrict__ data, const interpCFType* __restrict__ D) {

	//this is *the* amplitude calculation function
	//called from HybridOA.cu AddAmplitudeAtPoint (which is called by both MC and Vegas)
	//and called from OAJacob.cu MCOAJacobianKernel

	typedef typename std::conditional<sizeof(resCFType) == 2*sizeof(double), double, float>::type dfType;


	int tI, pI;
	int phiPoints = phiDivs * shellIndex;
	int thePoints = thetaDivs * shellIndex;
	
	double edge = M_2PI / double(phiPoints);

	tI = int((theta / M_PI) * double(thePoints));
	pI = int((phi  / M_2PI) * double(phiPoints));

	// The value [0, 1] representing the location ratio between the two points
	dfType t = (phi / edge) - pI;//fmod(phi, edge) / edge;
	
	pI = (pI == phiPoints) ? 0 : pI;

	resCFType nn[4];
	resCFType p1, p2;
	interpCFType d1, d2;

	// The index of the point above the value in the phi axis
	int pI1 = (pI + 1) % phiPoints;

#pragma unroll 4
	for(int rl = -1; rl < 3; rl++) {
		int theInd = (thePoints + 1 + tI + rl) % (thePoints + 1);
		// Calculate using spline
		int pos1;
		int pos2;

		// Check to see if theta passed [0, \pi] and the phi index needs to be adjusted
		if(theInd != tI + rl) {
			// There is a "2*" and a "/2" that have been removed (canceled out)
			// Assumes phiPoints is even
			if(tI + rl > thePoints)
				theInd = (tI + rl) - (thePoints+1);
			pos1 = 2*(phiPoints * theInd + pI );
			pos2 = 2*(phiPoints * theInd + pI1);

			pos1 += ((2*pI  >= phiPoints) ? -phiPoints : phiPoints);
			pos2 += ((2*pI1 >= phiPoints) ? -phiPoints : phiPoints);
		} else {
			pos1 = 2*(phiPoints * theInd + pI );
			pos2 = 2*(phiPoints * theInd + pI1);
		}
		

		d1 = *((interpCFType*)(&D[pos1 / 2]));	// pos1 >> 1 == pos1 / 2
		d2 = *((interpCFType*)(&D[pos2 / 2]));	// pos2 >> 1 == pos2 / 2

		p1 = *((resCFType *)(&data[pos1]));
		p2 = *((resCFType *)(&data[pos2]));

		nn[rl+1].x = p1.x + (d1.x) * t + 
				(3.0 * (p2.x - p1.x) - 2.0 * (d1.x) - (d2.x)) * (t*t) + 
				(2.0 * (p1.x - p2.x) + (d1.x + d2.x)) * (t*t*t);
		nn[rl+1].y = p1.y + (d1.y) * t + 
				(3.0 * (p2.y - p1.y) - 2.0 * (d1.y) - (d2.y)) * (t*t) + 
				(2.0 * (p1.y - p2.y) + (d1.y + d2.y)) * (t*t*t);
	}

	t = (theta / edge) - tI; //fmod(theta, edge) / edge;

	FourPointEvenlySpacedSpline<resCFType, interpCFType>(nn[0], nn[1], nn[2], nn[3], &d1, &d2);

	resCFType res;
	res.x = nn[1].x + d1.x * t +
		  (3.0 * (nn[2].x - nn[1].x) - 2.0 * d1.x - d2.x) * (t*t) + 
		  (2.0 * (nn[1].x - nn[2].x) + d1.x + d2.x) * (t*t*t);
	res.y = nn[1].y + d1.y * t +
		  (3.0 * (nn[2].y - nn[1].y) - 2.0 * d1.y - d2.y) * (t*t) + 
		  (2.0 * (nn[1].y - nn[2].y) + d1.y + d2.y) * (t*t*t);

	return res;
}

__device__ __host__ __forceinline__ long long IndexFromIndices( int qi, long long ti, long long pi, int thetaDivisions, int phiDivisions ) {
	if(qi == 0)
		return 0;
	qi--;
	long long base = ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;
	return (base + ti * phiDivisions * (qi+1) + pi + 1);	// The +1 is for the origin
}

template<typename fType, typename iType>
__device__ __host__ void GetQVectorFromIndex(long long index, const iType thetaDivs,
									const iType phiDivs, const fType stepSize,
									fType *q, fType *qx, fType *qy, fType *qz, int *lqiOut = NULL, long long *botOut = NULL) {
		
	//////////////////////////////////////////
	// Determine q_vec from single index
	// "IndicesFromIndex"

	int lqi = int(cbrtf(((3*index) / (thetaDivs*phiDivs)) + 0.5f));

	long long bot = (lqi * phiDivs * (lqi + 1) * (3 + thetaDivs + 2 * thetaDivs * (long long)lqi)) / 6; //(lqi*(28 + lqi*(60 + 32*(long long)lqi))) / 3;
	if(index > bot) {
		lqi++;
	}
	lqi--;
	bot = (lqi * phiDivs * (lqi + 1) * (3 + thetaDivs + 2 * thetaDivs * (long long)lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*(long long)lqi))) / 3;
	lqi++;

	if(lqiOut)
		*lqiOut = lqi;
	if(botOut)
		*botOut = bot;

	// End of IndicesFromIndex
	///////////////////////////////////////////////////////////////////
	fType sn, cs;

	*q = fType(lqi) * stepSize;

	sincospi(fType((index - bot - 1) / (lqi * phiDivs)) / fType(thetaDivs * lqi), &sn, &cs);//, ph = 2.0 * M_PI * double(in1->y) / double(in1->x);

	*qx = *q * sn;
	*qy = *q * sn;
	*qz = *q * cs;

	sincospi(fType(2 * ((index - bot - 1) % (lqi * phiDivs))) / fType(phiDivs * lqi), &sn, &cs);

	*qx *= cs;
	*qy *= sn;
			
	if(index == 0) {
		*qx = 0.0;
		*qy = 0.0;
		*qz = 0.0;
	}

}

template <typename fType>
__device__ __host__ bool closeToZero(fType x) {
	return (fabs(x) < 1.0e-7);
}

template <typename T, int NPerThread>
__global__ void ScaleKernel(T* startPoint, T scale, int lastElement)
{
	int tid = threadIdx.x;
	long long idx = (blockIdx.x * blockDim.x + tid);

	if( (NPerThread * (idx+1)) < lastElement ) {
#pragma unroll
		for(int i = 0; i < NPerThread; i++)
		{
			startPoint[NPerThread * idx + i] *= scale;
		}
	}
	else
	{
#pragma unroll
		for(int i = 0; i < NPerThread; i++)
		{
			if( (NPerThread * idx + i) < lastElement )
				startPoint[NPerThread * idx + i] *= scale;
		}
	}
}


#endif // __COMMONJACOBGPUMETHODS__