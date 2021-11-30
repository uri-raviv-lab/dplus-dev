#ifndef __CALCULATEJACOBIANSPLINES_CUH__
#define __CALCULATEJACOBIANSPLINES_CUH__

#include <cuda_runtime.h>

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <assert.h>


// CUDA runtime
#include <cuComplex.h>

#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0 ) // Turn off all warnings (bypasses #pragma warning(default : X )
#endif
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif
#include <typeinfo>  //for 'typeid'

#include "CommonJacobGPUMethods.cu"

#include "CommonCUDA.cuh"

#define CHKRANDERR(expr) do {\
    ranErr = expr;\
    if(ranErr != CURAND_STATUS_SUCCESS) {                                            \
         fprintf(stderr, "CURAND ERROR in file %s, line %d: %s (%d)\n", __FILE__, __LINE__, "cudaGetErrorString(ranErr)", ranErr);\
    }                \
} while(0)

// Need the namespace so that the operator overloads don't interfere with other stuff
namespace CudaJacobianSplines {
	
	// A template for calculating the cubic splines of a Jacobian Grid on a CUDA device.
	// The parameter pointers are device pointers that are preallocated by the calling
	// function.
	template <typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
	int GPUCalculateSplinesJacobSphrOnCardTempl(int thDivs, int phDivs, int outerLayerIndex,
									RES_FLOAT_TYPE *dInAmpData,  DCF_FLOAT_TYPE *dOutD, const cudaStream_t& memStream = 0, const cudaStream_t& comStream = 0);

} // namespace

#endif