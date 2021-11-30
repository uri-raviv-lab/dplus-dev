#ifndef __COMMONCUDA_CUH
#define __COMMONCUDA_CUH

#include <cuda_runtime.h>
#include "LocalBackend.h"

#define freePointer(p) {if(p) delete[] p; p = NULL;}
#define deviceFreePointer(p) {if(p) CHKERR(cudaFree(p)); p = NULL;}

static inline void PrintErrorWithLocation(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR in file %s, line %d: %s (%d)\n", file, line, cudaGetErrorString(err), err);
		fprintf(stdout, "ERROR in file %s, line %d: %s (%d)\n", file, line, cudaGetErrorString(err), err);
		throw backend_exception(ERROR_GENERAL, cudaGetErrorString(err));
//		cudaDeviceReset();
//		exit(0);
	}
}

#ifdef CHKERR
#undef CHKERR  // This is the implementation we prefer.
#endif
#define CHKERR(expr) do {\
    err = expr;\
    PrintErrorWithLocation(err, __FILE__, __LINE__);\
} while(0)


static inline void PrintKernelWithLocation(const char *name, const char *file, int line)
{
	static std::string lastName="", lastFile="";
	static int lastLine=-1;
	static bool printedDot = false;

	FILE *f = fopen("kernel.log", "a");
	if (f == NULL)
	{
		printf("Error opening kernel log file!\n");
		return;
	}

	if (lastName == name && lastFile == file && lastLine == line)
	{
		fprintf(f, ".");
		fflush(f);
		printedDot = true;
	}
	else
	{
		if (printedDot)
			fprintf(f, "\n");
		fprintf(f, "%s called from %s line %d\n", name, file, line);
		fflush(f);
		lastName = name;
		lastFile = file;
		lastLine = line;
		printedDot = false;
	}
	fclose(f);
}

//#define KERNEL_TRACING_ON
#ifdef KERNEL_TRACING_ON
#define TRACE_KERNEL(name) \
	do {\
		const char * kernelName = name;\
		PrintKernelWithLocation(kernelName, __FILE__, __LINE__); \
		} while (0)
#else
#define TRACE_KERNEL(name){}
#endif


// Frees, allocates, and copies a cpu vector to a device vector
#define CPU_VECTOR_TO_DEVICE(d,h,sz) {							\
	freePointer(d);												\
	CHKERR(cudaMalloc(&d, sz));									\
	CHKERR(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice)); }

#define CPU_VECTOR_TO_DEVICE_ASYNC(d,h,sz,st) {					\
	freePointer(d);												\
	CHKERR(cudaMalloc(&d, sz));									\
	CHKERR(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, st)); }

// CUDA 8.0 introduces a new built-in for fp64 atomicAdd(). Note that this
// built-in cannot be overridden with a custom function declared by the user
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else

template<typename T>
static __device__ double atomicAdd(double* address, T val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template<typename T>
__global__ void PrintDecimalDeviceMemory(T *mem, int length)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < length)
		printf("*p[%d] = %d\n", idx, mem[idx]);
}


template<typename T>
__global__ void PrintFloatDeviceMemory(T *mem, int length)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < length)
		printf("*p[%d] = %f\n", idx, mem[idx]);
}

template<typename T>
__global__ void PrintPairFloatDeviceMemory(T *mem, int length)
{
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < length)
		printf("*p[%d] = (%f, %f)\n", idx, mem[idx].x, mem[idx].y);
}

template<typename T>
__global__ void PrintFloat4DeviceMemory(T *mem, int length)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length)
		printf("*p[%d] = (%f, %f, %f, %f)\n", idx, mem[idx].x, mem[idx].y, mem[idx].z, mem[idx].w);
}

template<typename sType, int blockSize>
__device__ __inline__ void warpReduce( volatile sType* sdata, int tid ) {
	if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if(blockSize >= 16) sdata[tid] += sdata[tid +  8];
	if(blockSize >=  8) sdata[tid] += sdata[tid +  4];
	if(blockSize >=  4) sdata[tid] += sdata[tid +  2];
	if(blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

/*******************************************************************************
Copied with a few changes from Mark Harris' presentation at
http://gpgpu.org/static/sc2007/SC07_CUDA_5_Optimization_Harris.pdf
*******************************************************************************/
template<typename rtype, unsigned int blockSize>
__global__ void reduce(rtype *gIn, rtype *gOut, unsigned int n)
{
	extern __shared__ rtype sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while(i < n)
	{
		sdata[tid] += gIn[i] + gIn[i+blockSize];
		i += gridSize;
	}
	
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	if (tid < 32)
	{
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
	}

	if (tid == 0) gOut[blockIdx.x] = sdata[0];
}

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

template<typename rtype>
void cubReduce(rtype *gIn, rtype *gOut, unsigned int n, cudaStream_t &stream)
{
	cudaError_t err = cudaSuccess;

	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gIn, gOut, n, stream);

	// Allocate temporary storage
	CHKERR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	// Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gIn, gOut, n, stream);

	deviceFreePointer(d_temp_storage);
}

#endif // __COMMONCUDA_CUH
