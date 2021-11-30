#include "gpgpu.h"

#include <CL/cl.h>

#define PROGRAM ""

#ifdef _WIN32
#include <windows.h>
#define SHOW_ERROR(...) { char a[1024] = {0}; sprintf(a, __VA_ARGS__); MessageBoxA(NULL, a, "ERROR", NULL); }
#else
#include <cstdio>
#define SHOW_ERROR(...) printf(__VA_ARGS__)
#endif

// Global, NULL-by-default OpenCL objects
cl_device_id g_device = NULL;
cl_context g_context = NULL;
cl_command_queue g_cmdq = NULL;
cl_program g_models = NULL;

// Called when GPU backend is chosen
void InitializeOpenCL() {
	cl_uint numPlatforms, deviceCount;
    int errr = 0;

    // Creating stuff
    errr = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(errr != CL_SUCCESS) {
        SHOW_ERROR("ERROR GETTING PLATFORM NUM: %d\n", errr);
        return;
    }

    cl_platform_id* platforms = new cl_platform_id[numPlatforms];

    errr = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(errr != CL_SUCCESS) {
        SHOW_ERROR("ERROR GETTING PLATFORM: %d\n", errr);
        return;
    }

    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
    if(deviceCount < 1) {
        SHOW_ERROR("NO DEVICES FOUND\n");
        return;
    }

    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0 };

    g_context = clCreateContext(properties, 1, &g_device, NULL, NULL, NULL);  

	g_cmdq = clCreateCommandQueue(g_context, g_device, CL_QUEUE_PROFILING_ENABLE, NULL);

    delete [] platforms;

	// Compiling program
	g_models = clCreateProgramWithSource(g_context, 1,
                                         (const char **) &PROGRAM, 
                                         NULL, NULL); 

    // Compiling the program
    errr = clBuildProgram(g_models, 0, NULL, "", NULL, NULL);
		
    // Print out compiler errors
    if(errr) {
		char errors[16384] = {0};
        clGetProgramBuildInfo(g_models, g_device,
                              CL_PROGRAM_BUILD_LOG, sizeof(char) * 16384,
                              errors, NULL);

		SHOW_ERROR("Program compilation failed, details:\n%s\n", errors);
        return;
    }
}

// Called when CPU backend is chosen
void DestroyOpenCL() {
	if(g_cmdq)
		clReleaseCommandQueue(g_cmdq);
	g_cmdq = NULL;
	if(g_context)
		clReleaseContext(g_context);
	g_context = NULL;
	g_device = NULL;
}

bool GenerateGPUModel(const char *modelName, const VectorXf& x, const VectorXf& params, VectorXf& y, int nd, int extraParams) {
	int err = 0;
	if(x.size() != y.size())
		y = VectorXf::Zero(x.size());

	// No OpenCL
	if(!g_context)
		return false;

	int pow = 512;
	for(pow = 512; pow > 1; pow /= 2)
		if(x.size() % pow == 0)
			break;

	size_t dims[]   = {x.size(), 1, 1};
	size_t locals[] = {pow, 1, 1};

	// Creating the kernel
	cl_kernel kernel = clCreateKernel(g_models, modelName, &err);
	if(err) {
		SHOW_ERROR("Error creating kernel: %d", err);
		return false;
	}

	cl_mem gx, gy, gp;

	// Creating buffers
	gx = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * x.size(), (void *)x.data(), &err);
	if(err) {
		SHOW_ERROR("Error creating X buffer: %d", err);
		clReleaseKernel(kernel);
		return false;
	}
	
	gy = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * y.size(), (void *)y.data(), &err);
	if(err) {
		SHOW_ERROR("Error creating Y buffer: %d", err);
		clReleaseMemObject(gx);
		clReleaseKernel(kernel);
		return false;
	}

	gp = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * params.size(), (void *)params.data(), &err);
	if(err) {
		SHOW_ERROR("Error creating param buffer: %d", err);
		clReleaseMemObject(gx);
		clReleaseMemObject(gy);
		clReleaseKernel(kernel);
		return false;
	}

	// Setting the kernel's arguments
	int ma = params.size();

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gx);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gp);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &ma);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &nd);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &extraParams);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &gy);
	if(err) {
		SHOW_ERROR("Error setting arguments: %d", err);
		clReleaseMemObject(gx);
		clReleaseMemObject(gy);
		clReleaseMemObject(gp);
		clReleaseKernel(kernel);
		return false;
	}
			
	// Actually running the kernel
    cl_event ev;
    err = clEnqueueNDRangeKernel(g_cmdq, kernel, 1, NULL, dims, locals,
                                 0, NULL, &ev);
	if(err) {
		SHOW_ERROR("Error running kernel: %d", err);
		clReleaseMemObject(gx);
		clReleaseMemObject(gy);
		clReleaseMemObject(gp);
		clReleaseKernel(kernel);
		return false;
	}

	clWaitForEvents(1, &ev);
		
	// Getting results
	err = clEnqueueReadBuffer(g_cmdq, gy, CL_TRUE, 0, sizeof(float) * dims[0], 
							  y.data(), 0, NULL, NULL);
	if(err) {
		SHOW_ERROR("Error reading results: %d", err);
		clReleaseMemObject(gx);
		clReleaseMemObject(gy);
		clReleaseMemObject(gp);
		clReleaseKernel(kernel);
		return false;
	}

	clReleaseMemObject(gx);
	clReleaseMemObject(gy);
	clReleaseMemObject(gp);
	clReleaseKernel(kernel);

	return true;
}
