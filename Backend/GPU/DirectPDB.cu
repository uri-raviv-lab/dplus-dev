#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "CommonCUDA.cuh"
#include "GPUDirectCalc.cuh"

#include <cuda_runtime.h>

__constant__ float dev_a1[208];
__constant__ float dev_b1[208];
__constant__ float dev_a2[208];
__constant__ float dev_b2[208];
__constant__ float dev_a3[208];
__constant__ float dev_b3[208];
__constant__ float dev_a4[208];
__constant__ float dev_b4[208];
__constant__ float dev_c [208];

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

#ifdef _DEBUG
	printf("idx=%d\tq=%f\tnumCoeffs=%u\tsqq=%f\tnumQ=%u\n\taffs[idx]=%f\n", idx, q, numCoeffs, sqq, numQ, affs[idx]);
#endif // _DEBUG
}

bool GPUDirect_SetPDB(Workspace& work, const float4 *atomLocs,
                      const unsigned char *ionInd,
					  size_t numAtoms, const float *coeffs,
					  const unsigned int *atomsPerIon,
                      size_t numCoeffs)
{
    cudaError_t err = cudaSuccess;

	cudaStream_t stream = (cudaStream_t)work.stream;
 
	CHKERR(cudaGetLastError());

	CHKERR(cudaSetDevice(work.gpuID));

	work.numAtoms = numAtoms;
	work.numCoeffs = numCoeffs;
    
    // Copy the PDB memory to the device
    CHKERR(cudaMemcpyAsync(work.d_atomLocs, atomLocs, numAtoms * sizeof(float4),
                           cudaMemcpyHostToDevice, stream));

	// No stream support for now (no gain, no pain)
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.stream));

	CHKERR(cudaMemcpy(work.h_atomsPerIon, atomsPerIon, numCoeffs * sizeof(unsigned int),
                      cudaMemcpyHostToHost));

    if(coeffs)
    {
		// if we ever add streams, think about the case where two or more streams coexist and compute different PDBs
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
		dimGrid.x = ((work.numQ % BLOCK_WIDTH == 0) ? (work.numQ / BLOCK_WIDTH) : 
						(work.numQ / BLOCK_WIDTH + 1));
		dimGrid.y = 1;

		TRACE_KERNEL("AtomFormFactorKernel");
		AtomicFormFactorKernel <<<dimGrid, dimBlock, 0, stream >>>(work.d_qPoints, work.numQ, work.d_affs, numCoeffs);

		CHKERR(cudaGetLastError());
    }
    
	// No stream support for now (no gain, no pain)
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.stream));

    return true;
}

__global__ void PDBAmplitudeKernel(const float2 *angles, const float4 *atomLocs,
									  const float *gAffs,
									  unsigned char ionInd, 
									  unsigned int numAtoms, 
									  const float *qPoints,
									  float3 rot,
									  unsigned int numAngles, unsigned int numQ,
									  unsigned int ampPitch, double2 *outAmp, const bool bAppend)
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
		affs[threadIdx.y] = gAffs[ionInd * numQ + y];
	}

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

	// Reuse
	//sinth = 1.0f;
	//sinth = sqrtf(sinth);

	float2 res = make_float2(0.0f, 0.0f);
		
	float4 coord;
	float phase;
	float sn, cs;

	__syncthreads();
	
	for(int i = 0; i < numAtoms; i++)
	{
		coord = atomLocs[i];
		phase = qx * coord.x + qy * coord.y + qz * coord.z;
			
		sincosf(phase, &sn, &cs);

		res.x += cs;
		res.y += sn;			
	}				

	

	double2* pElement = (double2*)((char*)outAmp + y * ampPitch) + x;

	//sinth *= affs[threadIdx.y]; //gAffs[ionInd * numQ + y];
	sinth = affs[threadIdx.y];

	if(bAppend)
	{
		pElement->x += res.x * sinth;
		pElement->y += res.y * sinth;
	}
	else
	{
		pElement->x = res.x * sinth;
		pElement->y = res.y * sinth;
	}
}

__global__ void RotateAtomsKernel(float4 *atomLocs, unsigned int numAtoms, 
								  float3 rotmat0, float3 rotmat1, float3 rotmat2,
								  float4 *outAtomLocs)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	
	if(x >= numAtoms)
		return;

	/*
	if(x == 0)
	{
		printf("DEV: %f %f %f %f %f %f %f %f %f\n", rotmat0.x, rotmat0.y, rotmat0.z,
		   rotmat1.x, rotmat1.y, rotmat1.z, rotmat2.x, rotmat2.y, rotmat2.z);
	}*/

	float4 a = atomLocs[x];
	float4 outLoc;
	
	outLoc.x = rotmat0.x * a.x + rotmat0.y * a.y + rotmat0.z * a.z;
	outLoc.y = rotmat1.x * a.x + rotmat1.y * a.y + rotmat1.z * a.z;
	outLoc.z = rotmat2.x * a.x + rotmat2.y * a.y + rotmat2.z * a.z;

	outAtomLocs[x] = outLoc;
}

#define DEG2RAD(DEG) ((DEG)*((M_PI)/(180.0f)))

bool GPUDirect_PDBAmplitude(Workspace& work, float3 rotation)
{
	cudaError_t err = cudaSuccess;
	
	bool bShouldRotate = (fabs(rotation.x) > 1e-9 || fabs(rotation.y) > 1e-9 || fabs(rotation.z) > 1e-9);

	cudaStream_t stream = (cudaStream_t)work.stream;

	if(bShouldRotate)
	{
		// Compute rotation matrix
		float sx = sinf(rotation.x), cx = cosf(rotation.x);
		float sy = sinf(rotation.y), cy = cosf(rotation.y);
		float sz = sinf(rotation.z), cz = cosf(rotation.z);
	
		float3 rotmat[3];
		rotmat[0].x = cy * cz;        rotmat[0].y = -cy * sz;       rotmat[0].z = sy;
		rotmat[1].x = cx*sz+cz*sx*sy; rotmat[1].y = cx*cz-sx*sy*sz; rotmat[1].z = -cy*sx;
		rotmat[2].x = sx*sz-cx*cz*sy; rotmat[2].y = cz*sx-cx*sy*sz; rotmat[2].z = cx*cy;

		/*printf("Rotation: %f, %f, %f\n", rotation.x, rotation.y, rotation.z);
		printf("HOST:\n \t%f %f %f \n\t%f %f %f \n\t%f %f %f\n", rotmat[0].x, rotmat[0].y, rotmat[0].z,
			  rotmat[1].x, rotmat[1].y, rotmat[1].z, rotmat[2].x, rotmat[2].y, rotmat[2].z);
		printf("Num atoms: %d\n", work.numAtoms);*/


		dim3 dimRotGrid, dimRotBlock;
		dimRotBlock.x = 256;
		dimRotGrid.x = ComputeGridSize(work.numAtoms, 256);

		TRACE_KERNEL("RotateAtomsKernel");
		RotateAtomsKernel <<<dimRotGrid, dimRotBlock, 0, stream >>>(work.d_atomLocs, work.numAtoms,
																  rotmat[0], rotmat[1], rotmat[2], work.d_rotAtomLocs);
	}

	cudaStreamSynchronize(stream);

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = BLOCK_HEIGHT;
	dimGrid.x = ((work.numAngles % BLOCK_WIDTH == 0) ? (work.numAngles / BLOCK_WIDTH) : 
					(work.numAngles / BLOCK_WIDTH + 1));
	dimGrid.y = ((work.numQ % BLOCK_HEIGHT == 0) ? (work.numQ / BLOCK_HEIGHT) : 
					(work.numQ / BLOCK_HEIGHT + 1));


	unsigned int blockStart = 0;
	for(int i = 0; i < work.numCoeffs; i++)
	{
		TRACE_KERNEL("PDBAmplitudeKernel");
		PDBAmplitudeKernel<<<dimGrid, dimBlock, 0, (cudaStream_t)work.stream>>>(work.d_angles, 
													(bShouldRotate ? work.d_rotAtomLocs : work.d_atomLocs) + blockStart, work.d_affs, i,
													 work.h_atomsPerIon[i],
													 work.d_qPoints,
													 rotation, 
													 work.numAngles, work.numQ,
													 work.workPitch, work.d_work, (i != 0));
		blockStart += work.h_atomsPerIon[i];
	}

	// No stream support for now (no gain, no pain)
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.stream));

	/*
	double2 *amps = new double2[work.numAngles * work.numQ];
	cudaMemcpy2D(amps, work.numAngles * sizeof(double2), work.d_work, work.workPitch, work.numAngles, work.numQ, cudaMemcpyDeviceToHost);

	float *qPoints = new float[work.numQ];
	cudaMemcpy(qPoints, work.d_qPoints, sizeof(float) * work.numQ, cudaMemcpyDeviceToHost); 

	FILE *fp = fopen("S:\\BLAH.dat", "wb");
	if(fp)
	{
		for(int i = 0; i < work.numQ; i++)
		{
			fprintf(fp, "%f: ", qPoints[i]);
			for(int j = 0; j < work.numAngles; j++)
			{
				fprintf(fp, "%lf,%lf ", amps[i * work.numAngles + j].x, amps[i * work.numAngles + j].y);
			}
			fprintf(fp, "\n");
		}

		fclose(fp);
	}*/

	CHKERR(cudaGetLastError());

    return true;
}


