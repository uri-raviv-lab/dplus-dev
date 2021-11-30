#pragma once

#include "GPUInterface.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif


class GPUDirectCalculator : public IGPUCalculator
{
public:
	GPUDirectCalculator();
    virtual ~GPUDirectCalculator();
    
    virtual bool Initialize(int gpuID, const float2 *angles, size_t numAngles,
							const float *qPoints, size_t numQ,
							size_t maxNumAtoms, size_t maxNumCoeffs, 
							size_t maxTranslations, Workspace& res);


    virtual bool TranslateWorkspace(Workspace& workspace, float3 *translations, unsigned int numTrans);

    
    virtual bool ComputeIntensity(Workspace *workspaces, unsigned int numWorkspaces, double *outData);

	virtual bool FreeWorkspace(Workspace& workspace);
};

