#pragma once

#include "GPUInterface.h"


class GPUJacobainDomainCalculator : public IGPUGridCalculator
{
public:
	GPUJacobainDomainCalculator();
	virtual ~GPUJacobainDomainCalculator();

	// IGPUGridCalculator methods
	virtual bool Initialize(int gpuID, const std::vector<float>& qPoints,
		long long totalSize, int thetaDivisions, int phiDivisions, int qLayers,
		double qMax, double stepSize, GridWorkspace& res);

	virtual bool FreeWorkspace(GridWorkspace& workspace);

	// 
	int AssembleAmplitudeGrid(GridWorkspace& workspace, double **subAmp,
		float **subInt, double **transRot, int numSubAmps);
	
	int CalculateSplines(GridWorkspace& workspace);

	int OrientationAverageMC(GridWorkspace& workspace, long long maxIters,
						double convergence,  double *qVals, double *iValsOut);

};
