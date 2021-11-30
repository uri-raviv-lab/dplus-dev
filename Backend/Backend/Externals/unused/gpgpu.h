#ifndef __GPGPU_H
#define __GPGPU_H

#include "Eigen/Core"
using namespace Eigen;

void InitializeOpenCL();
void DestroyOpenCL();

bool GenerateGPUModel(const char *modelName, const VectorXf& x, const VectorXf& params, VectorXf& y, int nd, int extraParams);

#endif
