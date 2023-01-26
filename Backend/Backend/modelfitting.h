#ifndef __MODELFITTING_H
#define __MODELFITTING_H

#include <vector>
#include "Common.h"
#include "Model.h"
#include "../../BackendCommunication/LocalCommunication/LocalComm.h"

// Forward declarations
class LocalBackend;

struct fitJobArgs {
	LocalBackend *backend;
	JobPtr jobID;

	std::vector<double> x, y;
	std::vector<int> mask;
	FittingProperties fp;
};

ErrorCode PerformModelFitting(fitJobArgs *args);

ErrorCode PerformModelGeneration(fitJobArgs *args);
ErrorCode PerformModelGeneration2D(fitJobArgs* args);

#endif // __MODELFITTING_H

