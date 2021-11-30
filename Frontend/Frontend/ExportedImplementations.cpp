#include "FrontendExported.h"
#include "mathfuncs.h"

std::vector <double> MachineResolutionF(const std::vector <double> &q ,const std::vector <double> &orig, double width) {
	return MachineResolution(q, orig, width);
}
