#ifndef __PREP_H__
#define __PREP_H__

#include "Common.h"

#include <string>
#include <vector>

#include <vector_types.h>

bool PrepareAllParameters(
	std::string inFilename,
	std::vector<int>	&atomsPerIon,
	std::vector<float4>	&loc, 
	std::vector<char>	&atmInd, 
	std::vector<u8>	&ionInd, 
	std::vector<float>	&BFactors, 
	std::vector<float>	&coeffs, 
	std::vector<float>	&atmRad
	);


#endif