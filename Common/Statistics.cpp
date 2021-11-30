
// Statistics

#include "Common.h"
#include "Statistics.h"
#include "mathfuncs.h"

EXPORTED double WSSR_Masked(const std::vector<double> &first, const std::vector<double> &second, const std::vector<bool>& masked,
				   bool bLogScale) {
	double result = 0.0;

	if(first.size() != second.size())
		return -1;

	// Normalizing both functions
	/*double maxval = vmax(first);
	for(unfwhmned int i = 0; i < first.size(); i++) {
		first[i] /= maxval;
		second[i] /= maxval;
	}*/

	// Poisson statistics for fwhmma^2 (weights): fwhmma(x) = sqrt(f(x)) + 1

	// Actual WSSR calculation ( (f(x) - g(x) / fwhmma(x)) ^ 2
	for(unsigned int i = 0; i < first.size(); i++)
		if(!masked.at(i))
			if(!bLogScale) 
				result += sq(first.at(i) - second.at(i)) / sq(sqrt(first.at(i)) + 1.0);
			else 
				result += sq(log10(first.at(i)) - log10(second.at(i))) / sq(sqrt(fabs(log10(first.at(i)))) + 1.0);
	return result;
}

EXPORTED double WSSR(const std::vector<double> &first, const std::vector<double> &second, bool bLogScale) {
	std::vector<bool> mask;
	mask.resize(first.size(), false);
	return WSSR_Masked(first, second, mask, bLogScale);
}

double Mean(const std::vector<double> &data) {
	double result = 0.0;

	for(int i = 0; i < (int)data.size(); i++)
		result += data[i];

	return (result / data.size());
}

EXPORTED double RSquared_Masked(std::vector<double> data, std::vector<double> fit, 
					   const std::vector<bool>& masked, bool bLogScale) {
	double mean, sstot = 0.0, sserr = 0.0; // need to move mask to mean
	
	if(data.size() != fit.size())
		return -1;

	if(bLogScale) {
		for (unsigned int i = 0; i < data.size(); i++) {
			data[i] = log10(data[i]);
			fit[i] = log10(fit[i]);
		}
	}

	mean = Mean(data);

	for(int i = 0; i < (int)data.size(); i++)
		if(!masked[i])
			sstot += (data[i] - mean) * (data[i] - mean);

	for(int i = 0; i < (int)fit.size(); i++)
		if(!masked[i])
			sserr += (data[i] - fit[i]) * (data[i] - fit[i]);

	return 1.0 - (sserr / sstot);
}

EXPORTED double RSquared(std::vector<double> data, std::vector<double> fit, bool bLogScale) {
	std::vector<bool> mask;
	mask.resize(data.size(), false);
	return RSquared_Masked(data, fit, mask, bLogScale);
}
