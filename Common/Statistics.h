#ifndef __STATISTICS_H
#define __STATISTICS_H

// General statistic functions

double Mean(const std::vector<double> &data);

/* The following four functions are declared as EXPORTED in FronendExported.h*/
#ifdef BACKEND
double WSSR(const std::vector<double> &first, const std::vector<double> &second, bool bLogScale = false);

double RSquared(std::vector<double> data, std::vector<double> fit, bool bLogScale = false);

/*
double WSSR(const Eigen::VectorXd &first, const Eigen::VectorXd &second, bool bLogScale = false);

double RSquared(const Eigen::VectorXd& data, const Eigen::VectorXd& fit, bool bLogScale = false);
*/

double WSSR_Masked(const std::vector<double> &first, const std::vector<double> &second, const std::vector<bool>& masked,
							bool bLogScale = false);

double RSquared_Masked(std::vector<double> data, std::vector<double> fit, 
								const std::vector<bool>& masked, bool bLogScale = false);
#endif

#endif
