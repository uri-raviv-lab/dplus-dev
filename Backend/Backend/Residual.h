
#ifndef RESIDUAL_H
#define RESIDUAL_H

#include <functional>
#include "ceres/cost_function.h"
//#include "ceres/dynamic_adaptive_numeric_diff_cost_function.h"
#include "ceres/dynamic_numeric_diff_cost_function.h"
#include "ceres/numeric_diff_options.h"
using namespace Eigen;
class Residual
{
	struct Options {
		Options(const double *x, double const* const* p, double* residual, int numResiduals) : x_(x),
			p_(p),
			residual_(residual),
			numResiduals_(numResiduals) {}
		const double *x_;
		double const* const* p_;
		double* residual_; 
		int numResiduals_;
	};
public:

	Residual(const double *x, const double* y, int numParams, int numResiduals, std::function<bool(const double*, double const* const*, double*, int, int)> calcVector,
		std::vector<double> pBestParams, double *pBestEval)
		: x_(x), y_(y), numParams_(numParams), numResiduals_(numResiduals), calcVector_(calcVector), pBestEval_(pBestEval),
		bestEval_(std::numeric_limits<double>::infinity())
	{
		Eigen::Map<VectorXd> pBestParams_(pBestParams.data(), pBestParams.size());
	}

	static ceres::CostFunction *GetCeresCostFunction(const double *x, const double *y,
		int numParams, int numResiduals, std::function<bool(const double*, double const* const*, double*, int, int)> calcVector, double stepSize = 1e-2,
		double eps = 1e-6, std::vector<double> pBestParams = std::vector<double>(), double *pBestEval = NULL)
	{
		//auto *res =
		//	new ceres::DynamicAdaptiveNumericDiffCostFunction<Residual, ceres::CENTRAL, 10, true>(
		//		new Residual(x, y, numParams, numResiduals, calcVector, pBestParams, pBestEval),
		//		ceres::TAKE_OWNERSHIP, stepSize, eps);

		ceres::NumericDiffOptions opt;
		// the previus ceres default param
		opt.ridders_step_shrink_factor = 1.4;
		opt.ridders_epsilon = 1e-6;
		opt.relative_step_size = 1e-2;
		opt.max_num_ridders_extrapolations = 10;
		auto *res =
			new ceres::DynamicNumericDiffCostFunction<Residual, ceres::RIDDERS>(
				new Residual(x, y, numParams, numResiduals, calcVector, pBestParams, pBestEval),
				ceres::TAKE_OWNERSHIP, opt);

		res->AddParameterBlock(numParams);
		res->SetNumResiduals(numResiduals);

		return res;
	}

	bool operator()(double const* const* p, double* residual) const {
		return calcVector_(x_, p, residual, numResiduals_, numParams_);

	}

	const double* operator()() const {
		return x_;
	}

private:
	std::function<bool(const double* , double const* const* ,double* , int, int )> calcVector_;
	const double *x_;
	const double *y_;
	int numResiduals_;
	int numParams_;

	VectorXd *pBestParams_;
	double *pBestEval_;
	mutable double bestEval_;
};

#endif