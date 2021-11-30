#include "Externals/DESolver.h"
#include "DEFitting.h"
#include "Model.h"
#include "mathfuncs.h"
#include "Statistics.h"

double DEModelSolver::EnergyFunction(VectorXd& trial)
{
	std::vector<double> model (dataSize, 0.0);

	VectorXd dummy = FitModel->CalculateVector(x, nLayers, initialSolution);
	for(int i = 0; i < dummy.size(); i++)
		model[i] = dummy(i);
	//FitModel->PreCalculate(trial, nLayers);
	//for(int i = 0; i < dataSize; i++)
	//	model[i] = FitModel->Calculate(x[i], nLayers, dummy);

	if(props.wssrFitting)
		return WSSR(y, model, props.logScaleFitting);
	else
		return 1.0 - RSquared(y, model, props.logScaleFitting);
}

/*void DEModelSolver::calcErrors() {
	return;
}*/
