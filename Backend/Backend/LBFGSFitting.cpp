#include "LBFGSFitting.h"
#include <ctime>

#include "liblbfgs/lbfgs.h"

#define MAX_EVALS_PER_ITERATION 10000


LBFGSFitter::LBFGSFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
           const std::vector<double>& datay,
           const std::vector<double>& factor, 
           const std::vector<double>& bg,
           const std::vector<double>& fitWeights, VectorXd& p,
           const VectorXi& pmut, cons *pMin, cons *pMax,
		   int layers)  : ModelFitter(model, fp, datax, datay, factor, bg,
                                    fitWeights, p, pmut, pMin, pMax, layers) {
	h_lbfgs = new lbfgs_parameter_t;
	//maxIters = 0; // Tells the fitter to work to convergence
	maxIters = 20;

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init((lbfgs_parameter_t *)h_lbfgs);
	((lbfgs_parameter_t *)h_lbfgs)->max_iterations = maxIters;
	//((lbfgs_parameter_t *)h_lbfgs)->epsilon = 1e-7;

	params = p;
	curEval = Evaluate(p);

	/*
	printf("Init Parameters (eval=%f): ", curEval);
	for(int i = 0; i < p.size();i++)
		printf("%f%s, ", p[i], paramMut[i] ? " (M)" : "");
	printf("\n");
	*/

	// TODO: No clue what this means
	delPmax = delPmin = false;
}

double LBFGSFitter::FitIteration() {
    VectorXd curParams = params;
	VectorXd pnew;

	if(mutables == 0 || GetError()) {
		return (curEval = Evaluate(curParams));
	}

	pnew = curParams;

	// Apply/enforce local constraints
	if(!EnforceConstraints(pnew)) {
		return (curEval = Evaluate(pnew)); // If cannot enforce constraints
	}

    int ret = 0;
    lbfgsfloatval_t fx;
    
	// Only use the mutable parameters
	lbfgsfloatval_t *pdata = lbfgs_malloc(mutables);
	int j = 0;
	for(int i = 0; i < nParams; i++)
		if(paramMut[i])
			pdata[j++] = pnew[i];
/*
	printf("IN");
	for(int i = 0; i < mutables; i++)
		printf(", p[%d] = %f", i, pdata[i]);
	printf("\n");
*/

    // Start the L-BFGS optimization; this will invoke the callback functions
    // evaluate() and progress() when necessary.
	ret = lbfgs(mutables, pdata, &fx, &LBFGSFitter::LBEvaluate, NULL, this, (lbfgs_parameter_t *)h_lbfgs);

    /* Report the result. */
    printf("L-BFGS optimization terminated with status code = %d\n", ret);
    printf("  fx = %f", fx);
	for(int i = 0; i < mutables; i++)
		printf(", p[%d] = %f", i, pdata[i]);
	printf("\n");

	// Rewrite the parameters back
	j = 0;
	for(int i = 0; i < nParams; i++)
		if(paramMut[i])
			pnew[i] = pdata[j++];

	lbfgs_free(pdata);

	params = pnew;
	curEval = fx;

    return fx;
}

inline double derF(ModelFitter *obj, VectorXd& p, int ai, double h) {
	double result = 0.0;
	double porig = p[ai];
	p[ai] += h;
	result = obj->Evaluate(p);
	p[ai] = porig;

	return result;
}

void NumericalGradient(ModelFitter *obj, VectorXd& p, const VectorXi& pMut, int nLayers,
					   lbfgsfloatval_t *g) 
{
	double h = 1.0e-10;

	// f'(x) ~ [f(x-2h) - f(x+2h)  + 8f(x+h) - 8f(x-h)] / 12h
	int nParams = p.size();

	int j = 0;
	for(int i = 0; i < nParams; i++) 
	{
		if(!pMut[i])
			continue;

		g[j] = 0.0;
		while(fabs(g[j]) < 1.0e-5) //1e-5 is the gnorm epsilon in the LBFGS algorithm
		{
			g[j]  = (1.0 / (12.0 * h)) * derF(obj, p, i, -2.0 * h);
			g[j] += (8.0 / (12.0 * h)) * derF(obj, p, i,  h      );
			g[j] -= (8.0 / (12.0 * h)) * derF(obj, p, i, -h      );
			g[j] -= (1.0 / (12.0 * h)) * derF(obj, p, i,  2.0 * h);

			// Try to perform coarser derivation until h is too large
			h *= 100.0;
			if(h > 10.0)
				break;
		}
		
		j++;
	}
}

lbfgsfloatval_t LBFGSFitter::LBEvaluate(void *instance, const lbfgsfloatval_t *x,
										lbfgsfloatval_t *g, const int n,
										const lbfgsfloatval_t step) {
	LBFGSFitter *opt = (LBFGSFitter *)instance;	
	if(!opt)
		return POSINF;

	IModel *obj = opt->FitModel;
	
	// Create the full candidate vector
	VectorXd trial = opt->params;
	int j = 0;
	for(int i = 0; i < opt->nParams; i++)
		if(opt->paramMut[i])
			trial[i] = x[j++];

	// Apply/enforce local constraints
	if(!opt->EnforceConstraints(trial)) {
		// If cannot enforce constraints, produce a bad value with 0 gradient (plateau)
		for(int i = 0; i < n; i++)
			g[i] = 0.0;
		return opt->Evaluate(trial) * 1.0e10; 
	}

	// Evaluate the function
	double eval = opt->Evaluate(trial);

	// Compute the gradient numerically
	NumericalGradient(opt, trial, opt->paramMut, opt->nLayers, g);

	printf("EVAL = %f\n", eval);
	for(int i = 0; i < n; i++)
		printf("GRADIENT[%d] = %f\n", i, g[i]);

    return eval;
}

static int lb_progress(void *instance, const lbfgsfloatval_t *x,
					   const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
					   const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
					   const lbfgsfloatval_t step, int n, int k, int ls) {
    
	/*printf("Iteration %d:\n", k);
	printf("  fx = %f", fx);
	for(int i = 0; i < n; i++)
		printf(", p[%d] = %f", i, x[i]);
	printf("\n");
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");*/

    return 0;
}

LBFGSFitter::~LBFGSFitter()
{
	if(h_lbfgs)
		delete ((lbfgs_parameter_t *)h_lbfgs);

	if(delPmin)
		delete p_min;
	if(delPmax)
		delete p_max;
}
