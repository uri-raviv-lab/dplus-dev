#include <cstdlib>
#include <limits>
#include <sstream>
#include <iostream>	// cout
#include <stdio.h>	// fprintf
#include "Fitter.h"
#include "exportedFuncs.h"

#include "Eigen/LU"
#include "Eigen/SVD"

using namespace Eigen;

//////////////////////////////
// Copied functions from X+ //
//////////////////////////////
#pragma region Copied functions
double Mean(ArrayXd& data) {
	double result = 0.0;

	for(int i = 0; i < (int)data.size(); i++)
		result += data[i];

	return (result / data.size());
}
#pragma endregion

ArrayXd PopulationFitter::Calc() {
	return pm->CalculateExp();
	ArrayXd res = ArrayXd::Zero(intensities.rows());
}
double PopulationFitter::Calc(int ind) {
	double res = 0.0;

	for(int i = 0; i < params.size(); i++) {
		res += params(i) * intensities(i, ind);	// TODO: Make sure that I didn't mean (ind, i)
	}

	return res;
}

void LMPopFitter::calcErrors() {
	
	//VectorXd tmp = alpha.inverse().diagonal();
	lambda = 0.0;

	FitIteration();

	MatrixXd bl, bltmp(J.rows(), mutables);
	VectorXd blah;
	VectorXd tmp = alpha.diagonal();

	*err = (tmp.array().abs()).sqrt();

	if(errY) {
		int j = 0;
		for (int i = 0; i < J.cols(); i++) {
			if(!J.col(i).isZero())
				bltmp.col(j++) = J.col(i);
		}

		bl = bltmp * alpha * bltmp.transpose();
		blah = bl.diagonal();

		*errY = blah.array().abs().sqrt();
	}

}

double LMPopFitter::FitIteration() {
	VectorXd curParams, curBeta;
	MatrixXd curAlpha;

	if(mutables == 0 || GetError())
		return 0.0;

	// We need to solve the linear equation system curAlpha*delta = curBeta, where:
	// alpha = J'J                       where J is the Jacobian Matrix and J' is its transpose
	// curAlpha = J'J + lambda*diag(J'J)
	// curBeta  = J'(y-f(b))             where b is the initial guess vector, and y is the data

    curAlpha = alpha;
	curAlpha.diagonal() = alpha.diagonal() * (1.0 + lambda);
       
	curBeta = beta;


	//////////////////////
	// SVD only for now //
	//////////////////////
	try {
		// I don't know if using ComputeFullU/V would be better 
		JacobiSVD<MatrixXd> svd(curAlpha, ComputeThinU  | ComputeThinV );

		MatrixXd sigma = MatrixXd::Zero(mutables, mutables);
		sigma.diagonal() = svd.singularValues();
		
		for(int i = 0; i < mutables; i++)
			if(sigma(i, i) != 0.0)
				sigma(i, i) = 1.0 / sigma(i, i);

		curAlpha = svd.matrixV() * sigma * svd.matrixU().transpose();
		curBeta = curAlpha * curBeta;
	} catch(...) {
		// Invalid matrix values
		error = true;
		return -1.0;
	}

	// Testing the results of the equation solution
	curParams = params;
	
	int j = 0;
    for (int i = 0; i < nParams; i++)
        if (paramMut[i]) 
			curParams[i] = params[i] + curBeta[j++];

	if(fabs(lambda) < 1e-14)
		alpha = curAlpha;

	double wssr = CalculateCoefficients(curParams, curAlpha, curBeta);

	// If better, keep them, otherwise, don't, and increase damping factor
    if (wssr < curWssr && wssr >= 0.0) {
        lambda *= 0.1;
        curWssr = wssr;

		alpha = curAlpha;
        beta = curBeta;
        params = curParams;

		resY = resX;
    } 
    else
		lambda *= 10.0;

	return curWssr;
}

double LMPopFitter::CalculateCoefficients(VectorXd& p, MatrixXd& alphaMat,
                                       VectorXd& betaVec) {
    double wssr = 0.0, mean = 0.0, sstotal = 0.0, tmpRsq = 0.0;

	int size = y.size();

	VectorXd dy = VectorXd::Zero(y.size());
    MatrixXd dyda = MatrixXd::Zero(y.size(), nParams);

    betaVec.setZero();
    
    for (int i = 0; i < mutables; i++) 
        for (int j = 0; j <= i; j++) 
            alphaMat(i, j) = 0.0;

	// Link the Linked parameters 
	if (p_max) {
		for(int b = 0; b < p_max->link.size(); b++) {
			if(p_max->link[b] >= 0 && p_max->link[b] != b)
				p[b] = p[p_max->link[b]];
		}
	}
    
    // Constraints ("Penalty Method")
    for (int k = 0; k < nParams; k++) {
        if(p_min && p_max) {
            if(!(fabs(p_min->num[k] - p_max->num[k]) <= 1e-6)) {
                if (((p[k] < 0.0 && p_min->num[k] >= 0) || 
                     (p_min->num[k] > p[k]) ||
                     (p_max->num[k] < p[k]) ||  
                     (p_min->index[k] >= 0 && p[p_min->index[k]] > p[k])   ||
                     (p_max->index[k] >= 0 && p[p_max->index[k]] < p[k]) )  && paramMut[k]  ) {
                    return -1.0;
                }
            }
            
        }
    }

	ArrayXd ResY = y;

	if(isLogFitting())
		ResY = ResY.log();

    // Calculating the R-Squared coefficients
    if(true/*!isWSSRFitting()*/) {	// TODO: Uncomment when reintegrating
		mean = Mean(ResY);	

		for(int i = 0; i < size; i++)
			sstotal += (ResY[i] - mean) * (ResY[i] - mean);
	
    }
    
	ArrayXd pDummy = p.array();

	pm->ChangeCoefficients(pDummy);
	resX = Calc();

	if(isLogFitting())
		dy = (ResY.array() - resX.log()).matrix();
	else
		dy = (ResY.array() - resX).matrix();

    // Calculate trial alpha and beta coefficients according to the
    // gradient
    int j = 0;
    for (int l = 0; l < nParams; l++) {
        if (paramMut[l]) {
			// Partial derivative calculation
			dyda.col(l) = Derivative(p, l);

            for(int i = 0; i < size; i++) {
                
				double wt = dyda(i, l) / sqWeights[i];
                
                int k = 0;
                for (int m = 0; m <= l; m++) {
                    if(paramMut[m]) { 
                        alphaMat(j, k) += wt * dyda(i, m);
                        k++;
                    }
                }
                betaVec[j] += dy[i] * wt;
            }
            j++;
        }
    }
    
	J = dyda;

    // Makes matrix symmetric
    for (int i = 1; i < mutables; i++)
        for (int k = 0; k < i; k++) 
            alphaMat(k, i) = alphaMat(i, k);

    // Calculate WSSR
	mse = 0.0;
	for(int i = 0; i < size; i++) {
		mse += (dy[i] * dy[i]);
		wssr += (dy[i] * dy[i]) / sqWeights[i];
		tmpRsq += (dy[i] * dy[i])  / sstotal;
		//if(true/*isWSSRFitting()*/)	// TODO: return condition when reintegrating
		//	wssr += (dy[i] * dy[i]) / sqWeights[i];
  //      else
		//	wssr += (dy[i] * dy[i])  / sstotal;
	}
	if (wssr < curWssr && wssr >= 0.0) {
		curRsq = tmpRsq;
		(*modelEr)(0) = wssr;
		(*modelEr)(1) = 1.0 - curRsq;
	}

    mse /= double(size);
    return wssr;
}


ArrayXd PopulationFitter::Derivative(VectorXd param, int ai) {
	return pm->Derivative(tmpX, param, nLayers, ai);
}
