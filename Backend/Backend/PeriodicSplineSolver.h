#ifndef _PERIODIC_SPLINE_SOLVER__H
#define _PERIODIC_SPLINE_SOLVER__H

#pragma once
#include "Eigen/Core"

template <typename ComplexRealType, typename ResComplexRealType>
void EvenlySpacedPeriodicSplineSolver(const ComplexRealType *points, ResComplexRealType *res, unsigned long long sz) {
	typedef Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> vect;
	// std::complex<float> doens't have operator* for double and vise versa
	typedef typename std::conditional<sizeof(ComplexRealType) == 2*sizeof(double), double, float>::type fType;

	vect as;
	as.resize(sz);

	// Initialize  "matrix"
	Eigen::VectorXf diag, rightCol;
	diag = Eigen::VectorXf::Constant(sz, 4.0f);
	rightCol = Eigen::VectorXf::Constant(sz-1, 0.0f);

	// Initialize points
	for(int i = 0; i < sz; i++) {
		as(i) = fType(3.0) * (points[(i+1)%sz] - points[(sz+i-1)%sz]);
	}

	/************************************************************************/
	/* Calculate the spline coefficients                                    */
	/************************************************************************/ 
	fType fac = 0.0;
	// Remove bottom off diagonal set of ones
	rightCol(0) = 1.0f;
	rightCol(sz-2) = 1.0;
	fType vp = 0.25;
	fType vc;
	for(int i = 1; i < sz - 1; i++) {
		vc = 1.0 / (diag(i) - vp);
		diag(i) -= vp;
		rightCol(i) -= vp * rightCol(i-1);
		as (i) -= as(i-1) * vp;
		vp = vc;
	}
	diag(sz-1) -= vp * rightCol(sz-2);
	as (sz-1) -= vp * as(sz-2);

	// Remove left most value from bottom row
	float bottomLeft = 1.0f;
	for(int i = 0; i < sz-1; i++) {
		vc = bottomLeft / (diag(i));
		diag(sz-1) -= vc * rightCol(i);
		as (sz-1) -= vc * as (i);
		bottomLeft = - vc;
	}

	// Backwards substitution
	fac = (rightCol(sz-2) / diag(sz-1));
	as(sz-2) -= (as(sz-1)) * fac;	// Second to bottom row
	for(int i = sz-3; i >= 0; i--) {
		as(i) -= as(sz-1) * fType(rightCol(i) / diag(sz-1));	// Remove right column
		as(i) -= as( i+1) * fType(1.0 / diag(i+1));	// Remove the upper off diagonal
	}

	// Get the coefficients
	for(int i = 0; i < sz; i++) {
		res[i] = as(i) / fType(diag(i));
	}

}

/**
 * @name	EvenlySpacedPeriodicSplineSolver
 * @brief	Calculates the cubic spline of a evenly spaced periodic table of function values.
	 The logic follows that layed down in "An Introduction to Splines for Use in Computer
	 Graphics and Geometric Modeling" (Bartels, 1987, ch 3.4)

 * @param	ComplexRealType n array of type {float, double, std::complex<float/double>}
			representing the values of the function upon which the spline is to be calculated.
			No checks are done to ensure that the array exists. Make sure it is of length
			sz*sizeof(ComplexRealType).
 * @param	sz The length of the array.
 * @ret		An Eigen::VectorXt where 't' is {f,d,cf,cd} of the solution of the periodic cubic spline
 */
template <typename ComplexRealType>
Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> 
	EvenlySpacedPeriodicSplineSolver(const ComplexRealType *points, unsigned long long sz) {
		typedef Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> vect;

		vect res;
		res.resize(sz);
		EvenlySpacedPeriodicSplineSolver(points, &res[0], sz);

		return res;
}


template <typename ComplexRealType, typename ResComplexRealType>
void EvenlySpacedSplineSolver(const ComplexRealType *points, ResComplexRealType *res, unsigned long long sz) {
		typedef Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> vect;
		// std::complex<float> doens't have operator* for double and vise versa
		typedef typename std::conditional<sizeof(ComplexRealType) == 2*sizeof(double), double, float>::type fType;

		vect as;
		as.resize(sz);

		// Initialize  "matrix"
		Eigen::VectorXf diag;
		diag = Eigen::VectorXf::Constant(sz, 4.0f);
		diag[sz-1] = diag[0] = 2.0f;

		// Initialize points
		as(0) = fType(3.0) * (points[1] - points[0]);
		for(int i = 1; i < sz-1; i++) {
			as(i) = fType(3.0) * (points[i+1] - points[i-1]);
		}
		as(sz-1) = fType(3.0) * (points[sz-1] - points[sz-2]);

		/************************************************************************/
		/* Calculate the spline coefficients                                    */
		/************************************************************************/ 
		fType fac = 0.0;
		// Remove bottom off diagonal set of ones
		fType vp = 0.5, vc;
		for(int i = 1; i < sz - 1; i++) {
			vc = 1.0 / (diag(i) - vp);
			diag(i) -= vp;
			as (i) -= vp * as(i-1);
			vp = vc;
		}
		diag(sz-1) -= vp;
		as (sz-1) -= vp * as(sz-2);

		// Backwards substitution
/*
		fac = (1.0f / diag(sz-1));
		as(sz-2) -= as(sz-1) * fac;	// Second to bottom row
*/
		for(int i = sz-2; i >= 0; i--) {
			as(i) -= as( i+1) * fType(1.0 / diag(i+1));	// Remove the upper off diagonal
		}

		// Get the coefficients
		for(int i = 0; i < sz; i++) {
			res[i] = as(i) / fType(diag(i));
		}
}

//************************************
// Method:    EvenlySpacedSplineSolver
// FullName:  EvenlySpacedSplineSolver
// Description:     Calculates the cubic spline of an evenly spaced table of function values. The logic		\
					follows that layed down in "An Introduction to Splines for Use in Computer Graphics and	\
					Geometric Modeling" (Bartels, 1987, ch 3.1)
// Returns:   An Eigen::VectorXt where 't' is {f,d,cf,cd} of the solution of the periodic cubic spline
// Parameter: const ComplexRealType * points An array of type {float, double, std::complex<float/double>}	\
					representing the values of the function upon which the spline is to be calculated.		\
					No checks are done to ensure that the array exists. Make sure it is of length			\
					sz*sizeof(ComplexRealType).
// Parameter: unsigned long long sz The length of the array.
//************************************
template <typename ComplexRealType>
Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> 
	EvenlySpacedSplineSolver(const ComplexRealType *points, unsigned long long sz) {
		typedef Eigen::Matrix<ComplexRealType,Eigen::Dynamic,1> vect;

		vect res;
		res.resize(sz);

		EvenlySpacedSplineSolver(points, &res[0], sz);
		return res;
}

/**	
	@name	EvenlySpacedFourPointSpline
	Calculates the cubic spline of four evenly spaced function values. The logic
	follows that layed down in "An Introduction to Splines for Use in Computer Graphics and
	Geometric Modeling" (Bartels, 1987, ch 3.1). In short, solves
	\f[
		\begin{bmatrix} 2 & 1 & 0 & 0 \\ 1 & 4 & 1 & 0 \\ 0 & 1 & 4 & 1 \\0 & 0 & 1 & 2
		\end{bmatrix} \cdot
		\begin{bmatrix} D_1 \\ D_2 \\ D_3 \\ D_4 \end{bmatrix}
		=
		\begin{bmatrix} 3\left(y_2-y_1\right) \\ 3\left(y_3-y_1\right) \\
		3\left(y_4-y_2\right) \\ 3\left(y_4-y_3\right)
		\end{bmatrix}
	\f]
	gives
	\f[
		\begin{bmatrix} D_1 \\ D_2 \\ D_3 \\ D_4 \end{bmatrix} =
		\frac{1}{15} \begin{bmatrix}
		26 (y2-y1) & -7 (y2-y1) & 2 (y2-y1) &   y1-y2 \\
		-7 (y3-y1) & 14 (y3-y1) & -4 (y3-y1) & 2 (y3-y1) \\
		2 (y4-y2) & -4 (y4-y2) & 14 (y4-y2) & -7 (y4-y2) \\
		y3-y4 & 2 (y4-y3) & -7 (y4-y3) & 26  (y4-y3)
		\end{bmatrix}
	\f]
	so
	\f[
		\begin{aligned}
		D_2 &= \frac{
		-7 (y3-y1) & 14 (y3-y1) & -4 (y3-y1) & 2 (y3-y1)
		}{15}\\
		D_3 &= \frac{
		2 (y4-y2) & -4 (y4-y2) & 14 (y4-y2) & -7 (y4-y2)
		}{15}
		\end{aligned}
	\f]
			
	Values for \f$x\f$ between points 2 and 3 are calculated as
	\f[
	y\left(x\right) = y_2 + D_2 t + 
	\left(3 \left(y_3-y_2\right) - 2 D_2 - D_3\right) t^2 +
	\left(2 \left(y_2-y_3\right) + D_2 + D_3\right) t^3
	\f]
	where 
	\f[
		t=\frac{x-x_2}{x_3-x_2}
	\f]

	@param[in]  y1 The first value
	@param[in]  y2 The second value
	@param[in]  y3 The third value
	@param[in]  y4 The fourth value
	@param[out] d2 The first resulting value used to interpolate between points y2 and y3
	@param[out] d3 The second resulting value used to interpolate between points y2 and y3
	@tparam ComplexRealType The type of the input can be real (double/float) or complex<double/float>
	@tparam ResComplexRealType The type of the result can be real (double/float) or complex<double/float>
*/
template <typename ComplexRealType, typename ResComplexRealType>
void EvenlySpacedFourPointSpline(const ComplexRealType y1, const ComplexRealType y2, const ComplexRealType y3,
								 const ComplexRealType y4, ResComplexRealType *d2, ResComplexRealType *d3) {
	typedef typename std::conditional<sizeof(ComplexRealType) == 2*sizeof(double), double, float>::type fType;
									 
	*d2 = -fType(7. / 15.)*(y2 - y1) + fType(14. / 15.)*(y3 - y1) - fType(4. / 15.)*(y4 - y2) + fType(2. / 15.)*(y4 - y3);
	*d3 = fType(2. / 15.)*(y2 - y1) - fType(4. / 15.)*(y3 - y1) + fType(14. / 15.)*(y4 - y2) - fType(7. / 15.)*(y4 - y3);
}


/**
@name    TempertonEvenlySpacedPeriodicSplineSolver
Calculates the cubic spline of an evenly spaced table of function values. The logic is   
that of Clive Temperton (Algorithms for the Solution of Cyclic Tridiagonal Systems, 1975)
with the exception of his term for \sigma which is doens't work. The rest of the solution
is the basic Tridiagonal matrix algorithm (Thomas algorithm) assuming evenly spaced and  
cyclic conditions

@tparam TData either real or complex 
@tparam TCoeffs either real or complex
@ret   void
@param const TData* data An array of type {float, double, std::complex<float/double>}
					representing the values of the function upon which the spline is to be calculated.
					No checks are done to ensure that the array exists. Make sure it is of length
					sz*sizeof(ComplexRealType).
 @param TCoeffs* coeffs The pointer to where the results are to be written. Must be preallocated.
 @param int matSize The length of the array.
*/


//template <typename TData, typename TCoeffs>
//void TempertonEvenlySpacedPeriodicSplineSolver(const TData* data, TCoeffs* coeffs, int matSize) {
//	typedef typename std::conditional<sizeof(TData) == 2*sizeof(double), double, float>::type fType;
//	typedef typename std::conditional<sizeof(TCoeffs) == 2*sizeof(double), double, float>::type cofType;
//
//	// Only the first 14 have changes in their first 17 digits; in the advent of long double becoming relevant, deal
//	// Numbers were calculated in Mathematica with 20 significant figures
//	const double cPrime[16] = {0.25000000000000000000,0.266666666666666666666,0.26785714285714285714,0.26794258373205741627,0.26794871794871794872,0.26794915836482308485,0.26794918998527245950,0.26794919225551855963,0.26794919241851489598,0.26794919243021750641,0.26794919243105771603,0.26794919243111804037,0.26794919243112237146,0.26794919243112268242,0.26794919243112270475,0.26794919243112270635};
//
//
//	Eigen::Matrix<TData,Eigen::Dynamic,1> dPrime(matSize);
//
//	for(int i = 0; i < matSize; i++) {
//		dPrime(i) = TData(fType(3.0) * (data[(i+1)%matSize] - data[(matSize+i-1)%matSize]));
//	}
//
//	// Find first coefficient as per Algorithm 4
//	// (-lambda + sqrt(lambda^2 - 4) ) / 2
//	const fType alpha = -0.26794919243112270647;
//	const fType sigma = 1.0717967697244908259 / (3.7128129211020366964 * (1. - ipow(-0.26794919243112270647, matSize)));
//	TCoeffs x0 = (cofType)(sigma * (1.0 + ipow(alpha, matSize))) * TCoeffs(dPrime[0]);
//	int m = (matSize+1) / 2;
//	for(int i = 1; i < m; i++) {
//		x0 += sigma * (ipow(alpha, i) + ipow(alpha, matSize-(i))) * (dPrime[i] + dPrime[matSize-i]);
//	}
//	if(matSize % 2 == 0) {
//		x0 += sigma * (ipow(alpha, m) + ipow(alpha, matSize-m)) * dPrime[m];
//	}
//	dPrime[1] -= x0;
//	dPrime[matSize - 1] -= x0;
//
//	// Copy data so that it isn't written over
//	Eigen::Matrix<TData,Eigen::Dynamic,1> ds = dPrime;
//
//	coeffs[0] = TCoeffs(x0);
//
//	// Tridiagonal matrix algorithm, the c' are precalculated
//	dPrime[1] /= 4.0;
//	for(int i = 2; i < matSize; i++) {
//		dPrime[i] = (ds[i] - dPrime[i-1]) / fType(4. - cPrime[std::min(i-2, 15)]);
//	}
//	// Back substitution for the coefficients
//	coeffs[matSize-1] = TCoeffs(dPrime[matSize-1]);
//	for(int i = matSize - 2; i > 0; i--) {
//		coeffs[i] = TCoeffs(dPrime[i] - TData(cofType(cPrime[std::min(i-1, 15)]) * coeffs[i+1]));
//	}
//	// Done!
//}

#endif
