#include "HelicalModels.h"
#include "Quadrature.h" // For SetupIntegral

#include "mathfuncs.h" // For bessel functions and square
#include "LocalBackend.h"

#include <iostream>

HelicalModel::HelicalModel(std::string st, int extraParams) : FFModel(st, extraParams, 3, 2, -1,
													 EDProfile(NONE))
{}

ExtraParam HelicalModel::GetExtraParameter(int index) {
	return GetExtraParameterStatic(index);
}
ExtraParam HelicalModel::GetExtraParameterStatic(int index) {
	switch (index) {
		case 2:
			return ExtraParam("Height", 10.0, true, true);
		case 3:
			return ExtraParam("Helix Radius", 10.0, false, true);
		case 4:
			return ExtraParam("Pitch", 8.0, false, true);
		default:
			return Geometry::GetExtraParameter(index);	
	}
	return Geometry::GetExtraParameter(index);		
}

bool HelicalModel::IsLayerBased() {
	return false;
}

std::string HelicalModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
	return GetLayerParamNameStatic(index, edpfunc);
}
std::string HelicalModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {

	switch(index) {
		default:
			return Geometry::GetLayerParamName(index, edpfunc);
		case 0:
			return "Phase";
		case 1:
			return "E.D.";
		case 2:
			return "Cross Section";
	}
}

bool HelicalModel::IsParamApplicable(int layer, int lpindex) {
	if(layer < 0 || lpindex < 0 || lpindex >= 3)
		return false;

	//The solvent layer has no cross section or phase
	if ((layer == 0) && ((lpindex == 0) || (lpindex == 2)))
		return false;
	//The first helix's phase is 0 and not mutable
	if ((layer == 1) && (lpindex == 0))
		return false;
	return true;
}

std::string HelicalModel::GetLayerName(int layer)
{
	return GetLayerNameStatic(layer);
}
std::string HelicalModel::GetLayerNameStatic(int layer) {
	if(layer < 0)
		return "N/A";

	if(layer == 0)
		return "Solvent";

	return "Helix %d";
}

double HelicalModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
	switch(paramIndex) {
		default:
		case 0:
			// Phase
			if(layer <= 1)
				return 0.0;

			return 1.0;

		case 1:			
			// Electron Density
			if(layer == 0)
				return 333.0;

			return 400.0;
		case 2:
			// Cross Section
			if(layer == 0)
				return 0.0;

			return 3.0;
	}
}

void HelicalModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	Geometry::OrganizeParameters(p, nLayers);
	(*parameters)(0,0) = 0.0;	//Solvent phase
	(*parameters)(0,2) = 0.0;	//Solvent cross section
	rHelix = (*extraParams)[3];
	P =  (*extraParams)[4];
	edSolvent = (*parameters)(0,1);
	delta = (*parameters).col(0);
	deltaED = (*parameters).col(1);
	rCs = (*parameters).col(2);

}


HelixModel::HelixModel(std::string st, int integralStepsIn, int integralStepsOut, int extras) : HelicalModel(st, extras),
						steps(integralStepsOut), steps1(integralStepsIn) {}

void HelixModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	HelicalModel::OrganizeParameters(p, nLayers);
	height = (*extraParams)[2];
}

bool HelixModel::IsSlow() {
	return (!std::isinf(height) == 0);
}


void HelixModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
	steps	= int(8.0 * (rHelix + P));		//inner
	steps1	= int(std::max(1.0, 15.0 * height));	//outer
	
	if(!std::isinf(height)) {
#pragma omp parallel sections
		{			
#pragma omp section
			{
				SetupIntegral(xIn, wIn, 0.00001, 2.0 * PI + 0.00001, steps);	// x = theta_r
			}
#pragma omp section
			{
				subSteps = int(sqrt((double)steps1));
				SetupIntegral(xOut, wOut, -1.0, 1.0 , steps1); // Orientational Average: x = cos(theta_q)
			}
		}
	}
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void HelixModel::PreCalculateFF(VectorXd& p, int nLayers){
	OrganizeParameters(p,nLayers);
	steps = int(8.0 * (rHelix + P));	//inner

	SetupIntegral(xIn, wIn, 0.00001, 2.0 * PI + 0.00001, steps);	// x = theta_r
	csxIn = xIn.array().cos();

	// Set up voxels for first pitch
	double max_radius = 0;
	Eigen::ArrayXd tt, uu, aa, tt_sin, tt_cos, uu_sin, uu_cos;
	double t_step_size, u_step_size, a_step_size;
	double xx, yy, zz;
	double t_length = sqrt(P * P + sq(2. * M_PI * rHelix));

	try
	{
		max_radius = (rCs.array().maxCoeff() + rHelix);
		number_xy_voxels = int((max_radius * 2. * 10. / (rCs.array().maxCoeff())) + 0.99);
		if (number_xy_voxels % 2 == 0) number_xy_voxels++; // make odd so that the origin is included
		step_size = max_radius * 2. / (number_xy_voxels - 1 - 2);
		number_z_voxels = int((P + delta.array().maxCoeff() + rCs.array().maxCoeff() * 2.) / step_size + 1.99);
		if (number_z_voxels % 2 == 0) number_z_voxels++; // make odd so that the origin is included

		xy_origin = 1 + number_xy_voxels / 2;
		//z_origin = 1 + number_z_voxels / 2;
		z_origin = ((rCs.array().maxCoeff() / step_size) + 0.5);
		//z_origin = 0;

		space.resize(number_xy_voxels, number_xy_voxels * number_z_voxels);
		space.setZero();
		tt.setLinSpaced(std::max(0.5 + 3 * t_length / step_size, 10 / step_size), 0., 1.0);
		uu.setLinSpaced(std::max(0.5 + 3 * 2 * M_PI * rCs.array().maxCoeff() / step_size, 2 / step_size), 0., 2 * M_PI);
	}
	catch (const std::bad_alloc& e)
	{
		printf("bad_alloc:\n"
			"number_xy_voxels = %d\n"
			"number_z_voxels = %d\n"
			,
			number_xy_voxels,
			number_z_voxels
			);
		throw backend_exception(ERROR_INSUFFICIENT_MEMORY);
	}

	tt_cos = (tt * 2 * M_PI).cos();
	tt_sin = (tt * 2 * M_PI).sin();
	uu_cos = uu.cos();
	uu_sin = uu.sin();
	double rcp_rt_rp = 1. / sqrt(P * P + rHelix * rHelix * 2 * M_PI * 2 * M_PI);
	
	for (int r = 1; r < rCs.size(); r++)
	{
		double a_length = rCs(r);
		aa.setLinSpaced(std::max(0.5 + 3 * a_length / step_size, 2 / step_size), 0., a_length);
		for (int t = 0; t < tt.size(); t++)
		{
			for (int u = 0; u < uu.size(); u++)
			{
				for (int a = 0; a < aa.size(); a++)
				{
					xx = rHelix * tt_cos(t) - aa(a) * uu_cos(u) * tt_cos(t) + aa(a) * P * uu_sin(u) * tt_sin(t) * rcp_rt_rp;
					yy = rHelix * tt_sin(t) - aa(a) * uu_cos(u) * tt_sin(t) + aa(a) * P * uu_sin(u) * tt_cos(t) * rcp_rt_rp;
					zz = delta(r) + P * tt(t) + aa(a) * rHelix * uu_sin(u) * rcp_rt_rp * 2 * M_PI;
					int x_ind = xx / step_size;
					int y_ind = yy / step_size;
					int z_ind = zz / step_size;
					space(xy_origin + y_ind, (xy_origin + x_ind) + (z_ind + z_origin) * number_xy_voxels) = deltaED(r) - edSolvent;
				} // for a
			} // for u
		} // for t
	} // for r

	int non_zero_voxels = (space != 0).count();
	printf("Number of helix voxels: %d / %d \n", non_zero_voxels, space.size());
	printf("Volume: %fnm^3\n", non_zero_voxels * step_size*step_size*step_size);

	// TODO: Reduce to sparser representation of space

	voxel_COMs.resize(Eigen::NoChange, non_zero_voxels);
	voxel_COMs.setZero();
	voxel_contrast.resize(non_zero_voxels);
	int voxel_index = 0;

	for (int z = 0; z < number_z_voxels; z++)
	{
		for (int x = 0; x < number_xy_voxels; x++)
		{
			for (int y = 0; y < number_xy_voxels; y++)
			{
				if (space(y, x + z * number_xy_voxels) != 0)
				{
					voxel_COMs.col(voxel_index) =
						Eigen::Vector4d(
						(x - xy_origin)*step_size,
						(y - xy_origin)*step_size,
						(z - z_origin)*step_size,
						0.f
						);
					voxel_contrast(voxel_index) = space(y, x + z * number_xy_voxels);
					voxel_index++;
				}
			}
		}
	}

/*
	for (int i = 0; i < std::min(number_z_voxels, 40); i++)
	{
		std::cout << "layer " << i << "\n" << (space > 0).middleCols(i * number_xy_voxels, std::min(number_xy_voxels,40)) << "\n";
	}

	system("PAUSE");
	
*/

	Eigen::ArrayXXi side, top;
	side.resize(number_xy_voxels, number_z_voxels);
	top.resize(number_xy_voxels, number_xy_voxels);
	side.setZero();
	top.setZero();

	for (int i = 0; i < number_xy_voxels; i++)
	{
		Eigen::Map<Eigen::ArrayXXf, 0, Eigen::Stride<-1, 1>> mp(space.data() + i *number_xy_voxels, number_xy_voxels, number_z_voxels, Eigen::Stride<-1, 1>(number_xy_voxels*number_xy_voxels, 1));
		side += (mp > 0).cast<int>();
	}
	Eigen::Map<Eigen::ArrayXXf, 0, Eigen::Stride<-1, 1>> mp(space.data() + (number_xy_voxels / 2) *number_xy_voxels, number_xy_voxels, number_z_voxels, Eigen::Stride<-1, 1>(number_xy_voxels*number_xy_voxels, 1));
	std::cout << "\n" << (mp > 0) << "\n\n";


	std::cout << "\n" << (side > 0) << "\n\n";

	//system("PAUSE");

	for (int i = 0; i < number_z_voxels; i++)
	{
		top += (space > 0).middleCols(i * number_xy_voxels, number_xy_voxels).cast<int>();
	}
	std::cout << "\n" << (top > 0) << "\n\n";


}

double HelixModel::Calculate(double q, int nLayers, Eigen::VectorXd &a) {
	//ma - size of a
	//nLayers - number of helices
	//a[1]...a[nLayers - 1] - Delta from Helix 1
	//a[nLayers] - solvent ED
	//a[nLayers + 1]...a[2*nLayers - 1] - Helix ED
	//a[2*nLayers + 1]...a[3*nLayers - 1] - Helix Cross section
	// Extra params (in correct order): (0)scale, (1)background, (2)height, (3)R, (4)P
	if(a.size() > 0)
		OrganizeParameters(a, nLayers);

	VectorXd bess1 = VectorXd::Zero(nLayers);
	
	int hMax;
	if(delta[0] < 0.0) 
		delta[0] = 0.0;

	
//	static VectorXd xIn, wIn, xOut, wOut;
//	static int subSteps;
//	int steps, steps1;

	int N = int(height / P);
		
	hMax = int(floor(q * P / (2.0 * PI)));
	
	double intensity = 0.0;
		//Model from 02/02/2010 - Convolution of a thin helix and a disc
	if(std::isinf(height)) {	// Infinite model
#pragma omp parallel for reduction(+ : intensity)
		for(int m = 0; m <= hMax; m++) {
			double  root = sqrt(1.0 - sq(2.0 * PI * double(m) / (q * P)));
			std::complex <double> sum(0.0,0.0), i(0.0,1.0);
			for(int j = 1; j < nLayers; j++) {
 					sum += rCs[j] * (bessel_j1(q * rCs[j] * root) / root) * (deltaED[j] - edSolvent) 
						* exp(i * 2.0 * PI * double(m) * delta[j] / P);
			}
			sum *= bessel_jn(m, q * rHelix * root) ;
			intensity += norm(sum);
		}
		intensity *= 8.0 * PI * sq(PI)/ (sq(q) * q);
	}
	else {	// Finite model
		if(height < 1.0e-12)
			return (*extraParams)[1];
		std::complex <double>  i(0.0,1.0);
		
#pragma omp parallel for reduction(+ : intensity)
		for(int outest = 0; outest <= steps1 / subSteps ; outest++) {
			double subIntensity = 0.0;
			for (int ou = outest * subSteps; ou < (subSteps * outest) + subSteps; ou++) {
				if(!(ou < steps1)) continue;
				std::complex <double> sst(0.0,0.0),st(0.0,0.0);
				double outroot = sqrt(1.0 - sq(xOut[ou]));
				for (int s = 1; s < nLayers; s++) 
					st += (deltaED[s] - edSolvent)* exp(i * q * xOut[ou] * delta[s]) 
						* bessel_j1(q * rCs[s] * outroot) * rCs[s] / outroot ;	
				for (int in = 0; in < steps; in++) 
					sst += wIn[in] * exp(i * q * rHelix * outroot * cos(xIn[in]) 
						+ i * q * xOut[ou] * P * xIn[in] / (2.0 * PI)  );
				subIntensity += wOut[ou] * norm((fabs(xOut[ou]) < 1.0e-20) ? (N) : (sin(N * q * xOut[ou] * P / 2.0) / sin( q * xOut[ou] * P / 2.0)) * sst * st);
			}
			intensity += subIntensity;
		}
		intensity *= 2.0 * PI * sq(P) / sq (q);
	
	}

	/* End of 02/02/2010*/

	intensity *= (*extraParams)[0];   // Multiply by scale
	intensity += (*extraParams)[1];  // Add background
	
	return intensity;
}

std::complex<double> HelixModel::CalculateFF(Vector3d qvec, int nLayers, double w, double precision, VectorXd* a) {
	double qx = qvec(0), qy = qvec(1), qz = qvec(2), q = sqrt(qx*qx + qy*qy + qz*qz);
	double qperp = sqrt(qvec[0] * qvec[0] + qvec[1] * qvec[1]);
	const Eigen::Vector4d qVec4(qvec.x(), qvec.y(), qvec.z(), 0);
	
	double voxel_form_factor =
		sinc(step_size * qx) * step_size *
		sinc(step_size * qy) * step_size *
		sinc(step_size * qz) * step_size;
	std::complex<double> helical_amp(0., 0.);

 	int ma = 0;
    if(a)
        ma = a->size();

	int num_pitches = height / P;
	double remainder_of_pitch = num_pitches * P - height;

	for (int i = 0; i < voxel_COMs.cols(); i++)
	{
		helical_amp +=
			voxel_contrast(i) *
			voxel_form_factor *
			exp(std::complex<double>(0, 1) * qVec4.dot(voxel_COMs.col(i)));

	}

// 	for (int z = 0; z < number_z_voxels; z++)
// 	{
// 		for (int x = 0; x < number_xy_voxels; x++)
// 		{
// 			for (int y = 0; y < number_xy_voxels; y++)
// 			{
// 				if (space(y, x + z * number_xy_voxels) != 0)
// 				{
// 					const Eigen::Vector4d rvec((x - xy_origin)*step_size, (y - xy_origin)*step_size, (z - z_origin)*step_size, 0);
// 					helical_amp += 
// 						space(y, x + z * number_xy_voxels) * /* this is the contrast*/
// 						voxel_form_factor *
// 						exp(-std::complex<double>(0, 1) * qVec4.dot(rvec));
// 				}
// 			} // for a
// 		} // for u
// 	} // for t

	helical_amp *= !closeToZero(qz * P) ? (exp(std::complex<double>(0, 1) * qz * (P * num_pitches)) - 1.0) / (exp(std::complex<double>(0, 1) * qz * P) - 1.0) : num_pitches;

	return helical_amp * (*extraParams)(0) + (*extraParams)(1); // Multiply by scale and add background


	VectorXd bess1 = VectorXd::Zero(nLayers);
	
	int hMax;
	if(delta[0] < 0.0) 
		delta[0] = 0.0;

	int N = int(height / P);
	double N_double = height / P;
		
	hMax = int(floor(qz * P / (2.0 * PI)));
	double qphi = atan2(qy, qx);
	
	// Amplitude real and complex parts
	double ampr = 0.0, ampim = 0.0;

	if(height < 0.0) {	// Infinite model	-- For now, no support for this.
		if(closeToZero(q)) {
			return std::complex<double>(1.0,0.0);	// TODO::AT_ZERO
		}
		if(closeToZero(qperp)) {
			return std::complex<double>(1.0,0.0);	// TODO::AT_ZERO
		}
		if(closeToZero(qz)) {
			return std::complex<double>(1.0,0.0);	// TODO::AT_ZERO
		}

#pragma omp parallel for reduction(+ : ampr, ampim)
		for(int m = 0; m <= hMax; m++) {
			//Dirac Delta Function condition
			if(fabs(qz - 2.0 * PI * double(m) / P ) < precision) {
				std::complex <double> cof(0.0,0.0), sum(0.0,0.0), i(0.0,1.0);
				//double  root = sqrt(1.0 - sq(2.0 * PI * double(m) / (q * P)));
				cof = 4.0 * PI * PI * exp(-i * double(m) * qphi) * bessel_jn(m, qperp * rHelix) / qperp ;	// TODO::AT_ZERO
				if (m % 4 == 0)
					cof *= 1.0;
				else if (m % 4 == 1)
					cof *= -i;
				else if (m % 4 == 2)
					cof *= -1.0;
				else 
					cof *= i;
				for(int j = 1; j < nLayers - 1; j++) {
 						//sum += rCs[j] * (bessel_j1(q * rCs[j] * root) / root) * (deltaED[j] - edSolvent) 
						//	* exp(i * 2.0 * PI * double(m) * delta[j] / P);
					sum += (deltaED[j] - edSolvent) * rCs[j] * exp(i * (double)m * delta[j]) * bessel_j1(qperp * rCs[j]);	// TODO::AT_ZERO
					sum *= cof;
				}
				ampr += sum.real();
				ampim += sum.imag();
			}
		}	

		return std::complex<double>(ampr, ampim) / w;
	
	
	}
	else {	// Finite model
		if(height < 1.0e-12)
			return (*extraParams)[1];	// Background
		const std::complex<double> i(0.0,1.0);
		std::complex<double> pre(0.0,0.0), amplitude(0.0,0.0);
		std::vector<double> some_variable;
		some_variable.resize(nLayers);

		double st_real = 0.0, st_imag = 0.0, sst_real = 0.0, sst_imag = 0.0;

		{
			std::complex<double> I_in_the_paper = 0, S_in_the_paper = 0;
			I_in_the_paper = (((qperp * rHelix * csxIn.array() + qz * P * xIn.array() * 0.1591549430918953 /*1/(2\pi)*/).cast<std::complex<double>>() * (-i)).exp() * wIn.array()).sum();
// 			for (int in = 0; in < steps; in++) {
// 				std::complex <double> sst = 
// 					wIn[in] * exp(-i * qperp * rHelix * cos(xIn[in]) - i * qz * P * xIn[in] / (2.0 * PI));
// 				I_in_the_paper += sst;
// 			}
			I_in_the_paper *= exp(i * qz * P * qphi * 0.1591549430918953 /*1/(2\pi)*/);

			if (closeToZero(qperp))
				for (int o = 1; o < some_variable.size(); o++)
					some_variable[o] = 0.5 * sq(rCs[o]);
			else
				for (int o = 1; o < some_variable.size(); o++)
					some_variable[o] = bessel_j1(qperp * rCs[o]) * (rCs[o] / qperp);

			for (int j = 1; j < nLayers; j++)
			{
				S_in_the_paper += some_variable[j] * (deltaED[j] - edSolvent) * exp(i * qz * delta[j]);
			}
			double sinnx = closeToZero(N_double * qz) ? N_double : sin(N_double * qz * 0.5 * P) / sin(qz * 0.5 * P);
			return P * S_in_the_paper * I_in_the_paper * sinnx * exp(i * (N_double - 1) * qz * 0.5 * P);
		}

#pragma omp parallel for reduction(+ : st_real, st_imag)	
		for (int s = 1; s < nLayers; s++) {
			double coef = rCs[s] * (deltaED[s] - edSolvent)* P;
			if(closeToZero(qperp)) {
				coef *= rCs[s] / 2.0;
			} else {
				coef *= bessel_j1(qperp * rCs[s]);
			}
			double innerExp = qz * P * delta[s] / (2.0 * PI) - qz * P * qphi / (2.0 * PI);
			st_real += coef * cos(innerExp);
			st_imag += coef * sin(innerExp);
		}

		if(closeToZero(qz)) {
			pre =  (double(N) + 0.5) / (2.0);
		} else {
			pre = sin(qz * P * (double(N) + 0.5)) / (sin(qz * P / 2.0));
		}
		if(!closeToZero(qperp))
			pre /= qperp;

#pragma omp parallel for reduction(+ : sst_real, sst_imag)
		for (int in = 0; in < steps; in++) {
			// i.e. sst_real += ...; sst_imag += ...;
			std::complex <double> sst (0.0,0.0);
			sst = wIn[in] * exp(-i * qperp * rHelix * cos(xIn[in]) - i * qz * P * xIn[in] / (2.0 * PI)  );
			sst_real += sst.real();
			sst_imag += sst.imag();
		}
	
		return pre * std::complex<double>(st_real, st_imag) * std::complex<double>(sst_real, sst_imag);	
	
	}
}

VectorXd HelixModel::Derivative(const std::vector<double>& x, VectorXd param, 
										int nLayers, int ai) {
	int nParams = (int)param.size();
	
	// Finite helix is numeric
	if(height >= 0.0)
		return FFModel::Derivative(x, param, nLayers, ai);

	// TODO::AT_ZERO
	OrganizeParameters(param, nLayers);
	VectorXd Der(x.size());
	
	if ((ai > 0) && (ai < nLayers)) {//Partial Phase Derivation
#pragma omp parallel for
		for (int c = 0; c < int(x.size()); c++) {
			std::complex<double> sum(0.0, 0.0), sum1(0.0, 0.0), i(0.0,1.0);
			double sq = 0;
			int hMax = int(floor(x[c] * P / (2.0 * PI)));
			for (int i = 0; i < (nLayers-1); i++) {
				for (int m = 0; m < hMax ; m++) {
					sq = (1 - (2 * PI * m / (P * x[c])) * (2 * PI * m / (P * x[c])));
					sum1 += (1 /sq) * (bessel_jn(m, x[c] * rHelix * sqrt(sq))) * (bessel_jn(m, x[c] 
						* rHelix * sqrt(sq))) * 2 * m * sin(m * (delta[ai - 1]-delta[i]))
							* bessel_j1(x[c] * rCs[ai - 1] * sqrt(sq)) * bessel_j1(x[c] * rCs[i] * sqrt(sq));
				}
			sum += rCs[i] * rCs[ai - 1] * (deltaED[i] - edSolvent) * (deltaED[ai - 1] - edSolvent) * sum1;
			}
			Der[c] = 32 * PI * PI * PI * PI * PI / (x[c] * x[c] * x[c]) * norm(sum);
		}
		return ((*extraParams)[0] * Der);
	}
	else if  ((ai > nLayers) && (ai < (2 * nLayers))) {//Partial E.D. Derivation
#pragma omp parallel for
		for (int c = 0; c < int(x.size()); c++) {
			std::complex<double> sum(0.0, 0.0), sum1(0.0, 0.0), i(0.0,1.0);
			double sq = 0;
			int hMax = int(floor(x[c] * P / (2.0 * PI)));
			for (int i = 0; i < (nLayers-1); i++) {
				for (int m=0; m < hMax ; m++) {
					sq = (1 - (2 * PI * m / (P * x[c])) * (2 * PI * m / (P * x[c])));
					sum1 += (1 /sq) * (bessel_jn(m, x[c] * rHelix * sqrt(sq))) * (bessel_jn(m, x[c] 
						* rHelix * sqrt(sq))) * 2 * cos(m * (delta[ai - nLayers - 2]-delta[i]))
							* bessel_j1(x[c] * rCs[ai - nLayers - 2] * sqrt(sq)) * bessel_j1(x[c] * rCs[i] * sqrt(sq));
				}
				sum += rCs[i] * rCs[ai - nLayers - 2] * (deltaED[i] - edSolvent) * sum1;
			}
			Der[c] = 32 * PI * PI * PI * PI * PI / (x[c] * x[c] * x[c]) * norm(sum);
		}
		return ((*extraParams)[0] * Der);
	}
	else if  ((ai > 2 * nLayers) && (ai < (nLayerParams * nLayers))) {//Partial Cross Section Derivation
#pragma omp parallel for
		for (int c = 0; c < int(x.size()); c++) {
			std::complex<double> sum(0.0, 0.0), sum1(0.0, 0.0), i(0.0,1.0);
			double sq = 0;
			int hMax = int(floor(x[c] * P / (2.0 * PI)));
			for (int i = 0; i < (nLayers-1); i++) {
				for (int m=0; m < hMax ; m++) {
					sq = (1 - (2 * PI * m / (P * x[c])) * (2 * PI * m / (P * x[c])));
					sum1 += (1 /sqrt(sq)) * (bessel_jn(m, x[c] * rHelix * sqrt(sq))) * (bessel_jn(m, x[c] 
						* rHelix * sqrt(sq))) * 2 * cos(m * (delta[ai - 2 * nLayers - 3]-delta[i]))
							* x[c] * (bessel_j0(x[c] * rCs[ai - 2 * nLayers - 3] * sqrt(sq)) - bessel_jn(2, x[c] * rCs[ai - 2 * nLayers - 3])) * bessel_j1(x[c] * rCs[i] * sqrt(sq));
				}
				sum += rCs[i] * (deltaED[i] - edSolvent) * (deltaED[ai - 2 * nLayers - 3] - edSolvent) * sum1;
			}
			Der[c] = 32 * PI * PI * PI * PI * PI / (x[c] * x[c] * x[c]) * norm(sum);
		}
		return ((*extraParams)[0] * Der);
	}
	else if (ai == (nParams - nExtraParams) + 3) {//Partial Helix Radius Derivative
#pragma omp parallel for
		for (int c = 0; c < int(x.size()); c++) {
			std::complex<double> sum(0.0, 0.0), sum1(0.0, 0.0), i(0.0,1.0);
			double sq = 0;
			int hMax = int(floor(x[c] * P / (2.0 * PI)));
			for (int i = 0; i < (nLayers-1); i++) {
				for (int j = 0; j < (nLayers-1); j++) {
				for (int m=0; m < hMax ; m++) {
					sq = (1 - (2 * PI * m / (P * x[c])) * (2 * PI * m / (P * x[c])));
					sum1 += (1 / sqrt(sq)) * (bessel_jn(m, x[c] * rHelix * sqrt(sq))) * (bessel_jn(m - 1, x[c] 
						* rHelix * sqrt(sq)) - bessel_jn(m + 1, x[c] * rHelix * sqrt(sq))) * x[c]
							* exp(i * m * (delta[j]-delta[i]))* bessel_j1(x[c] * rCs[j] * sqrt(sq)) * bessel_j1(x[c] * rCs[i] * sqrt(sq));
				}
				sum += rCs[j] * rCs[i] * (deltaED[i] - edSolvent) * (deltaED[j] - edSolvent) * sum1;
				}
			}
			Der[c] = 32 * PI * PI * PI * PI * PI / (x[c] * x[c] * x[c]) * norm(sum);
		}
		return ((*extraParams)[0] * Der);
	}
	else if (ai == (nParams - nExtraParams) + 4) {//Partial Pitch Derivative		
#pragma omp parallel for
		for (int c = 0; c < int(x.size()); c++) {
			std::complex<double> sum(0.0, 0.0), sum1(0.0, 0.0), i(0.0,1.0);
			double bi1, bj1, bRm, bmin1mR, bplus1mR, bi0, bi2, bj0, bj2;
			double sq = 0;
			int hMax = int(floor(x[c] * P / (2.0 * PI)));
			for (int i = 0; i < (nLayers-1); i++) {
				for (int j = 0; j < (nLayers-1); j++) {
				for (int m=0; m < hMax ; m++) {
					sq = (1 - (2 * PI * m / (P * x[c])) * (2 * PI * m / (P * x[c])));
					bRm = bessel_jn(m, x[c] * rHelix * sqrt(sq));
					bi1 = bessel_j1(x[c] * rCs[i] * sqrt(sq));
					bj1 = bessel_j1(x[c] * rCs[j] * sqrt(sq));
					bi0 = bessel_j0(x[c] * rCs[i] * sqrt(sq));
					bj0 = bessel_j0(x[c] * rCs[j] * sqrt(sq));
					bi2 = bessel_jn(2, x[c] * rCs[i] * sqrt(sq));
					bj2 = bessel_jn(2, x[c] * rCs[j] * sqrt(sq));
					bmin1mR = bessel_jn(m - 1, x[c] * rHelix * sqrt(sq));
					bplus1mR = bessel_jn(m + 1, x[c] * rHelix * sqrt(sq));
					sum1 += - (1 / (sq * sq * P * P * P * x[c] * x[c])) * 8 * exp(i * m * (delta[i]-delta[j])) * m * m * PI * PI * bi1 * bj1 * bRm * bRm
						+ (1 / (sq * sqrt(sq) * P * P * P * x[c])) * 4 * exp(i * m * (delta[i]-delta[j])) * m * m * PI * PI * rHelix * bi1 * bj1 * bRm * (bmin1mR - bplus1mR)
						+ (1 / (sq * sqrt(sq) * P * P * P * x[c])) * 2 * exp(i * m * (delta[i]-delta[j])) * m * m * PI * PI * rCs[i] * bj1 * bRm * bRm * (bi0 - bi2)
						+ (1 / (sq * sqrt(sq) * P * P * P * x[c])) * 2 * exp(i * m * (delta[i]-delta[j])) * m * m * PI * PI * rCs[j] * bi1 * bRm * bRm * (bj0 - bj2);
				}
				sum += rCs[j] * rCs[i] * (deltaED[i] - edSolvent) * (deltaED[j] - edSolvent) * sum1;
				}
			}
			Der[c] = 32 * PI * PI * PI * PI * PI / (x[c] * x[c] * x[c]) * norm(sum);
		}
		return ((*extraParams)[0] * Der);
	}
	else //Default Numerical Derivation
		return Geometry::Derivative(x, param, nLayers, ai);
}
