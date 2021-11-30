#include "SlabModels.h"
#include "Quadrature.h" // For SetupIntegral

#include "mathfuncs.h" // For bessel functions and square

#pragma region Abstract Slab
	SlabModel::SlabModel(std::string st, EDProfile edp, int nlp, int maxlayers) : FFModel(st, 
															4, nlp, 2, maxlayers, edp)
	{}

	bool SlabModel::IsParamApplicable(int layer, int lpindex)
	{
		if(layer == 0)	//Solvent
			if(lpindex == 0)	// Width
				return false;
		return true;
	}

	std::string SlabModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
		return GetLayerParamNameStatic(index, edpfunc);
	}
	std::string SlabModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {
		switch (index) {
			default:
				return Geometry::GetLayerParamName(index, edpfunc);
			case 0:
				return "Width";
			case 1:
				return "E.D.";
		}
	}

	std::string SlabModel::GetLayerName(int layer) {
		return GetLayerNameStatic(layer);
	}
	std::string SlabModel::GetLayerNameStatic(int layer) {
		if (layer < 0)
			return "N/A";

		if(layer == 0)
			return "Solvent";

		return "Layer %d";
	}


	double SlabModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
		switch(paramIndex) {
			default:
			case 0:
				// Width
				if(layer < 1)
					return 0.0;

				return 1.0;

			case 1:			
				// Electron Density
				if(layer == 0)
					return 333.0;
				if(layer == 1)
					return 280.0;
				
				return 400.0;
		}
	}

	void SlabModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers)
	{
		Geometry::OrganizeParameters(p, nLayers);
		width		= (*parameters).col(0);
		ED			= (*parameters).col(1);
		edSolvent	= (*parameters)(0,1);
		xDomain		= (*extraParams)(2);
		yDomain		= (*extraParams)(3);
		centers.resizeLike(width);
		
		// DEBUG LINES
		//std::string str = debugMatrixPrintM(*parameters);
		//MessageBoxA(NULL, str.c_str(), "Parameters Matrix", NULL);
		// END DEBUG LINES
	}

	ExtraParam SlabModel::GetExtraParameter(int index) {
		return GetExtraParameterStatic(index);
	}
	ExtraParam SlabModel::GetExtraParameterStatic(int index) {
		switch (index) {
			default:
				return FFModel::GetExtraParameterStatic(index);
			case 2:
				return ExtraParam("X Domain Size", 10.0, true, true);
			case 3:
				return ExtraParam("Y Domain Size", 10.0, true, true);
		}
	}


#pragma endregion

#pragma region Uniform Slab

	UniformSlabModel::UniformSlabModel(std::string st, ProfileType t) : SlabModel(st, EDProfile(t))
		{}

	double UniformSlabModel::Calculate(double q, int nLayers, VectorXd& p)
	{
		if(p.size() > 0)	// For PD model
			OrganizeParameters(p, nLayers);
		
		double intensity = 0.0;
		
		if(this->profile.type == SYMMETRIC) {
			for(int i = 1; i < nLayers; i++)
				intensity += 2 * (ED[i] - edSolvent) * (sin(q * width[i]) - sin(q * width[i - 1]));
			
			intensity *= intensity;
		}
		if(this->profile.type == ASYMMETRIC) {
			for(int i = 1; i < nLayers; i++) {
				for(int j = 1; j < nLayers; j++) {
					intensity += (ED[i] - edSolvent) * (ED[j] - edSolvent) *
					(cos(q * (width[j] - width[i])) - 2.0 * cos(q * (width[j - 1] - width[i]))
					  + cos(q * (width[j - 1] - width[i - 1])));
				}
			}
		}
		
		intensity *= (2.0 / sq(sq(q)));
		
		intensity *= (*extraParams)(0);	// Multiply by scale
		intensity += (*extraParams)(1); // Add background

		return intensity;
	}

	std::complex<double> UniformSlabModel::CalculateFF(Vector3d qvec, 
									 int nLayers, double w, double precision, VectorXd* p)
	{
		double qx = qvec(0), qy = qvec(1), qz = qvec(2), q = sqrt(qx*qx + qy*qy + qz*qz);

		std::complex<double> res(0.0, 0.0);
		const std::complex<double> im(0.0, 1.0);

		if(this->profile.type == SYMMETRIC) {
			if(closeToZero(qz)) {
				for(int i = 1; i < nLayers; i++)
					res += (ED[i] - ED[0]) * 2. * (width(i) - width(i - 1));
				return res * 4. * sinc(qx * xDomain) * xDomain * sinc(qy * yDomain) * yDomain ;
			}

			double prevSin = 0.0;
			double currSin;
			for(int i = 1; i < nLayers; i++) {
				currSin = sin(width(i) * qz);
				res += (ED[i] - ED[0]) * 2. * (currSin - prevSin) / qz;
				prevSin = currSin;
			}
			res *= 4. * sinc(qx * xDomain) * xDomain * sinc(qy * yDomain) * yDomain;
		} else {
			// Assymetric profile
			for(int i = 1; i < nLayers; i++) {
				const auto sw = sinc(qz * width(i) * 0.5) * width(i);
				const auto ex = exp(im * qz * centers(i) );
				res += (ED[i] - ED[0]) * sw * ex;
			}
			res *= 4. * sinc(qx * xDomain) * xDomain * sinc(qy * yDomain)  * yDomain;

		}	// else

		return res * (*extraParams)(0) + (*extraParams)(1); // Multiply by scale and add background
	}

	void UniformSlabModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers)
	{
		SlabModel::OrganizeParameters(p, nLayers);

		width(0) = 0.0;
		centers.setZero();

		xDomain *= 0.5;
		yDomain *= 0.5;

		if (this->profile.type == SYMMETRIC)
			for (int i = 2; i < nLayers; i++)
				width[i] += width[i - 1];
		else
		{
			const auto middle = width.sum() * 0.5;
			centers(0) = -middle;
			for (int i = 1; i < nLayers; i++)
				centers(i) = centers(i - 1) + (width(i) + width(i - 1))* 0.5;
		}
		if (false)
		{	// This should be enabled if/when we decide to move the drawing to the center.
			if (this->profile.type == ASYMMETRIC)
				centers.array() -= width.sum() * 0.5;
		}
	}

	void UniformSlabModel::PreCalculate(VectorXd& p, int nLayers)
	{
		OrganizeParameters(p, nLayers);
	}

	void UniformSlabModel::PreCalculateFF(VectorXd& p, int nLayers) {
		// OrganizeParameters(p, nLayers);	// Already called explicitly by D+
	}


#pragma endregion
