#include "DUMMY_HEADER_FILE.h"


/****************************************************************************/
/* THIS IS A TEMPORARY FILE THAT IS TO BE USED ONLY FOR DEFINING STRUCTS	*/
/* OR METHODS THAT HAVE YET TO BE IMPLEMENTED IN THE FRONTEND OR BACKEND.	*/
/*																			*/
/* REMEMBER TO IMPLEMENT ITEMS BEFORE DELETEING THEM!!!!!!!!				*/
/****************************************************************************/
#ifndef PI
	#define PI 3.14
#endif

double GetResolution() {
	return 0.0;
}

void SetMinimumSig(double eefd) {}

bool hasGPUBackend() {
	return false;
}

bool isGPUBackend() {
	return false;
}

void SetGPUBackend(bool asdrg) {}

int vmin(const std::vector<double>& vec) {
	double y_min = vec[0];
	int res = 0;
	for (int i = 1; i < (int)vec.size(); i++) {
		if (y_min > vec[i]) {
			y_min = vec[i];
			res = i;
		}
	}

	return res;
}

inline double vminfrom(vector<double>& v, int from, int *m) {
	if(v.size() == 0)
		return 0.0;
	double val = v.at(0);
	for(unsigned int i = from; i < v.size(); i++) {
		if(v[i] < val) {
			val = v[i];
			if(m)
				*m = i;
		}
	}
	return val;
}

inline double vmax(vector<double>& v, int *m) {
	if(v.size() == 0)
		return 0.0;
	double val = v.at(0);
	for(unsigned int i = 0; i < v.size(); i++) {
		if(v[i] > val) {
			val = v[i];
			if(m)
				*m = i;
		}
	}
	return val;
}

double InterpolatePoint(double x0, const std::vector<double>& x, const std::vector<double>& y) {
	for(int i = 0; i < (int)x.size(); i++) {
		if(x0 <= x[i] && i != 0)
			return y[i - 1] + (x0 - x[i - 1]) * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]));
	}
	return 0.0;
}

void ImportBackground(const wchar_t *filename, const wchar_t *datafile,
							   const wchar_t *savename, bool bFactor) {
	vector<double> datax, datay, bgx, bgy, bgyTemp, resx, resy;
	
	ReadDataFile(datafile, datax, datay);

	ReadDataFile(filename, bgx, bgy);
	
	// If the data files don't match in size or exact q-range crop and/or interpolate
	if((bgx.size() != datax.size()) || (fabs(bgx.at(0) - datax.at(0)) < 1.0e-8) || (fabs(bgx.at(bgx.size() - 1) - datax.at(datax.size() - 1)) < 1.0e-8)) {
		//Crop
		int j = 0, k = 0;
		for(; j < (int)datax.size() && datax[j] < bgx[0]; j++);
		for(k = int(datax.size()) - 1; k >= j && datax[k] > bgx[bgx.size() - 1]; k--);

		if(k - j < 1) { // less than 2 points left
			;
			return;
		}

		datax.erase(datax.begin() + k + 1, datax.end());
		datay.erase(datay.begin() + k + 1, datay.end());
		datax.erase(datax.begin(), datax.begin() + j);
		datay.erase(datay.begin(), datay.begin() + j);



		//Interpolate
		bgyTemp = datax;
		for(int i = 0; i < (int)datax.size(); i++)
			bgyTemp.at(i) = InterpolatePoint(datax.at(i), bgx, bgy);

		bgx = datax;
		bgy = bgyTemp;

	}

	if(!bFactor) {
		int q2 = 0;
		double fmin, fmax, gmin, gmax;
		gmax = vmax(bgy, &q2);
		fmax = InterpolatePoint(bgx.at(q2), datax, datay);
		fmin = datay[vmin(datay)];
		gmin = bgy[vmin(bgy)];

		resx.resize(datax.size());
		resy.resize(datay.size());
		// Fit background to data
		for(unsigned int i = 0; i < resx.size(); i++) {
			double x = datax[i], fx = datay[i];
			resx[i] = x;
			// We will now allow negative points
			/*bgy[i] = bgy[i] - gmin; /*((InterpolatePoint(x, bgx, bgy) - gmin) *
									((fmax - fmin) / (gmax - gmin)));*/
			resy[i] = fx /*- fmin*/ - bgy[i];
		}

		// Again, allowing negative values (for true background substraction
		//double minff = vminfrom(resy, q2 + 1, NULL);
		//for(unsigned int i = 0; i < resx.size(); i++)
		//	resy[i] = resy[i] /*+ GetMinimumSig()*/ - minff;
	} else { // Find the factor
		// Find the ratios for the second half of the graph
		// Take the lowest ratio (sig/BG)
		// resy.at(i) = sig[i] - (ratio * BG[i]) -->interpolate BG if need be
		std::vector<double> ratio;
		
		ratio.resize(datax.size(), std::numeric_limits<double>::max());
		for(int i = datax.size() / 5; i < (int)datax.size(); i++)
			if(!(fabs(bgy.at(i)) <= 1e-10))
				ratio.at(i) = datay.at(i) / bgy.at(i);
		
		int minRatio = 0;
		for(int i = 0; i < (int)ratio.size(); i++)
			if(ratio.at(i) < ratio.at(minRatio))
				minRatio = i;
		double rat = ratio.at(minRatio);

		// Fill out the result
		resx = datax; 
 		for(int i = 0; i < (int)datay.size(); i++)
			resy.push_back(datay.at(i) - rat * bgy.at(i));

	}

	WriteDataFile(savename, resx, resy);
}

double GetMinimumSig() {
	return 0.0;
}

//void SetMinimumSig() {}

void ClassifyQuadratureMethod(QuadratureMethod method) {
/*
	switch(method) {
	default:
	case QUAD_GAUSSLEGENDRE:
		Quadrature = GaussLegendre2D;
		break;
	case QUAD_MONTECARLO:
		Quadrature = MonteCarlo2D;
		break;
	case QUAD_SIMPSON:
		Quadrature = SimpsonRule2D;
		break;
	}
*/
}

void SetPDFunc(PeakType shape) {
	if(shape < SHAPE_GAUSSIAN || shape > SHAPE_LORENTZIAN_SQUARED)
		return;

//	g_pdFunc = shape;
}

void GenerateBGLinesandFormFactor(const wchar_t *workspace, 
								  const wchar_t *datafile,
								  std::vector <double>& bglx,
								  std::vector <double>& bgly,
								  std::vector <double>& ffy, bool ang) {
	vector<double> sx, datax, datay;
	double slope, intercept;

	vector<int> a;

	Read1DDataFile(datafile, sx);
	ReadDataFile(workspace, datax, datay);
	if(ang) {
		for (unsigned int i=0; i < datax.size(); i++) 
			datax[i] *= 10.0;
	}

	int ctr = 0;
	for (int i = 0; i < (int)datax.size(); i++){
		if (fabs(  datax.at(i) - sx.at(ctr))<=0.0001){
			a.push_back(i); 
			ctr++;
			if(ctr == (int)sx.size())
				break;
		}
	}

	vector<double> bx (a.back(), 0.0), by (a.back(), 0.0);

	double sya = InterpolatePoint(sx.at(0), datax, datay), 
		syb = InterpolatePoint(sx.at(1), datax, datay);
	for (int j = 0; j < (int)sx.size() - 1; j++) {
		slope = ( log10(syb) - log10(sya) ) / ( log10(sx[j+1]) - log10(sx[j]) );

		intercept = log10(sya) - slope*log10(sx[j]);

		for (int i = a[j]; i < a[j+1]; i++) {
			bx[i] = datax[i];
			by[i] = pow(10.0, intercept) * pow(datax[i],slope);
		}

		sya = syb;
		if(j + 2 < (int)sx.size())
			syb = InterpolatePoint(sx.at(j + 2), datax, datay);
	}

	for (int i = a[0]; i < a[a.size() - 1]; i++) {
		bglx.push_back(bx[i]);
		bgly.push_back(by[i]);
		ffy.push_back(GetMinimumSig() + datay[i] - by[i]);
	}
}

int GetPeakType() {
	return 17;
}


int phaseDimensions(PhaseType phase) {
	switch (phase) {
		case PHASE_LAMELLAR_1D:
			return 1;
		case PHASE_2D:
		case PHASE_HEXAGONAL_2D:
		case PHASE_CENTERED_RECTANGULAR_2D:
		case PHASE_RECTANGULAR_2D:
		case PHASE_SQUARE_2D:
			return 2;

		default:
		case PHASE_NONE:
		case PHASE_3D:
		case PHASE_RHOMBO_3D:
		case PHASE_HEXA_3D:
		case PHASE_MONOC_3D:
		case PHASE_ORTHO_3D:
		case PHASE_TETRA_3D:
		case PHASE_CUBIC_3D:
			return 3;
	}
}

void SetPeakType(PeakType sgh) {}

//documentation :: 
// q is the peak index we are dealing with
// &a are the params, ma is the phase and nd the dimention
void BuildAMatrixForPhases(VectorXd  &a, MatrixXd &g, PhaseType phase) {
	double sqroot;

	switch (phase) {
		case PHASE_RHOMBO_3D:
			a[1] = a[3] = a[0];
			a[4] = a[5] = a[2];

			sqroot = sqrt( sin(a[4])*sin(a[4]) - 
				( cos(a[5]) - cos(a[4]) * cos (a[2])) / sin(a[2]) 
				* ( cos(a[5]) - cos(a[4]) * cos (a[2]) )/ sin(a[2]) );

			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(0,2) =   2.0 * PI / (a[0] * sin (a[2]) * sqroot ) *   
				( (cos(a[5]) - cos(a[4]) * cos (a[2]) )/ tan (a[2]) - cos(a[4]) * sin (a[2]) )  ;
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]) );
			g(2,1) = - 2.0 * PI / (a[1] * sin (a[2]) * sin (a[2]) * sqroot ) * (cos(a[5]) - cos(a[4]) *cos (a[2]));
			g(2,0) =   0;
			g(2,1) =   0;
			g(2,2) =   2.0 * PI / (a[3] * sqroot );
			break;


		case PHASE_HEXA_3D:
			a[1] = a[0];
			a[4] = a[5] = PI / 2.0;
			a[2] = 2.0 * PI / 3.0;

			sqroot = sqrt( sin(a[4])*sin(a[4]) - 
				( cos(a[5]) - cos(a[4]) * cos (a[2])) / sin(a[2]) 
				* ( cos(a[5]) - cos(a[4]) * cos (a[2]) )/ sin(a[2]) );

			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(0,2) =   2.0 * PI / (a[0] * sin (a[2]) * sqroot ) *   
				( (cos(a[5]) - cos(a[4]) * cos (a[2]) )/ tan (a[2]) - cos(a[4]) * sin (a[2]) )  ;
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]) );
			g(2,1) = - 2.0 * PI / (a[1] * sin (a[2]) * sin (a[2]) * sqroot ) * (cos(a[5]) - cos(a[4]) *cos (a[2]));
			g(2,0) =   0;
			g(2,1) =   0;
			g(2,2) =   2.0 * PI / (a[3] * sqroot );
			break;



		case PHASE_CUBIC_3D:
			a[3] = a[1] = a[0];
			a[4] = a[5] = a[2] = PI / 2.0;
		case PHASE_TETRA_3D:
			a[4] = a[5] = a[2] = PI / 2.0;
			a[1] = a[0];
		case PHASE_ORTHO_3D:
			a[4] = a[5] = a[2] = PI / 2.0;
		case PHASE_MONOC_3D:
			a[2] = a[5] = PI / 2.0;
		case PHASE_3D:
		case PHASE_NONE:
		default:	
			sqroot = sqrt( sin(a[4])*sin(a[4]) - 
				( cos(a[5]) - cos(a[4]) * cos (a[2])) / sin(a[2]) 
				* ( cos(a[5]) - cos(a[4]) * cos (a[2]) )/ sin(a[2]) );

			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(0,2) =   2.0 * PI / (a[0] * sin (a[2]) * sqroot ) *   
				( (cos(a[5]) - cos(a[4]) * cos (a[2]) )/ tan (a[2]) - cos(a[4]) * sin (a[2]) )  ;
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]) );
			g(2,1) = - 2.0 * PI / (a[1] * sin (a[2]) * sin (a[2]) * sqroot ) * (cos(a[5]) - cos(a[4]) *cos (a[2]));
			g(2,0) =   0;
			g(2,1) =   0;
			g(2,2) =   2.0 * PI / (a[3] * sqroot );
			break;
		case PHASE_LAMELLAR_1D:
			g(0,0) =   2.0 * PI / (a[0]);
			break;
		case PHASE_HEXAGONAL_2D:
			a[1] = a[0];
			a[2] = PI / 1.5;			
		case PHASE_2D:
			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]));
			break; 
		case PHASE_SQUARE_2D:
			a[1] = a[0];
		case PHASE_RECTANGULAR_2D:
			a[2] = PI / 2.0;
			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]));
			break; 
		case PHASE_CENTERED_RECTANGULAR_2D:
			a[2] = PI / 4.0;			
			g(0,0) =   2.0 * PI / (a[0]);
			g(0,1) = - 2.0 * PI / (a[0] * tan (a[2]));
			g(1,0) =   0;
			g(1,1) =   2.0 * PI / (a[1] * sin (a[2]));
			break; 
	}

	//MessageBoxA(NULL, debugMatrixPrint(g).c_str(), "G", NULL);
}

// Creating matrix of indices
MatrixXi GenerateIndicesMatrix(int dim, int length) {
	MatrixXi result = MatrixXi::Zero(length, dim);



	int help = 0;
	int help1 = 0;
	int base = 2;
	for(int i = 0; i < result.rows(); i++) {
		help1++;
		help = help1;
		for (int j = dim - 1; j >= 0 ; j--) {
			result(i, j) = (help % base);
			help /= base;
		}
		for (int m=0; m < i; m++ ) {
			if (result.row(m) == result.row(i)) {
				i--;
				break;
			}
		}

		bool equals = (result(i, 0) ==  base - 1);
		for (int j = dim - 1; j > 0 ; j--) 
			equals &= ((result(i, j) == result(i, j-1)) &&
			(result(i, j-1) ==  base - 1));
		if(equals) {
			base++;
			help = 0;
			help1 = 0;
		}

	}

	//MessageBoxA(NULL, debugMatrixPrint(result.cast<double>()).c_str(), "indices", NULL);

	return result;
}

void uniqueIndices(const VectorXd& G_norm, std::vector<double>& result, std::vector<std::string> &locs) {
	result.clear();

	int s = 1;
	for (int i = 0; i<G_norm.size(); i++)
		result.push_back(G_norm[i]);

	while (s < (int)result.size()) 	{
		if ( fabs(result[s-1] - result[s]) <= 1e-7) {		
			result.erase(result.begin() + s);
			locs[s - 1].append(" , " + locs[s]);
			locs.erase(locs.begin() + s);
			continue;

		}
		s++;
	}

}

std::vector <double> GenPhases (PhaseType phase , phaseStruct *p,
								std::vector<std::string> &locs) {
	// Initialization
	int ma,dim;
	// Initializing vectors
	VectorXd a;  // Parameter vector

	VectorXd params; 
	VectorXi ia; // Mutability vector
	cons a_min(7); // Fit range/constraint vector
	cons a_max(7); // Fit range/constraint vector

	// a paramter to pass to ma 
	// of the dummy function that should be like a numerator for the different phases

	ma = 7;
	a  = VectorXd::Zero(ma);  
	ia = VectorXi::Zero(ma);
	a[0] = p->a;
	a[1] = p->b;
	a[2] = p->gamma;
	a[3] = p->c;
	a[4] = p->alpha;
	a[5] = p->beta;
	//a[6] = p->um;

	ia[0] = p->aM == 'Y';
	ia[1] = p->bM == 'Y';
	ia[2] = p->gammaM == 'Y';
	ia[3] = p->cM == 'Y';
	ia[4] = p->alphaM == 'Y';
	ia[5] = p->betaM == 'Y';
	ia[6] = p->umM == 'Y';

	a_min.num[0] = p->amin;
	a_min.num[1] = p->bmin;
	a_min.num[2] = p->gammamin;
	a_min.num[3] = p->cmin;
	a_min.num[4] = p->alphamin;
	a_min.num[5] = p->betamin;
	a_min.num[6] = p->ummin;

	a_max.num[0] = p->amax;
	a_max.num[1] = p->bmax;
	a_max.num[2] = p->gammamax;
	a_max.num[3] = p->cmax;
	a_max.num[4] = p->alphamax;
	a_max.num[5] = p->betamax;
	a_max.num[6] = p->ummax;

	dim = phaseDimensions(phase);

	params = VectorXd::Zero(ma);

	MatrixXd g = MatrixXd::Zero(dim,dim);
	BuildAMatrixForPhases(a,g, phase);

	MatrixXi ind = GenerateIndicesMatrix(dim, dim * dim * 100);

	MatrixXd G = ind.cast<double>()*g;
	VectorXd G_norm = VectorXd::Zero(G.rows());
	for (int i = 0; i < G.rows(); i++) 
		G_norm[i] = G.row(i).norm();

	//Sort the rows of G according to the norm
	for (int a = 0; a < G.rows(); a++) {
		for (int b = a + 1; b < G.rows(); b++) {
			if(G_norm[a] > G_norm[b]) {
				double c = G_norm[a];
				G_norm[a] = G_norm[b];
				G_norm[b] = c;
				G.row(a).swap(G.row(b));
				ind.row(a).swap(ind.row(b));
			}
		}
		std::stringstream s;

		s << "(" <<ind(a,0);
		for (int d = 1; d < dim; d++) 
			s << ","<< ind(a,d);
		s << ")";
		locs.push_back(s.str());
	}

	std::vector <double> G_N;	
	uniqueIndices(G_norm, G_N, locs);

	return G_N;

}

void SetSigma(bool sdfg) {}

void SetD(double weru) {}

void SetPDResolution(double njhsdf) {}

void ReadPeaks(std::wstring filename, std::string type, peakStruct *peaks, void* iniFile) {}

void ReadBG(std::wstring filename, std::string type, bgStruct *bg, void* iniFile) {}

void ReadPhases(std::wstring filename, std::string type, phaseStruct *bg, int* sdftfth, void* iniFile) {}

bool GenerateStructureFactor(const std::vector<double> x, std::vector<double>& y, peakStruct *p) {return false;}

bool GenerateBackground(const std::vector<double> bgx, std::vector<double>& genY,
						bgStruct *p) { return false; }

void setAccuracySettings(bool accurateFitting, bool accurateDerivative, bool wssrFitting, bool logFitting) {}

void SetQuadResolution(int res) {}

void SetFitIterations(int iterations) {}

void SetFitMethod(FitMethod method) {}

PeakType GetPDFunc() {return SHAPE_GAUSSIAN;}

int GetPDResolution() { return DEFAULT_PDRES;}

void WritePeaks(const std::wstring &filename, std::string obj, const peakStruct *peaks, void* ini) {}
void WritePeaks(const std::wstring &filename, std::string obj, const paramStruct *peaks, void* ini) {}

void WritePhases(const std::wstring &filename, std::string obj, const phaseStruct *ph, int pt, void* ini) {}

void WriteBG(const std::wstring &filename, std::string obj, const bgStruct *BGs, void* ini) {}

void SetResolution(double e3g) {}

void SetUseFrozenFF(bool crap) {}

bool GenerateBackground(const std::wstring datafile,std::vector <double>& bgx,
						std::vector <double>& bgy, std::vector <double>& ffy, bool angstrom) {return false;}
