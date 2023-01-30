#define AMP_EXPORTER

#include "../backend_version.h"

#include "PDBAmplitude.h"
#include <math.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include "boost/filesystem/fstream.hpp"
#include <random>
// #include "boost/random/uniform_real_distribution.hpp"
// #include "boost/random/uniform_int_distribution.hpp"
// #include "boost/random/random_device.hpp"
// #include "boost/random/mersenne_twister.hpp"
#include "boost/filesystem.hpp"
#include "boost/multi_array.hpp"
#include "boost/cstdint.hpp"
#include <boost/lexical_cast.hpp>
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iomanip>
#include <queue>
#include <algorithm>    // std::sort
#include "md5.h"
#include "../../../Common/ZipLib/Source/ZipFile.h"
#include "../../../Common/ZipLib/Source/streams/memstream.h"
#include "../../../Common/ZipLib/Source/methods/StoreMethod.h"
#include "../GPU/GPUInterface.h"
#include <cuda_runtime.h> // For getdevicecount and memory copies
#include <vector_functions.h>
#include <rapidjson/document.h>
#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>


#ifdef _WIN32
#include <windows.h> // For LoadLibrary
#pragma comment(lib, "user32.lib")
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym 
#endif

#include "Grid.h"

#include <boost/math/special_functions/next.hpp>

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include "AmplitudeCache.h"
#include "BackendInterface.h"
#include "LocalBackend.h"

#include <GPUHeader.h>

using Eigen::ArrayXd;
using Eigen::Matrix3d;
using std::stringstream;

#include "UseGPU.h"



// Let's limit the number of multiple declarations to 1
#include "declarations.h"
#include "UseGPU.h"

/* C++11...
template<typename dataFType, typename interpFType>
using GPUCalculateMCOA_t = int (*)(long long voxels, int thDivs, int phDivs, dataFType stepSz,
dataFType *inAmpData,  interpFType *inD,
dataFType *qs, dataFType *intensities, int qPoints,
long long maxIters, dataFType convergence,
progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/
typedef double dataFType;
typedef double interpFType;
typedef int(*GPUCalculateMCOA_t)(long long voxels, int thDivs, int phDivs, dataFType stepSz,
	dataFType *inAmpData, interpFType *inD,
	dataFType *qs, dataFType *intensities, int qPoints,
	long long maxIters, dataFType convergence,
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);

typedef int(*GPUSumGridsJacobSphr_t) (long long voxels, int thDivs, int phDivs, dataFType stepSz,
	dataFType **inAmpData, interpFType **inD, dataFType *trans, dataFType *rots,
	int numGrids, dataFType *outAmpData,
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);

typedef bool(*GPUHybridCalcAmpGrid_t)(GridWorkspace& work, double*);
GPUHybridCalcAmpGrid_t gpuAmpGridHybridAmplitude = NULL;

template <typename K, typename V>
class pairfirst_less : public std::binary_function<std::pair<K, V>, std::pair<K, V>, bool>
{
public:
	bool operator()(const std::pair<K, V> &left, const std::pair<K, V> &right) const
	{
		return left.first < right.first;
	}
};

template <typename TT>
class rotation_less : public std::binary_function<TT, TT, bool>
{
public:
	rotation_less(double arg_ = 1e-7) : epsilon(arg_) {}
	bool operator()(const TT &left, const TT &right) const
	{
		if (abs(left.x - right.x) > epsilon)
			return left.x < right.x;
		if (abs(left.y - right.y) > epsilon)
			return left.y < right.y;
		if (abs(left.z - right.z) > epsilon)
			return left.z < right.z;

		// They're equal
		return false;
	}
	double epsilon;
};

class rotation_ulp_less : public std::binary_function<float4, float4, bool>
{
public:
	rotation_ulp_less(int arg_ = 64) : ulps(arg_) {}
	bool operator()(const float4 &left, const float4 &right) const
	{
		if (abs(boost::math::float_distance(left.x, right.x)) > ulps)
			return left.x < right.x;
		if (abs(boost::math::float_distance(left.y, right.y)) > ulps)
			return left.y < right.y;
		if (abs(boost::math::float_distance(left.z, right.z)) > ulps)
			return left.z < right.z;

		// They're equal
		return false;
	}
	int ulps;
};

class paramvec_less : public std::binary_function<VectorXd, VectorXd, bool>
{
public:
	paramvec_less(double arg_ = 1e-10) : epsilon(arg_) {}
	bool operator()(const VectorXd &left, const VectorXd &right) const
	{
		// Number of parameters (layers)
		if (left.size() != right.size())
			return left.size() < right.size();

		// Parameters
		unsigned int leftSize = left.size();
		for (int k = 0; k < leftSize; k++)
		{
			if (abs(left[k] - right[k]) > epsilon)
				return (left[k] < right[k]);
		}

		// They're equal
		return false;
	}
	double epsilon;
};

typedef std::tuple<std::string, VectorXd> UniqueModels;

class UniqueModels_less : public std::binary_function<UniqueModels, UniqueModels, bool>
{
public:
	UniqueModels_less() {}
	bool operator()(const UniqueModels &left, const UniqueModels &right) const
	{
		int strRes = std::get<0>(left).compare(std::get<0>(right));
		if (strRes != 0)
			return (strRes > 0);

		paramvec_less cmp;
		return cmp(std::get<1>(left), std::get<1>(right));
	}
};


namespace fs = boost::filesystem;

#pragma region General functions and templates
typedef std::vector<std::vector<std::vector<std::complex<FACC> > > >::size_type	stx;
typedef std::vector<std::vector<std::complex<FACC> > >::size_type				sty;
typedef std::vector<std::complex<FACC> >::size_type								stz;


#pragma region Gauss Legendre
/*Copied from alglib*/
void buildgausslegendrequadrature(int n, double a, double b,
	ArrayXd& x, ArrayXd& w)
{
	double r;
	double r1;
	double p1;
	double p2;
	double p3;
	double dp3;
	const double epsilon = std::numeric_limits<double>::epsilon();

	x = ArrayXd::Zero(n);
	w = ArrayXd::Zero(n);
	for (int i = 0; i <= (n + 1) / 2 - 1; i++)
	{
		r = cos(M_PI*(4 * i + 3) / (4 * n + 2));
		do
		{
			p2 = 0;
			p3 = 1;
			for (int j = 0; j < n; j++)
			{
				p1 = p2;
				p2 = p3;
				p3 = ((2 * j + 1)*r*p2 - j * p1) / (j + 1);
			}
			dp3 = n * (r*p3 - p2) / (r*r - 1);
			r1 = r;
			r = r - p3 / dp3;
		} while (fabs(r - r1) >= epsilon * (1 + fabs(r)) * 100);
		x(i) = r;
		x(n - 1 - i) = -r;
		w(i) = 2 / ((1 - r * r)*dp3*dp3);
		w(n - 1 - i) = 2 / ((1 - r * r)*dp3*dp3);
	}

	for (int i = 0; i < n; i++)
	{
		x[i] = 0.5*(x[i] + 1)*(b - a) + a;
		w[i] = 0.5*w[i] * (b - a);
	}
}

void SetupIntegral(ArrayXd& x, ArrayXd& w,
	double s, double e, int steps) {
	if (steps == 0)
		return;
	if (x.size() != steps ||
		x.size() != w.size() ||
		fabs(x[steps - 1] - e) > x[steps - 1] / steps ||
		fabs(x[0] - s) > (x[steps - 1] - x[0]) / steps)
		buildgausslegendrequadrature(steps, s, e, x, w);
}
#pragma endregion	// Gauss Legendre

void PrintTime() {
	char buff[100];
	time_t now = time(0);
	strftime(buff, 100, "%Y-%m-%d %H:%M:%S", localtime(&now));
	std::cout << buff;
}


#pragma endregion	// General functions and templates

#pragma region Geometric Amplitude class

GeometricAmplitude::GeometricAmplitude(FFModel *mod) : model(mod) {
}

GeometricAmplitude::~GeometricAmplitude() {
}

void GeometricAmplitude::OrganizeParameters(const VectorXd& p, int nLayers) {
	Amplitude::OrganizeParameters(p, nLayers);

	modelLayers = nLayers;

	size_t origsize = p.size();
	if (origsize == 0)
		return;

	if (model)
		model->OrganizeParameters(p, nLayers);
}

void GeometricAmplitude::PreCalculate(VectorXd& p, int nLayers) {
	Amplitude::PreCalculate(p, nLayers);

	OrganizeParameters(p, nLayers);

	if (model)
		model->PreCalculateFF(p, nLayers);
}


void GeometricAmplitude::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String(this->model->GetName().c_str());


	writer.Key("Position");
	writer.StartArray();
	writer.Double(tx);
	writer.Double(ty);
	writer.Double(tz);
	writer.EndArray();

	writer.Key("Rotation");
	writer.StartArray();
	writer.Double(ra);
	writer.Double(rb);
	writer.Double(rg);
	writer.EndArray();

	writer.Key("Used Grid");
	writer.Bool(bUseGrid);

	if (bUseGrid && grid) {
		writer.Key("N");
		writer.Double(GetGridSize());
		writer.Key("qMax");
		writer.Double(grid->GetQMax());
		writer.Key("StepSize");
		writer.Double(grid->GetStepSize());
	}

	writer.Key("Parameter Names");
	writer.StartArray();
	for (int i = 0; i < model->GetNumLayerParams(); i++)
		writer.String(model->GetLayerParamName(i, NULL).c_str());
	writer.EndArray();

	writer.Key("Layer Parameters");
	writer.StartArray();
	std::vector<double> pars;
	int cnt = 0;
	pars = model->GetAllParameters();
	for (int i = 0; i < modelLayers; i++) {
		writer.StartArray();
		for (int j = 0; j < model->GetNumLayerParams(); j++) {
			writer.Double( pars[j * modelLayers + i]);
			cnt++;
		}
		writer.EndArray();
	}
	writer.EndArray();

	writer.Key("Extra Parameters");
	writer.StartArray();
	// Extra parameters
	for (int i = 0; i < model->GetNumExtraParams(); i++) {
		std::string ename(model->GetExtraParameter(i).name);
		writer.Key(ename.c_str());
		writer.Double(pars[cnt++]);
	}
	writer.EndArray();
}

void GeometricAmplitude::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers;
	ampers.resize(depth + 1, '#');
	ampers.append(" ");

	std::stringstream ss;

	if (depth == 0) {
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
	}

	header.append(ampers + "//////////////////////////////////////\n");

	ss << "Geometric model\t" << this->model->GetName() << "\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Position (" << tx << "," << ty << "," << tz << ")\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Rotation (" << ra << "," << rb << "," << rg << ")\n";
	header.append(ampers + ss.str());
	ss.str("");

	ss << "Grid was " << (bUseGrid ? "" : "not ") << "used.\n";
	header.append(ampers + ss.str());
	ss.str("");

	if (bUseGrid && grid) {
		ss << "N^3; N = " << GetGridSize() << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "qMax = " << grid->GetQMax() << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Grid step size = " << grid->GetStepSize() << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

	ss << "Parameters:";
	for (int i = 0; i < model->GetNumLayerParams(); i++)
		ss << "\t" << model->GetLayerParamName(i, NULL);
	ss << "\n";
	header.append(ampers + ss.str());
	ss.str("");

	std::vector<double> pars;
	int cnt = 0;
	pars = model->GetAllParameters();
	char tempBuf[20];
	for (int i = 0; i < modelLayers; i++) {
		std::string layerName = model->GetLayerName(i);
		if (std::string::npos != layerName.find("%d"))
		{
			sprintf(tempBuf, layerName.c_str(), i);
			ss << tempBuf << ":";
		}
		else {
			ss << model->GetLayerName(i) << ":";
		}

		for (int j = 0; j < model->GetNumLayerParams(); j++) {
			ss << "\t" << pars[j * modelLayers + i];
			cnt++;
		}
		ss << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}
	// Extra parameters
	for (int i = 0; i < model->GetNumExtraParams(); i++) {
		ss << model->GetExtraParameter(i).name << "\t" << pars[cnt++] << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

}

std::string GeometricAmplitude::Hash() const
{
	std::string str = BACKEND_VERSION "Geometric: ";
	if (model)
		str += model->GetName();
	else
		str += "N/A";
	auto ps = model->GetAllParameters();
	for (const auto& pp : ps)
		str += std::to_string(pp);

	return md5(str);
}

std::string GeometricAmplitude::GetName() const {
	return model->GetName();
}

bool GeometricAmplitude::SetModel(Workspace& workspace)
{
	IGPUCalculable *calc = dynamic_cast<IGPUCalculable *>(model);
	if (NULL != calc)
	{
		return calc->SetModel(workspace);
	}

	printf("%s is not a GPU calculable geometric model, skipping\n", model->GetName().c_str());
	return false;
}

bool GeometricAmplitude::SetParameters(Workspace& workspace, const double *params, unsigned int numParams)
{
	IGPUCalculable *calc = dynamic_cast<IGPUCalculable *>(model);
	if (NULL != calc)
	{
		return calc->SetParameters(workspace, params, numParams);
	}
	return false;
}

bool GeometricAmplitude::ComputeOrientation(Workspace& workspace, float3 rotation)
{
	IGPUCalculable *calc = dynamic_cast<IGPUCalculable *>(model);
	if (NULL != calc)
	{
		return calc->ComputeOrientation(workspace, rotation);
	}
	return false;
}

void GeometricAmplitude::CorrectLocationRotation(double& x, double& y, double& z, double& alpha, double& beta, double& gamma)
{
	IGPUCalculable *calc = dynamic_cast<IGPUCalculable *>(model);
	if (NULL != calc)
	{
		calc->CorrectLocationRotation(x, y, z, alpha, beta, gamma);
	}
}

bool GeometricAmplitude::CalculateGridGPU(GridWorkspace& workspace)
{
	IGPUGridCalculable *calc = dynamic_cast<IGPUGridCalculable *>(model);
	if (NULL != calc)
	{
		return calc->CalculateGridGPU(workspace);
	}

	printf("%s is not a hybrid GPU calculable geometric model, will be calculated "
		"on the CPU and transfered\n", model->GetName().c_str());
	{
		bUseGrid = true;
		calculateGrid(workspace.qMax, 2 * (workspace.qLayers - 4));
		//bUseGrid = false;

		double3 tr, rt;
		GetTranslationRotationVariables(tr.x, tr.y, tr.z, rt.x, rt.y, rt.z);

		if (!g_useGPUAndAvailable)
			return false;

		if (!gpuAmpGridHybridAmplitude)
			gpuAmpGridHybridAmplitude = (GPUHybridCalcAmpGrid_t)GPUHybrid_AmpGridAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_AmpGridAmplitudeDLL");
		if (!gpuAmpGridHybridAmplitude)
			return false;

		return gpuAmpGridHybridAmplitude(workspace, grid->GetDataPointer());

	}
	return false;
}

bool GeometricAmplitude::SetModel(GridWorkspace& workspace)
{
	IGPUGridCalculable *calc = dynamic_cast<IGPUGridCalculable *>(model);
	if (NULL != calc)
	{
		return calc->SetModel(workspace);
	}

	printf("%s is not a hybrid GPU calculable geometric model, will be calculated "
		"on the CPU and transfered\n", model->GetName().c_str());
	return false;
}

bool GeometricAmplitude::ImplementedHybridGPU() {
	return dynamic_cast<IGPUGridCalculable*>(model) != NULL;
}

#pragma endregion	// Geometric Amplitude class

#pragma region Domain (Intensity Calculator) class

Eigen::Matrix4f Euler4D(Radian theta, Radian phi, Radian psi) {
	float ax, ay, az, c1, c2, c3, s1, s2, s3;
	ax = theta;
	ay = phi;
	az = psi;
	c1 = cos(ax); s1 = sin(ax);
	c2 = cos(ay); s2 = sin(ay);
	c3 = cos(az); s3 = sin(az);
	Eigen::Matrix4f rot = Eigen::MatrixXf::Identity(4, 4);

	// Tait-Bryan angles X1Y2Z3 (x-alpha, y-beta, z-gamma)
	rot(0, 0) = c2 * c3;			rot(0, 1) = -c2 * s3;			rot(0, 2) = s2;
	rot(1, 0) = c1 * s3 + c3 * s1*s2;	rot(1, 1) = c1 * c3 - s1 * s2*s3;	rot(1, 2) = -c2 * s1;
	rot(2, 0) = s1 * s3 - c1 * c3*s2;	rot(2, 1) = c3 * s1 + c1 * s2*s3;	rot(2, 2) = c1 * c2;
	return rot;
}

typedef std::pair<LocationRotation, double> LocRotScale;

template <bool bHybrid>
void Flatten(Amplitude *amp, const VectorXd& ampParams, int ampLayers,
	const Eigen::Matrix4f &tMatC, std::vector<Amplitude *>& flatVec,
	std::vector<LocRotScale>& flatLocrot,
	std::vector<VectorXd>& flatParams,
	double totalScale = 1.0
)
{
	// Multiply new matrix from the left => Appends transformation
	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f R = Euler4D(amp->ra, amp->rb, amp->rg);

	//std::cout << "XYZABG(): " << amp->tx << " " << amp->ty << " " << amp->tz << " " << amp->ra << " " << amp->rb << " " << amp->rg << std::endl;
	//std::cout << "before: " << std::endl << tMat << std::endl;

	T(0, 3) = amp->tx; T(1, 3) = amp->ty; T(2, 3) = amp->tz;
	Eigen::Matrix4f tMat = tMatC * T * R;

	//std::cout << "after: " << std::endl << tMat << std::endl;

	// If symmetry, get root positions
	if ((!bHybrid && dynamic_cast<ISymmetry *>(amp)) ||	// Direct
		(bHybrid && !amp->GetUseGridWithChildren() && dynamic_cast<ISymmetry *>(amp))	// Hybrid
		)
	{
		ISymmetry *sym = dynamic_cast<ISymmetry *>(amp);
		sym->Populate(ampParams, ampLayers);
		unsigned int totalLocrots = sym->GetNumSubLocations();
		unsigned int subamps = sym->GetNumSubAmplitudes();

		std::vector< std::tuple<Amplitude *, VectorXd, int> > children;
		for (unsigned int i = 0; i < subamps; i++)
		{
			VectorXd subparams;
			int sublayers;
			sym->GetSubAmplitudeParams(i, subparams, sublayers);
			children.push_back(std::make_tuple(sym->GetSubAmplitude(i), subparams, sublayers));
		}

		for (unsigned int i = 0; i < totalLocrots; i++)
		{
			LocationRotation subloc = sym->GetSubLocation(i);
			T(0, 3) = subloc.x; T(1, 3) = subloc.y; T(2, 3) = subloc.z;
			R = Euler4D(subloc.alpha, subloc.beta, subloc.gamma);

			// First rotate, then translate
			Eigen::Matrix4f newmat = tMat * T * R;

			// Multiply new matrix from the left - Append transformation
			for (unsigned int j = 0; j < subamps; j++)
			{
				Flatten<bHybrid>(std::get<0>(children[j]), std::get<1>(children[j]),
					std::get<2>(children[j]), newmat, flatVec, flatLocrot, flatParams, totalScale * amp->scale);
			}
		}
	}
	else
	{
		// Obtain final entity position (by transforming (0,0,0[,1]) with the matrix)
		Eigen::Vector4f origin = Eigen::Vector4f::Zero();
		origin[3] = 1.0f; // (0,0,0,1)
		origin = tMat * origin;

		// For rotation, transform (1,0,0[,1]), (0,1,0[,1]) and (0,0,1[,1]).
		// The [,1] is for homogeneous coordinates
		// From (0,0,0) derive position
		// From the rest of the vectors, derive orientation	
		Eigen::Vector4f xbase = Eigen::Vector4f::Zero(), xbasenew;
		Eigen::Vector4f ybase = Eigen::Vector4f::Zero(), ybasenew;
		Eigen::Vector4f zbase = Eigen::Vector4f::Zero(), zbasenew;
		xbase[0] = 1.0f; xbase[3] = 1.0f; // (1,0,0,1)
		ybase[1] = 1.0f; ybase[3] = 1.0f; // (0,1,0,1)
		zbase[2] = 1.0f; zbase[3] = 1.0f; // (0,0,1,1)
		xbasenew = (tMat * xbase) - origin;
		ybasenew = (tMat * ybase) - origin;
		zbasenew = (tMat * zbase) - origin;

		Eigen::Matrix3f basemat;
		basemat(0, 0) = xbasenew[0]; basemat(0, 1) = ybasenew[0]; basemat(0, 2) = zbasenew[0];
		basemat(1, 0) = xbasenew[1]; basemat(1, 1) = ybasenew[1]; basemat(1, 2) = zbasenew[1];
		basemat(2, 0) = xbasenew[2]; basemat(2, 1) = ybasenew[2]; basemat(2, 2) = zbasenew[2];

		//std::cout << basemat << std::endl;

		Radian alpha, beta, gamma;
		Eigen::Quaternionf qt(basemat);
		GetEulerAngles(qt, alpha, beta, gamma);

		// If beta is in the 2nd or 3rd quadrants, we should add 180 modulo 360 to GAMMA and ALPHA
		// but we cannot know because of how rotation matrices work with Euler/Tait-Bryan angles.

		flatVec.push_back(amp);
		flatLocrot.push_back(std::make_pair(
			LocationRotation(origin.x(), origin.y(), origin.z(),
				alpha, beta, gamma),
			totalScale)
		);
		flatParams.push_back(ampParams);
	}
}

template <bool bHybrid>
void FlattenTree(const std::vector<Amplitude *> amps, const std::vector<VectorXd>& ampParams,
	const std::vector<int>& ampLayers,
	Eigen::Matrix4f& tMat, std::vector<Amplitude *>& flatVec,
	std::vector<LocRotScale>& flatLocrot,
	std::vector<VectorXd>& flatParams)
{
	int i = 0;
	for (auto iter = amps.begin(); iter != amps.end(); ++iter)
	{
		Flatten<bHybrid>(*iter, ampParams[i], ampLayers[i], tMat, flatVec, flatLocrot, flatParams);
		i++;
	}
}

template< typename tPair >
struct first_t {
	typename tPair::first_type operator()(const tPair& p) const { return     p.first; }
};

template< typename tMap >
first_t< typename tMap::value_type > first(const tMap& m) { return first_t<     typename tMap::value_type >(); }

template <typename T>
PDB_READER_ERRS DomainModel::CalculateIntensityVector(const std::vector<T>& Q,
	std::vector<T>& res, T epsi, uint64_t iterations)
{
	clock_t gridBegin, aveBeg, aveEnd;

	/*
	FILE origSTDOUT = *stdout;

	static FILE* nul = fopen( "NUL", "w" );
	*stdout = *nul;
	setvbuf( stdout, NULL, _IONBF, 0 );
	*/

	if (only_scale_changed
		&& _previous_hash == Hash()
		&& _previous_intensity.size() > 1
		&& _previous_q_values.size() == Q.size() && (_previous_q_values == Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>()).all())
	{
		Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size()) = _previous_intensity.cast<T>();
		return PDB_OK;
	}
	_previous_intensity.resize(0);
	_previous_q_values.resize(0);

	gridBegin = clock();
	aveBeg = clock();

	// Determine if we have and want to use a GPU
	{
		int devCount;
		cudaError_t t = UseGPUDevice(&devCount);
		g_useGPUAndAvailable = (devCount > 0 && bDefUseGPU && t == cudaSuccess);
	}

	//split by calculation type:
	if (orientationMethod == OA_ADAPTIVE_MC_VEGAS)
	{
		if (!g_useGPUAndAvailable) //cpu
			throw backend_exception(ERROR_UNIMPLEMENTED_CPU, g_errorStrings[ERROR_UNIMPLEMENTED_CPU]);

		PDB_READER_ERRS errRes = PerformGPUHybridComputation(gridBegin, Q, aveBeg, res, epsi, iterations, aveEnd);
		if (errRes == PDB_OK)
			setPreviousValues(res, Q);
		else
		{
			if (errRes == ERROR_WITH_GPU_CALCULATION_TRY_CPU) //space filling
				throw backend_exception(ERROR_UNIMPLEMENTED_CPU, g_errorStrings[ERROR_UNIMPLEMENTED_CPU]);
			throw backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]); 
		}

		return errRes;

	}

	//some additional initialization needed for MC and GK
	srand((unsigned int)(time(NULL)));
	std::vector<unsigned int> seeds(Q.size());
	const double cProgMin = 0.3, cProgMax = 1.0;

	int prog = 0;
	std::mt19937 seedGen;
	std::uniform_int_distribution<unsigned int> sRan(0, UINT_MAX);
	seedGen.seed(static_cast<unsigned int>(std::time(0)));

	for (unsigned int i = 0; i < Q.size(); i++)
		seeds[i] = sRan(seedGen);

	if (orientationMethod == OA_MC)
	{
		if (!GetUseGridWithChildren() && g_useGPUAndAvailable) //use gpu without use grid at the top level--
			//this also includes use gpu without use grid *at all* - using gpu will always treat leaves as grids
		{
			PDB_READER_ERRS errRes = PerformGPUHybridComputation(gridBegin, Q, aveBeg, res, epsi, iterations, aveEnd);
			if (errRes == PDB_OK)
				setPreviousValues(res, Q);
			if (errRes != ERROR_WITH_GPU_CALCULATION_TRY_CPU)
				return errRes;
			// Else, continue
		}

		if (bDefUseGrid) {
			gridComputation();
		}
		if (pStop && *pStop)
			return STOPPED;

		if (g_useGPUAndAvailable && bDefUseGrid && GetUseGridWithChildren()) //use gpu with use grid at top level node
			//not that bDefUseGrid immediately means GetUseGridWithChildren()
			//hence the extra check is redundant, presumably to shield against something having gone wrong
		{
			PDB_READER_ERRS errRes = PerformgGPUAllGridsMCOACalculations(Q, res, iterations, epsi, aveEnd, aveBeg, gridBegin);
			if (errRes == PDB_OK)
			{
				setPreviousValues(res, Q);
			}
			if (errRes != ERROR_WITH_GPU_CALCULATION_TRY_CPU)
				return errRes;
			// Else, continue on the CPU
		}

		//cpu:

		// MC only: Attempt to do orientation averaging between layers
		bool canIntegratePointsBetweenLayersTogether = true;
		for (auto &subAmp : _amps)
		{
			canIntegratePointsBetweenLayersTogether &=
				subAmp->GetUseGridAnyChildren();
				//subAmp->GetUseGridWithChildren() // Has to use a grid and not be hybrid (for now)
				// The minimal requirement should be that all the leaves have grids
				//&
				// Has to be the JacobianSphereGrid (may also be a half grid in the future)
				//dynamic_cast<JacobianSphereGrid*>(subAmp->GetInternalGridPointer()) != nullptr;
				// This doesn't work if doing a hybrid calculation. I need to think of another test.
				
		}

		if (canIntegratePointsBetweenLayersTogether)
		{
			return IntegrateLayersTogether(seeds, sRan, seedGen, Q, res, epsi, iterations, cProgMax, cProgMin, prog, aveEnd, aveBeg, gridBegin);
		}

		
		return DefaultCPUCalculation(aveBeg, Q, res, epsi, seeds, iterations, cProgMax, cProgMin, prog, aveEnd, gridBegin);

	}

	if (orientationMethod == OA_ADAPTIVE_GK)
	{
		if (bDefUseGrid) {
			gridComputation();
		}

		if (pStop && *pStop)
			return STOPPED;

		return DefaultCPUCalculation(aveBeg, Q, res, epsi, seeds, iterations, cProgMax, cProgMin, prog, aveEnd, gridBegin);
	}

	return UNIMPLEMENTED; //we should never get here, but it's good to cover bases
}

template <typename T>
PDB_READER_ERRS DomainModel::CalculateIntensity2DMatrix(const std::vector<T>& Q,
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& res, T epsi, uint64_t iterations)
{
	std::cout << "!!!!!!! DomainModel::CalculateIntensity2DMatrix !!!!!!!" << std::endl;
	clock_t gridBegin, aveBeg, aveEnd;

	/*
	FILE origSTDOUT = *stdout;

	static FILE* nul = fopen( "NUL", "w" );
	*stdout = *nul;
	setvbuf( stdout, NULL, _IONBF, 0 );
	*/

	if (only_scale_changed
		&& _previous_hash == Hash()
		&& _previous_intensity_2D.rows() > 1
		&& _previous_q_values.size() == Q.size() && (_previous_q_values == Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>()).all())
	{
		res = _previous_intensity_2D.cast<T>();
		return PDB_OK;
	}
	_previous_intensity_2D = MatrixXd(0, 0);
	_previous_q_values.resize(0);

	gridBegin = clock();
	aveBeg = clock();

	// Determine if we have and want to use a GPU
	{
		int devCount;
		cudaError_t t = UseGPUDevice(&devCount);
		g_useGPUAndAvailable = (devCount > 0 && bDefUseGPU && t == cudaSuccess);
	}

	//split by calculation type:
	if (orientationMethod == OA_ADAPTIVE_MC_VEGAS)
	{
		if (!g_useGPUAndAvailable) //cpu
			throw backend_exception(ERROR_UNIMPLEMENTED_CPU, g_errorStrings[ERROR_UNIMPLEMENTED_CPU]);

		PDB_READER_ERRS errRes = PerformGPUHybridComputation2D(gridBegin, Q, aveBeg, res, epsi, iterations, aveEnd);
		if (errRes == PDB_OK)
			setPreviousValues2D(res, Q);
		else
		{
			if (errRes == ERROR_WITH_GPU_CALCULATION_TRY_CPU) //space filling
				throw backend_exception(ERROR_UNIMPLEMENTED_CPU, g_errorStrings[ERROR_UNIMPLEMENTED_CPU]);
			throw backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]);
		}

		return errRes;

	}

	//some additional initialization needed for MC and GK
	srand((unsigned int)(time(NULL)));
	std::vector<unsigned int> seeds(Q.size());
	const double cProgMin = 0.3, cProgMax = 1.0;

	int prog = 0;
	std::mt19937 seedGen;
	std::uniform_int_distribution<unsigned int> sRan(0, UINT_MAX);
	seedGen.seed(static_cast<unsigned int>(std::time(0)));

	for (unsigned int i = 0; i < Q.size(); i++)
		seeds[i] = sRan(seedGen);

	if (orientationMethod == OA_MC)
	{
		if (!GetUseGridWithChildren() && g_useGPUAndAvailable) //use gpu without use grid at the top level--
			//this also includes use gpu without use grid *at all* - using gpu will always treat leaves as grids
		{
			PDB_READER_ERRS errRes = PerformGPUHybridComputation2D(gridBegin, Q, aveBeg, res, epsi, iterations, aveEnd);
			if (errRes == PDB_OK)
				setPreviousValues2D(res, Q);
			if (errRes != ERROR_WITH_GPU_CALCULATION_TRY_CPU)
				return errRes;
			// Else, continue
		}

		if (bDefUseGrid) {
			gridComputation();
		}
		if (pStop && *pStop)
			return STOPPED;

		if (g_useGPUAndAvailable && bDefUseGrid && GetUseGridWithChildren()) //use gpu with use grid at top level node
			// not that bDefUseGrid immediately means GetUseGridWithChildren()
			// hence the extra check is redundant, presumably to shield against something having gone wrong
		{
			PDB_READER_ERRS errRes = PerformgGPUAllGridsMCOACalculations2D(Q, res, iterations, epsi, aveEnd, aveBeg, gridBegin);
			if (errRes == PDB_OK)
			{
				setPreviousValues2D(res, Q);
			}
			if (errRes != ERROR_WITH_GPU_CALCULATION_TRY_CPU)
				return errRes;
			// Else, continue on the CPU
		}

		//cpu:

		// MC only: Attempt to do orientation averaging between layers
		bool canIntegratePointsBetweenLayersTogether = true;
		for (auto& subAmp : _amps)
		{
			canIntegratePointsBetweenLayersTogether &=
				subAmp->GetUseGridAnyChildren();
			//subAmp->GetUseGridWithChildren() // Has to use a grid and not be hybrid (for now)
			// The minimal requirement should be that all the leaves have grids
			//&
			// Has to be the JacobianSphereGrid (may also be a half grid in the future)
			//dynamic_cast<JacobianSphereGrid*>(subAmp->GetInternalGridPointer()) != nullptr;
			// This doesn't work if doing a hybrid calculation. I need to think of another test.

		}

		std::vector<PolarCalculationData*> polarData = JacobianSphereGrid::QListToPolar(Q, qMin, qMax);

		if (canIntegratePointsBetweenLayersTogether)
		{

			IntegrateLayersTogether2D(seeds, sRan, seedGen, polarData, epsi, iterations, cProgMax, cProgMin, prog, aveEnd, aveBeg, gridBegin);
			res = JacobianSphereGrid::PolarQDataToCartesianMatrix(polarData, Q.size());
			
			_previous_intensity_2D = (Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size())).template cast<double>();
			_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>();
			
			return PDB_OK;

		}

		return DefaultCPUCalculation2D(aveBeg, Q, res, epsi, seeds, iterations, cProgMax, cProgMin, prog, aveEnd, gridBegin);

	}

	if (orientationMethod == OA_ADAPTIVE_GK)
	{
		if (bDefUseGrid) {
			gridComputation();
		}

		if (pStop && *pStop)
			return STOPPED;

		return DefaultCPUCalculation2D(aveBeg, Q, res, epsi, seeds, iterations, cProgMax, cProgMin, prog, aveEnd, gridBegin);
	}

	return UNIMPLEMENTED; //we should never get here, but it's good to cover bases
}

template<typename T>
PDB_READER_ERRS DomainModel::DefaultCPUCalculation(clock_t &aveBeg, const std::vector<T> & Q, std::vector<T> & res, T &epsi, std::vector<unsigned int> &seeds, const uint64_t &iterations, const double &cProgMax, const double &cProgMin, int &prog, clock_t &aveEnd, const clock_t &gridBegin)
{
	aveBeg = clock();

	bool noException = true;
	int exceptionInt = 0;
	std::string exceptionString = "";

#ifdef GAUSS_LEGENDRE_INTEGRATION
#pragma omp parallel sections
	{
#pragma omp section
		{
			SetupIntegral(theta_, wTheta, 0.0, M_PI, int(sqrt(double(iterations))));
		}
#pragma omp section
		{
			SetupIntegral(phi_, wPhi, 0.0, 2.0 * M_PI, int(sqrt(double(iterations))));
		}
}
#endif
	// TODO::Spherical - Simple trapezoidal integration should be considered, maybe spline


#pragma omp parallel for schedule(dynamic, Q.size() / 50)
	for (int i = 0; i < Q.size(); i++)
	{
		if (pStop && *pStop)
			continue;

		if (!noException)
			continue;

		try // This try/catch combo is here because of OMP
		{
			res[i] = CalculateIntensity(Q[i], epsi, seeds[i], iterations);
		}
		catch (backend_exception& ex)
		{
			exceptionInt = ex.GetErrorCode();
			exceptionString = ex.GetErrorMessage();
			noException = false;
		}

#pragma omp critical
		{
			if (progFunc)
				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(Q.size())) + cProgMin);
		}

	}

	if (!noException)
		throw backend_exception(exceptionInt, exceptionString.c_str());

	if (pStop && *pStop)
		return STOPPED;

	aveEnd = clock();

	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;

	/*
	*stdout = origSTDOUT;
	*/
	_previous_hash = Hash();
	_previous_intensity = (Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size())).template cast<double>();
	_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>();

	return PDB_OK;
}

template<typename T>
PDB_READER_ERRS DomainModel::DefaultCPUCalculation2D(clock_t& aveBeg, const std::vector<T>& Q, 
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& res, T& epsi, std::vector<unsigned int>& seeds, const uint64_t& iterations, const double& cProgMax, const double& cProgMin, int& prog, clock_t& aveEnd, const clock_t& gridBegin)
{
	std::cout << "DomainModel::DefaultCPUCalculation2D"<<std::endl;
	aveBeg = clock();

	bool noException = true;
	int exceptionInt = 0;
	std::string exceptionString = "";

	res.setZero();

#ifdef GAUSS_LEGENDRE_INTEGRATION
#pragma omp parallel sections
	{
#pragma omp section
		{
			SetupIntegral(theta_, wTheta, 0.0, M_PI, int(sqrt(double(iterations))));
		}
#pragma omp section
		{
			SetupIntegral(phi_, wPhi, 0.0, 2.0 * M_PI, int(sqrt(double(iterations))));
		}
	}
#endif
	// TODO::Spherical - Simple trapezoidal integration should be considered, maybe spline

	FACC qZ, qPerp, q, theta;
#pragma omp parallel for schedule(dynamic, Q.size() / 50)
	for (int i = 0; i < Q.size(); i++)
	{
		for (int j = 0; j < Q.size(); j++)
		{
			if (pStop && *pStop)
				continue;

			if (!noException)
				continue;

			// TODO: swap i,j ?
			qZ = Q[j];
			qPerp = Q[i];
			JacobianSphereGrid::qZ_qPerp_to_q_Theta(qZ, qPerp, q, theta);
			//std::cout <<"qZ="<<qZ<<", qP="<<qPerp<<", q="<<q<<", theta=" << theta << std::endl;

			try // This try/catch combo is here because of OMP
			{
				if (q <= qMax && q>= qMin)
					res(i,j) = CalculateIntensity(q, theta, epsi, seeds[i], iterations);
			}
			catch (backend_exception& ex)
			{
				exceptionInt = ex.GetErrorCode();
				exceptionString = ex.GetErrorMessage();
				noException = false;
			}
#pragma omp critical
			{
				if (progFunc)
					progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(Q.size())) + cProgMin);
			}
 		}
	}

	if (!noException)
		throw backend_exception(exceptionInt, exceptionString.c_str());

	if (pStop && *pStop)
		return STOPPED;

	aveEnd = clock();

	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;

	/*
	*stdout = origSTDOUT;
	*/
	_previous_hash = Hash();
	_previous_intensity_2D = res.template cast<double>();
	_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>();

	return PDB_OK;
}


template<typename T>
void DomainModel::setPreviousValues(std::vector<T> & res, const std::vector<T> & Q)
{
	_previous_hash = Hash();
	_previous_intensity = (Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size())).template cast<double>();
	_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >(Q.data(), Q.size()).template cast<double>();
}

template<typename T>
void DomainModel::setPreviousValues2D(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& res, const std::vector<T>& Q)
{
	_previous_hash = Hash();
	_previous_intensity_2D = res.template cast<double>();
	_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >(Q.data(), Q.size()).template cast<double>();
}

template<typename T>
PDB_READER_ERRS DomainModel::IntegrateLayersTogether(std::vector<unsigned int> &seeds, std::uniform_int_distribution<unsigned int> &sRan, std::mt19937 &seedGen, const std::vector<T> & Q, std::vector<T> & res, T &epsi, uint64_t &iterations, const double &cProgMax, const double &cProgMin, int &prog, clock_t &aveEnd, const clock_t &aveBeg, const clock_t &gridBegin)
{
	seeds.resize(1 + gridSize / 2);
	for (size_t i = 0; i < seeds.size(); i++)
		seeds[i] = sRan(seedGen);

	// Compute the value at q == 0
	if (Q[0] == 0.0)
	{
		std::complex<FACC> amp(0.0, 0.0);
		for (unsigned int j = 0; j < _amps.size(); j++)
			amp += _amps[j]->getAmplitude(0, 0, 0);
		res[0] = real(amp * conj(amp));
	}

	T stepSize = 2 * qMax / gridSize;
	int tmpLayer = 0;
	auto qtmpBegin = Q.begin();
	while (*qtmpBegin > (tmpLayer + 1) * stepSize) // Minimum q val
		tmpLayer++;

#pragma omp parallel for schedule(dynamic, 1)
	for (int layerInd = tmpLayer; layerInd < gridSize / 2; layerInd++)
	{
		if (pStop && *pStop)
			continue;

		auto qBegin = Q.begin();
		auto qEnd = qBegin + 1;

		while (*qBegin <= layerInd * stepSize) // Minimum q val
			qBegin++;

		qEnd = qBegin + 1;
		while (qEnd + 1 != Q.end() && *qEnd <= (layerInd + 1) * stepSize) // Maximum q val
			qEnd++;
		if (qEnd + 1 != Q.end())
			qEnd--;

		const std::vector<T> relevantQs(qBegin, qEnd + 1);
		std::vector<T> reses(relevantQs.size(), T(0));

		// Integrate until converged
		AverageIntensitiesBetweenLayers(relevantQs, reses, layerInd, epsi, seeds[layerInd], iterations);

		// Copy results
		std::copy_n(reses.begin(), reses.size(), res.begin() + (qBegin - Q.begin()));

		// Report progress
#pragma omp critical
		{
			if (progFunc)
				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(gridSize / 2)) + cProgMin);
		}

	}

	if (pStop && *pStop)
		return STOPPED;

	aveEnd = clock();

	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;

	_previous_hash = Hash();
	_previous_intensity = (Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size())).template cast<double>();
	_previous_q_values = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(Q.data(), Q.size()).template cast<double>();
	return PDB_OK;
}


template<typename T>
PDB_READER_ERRS DomainModel::IntegrateLayersTogether2D(std::vector<unsigned int>& seeds, std::uniform_int_distribution<unsigned int>& sRan, std::mt19937& seedGen, std::vector<PolarCalculationData*> qData, 
	 T& epsi, uint64_t& iterations, const double& cProgMax, const double& cProgMin, int& prog, clock_t& aveEnd, const clock_t& aveBeg, const clock_t& gridBegin)
{
	std::cout << "--------- DomainModel::IntegrateLayersTogether2D ---------" << std::endl;
	seeds.resize(1 + gridSize / 2);
	for (size_t i = 0; i < seeds.size(); i++)
		seeds[i] = sRan(seedGen);

	T stepSize = 2 * qMax / gridSize;
	int tmpLayer = 0;
	auto qtmpBegin = qData[0]->q;
	while (qtmpBegin > (tmpLayer + 1) * stepSize) // Minimum q val
		tmpLayer++;

#pragma omp parallel for schedule(dynamic, 1)
	for (int layerInd = tmpLayer; layerInd < gridSize / 2; layerInd++)
	{
		if (pStop && *pStop)
			continue;

		auto qBegin = qData.begin();
		auto qEnd = qBegin + 1;

		while (qBegin != qData.end() && (* qBegin)->q <= layerInd * stepSize) // Minimum q val
			qBegin++;

		if (qBegin != qData.end())
			qEnd = qBegin + 1;
		while (qEnd != qData.end() && qEnd + 1 != qData.end() && (* qEnd)->q <= (layerInd + 1) * stepSize) // Maximum q val
			qEnd++;
		if (qEnd == qData.end() || qEnd + 1 != qData.end())
			qEnd--;

		std::vector<PolarCalculationData*> relevantQData(qBegin, qEnd + 1);

		if (relevantQData.size() == 0)
			continue;


		// Integrate until converged
		AverageIntensitiesBetweenLayers2D(relevantQData, layerInd, epsi, seeds[layerInd], iterations);

		// Report progress
#pragma omp critical
		{
			if (progFunc)
				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(gridSize / 2)) + cProgMin);
		}

	}

	if (pStop && *pStop)
		return STOPPED;

	aveEnd = clock();

	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;

	_previous_hash = Hash();

	return PDB_OK;
}


PDB_READER_ERRS DomainModel::gridComputation()
{
	// Progress reporting: First 30% go to grid computation, other 70% go to intensity calculation
	const double progMin = 0.0, progMax = 0.3;


	// Find the minimal qMax from existing grids
	double effqMax = qMax;
	for (int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if (_amps[i]->GetUseGrid() && _amps[i]->GridIsReadyToBeUsed()) {
			// TODO::Spherical - Check to make sure this is still correct with SphereGrid
			double lqmax = _amps[i]->GetGridStepSize() * double(_amps[i]->GetGridSize()) / 2.0;
			effqMax = std::min(effqMax, lqmax);
		}
	}
	if ((effqMax / qMax) < (1.0 - 0.001)) {
		std::ostringstream oss;
		oss << "The qmax of the loaded grid does not match the qmax input by the user in Preferences. The loaded grid's qmax is: " << effqMax;
		std::string message = oss.str();
		throw backend_exception(ERROR_GENERAL, message.c_str());	// TODO::____ Need to either reassign the values in Q to reflect the maximum value, or something else
	}

	// If the grid size has changed or something else, we should recalculate the grid or see if one already exists
	for (int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if (pStop && *pStop)
			return STOPPED;

		if (_amps[i]->GetUseGrid() && _amps[i]->GridIsReadyToBeUsed()) {
			if (_amps[i]->GetGridSize() < gridSize) {
				_amps[i]->ResetGrid();	// TODO:Optimization - see if an existing grid exists as a file
			}
		}
	}

	// Calculate grids that need to be evaluated
	for (int i = 0; i < this->GetNumSubAmplitudes(); i++) {
		if (pStop && *pStop)
			return STOPPED;

		// Check to see if amplitude needs to be generated and bUseGrid
		double progSec = (progMax - progMin) / double(GetNumSubAmplitudes());
		if (_amps[i]->GetUseGrid() && !_amps[i]->GridIsReadyToBeUsed()) {
			std::cout << "*************** calc grid ! ********************\n";
			_amps[i]->calculateGrid(qMax, gridSize, progFunc, progArgs, progMin + double(i) * progSec,
				progMin + double(i + 1) * progSec, pStop);
		}
		else if (!_amps[i]->GetUseGrid()) {
			_amps[i]->calculateGrid(qMax, gridSize, progFunc, progArgs, progMin + double(i) * progSec,
				progMin + double(i + 1) * progSec, pStop);
		}
	}

	if (pStop && *pStop)
		return STOPPED;
	if (progFunc)
		progFunc(progArgs, progMax);
	return PDB_OK;
}

// Explicit instantiation for float and double
template EXPORTED_BE PDB_READER_ERRS DomainModel::CalculateIntensityVector<float>(
	const std::vector<float>& Q,
	std::vector<float>& res, float epsi, uint64_t iterations);

template EXPORTED_BE PDB_READER_ERRS DomainModel::CalculateIntensityVector<double>(
	const std::vector<double>& Q,
	std::vector<double>& res, double epsi, uint64_t iterations);

double DomainModel::GaussKron2DSphereRecurs(double q, double epsilon,
	int64_t maxDepth, int64_t minDepth) {
	return GaussLegTheta(q, 0.0, GetHasAnomalousScattering() ? M_PI : M_PI_2, epsilon, maxDepth - 1, minDepth - 1) / (2.0 * PI);
}

double DomainModel::GaussLegTheta(double q, double tMin, double tMax, double epsilon, int64_t maxDepth, int64_t minDepth) {
	double kVal = 0.0, gVal = 0.0, sT, ba = (tMax - tMin), apb = (tMax + tMin), ba2 = ba * 0.5, apb2 = apb * 0.5;
	double kVals[15] = { 0.0 }, gVals[7] = { 0.0 };
	for (int i = 0; i < 15; i++) {
		sT = sin(apb2 + ba2 * k15x[i]);
		kVals[i] = GaussLegPhi(q, sT, cos(apb2 + ba2 * k15x[i]), 0.0, PIx2, epsilon, maxDepth, minDepth);
		kVal += kVals[i] * k15w[i] * ba2 * sT;
		if (i % 2 == 1) {
			gVals[i / 2] = kVals[i];
			gVal += gVals[i / 2] * g7w[i / 2] * ba2 * sT;
		}
	}

	if (minDepth < 0 && (maxDepth < 0 || fabs(1.0 - (gVal / kVal)) <= epsilon || gVal == kVal)) {
		return kVal;
	}

	double mid = (tMax + tMin) / 2.0;
	return
		GaussLegTheta(q, tMin, mid, epsilon, maxDepth - 1, minDepth - 1) +
		GaussLegTheta(q, mid, tMax, epsilon, maxDepth - 1, minDepth - 1);

}

double DomainModel::GaussLegPhi(double q, double st, double ct, double pMin, double pMax, double epsilon, int64_t maxDepth, int64_t minDepth) {
	std::complex<FACC> kVals[15], gVals[7];
	double kVal = 0.0, gVal = 0.0, ba = (pMax - pMin), apb = (pMax + pMin), ba2 = ba * 0.5, apb2 = apb * 0.5, xi;
	for (int i = 0; i < 15; i++) {
		xi = apb2 + ba2 * k15x[i];
		for (unsigned int j = 0; j < _amps.size(); j++) {
			kVals[i] += _amps[j]->getAmplitude(q * st * cos(xi), q * st * sin(xi), q * ct);
		}
		kVal += real(kVals[i] * conj(kVals[i])) * ba2 * k15w[i];
		if (i % 2 == 1) {
			gVals[i / 2] = kVals[i];
			gVal += real(gVals[i / 2] * conj(gVals[i / 2])) * ba2 * g7w[i / 2];
		}
	}

	if (minDepth < 0 && (maxDepth < 0 || fabs(1.0 - (gVal / kVal)) <= epsilon || gVal == kVal)) {
		return kVal;
	}

	return
		GaussLegPhi(q, st, ct, pMin, apb2, epsilon, maxDepth - 1, minDepth - 1) +
		GaussLegPhi(q, st, ct, apb2, pMax, epsilon, maxDepth - 1, minDepth - 1);
}

FACC DomainModel::CalculateIntensity(FACC q, FACC epsi, unsigned int seed, uint64_t iterations) 
{
	return CalculateIntensity(q, -1, epsi, seed, iterations);
}

FACC DomainModel::CalculateIntensity(FACC q, FACC thetaIn, FACC epsi, unsigned int seed, uint64_t iterations) {
	FACC res = 0.0;

	if (q == 0.0) {
		std::complex<FACC> amp(0.0, 0.0);
		for (unsigned int j = 0; j < _amps.size(); j++)
			amp += _amps[j]->getAmplitude(q, q, q);
		return real(amp * conj(amp));
	}

	if (orientationMethod == OA_ADAPTIVE_GK) {
		return GaussKron2DSphereRecurs(q, epsi, iterations, 0);
	}
	else if (orientationMethod == OA_MC) {
		unsigned long long minIter = /*iterations / */20;
		std::vector<FACC> results, sins;
		results.resize(std::min(iterations, results.max_size()), 0.0);
		sins.resize(results.size() + 1);
		sins[0] = 0.0;

#ifdef GAUSS_LEGENDRE_INTEGRATION
#pragma omp parallel for reduction(+ : res)
		for (int i = 0; i < theta_.size(); i++) {
			for (int j = 0; j < phi_.size(); j++) {
				double theta = theta_[i];
				double phi = phi_[j];
				double st = sin(theta);
				double sp = sin(phi);
				double cp = cos(phi);
				double ct = cos(theta);
				std::complex<FACC> amp(0.0, 0.0);
				for (int j = 0; j < _amps.size(); j++)
					amp += _amps[j]->getAmplitude(q * st * cp, q * st * sp, q * ct);
				res += real(amp * conj(amp)) * st * wPhi[j] * wTheta[i];
			}
		}

		return res / (4.0 * M_PI);
#endif

		std::mt19937 rng;
		rng.seed(seed);
#define WOLFRAM_RANDOM_POINTS_ON_SPHERE	// See http://mathworld.wolfram.com/SpherePointPicking.html
#ifdef WOLFRAM_RANDOM_POINTS_ON_SPHERE
		std::uniform_real_distribution<FACC> ranU2(0.0, 2.0);
#else
		std::uniform_real_distribution<FACC> thRan(0.0, M_PI);
		std::uniform_real_distribution<FACC> phRan(0.0, 2.0 * M_PI);
#endif
		std::complex<FACC> phase, im(0.0, 1.0);

		for (uint64_t i = 0; i < iterations; i++) {
			FACC theta, phi, st, sp, cp, ct, u2, v2;

#ifndef WOLFRAM_RANDOM_POINTS_ON_SPHERE
			if (thetaIn < 0)
				theta = thRan(rng);
			else
				theta = thetaIn;
			phi = phRan(rng);
#else
			// See http://mathworld.wolfram.com/SpherePointPicking.html
			u2 = ranU2(rng);
			v2 = ranU2(rng);
			phi = u2 * M_PI;
			if (thetaIn < 0)
				theta = acos(v2 - 1.0);
			else
				theta = thetaIn;
#endif
			st = sin(theta);
			sp = sin(phi);
			cp = cos(phi);
#ifdef WOLFRAM_RANDOM_POINTS_ON_SPHERE
			ct = (v2 - 1.0);
#else
			ct = cos(theta);
#endif
			std::complex<FACC> amp(0.0, 0.0);
			for (int j = 0; j < _amps.size(); j++)
				amp += _amps[j]->getAmplitude(q * st * cp, q * st * sp, q * ct);
			res += real(amp * conj(amp))
#ifndef WOLFRAM_RANDOM_POINTS_ON_SPHERE
				* st;
			sins[i + 1] = sins[i] + st;
#else
				;
#endif

			// Convergence Place TODO FIXME
#pragma region Convergence test
			results[i] = res /
#ifdef WOLFRAM_RANDOM_POINTS_ON_SPHERE
				FACC(i + 1);
#else
				sins[i + 1];
#endif
			if (i >= minIter && epsi > 0.0) {
				if (fabs(1.0 - (results[i] / results[i >> 1])) < epsi) {
					if (fabs(1.0 - (results[i] / results[(i << 1) / 3])) < epsi) {
						if (fabs(1.0 - (results[i] / results[(3 * i) >> 2])) < epsi) {
							return results[i];
						}
					}
				}
			} // if i >=...
#pragma endregion
		} // for

		return results[results.size() - 1];
	}
	else {	// Incorrect orientationMethod
		throw backend_exception(ERROR_UNIMPLEMENTED_CPU, g_errorStrings[ERROR_UNIMPLEMENTED_CPU]);
	}
}

#pragma endregion

#pragma region Domain (IModel) class

DomainModel::DomainModel() {
	bDefUseGrid = true;
	bDefUseGPU = true;
	gridSize = 150;
	oIters = 100000;
	eps = 1.021e-3;
	qMax = 5.16;
	orientationMethod = OA_MC;

}

DomainModel::~DomainModel() {
}

void DomainModel::OrganizeParameters(const VectorXd& p, int nLayers) {
	_ampParams.resize(_amps.size());
	_ampLayers.resize(_amps.size());

	size_t nNodeParams = 0;
	const double *nodeparams = ParameterTree::GetNodeParamVec(p.data(), p.size(), nNodeParams, false);


	only_scale_changed =
		previousParameters.size() == p.size() &&
		(previousParameters.array() == p.array()).tail(p.size() - 1).all();

	// We start from nodeparams[1] to skip nLayers

	oIters = (unsigned long long)(nodeparams[1]);
	gridSize = (unsigned int)(nodeparams[2]);
	if (gridSize % 2 == 1) gridSize++;
	bDefUseGrid = fabs(nodeparams[3] - 1.0) <= 1e-6 ? true : false;
	eps = nodeparams[4];
	qMax = nodeparams[5];
	qMin = nodeparams[7];
	orientationMethod = OAMethod_Enum((int)(nodeparams[6] + 0.1));

	// Before incurring too many costs, check to make sure we have valid arguments
	if (!bDefUseGrid)
	{
		int devCount;
		cudaError_t t = UseGPUDevice(&devCount); // Trigger g_useGPU update

		if (g_useGPU && g_useGPUAndAvailable)
			throw backend_exception(ERROR_INVALIDARGS,
				"When using a GPU for computations, the grid must be enabled. "
				"Make sure either the Use Grid checkbox is checked, or Use GPU is unchecked.");
	}


	for (unsigned int i = 0; i < _amps.size(); i++)
	{
		size_t subParams = 0;
		const double *subpd = ParameterTree::GetChildParamVec(p.data(), p.size(), i, subParams,
			(ParameterTree::GetChildNumChildren(p.data(), i) > 0));

		if (ParameterTree::GetChildNumChildren(p.data(), i) > 0) {
			// Since we extracted the whole parameter vector, with info
			_ampLayers[i] = int(ParameterTree::GetChildNLayers(p.data(), i) + 0.001);
		}
		else {
			// Compute the number of layers
			_ampLayers[i] = int(*subpd++ + 0.001);
			subParams--;
		}

		_ampParams[i] = VectorXd::Zero(subParams);
		for (unsigned int j = 0; j < subParams; j++)
			_ampParams[i][j] = subpd[j];

		LocationRotation locrot;
		ParameterTree::GetChildLocationData(p.data(), i, locrot);
		_amps[i]->SetLocationData(locrot.x, locrot.y, locrot.z, locrot.alpha, locrot.beta, locrot.gamma);

		bool bUseGridHere;
		ParameterTree::GetChildUseGrid(p.data(), i, bUseGridHere);
		_amps[i]->SetUseGrid(bUseGridHere && bDefUseGrid);
	}

	previousParameters = p;
}

double DomainModel::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd()*/) {
	std::mt19937 seedGen;
	std::uniform_int_distribution<unsigned int> sRan(0, UINT_MAX);
	seedGen.seed(static_cast<unsigned int>(std::time(0)));

	unsigned int sd = sRan(seedGen);

	return CalculateIntensity(q, eps, sd, oIters);
}

void DomainModel::PreCalculate(VectorXd& p, int nLayers) {
	OrganizeParameters(p, nLayers);


	if (only_scale_changed && _previous_hash == Hash())
		return;

	for (unsigned int i = 0; i < _amps.size(); i++)
		_amps[i]->PreCalculate(_ampParams[i], _ampLayers[i]);
}



VectorXd DomainModel::CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */, progressFunc progress /*= NULL*/, void *progressArgs /*= NULL*/) {
	size_t points = q.size();
	VectorXd vec = VectorXd::Zero(points);
	std::vector<double> rVec(points);

	progFunc = progress;
	progArgs = progressArgs;

	PreCalculate(p, nLayers);

	if (CalculateIntensityVector<double>(q, rVec, eps, oIters) != PDB_OK) {
		// On error
		vec = VectorXd::Constant(points, -1.0);
		return vec;
	}

	// Copy data from rVec to vec
	memcpy(vec.data(), &rVec[0], points * sizeof(double));

	return vec;
}

MatrixXd DomainModel::CalculateMatrix(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */, progressFunc progress /*= NULL*/, void* progressArgs /*= NULL*/) 
{
	std::cout << "!!!!!!! DomainModel::CalculateMatrix !!!!!!!" << std::endl;
	
	size_t points = q.size();
	MatrixXd mat = MatrixXd::Zero(points, points);
	MatrixXd rMat = MatrixXd::Zero(points, points);

	progFunc = progress;
	progArgs = progressArgs;

	PreCalculate(p, nLayers);

	if (CalculateIntensity2DMatrix(q, rMat, eps, oIters) != PDB_OK) {
		// On error
		mat = MatrixXd::Constant(points, points, -1.0);
		return mat;
	}

	mat = rMat;

	return mat;
}

VectorXd DomainModel::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	// TODO::Fit: Figure out which indexes refer to translation and rotation, use special derivatives
	// h is so "large" because grids do not support that kind of small modifications
	return NumericalDerivative(this, x, param, nLayers, ai, 1.0e-2);
}

VectorXd DomainModel::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */) {
	// Not applicable
	return VectorXd::Zero(0);
}

void DomainModel::SetStop(int *stop) {
	pStop = stop;
}

int DomainModel::GetNumSubAmplitudes() {
	return int(_amps.size());
}

Amplitude *DomainModel::GetSubAmplitude(int index) {
	if (index < 0 || (unsigned int)(index) >= _amps.size())
		return NULL;

	return _amps[index];
}

void DomainModel::SetSubAmplitude(int index, Amplitude *subAmp) {
	if (index < 0 || (unsigned int)(index) >= _amps.size())
		return;

	_amps[index] = subAmp;
}

void DomainModel::AddSubAmplitude(Amplitude *subAmp) {
	if (subAmp)
		_amps.push_back(subAmp);
}

void DomainModel::RemoveSubAmplitude(int index) {
	if (index < 0 || (unsigned int)(index) >= _amps.size())
		return;

	_amps.erase(_amps.begin() + index);
}

void DomainModel::ClearSubAmplitudes() {
	_amps.clear();
}

void DomainModel::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String("Domain Model");

	writer.Key("qMax");
	writer.Double(qMax);

	if (bDefUseGrid) {
		writer.Key("gridSize");
		writer.Double(gridSize);
	}

	writer.Key("SubModels");
	writer.StartArray();
	for (unsigned int i = 0; i < _amps.size(); i++) {
		writer.StartObject();
		_amps[i]->GetHeader(depth + 1, writer);
		writer.EndObject();
	}
	writer.EndArray();


}
void DomainModel::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers;
	ampers.resize(depth + 1, '#');
	ampers.append(" ");

	std::stringstream ss;

	if (depth == 0)
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");

	header.append(ampers + "//////////////////////////////////////\n");
	ss << "Domain Model\n";
	header.append(ampers + ss.str());
	ss.str("");

	ss << "qMax = " << qMax << "\n";
	header.append(ampers + ss.str());
	ss.str("");

	if (bDefUseGrid) {
		ss << "Grid size = " << this->gridSize << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

	for (unsigned int i = 0; i < _amps.size(); i++) {
		_amps[i]->GetHeader(depth + 1, header);
	}


}

bool DomainModel::Populate(const VectorXd& p, int nLayers) {
	// Does nothing
	return true;
}

unsigned int DomainModel::GetNumSubLocations() {
	return 1;
}

LocationRotation DomainModel::GetSubLocation(int index) {
	return LocationRotation();
}

void DomainModel::GetSubAmplitudeParams(int index, VectorXd& params, int& nLayers) {
	if (index < 0 || index >= _amps.size())
		return;

	params = _ampParams[index];
	nLayers = _ampLayers[index];
}

bool DomainModel::GetUseGridWithChildren() const {
	for (int i = 0; i < _amps.size(); i++) {
		if (!_amps[i]->GetUseGridWithChildren())
			return false;
	}

	return bDefUseGrid;	// TODO::Hybrid Maybe this should be just 'true'?
}

bool DomainModel::SavePDBFile(std::ostream &output) {
	std::vector<std::string> lines;
	std::vector<Eigen::Vector3f> locs;

	bool res = AssemblePDBFile(lines, locs);

	if (lines.size() != locs.size()) {
		std::cout << "Mismatched sizes" << std::endl;
		return false;
	}

	for (int i = 0; i < locs.size(); i++) {
		std::string line = lines[i];
		std::string xst, yst, zst;
		char grr = line[54];
		xst.resize(24);
		yst.resize(24);
		zst.resize(24);

		sprintf(&xst[0], "%8f", locs[i].x() * 10.);
		sprintf(&yst[0], "%8f", locs[i].y() * 10.);
		sprintf(&zst[0], "%8f", locs[i].z() * 10.);

		sprintf(&line[30], "%s", xst.substr(0, 8).c_str());
		sprintf(&line[38], "%s", yst.substr(0, 8).c_str());
		sprintf(&line[46], "%s", zst.substr(0, 8).c_str());

		line[54] = grr;

		output << line << std::endl;
	}

	return true;
}

bool DomainModel::AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs, bool electronPDB) {
	for (int i = 0; i < _amps.size(); i++) {
		std::vector<std::string> subLines;
		std::vector<Eigen::Vector3f> subLocs;
		ISymmetry *symmCast = dynamic_cast<ISymmetry*>(_amps[i]);

		if (symmCast)
		{
			// Collect atoms
			symmCast->AssemblePDBFile(subLines, subLocs);

			// Append atoms
			lines.reserve(lines.size() + subLines.size());
			lines.insert(lines.end(), subLines.begin(), subLines.end());
			locs.reserve(locs.size() + subLocs.size());
			locs.insert(locs.end(), subLocs.begin(), subLocs.end());
		} //if symmCast

		if (electronPDB)
		{
			ElectronPDBAmplitude* pdbCast = dynamic_cast<ElectronPDBAmplitude*>(_amps[i]);
			if (pdbCast) 
			{
				pdbCast->AssemblePDBFile(subLines, subLocs);

				lines.reserve(lines.size() + subLines.size());
				lines.insert(lines.end(), subLines.begin(), subLines.end());
				locs.reserve(locs.size() + subLocs.size());
				locs.insert(locs.end(), subLocs.begin(), subLocs.end());
			} // if pdbCast
		}
		else
		{
			PDBAmplitude* pdbCast = dynamic_cast<PDBAmplitude*>(_amps[i]);
			if (pdbCast) 
			{
				pdbCast->AssemblePDBFile(subLines, subLocs);

				lines.reserve(lines.size() + subLines.size());
				lines.insert(lines.end(), subLines.begin(), subLines.end());
				locs.reserve(locs.size() + subLocs.size());
				locs.insert(locs.end(), subLocs.begin(), subLocs.end());
			} // if pdbCast
		}


		
	} // for i < _amps.size

	return true;
} // DomainModel::AssemblePDBFile

bool DomainModel::CalculateVectorForCeres(const double* qs, double const* const* p, double* residual, int points) {
	/*
	VectorXd ps;
	ps.resize(pSizes[0]);
	memcpy(ps.data(), p[0], sizeof(double) * ps.size());	// Or something. This won't work, as p is an uneven matrix; but you get the point
	*/
	/*
	VectorXd params = SOME_METHOD_TO_CONVERT_PARAMETERS_FROM_CERES_FORMAT(p);
	*/
	progFunc = NULL;

	for (auto par = mutParams.begin(); par != mutParams.end(); par++)
	{
		*(par->second) = p[0][par->first];
	}
	std::vector<double> ivec(points);
	std::vector<double> q(qs, qs + points);

	PreCalculate(*pVecCopy, 0);

	if (CalculateIntensityVector<double>(q, ivec, eps, oIters) != PDB_OK) {
		// On error
		return false;
	}

	memcpy(residual, &ivec[0], points * sizeof(double));

	return true;
}

void DomainModel::SetInitialParamVecForCeres(VectorXd* p, std::vector<int> &mutIndices) {
	pVecCopy = p;
	mutParams.clear();
	for (int i = 0; i < mutIndices.size(); i++) {
		mutParams[i] = &((*p)[mutIndices[i]]);
		std::cout << &((*p)[mutIndices[i]]) << "\n" << p->data() + mutIndices[i] << " = " <<
			(*pVecCopy)[mutIndices[i]] << std::endl;
	}
}

void DomainModel::GetMutatedParamVecForCeres(VectorXd* p) {
	p->resize(mutParams.size());
	for (auto it = mutParams.begin(); it != mutParams.end(); ++it) {
		(*p)[it->first] = *(it->second);
	}
}

void DomainModel::SetMutatedParamVecForCeres(const VectorXd& p) {
	for (auto it = mutParams.begin(); it != mutParams.end(); ++it) {
		*(it->second) = p[it->first];
	}
}

template <typename T>
PDB_READER_ERRS DomainModel::PerformGPUHybridComputation(clock_t &gridBegin, const std::vector<T> &Q, clock_t &aveBeg, std::vector<T> &res, T epsi, uint64_t iterations, clock_t &aveEnd)
{
#ifndef USE_JACOBIAN_SPHERE_GRID
	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
#endif

	gridBegin = clock();
	int devCount;
	cudaError_t t = UseGPUDevice(&devCount);
	if (devCount <= 0 || !g_useGPUAndAvailable || t != cudaSuccess) {
		printf("No GPU backend. Calculating on CPU.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Locate and create the GPU calculator
	gpuGridcalculator_t gpuCalcHybridGen = (gpuGridcalculator_t)GPUCreateCalculatorHybrid;
	//GetProcAddress((HMODULE)g_gpuModule, "GPUCreateCalculatorHybrid");
	if (!gpuCalcHybridGen) {
		printf("ERROR: Cannot find hybrid generator function\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Report indeterminate progress (for now)
	// TODO::Later: Meaningful progress bar in direct method
	if (progFunc)
		progFunc(progArgs, 0.0);

	IGPUGridCalculator *hybridCalc = gpuCalcHybridGen();

	// Flatten the symmetry/primitive tree to a single vector
	std::vector<Amplitude *> flatVec;
	std::vector<LocRotScale> flatLocrot;
	std::vector<VectorXd> flatParams;

	Eigen::Matrix4f tmat = Eigen::Matrix4f::Identity();
	FlattenTree<true>(_amps, _ampParams, _ampLayers, tmat, flatVec, flatLocrot, flatParams);
#ifdef _DEBUG
	{
		for (int ll = 0; ll < flatLocrot.size(); ll++) {
			std::cout << "[" << flatLocrot[ll].first.x << ", " << flatLocrot[ll].first.y << ", " << flatLocrot[ll].first.z << "]  ["
				<< flatLocrot[ll].first.alpha << ", " << flatLocrot[ll].first.beta << ", " << flatLocrot[ll].first.gamma <<
				"] scale: " << flatLocrot[ll].second << std::endl;
		}
	}
#endif // _DEBUG
	bool allImplemented = true;
	for (int i = 0; i < flatVec.size(); ++i)
	{
		IGPUGridCalculable *tst = dynamic_cast<IGPUGridCalculable*>(flatVec[i]);
		if (!tst) {
			allImplemented = false;
			break;
		}
	}
	if (!allImplemented) {
		printf("Not all models have been implemented to work as hybrid on a GPU. Continuing on CPU.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// An ugly but efficient fixed-height tree built from maps. Tree groups amplitudes by:
	// 1. type
	// 2. parameters (number of layers, then parameters themselves)
	// 3. orientation (alpha, then beta, then gamma)
	// 4(unordered). translation
	typedef std::map<std::string,
		std::map<VectorXd,
		std::map<float4, std::vector<float4>, rotation_ulp_less>,
		paramvec_less> > AmpCalcTree;

	typedef std::map<VectorXd,
		std::map<float4, std::vector<float4>, rotation_ulp_less>,
		paramvec_less> ParamOrientTree;

	/*
	/// Maps a param vector to a pointer to an Amplitude
	typedef std::map<VectorXd, Amplitude*, paramvec_less> VecToAmp;
	*/

	typedef std::map<UniqueModels, Amplitude*, UniqueModels_less> AmpsToCalc;
	/// var[workspaceid][hash][paramVec] is a pointer to an Amplitude
	typedef std::vector<AmpsToCalc> WorkspacesToCalc;

	// The work is divided to P (numWorkspaces) trees of parallel processing
	const unsigned int numWorkspaces = devCount;
	std::vector<AmpCalcTree> ampTrees(numWorkspaces);
	std::vector<GridWorkspace> workspaces(numWorkspaces);
	std::map<std::string, Amplitude*> hashToAmp;
	WorkspacesToCalc myAmps;
	// TODO::Hybrid - Hash should include the models parameters to distinguish between models


	std::vector<float> Qfloat(Q.size());
	for (int i = 0; i < Q.size(); ++i)
		Qfloat[i] = (float)Q[i];

	unsigned int ampsPerWorkspace = flatVec.size() / numWorkspaces;

	int curGPU = 0;
	int numAmpsToCalc = 0;
	myAmps.resize(numWorkspaces);
	// The loop that generates the trees and workspaces
	for (unsigned int p = 0; p < numWorkspaces; p++)
	{
		unsigned int lastAmp = ((p == numWorkspaces - 1) ? flatVec.size() : ((p + 1) * ampsPerWorkspace));

		for (unsigned int i = p * ampsPerWorkspace; i < lastAmp; ++i)
		{
			std::string hash = flatVec[i]->Hash();
			hashToAmp[hash] = flatVec[i];
			myAmps[p][std::make_tuple(hash, flatParams[i])] = flatVec[i];
			LocationRotation locrot = flatLocrot[i].first;

			// OPTIMIZATION: Conserve rotations and translations by letting 
			// rotation/translation-invariant models modify their Locrots
			// EXAMPLES: A sphere is invariant to rotation, a cylinder is invariant to one of the angles
			double aa = locrot.alpha, bb = locrot.beta, gg = locrot.gamma;
			IGPUCalculable *gpuModel = dynamic_cast<IGPUCalculable *>(flatVec[i]);
			if (gpuModel)
				gpuModel->CorrectLocationRotation(locrot.x, locrot.y, locrot.z,
					aa, bb, gg);
			locrot.alpha = Radian(aa);
			locrot.beta = Radian(bb);
			locrot.gamma = Radian(gg);

			//std::cout << "Model " << i << ": " << flatVec[i]->GetName() << " - XYZABG " 
			//		  << locrot.x << " " << locrot.y << " " << locrot.z << " " << locrot.alpha << " " << locrot.beta << " " << locrot.gamma << std::endl; 

			ampTrees[p][hash]
				[flatParams[i]]
			[make_float4(locrot.alpha, locrot.beta, locrot.gamma, flatLocrot[i].second)].
				push_back(make_float4(locrot.x, locrot.y, locrot.z, 0.));
		}

		// Obtain the maximal number of translations
		unsigned int maxTranslations = 0;
		for (AmpCalcTree::iterator iter = ampTrees[p].begin(); iter != ampTrees[p].end(); ++iter)
			for (ParamOrientTree::iterator piter = iter->second.begin(); piter != iter->second.end(); ++piter)
				for (auto oiter = piter->second.begin(); oiter != piter->second.end(); ++oiter)
				{
					if (oiter->second.size() > maxTranslations)
						maxTranslations = oiter->second.size();
				}
		// END of maximal translations

		// This should/could be obtained by allocating a grid and obtaining the numbers from the instance.
		// I don't want to allocate so much memory.
		double stepSize = qMax / double(gridSize / 2);
		int actualSz = (gridSize / 2) + 3 /*Extras*/;
		long long kk = actualSz;
		int thetaDivisions = 3;
		int phiDivisions = 6;
		long long totalsz = (phiDivisions * kk * (kk + 1) * (3 + thetaDivisions + 2 * thetaDivisions * kk)) / 6;
		totalsz++;	// Add the origin
		totalsz *= 2;	// Complex

		bool bsuccess = hybridCalc->Initialize(curGPU, Qfloat, totalsz, thetaDivisions, phiDivisions, actualSz + 1, qMax,
			stepSize, workspaces[p]);
		hybridCalc->SetNumChildren(workspaces[p], myAmps[p].size());

		for (auto hiter = myAmps[p].begin(); hiter != myAmps[p].end(); ++hiter)
		{
			numAmpsToCalc++;
			int childNum = 0;
			bool bsuccess = hybridCalc->Initialize(curGPU, Qfloat,
				totalsz, thetaDivisions, phiDivisions, actualSz + 1, qMax, stepSize,
				//std::get<1>(hiter->first).size(), std::get<1>(hiter->first).data(),
				workspaces[p].children[childNum++]);
		}
		curGPU++;
		if (curGPU == devCount)
			curGPU = 0;
	} // for(unsigned int p = 0; p < numWorkspaces; p++)

	printf("End of hybrid init...\n");

	int numAmpsCalced = 0;

	// Calculate all of the grids up to the point where there are no grids used
	for (unsigned int p = 0; p < numWorkspaces; p++)
	{
		int childNum = 0;
		for (auto iter = myAmps[p].begin(); iter != myAmps[p].end(); ++iter)
		{
			// Don't need to call PreCalculate, it was already called in DomainModel::CalculateVector
			//inIter->second->PreCalculate(VectorXd(inIter->first) /*Needs to be copied, otherwise will be const and can't pass by reference*/
			//	, inIter->first.size());
			IGPUGridCalculable* gpuModel = dynamic_cast<IGPUGridCalculable*>(iter->second);
			if (gpuModel)
			{
				gpuModel->SetModel(workspaces[p].children[childNum]);	// Running this in a separate loop may increase performance (maybe?)
				gpuModel->CalculateGridGPU(workspaces[p].children[childNum]);

				std::string	hash = std::get<0>(iter->first);
				VectorXd	parv = std::get<1>(iter->first);

				std::vector<float4> rots;
				int rotIndex = 0;
				// Copy the keys (rotations)
				std::transform(ampTrees[p][hash][parv].begin(),
					ampTrees[p][hash][parv].end(),
					std::back_inserter(rots),
					first(ampTrees[p][hash][parv]));

				hybridCalc->AddRotations(workspaces[p].children[childNum], rots);

				for (auto rotI = ampTrees[p][hash][parv].begin();
					rotI != ampTrees[p][hash][parv].end();
					++rotI)
				{
					hybridCalc->AddTranslations(workspaces[p].children[childNum], rotIndex++,
						rotI->second);

				}
				childNum++;
			}
			else {
				// TODO::Hybrid We can call Amplitude::CalculateGrid and transfer the grid back and forth 
				//				as an alternative to making all of the Amplitudes extend IGPUGridCalculable.
				printf("Model not IGPUGridCalculable.\n");
			}

			if (pStop && *pStop) {
				for (int j = 0; j < workspaces.size(); j++) {
					hybridCalc->FreeWorkspace(workspaces[j]);
				}
				delete hybridCalc;
				return STOPPED;
			} // if pStop
			numAmpsCalced++;
		} // for iter = myAmps[p].begin(); iter != myAmps[p].end(); ++iter
	} // for unsigned int p = 0; p < numWorkspaces; p++
	printf("End of hybrid grid calculation.\n");

	// TODO::Hybrid Figure out how to construct full amplitude
	int ii = 0;
	ii++;
	if (workspaces.size() > 1) {
		printf("\nWARNING!!! NO IMPLEMENTATION FOR MULTIPLE GPU SUPPORT YET!!! BREAKING!\n\n");
		for (int j = 0; j < workspaces.size(); j++) {
			hybridCalc->FreeWorkspace(workspaces[j]);
		}
		delete hybridCalc;
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Consolidate ampTrees[p][hash][ampParams][rotation].[translations] and myAmps[p][hash][ampParams].[*Amplitudes]

	// TODO::Hybrid Orientation Average

	// Master
	workspaces[0].intMethod = orientationMethod;
	aveBeg = clock();

	bool bDone = hybridCalc->ComputeIntensity(workspaces, (double*)res.data(), epsi, iterations,
		progFunc, progArgs, 0.3f, 1.0f, pStop);

	// Do we want to think about saving the amplitudes back to the CPU?
	for (int j = 0; j < workspaces.size(); j++) {
		hybridCalc->FreeWorkspace(workspaces[j]);
	}
	delete hybridCalc;
	aveEnd = clock();

	// This is to release the device (duh!). Important to allow the GPU to power
	// down and cool off. Otherwise the room gets hot...
	cudaDeviceReset();

	if (bDone)
	{
		std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
		std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;
		return PDB_OK;	// When done
	}
	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
}

template <typename T>
PDB_READER_ERRS DomainModel::PerformGPUHybridComputation2D(clock_t& gridBegin, const std::vector<T>& Q, clock_t& aveBeg, 
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& res, T epsi, uint64_t iterations, clock_t& aveEnd)
{
#ifndef USE_JACOBIAN_SPHERE_GRID
	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
#endif

	gridBegin = clock();
	int devCount;
	cudaError_t t = UseGPUDevice(&devCount);
	if (devCount <= 0 || !g_useGPUAndAvailable || t != cudaSuccess) {
		printf("No GPU backend. Calculating on CPU.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Locate and create the GPU calculator
	gpuGridcalculator_t gpuCalcHybridGen = (gpuGridcalculator_t)GPUCreateCalculatorHybrid;
	//GetProcAddress((HMODULE)g_gpuModule, "GPUCreateCalculatorHybrid");
	if (!gpuCalcHybridGen) {
		printf("ERROR: Cannot find hybrid generator function\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Report indeterminate progress (for now)
	// TODO::Later: Meaningful progress bar in direct method
	if (progFunc)
		progFunc(progArgs, 0.0);

	IGPUGridCalculator* hybridCalc = gpuCalcHybridGen();

	// Flatten the symmetry/primitive tree to a single vector
	std::vector<Amplitude*> flatVec;
	std::vector<LocRotScale> flatLocrot;
	std::vector<VectorXd> flatParams;

	Eigen::Matrix4f tmat = Eigen::Matrix4f::Identity();
	FlattenTree<true>(_amps, _ampParams, _ampLayers, tmat, flatVec, flatLocrot, flatParams);
#ifdef _DEBUG
	{
		for (int ll = 0; ll < flatLocrot.size(); ll++) {
			std::cout << "[" << flatLocrot[ll].first.x << ", " << flatLocrot[ll].first.y << ", " << flatLocrot[ll].first.z << "]  ["
				<< flatLocrot[ll].first.alpha << ", " << flatLocrot[ll].first.beta << ", " << flatLocrot[ll].first.gamma <<
				"] scale: " << flatLocrot[ll].second << std::endl;
		}
	}
#endif // _DEBUG
	bool allImplemented = true;
	for (int i = 0; i < flatVec.size(); ++i)
	{
		IGPUGridCalculable* tst = dynamic_cast<IGPUGridCalculable*>(flatVec[i]);
		if (!tst) {
			allImplemented = false;
			break;
		}
	}
	if (!allImplemented) {
		printf("Not all models have been implemented to work as hybrid on a GPU. Continuing on CPU.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// An ugly but efficient fixed-height tree built from maps. Tree groups amplitudes by:
	// 1. type
	// 2. parameters (number of layers, then parameters themselves)
	// 3. orientation (alpha, then beta, then gamma)
	// 4(unordered). translation
	typedef std::map<std::string,
		std::map<VectorXd,
		std::map<float4, std::vector<float4>, rotation_ulp_less>,
		paramvec_less> > AmpCalcTree;

	typedef std::map<VectorXd,
		std::map<float4, std::vector<float4>, rotation_ulp_less>,
		paramvec_less> ParamOrientTree;

	/*
	/// Maps a param vector to a pointer to an Amplitude
	typedef std::map<VectorXd, Amplitude*, paramvec_less> VecToAmp;
	*/

	typedef std::map<UniqueModels, Amplitude*, UniqueModels_less> AmpsToCalc;
	/// var[workspaceid][hash][paramVec] is a pointer to an Amplitude
	typedef std::vector<AmpsToCalc> WorkspacesToCalc;

	// The work is divided to P (numWorkspaces) trees of parallel processing
	const unsigned int numWorkspaces = devCount;
	std::vector<AmpCalcTree> ampTrees(numWorkspaces);
	std::vector<GridWorkspace> workspaces(numWorkspaces);
	std::map<std::string, Amplitude*> hashToAmp;
	WorkspacesToCalc myAmps;
	// TODO::Hybrid - Hash should include the models parameters to distinguish between models


	std::vector<float> Qfloat(Q.size());
	for (int i = 0; i < Q.size(); ++i)
		Qfloat[i] = (float)Q[i];

	unsigned int ampsPerWorkspace = flatVec.size() / numWorkspaces;

	int curGPU = 0;
	int numAmpsToCalc = 0;
	myAmps.resize(numWorkspaces);
	// The loop that generates the trees and workspaces
	for (unsigned int p = 0; p < numWorkspaces; p++)
	{
		unsigned int lastAmp = ((p == numWorkspaces - 1) ? flatVec.size() : ((p + 1) * ampsPerWorkspace));

		for (unsigned int i = p * ampsPerWorkspace; i < lastAmp; ++i)
		{
			std::string hash = flatVec[i]->Hash();
			hashToAmp[hash] = flatVec[i];
			myAmps[p][std::make_tuple(hash, flatParams[i])] = flatVec[i];
			LocationRotation locrot = flatLocrot[i].first;

			// OPTIMIZATION: Conserve rotations and translations by letting 
			// rotation/translation-invariant models modify their Locrots
			// EXAMPLES: A sphere is invariant to rotation, a cylinder is invariant to one of the angles
			double aa = locrot.alpha, bb = locrot.beta, gg = locrot.gamma;
			IGPUCalculable* gpuModel = dynamic_cast<IGPUCalculable*>(flatVec[i]);
			if (gpuModel)
				gpuModel->CorrectLocationRotation(locrot.x, locrot.y, locrot.z,
					aa, bb, gg);
			locrot.alpha = Radian(aa);
			locrot.beta = Radian(bb);
			locrot.gamma = Radian(gg);

			//std::cout << "Model " << i << ": " << flatVec[i]->GetName() << " - XYZABG " 
			//		  << locrot.x << " " << locrot.y << " " << locrot.z << " " << locrot.alpha << " " << locrot.beta << " " << locrot.gamma << std::endl; 

			ampTrees[p][hash]
				[flatParams[i]]
			[make_float4(locrot.alpha, locrot.beta, locrot.gamma, flatLocrot[i].second)].
				push_back(make_float4(locrot.x, locrot.y, locrot.z, 0.));
		}

		// Obtain the maximal number of translations
		unsigned int maxTranslations = 0;
		for (AmpCalcTree::iterator iter = ampTrees[p].begin(); iter != ampTrees[p].end(); ++iter)
			for (ParamOrientTree::iterator piter = iter->second.begin(); piter != iter->second.end(); ++piter)
				for (auto oiter = piter->second.begin(); oiter != piter->second.end(); ++oiter)
				{
					if (oiter->second.size() > maxTranslations)
						maxTranslations = oiter->second.size();
				}
		// END of maximal translations

		// This should/could be obtained by allocating a grid and obtaining the numbers from the instance.
		// I don't want to allocate so much memory.
		double stepSize = qMax / double(gridSize / 2);
		int actualSz = (gridSize / 2) + 3 /*Extras*/;
		long long kk = actualSz;
		int thetaDivisions = 3;
		int phiDivisions = 6;
		long long totalsz = (phiDivisions * kk * (kk + 1) * (3 + thetaDivisions + 2 * thetaDivisions * kk)) / 6;
		totalsz++;	// Add the origin
		totalsz *= 2;	// Complex

		bool bsuccess = hybridCalc->Initialize(curGPU, Qfloat, totalsz, thetaDivisions, phiDivisions, actualSz + 1, qMax,
			stepSize, workspaces[p]);
		hybridCalc->SetNumChildren(workspaces[p], myAmps[p].size());

		for (auto hiter = myAmps[p].begin(); hiter != myAmps[p].end(); ++hiter)
		{
			numAmpsToCalc++;
			int childNum = 0;
			bool bsuccess = hybridCalc->Initialize(curGPU, Qfloat,
				totalsz, thetaDivisions, phiDivisions, actualSz + 1, qMax, stepSize,
				//std::get<1>(hiter->first).size(), std::get<1>(hiter->first).data(),
				workspaces[p].children[childNum++]);
		}
		curGPU++;
		if (curGPU == devCount)
			curGPU = 0;
	} // for(unsigned int p = 0; p < numWorkspaces; p++)

	printf("End of hybrid init...\n");

	int numAmpsCalced = 0;

	// Calculate all of the grids up to the point where there are no grids used
	for (unsigned int p = 0; p < numWorkspaces; p++)
	{
		int childNum = 0;
		for (auto iter = myAmps[p].begin(); iter != myAmps[p].end(); ++iter)
		{
			// Don't need to call PreCalculate, it was already called in DomainModel::CalculateVector
			//inIter->second->PreCalculate(VectorXd(inIter->first) /*Needs to be copied, otherwise will be const and can't pass by reference*/
			//	, inIter->first.size());
			IGPUGridCalculable* gpuModel = dynamic_cast<IGPUGridCalculable*>(iter->second);
			if (gpuModel)
			{
				gpuModel->SetModel(workspaces[p].children[childNum]);	// Running this in a separate loop may increase performance (maybe?)
				gpuModel->CalculateGridGPU(workspaces[p].children[childNum]);

				std::string	hash = std::get<0>(iter->first);
				VectorXd	parv = std::get<1>(iter->first);

				std::vector<float4> rots;
				int rotIndex = 0;
				// Copy the keys (rotations)
				std::transform(ampTrees[p][hash][parv].begin(),
					ampTrees[p][hash][parv].end(),
					std::back_inserter(rots),
					first(ampTrees[p][hash][parv]));

				hybridCalc->AddRotations(workspaces[p].children[childNum], rots);

				for (auto rotI = ampTrees[p][hash][parv].begin();
					rotI != ampTrees[p][hash][parv].end();
					++rotI)
				{
					hybridCalc->AddTranslations(workspaces[p].children[childNum], rotIndex++,
						rotI->second);

				}
				childNum++;
			}
			else {
				// TODO::Hybrid We can call Amplitude::CalculateGrid and transfer the grid back and forth 
				//				as an alternative to making all of the Amplitudes extend IGPUGridCalculable.
				printf("Model not IGPUGridCalculable.\n");
			}

			if (pStop && *pStop) {
				for (int j = 0; j < workspaces.size(); j++) {
					hybridCalc->FreeWorkspace(workspaces[j]);
				}
				delete hybridCalc;
				return STOPPED;
			} // if pStop
			numAmpsCalced++;
		} // for iter = myAmps[p].begin(); iter != myAmps[p].end(); ++iter
	} // for unsigned int p = 0; p < numWorkspaces; p++
	printf("End of hybrid grid calculation.\n");

	// TODO::Hybrid Figure out how to construct full amplitude
	int ii = 0;
	ii++;
	if (workspaces.size() > 1) {
		printf("\nWARNING!!! NO IMPLEMENTATION FOR MULTIPLE GPU SUPPORT YET!!! BREAKING!\n\n");
		for (int j = 0; j < workspaces.size(); j++) {
			hybridCalc->FreeWorkspace(workspaces[j]);
		}
		delete hybridCalc;
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Consolidate ampTrees[p][hash][ampParams][rotation].[translations] and myAmps[p][hash][ampParams].[*Amplitudes]

	// TODO::Hybrid Orientation Average

	// Master
	workspaces[0].intMethod = orientationMethod;
	aveBeg = clock();

	bool bDone = hybridCalc->ComputeIntensity(workspaces, (double*)res.data(), epsi, iterations,
		progFunc, progArgs, 0.3f, 1.0f, pStop);

	// Do we want to think about saving the amplitudes back to the CPU?
	for (int j = 0; j < workspaces.size(); j++) {
		hybridCalc->FreeWorkspace(workspaces[j]);
	}
	delete hybridCalc;
	aveEnd = clock();

	// This is to release the device (duh!). Important to allow the GPU to power
	// down and cool off. Otherwise the room gets hot...
	cudaDeviceReset();

	if (bDone)
	{
		std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
		std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;
		return PDB_OK;	// When done
	}
	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
}


template <typename T>
PDB_READER_ERRS DomainModel::PerformgGPUAllGridsMCOACalculations(const std::vector<T> &Q, std::vector<T> &res, uint64_t iterations, T epsi, clock_t &aveEnd, clock_t aveBeg, clock_t gridBegin)
{
#ifndef USE_JACOBIAN_SPHERE_GRID
	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
#endif
	int devCount;
	cudaError_t t = UseGPUDevice(&devCount);
	if (devCount <= 0 || t != cudaSuccess) {
		printf("No compatible GPU detected. Proceeding with CPU.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	GPUSumGridsJacobSphr_t gpuSumGrids = (GPUSumGridsJacobSphr_t)GPUSumGridsJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUSumGridsJacobSphrDF");
	if (!gpuSumGrids)
	{
		printf("ERROR: Cannot find sum grids function in GPU DLL.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	GPUCalculateMCOA_t gpuCalcMCOA = (GPUCalculateMCOA_t)GPUCalcMCOAJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUCalcMCOAJacobSphrDF");
	if (!gpuCalcMCOA)
	{
		printf("ERROR: Cannot find MC OA function in GPU DLL.\n");
		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
	}

	// Create grid on GPU that is the sum of all of _amps including translation/orientation, keeping the result
	std::vector<double *> grids;
	std::vector<double *> derivatives;
	std::vector<double> trans, rots;
	for (int i = 0; i < _amps.size(); i++) {
		double x, y, z;
		double a, b, g;
		grids.push_back(_amps[i]->GetDataPointer());
		derivatives.push_back(((JacobianSphereGrid*)(_amps[i]->GetInternalGridPointer()))->GetInterpolantPointer());
		_amps[i]->GetTranslationRotationVariables(x, y, z, a, b, g);
		trans.push_back(x);
		trans.push_back(y);
		trans.push_back(z);
		rots.push_back(a);
		rots.push_back(b);
		rots.push_back(g);
	}

	JacobianSphereGrid *grd;
	JacobianSphereGrid tmpGrid((_amps.size() > 1 ? gridSize : 1), qMax);	// 
	long long voxels;
	if (_amps.size() > 1) {
		grd = &tmpGrid;
		voxels = grd->GetRealSize() / (sizeof(double) * 2);
		gpuSumGrids(voxels, grd->GetDimY(1) - 1, grd->GetDimZ(1, 1), grd->GetStepSize(), grids.data(),
			derivatives.data(), trans.data(), rots.data(), _amps.size(), grd->GetDataPointer(),
			progFunc, progArgs, 0.3, 0.35, pStop);

		// Calculate splines on GPU
		tmpGrid.CalculateSplines();
	}
	else {
		grd = (JacobianSphereGrid *)(_amps[0]->GetInternalGridPointer());
		voxels = grd->GetRealSize() / (sizeof(double) * 2);
	}
	// Calculate the Orientation Average
	int re = gpuCalcMCOA(voxels, grd->GetDimY(1) - 1, grd->GetDimZ(1, 1), grd->GetStepSize(), grd->GetDataPointer(),
		grd->GetInterpolantPointer(), (double*)Q.data(), (double*)res.data(), Q.size(), iterations, epsi,
		progFunc, progArgs, 0.35, 1.0, pStop);

	if (pStop && *pStop)
		return STOPPED;

	aveEnd = clock();

	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;

	// Reset the GPU to clear memory

	return PDB_OK;
}

template <typename T>
PDB_READER_ERRS DomainModel::PerformgGPUAllGridsMCOACalculations2D(const std::vector<T>& Q, 
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& res, uint64_t iterations, T epsi, clock_t& aveEnd, clock_t aveBeg, clock_t gridBegin)
{
////#ifndef USE_JACOBIAN_SPHERE_GRID
////	return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
////#endif
////	int devCount;
////	cudaError_t t = UseGPUDevice(&devCount);
////	if (devCount <= 0 || t != cudaSuccess) {
////		printf("No compatible GPU detected. Proceeding with CPU.\n");
////		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
////	}
////
////	GPUSumGridsJacobSphr_t gpuSumGrids = (GPUSumGridsJacobSphr_t)GPUSumGridsJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUSumGridsJacobSphrDF");
////	if (!gpuSumGrids)
////	{
////		printf("ERROR: Cannot find sum grids function in GPU DLL.\n");
////		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
////	}
////
////	GPUCalculateMCOA_t gpuCalcMCOA = (GPUCalculateMCOA_t)GPUCalcMCOAJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUCalcMCOAJacobSphrDF");
////	if (!gpuCalcMCOA)
////	{
////		printf("ERROR: Cannot find MC OA function in GPU DLL.\n");
////		return ERROR_WITH_GPU_CALCULATION_TRY_CPU;
////	}
////
////	// Create grid on GPU that is the sum of all of _amps including translation/orientation, keeping the result
////	std::vector<double*> grids;
////	std::vector<double*> derivatives;
////	std::vector<double> trans, rots;
////	for (int i = 0; i < _amps.size(); i++) {
////		double x, y, z;
////		double a, b, g;
////		grids.push_back(_amps[i]->GetDataPointer());
////		derivatives.push_back(((JacobianSphereGrid*)(_amps[i]->GetInternalGridPointer()))->GetInterpolantPointer());
////		_amps[i]->GetTranslationRotationVariables(x, y, z, a, b, g);
////		trans.push_back(x);
////		trans.push_back(y);
////		trans.push_back(z);
////		rots.push_back(a);
////		rots.push_back(b);
////		rots.push_back(g);
////	}
////
////	JacobianSphereGrid* grd;
////	JacobianSphereGrid tmpGrid((_amps.size() > 1 ? gridSize : 1), qMax);	// 
////	long long voxels;
////	if (_amps.size() > 1) {
////		grd = &tmpGrid;
////		voxels = grd->GetRealSize() / (sizeof(double) * 2);
////		gpuSumGrids(voxels, grd->GetDimY(1) - 1, grd->GetDimZ(1, 1), grd->GetStepSize(), grids.data(),
////			derivatives.data(), trans.data(), rots.data(), _amps.size(), grd->GetDataPointer(),
////			progFunc, progArgs, 0.3, 0.35, pStop);
////
////		// Calculate splines on GPU
////		tmpGrid.CalculateSplines();
////	}
////	else {
////		grd = (JacobianSphereGrid*)(_amps[0]->GetInternalGridPointer());
////		voxels = grd->GetRealSize() / (sizeof(double) * 2);
////	}
////	// Calculate the Orientation Average
////	int re = gpuCalcMCOA(voxels, grd->GetDimY(1) - 1, grd->GetDimZ(1, 1), grd->GetStepSize(), grd->GetDataPointer(),
////		grd->GetInterpolantPointer(), (double*)Q.data(), (double*)res.data(), Q.size(), iterations, epsi,
////		progFunc, progArgs, 0.35, 1.0, pStop);
////
////	if (pStop && *pStop)
////		return STOPPED;
////
////	aveEnd = clock();
////
////	std::cout << "Took " << double(aveBeg - gridBegin) / CLOCKS_PER_SEC << " seconds to calculate the grids." << std::endl;
////	std::cout << "Took " << double(aveEnd - aveBeg) / CLOCKS_PER_SEC << " seconds to calculate the orientational average." << std::endl;
////
////	// Reset the GPU to clear memory
//
	return PDB_OK;
}


template <typename T>
void DomainModel::AverageIntensitiesBetweenLayers(const std::vector<T> &TrelevantQs, std::vector<T> &reses, size_t layerInd, FACC epsi, unsigned int seed, uint64_t iterations)
{
	typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayT;
	if (orientationMethod == OA_ADAPTIVE_GK)
	{
		throw std::logic_error("The method or operation is not implemented for Gauss Kronrod.");
		//return GaussKron2DSphereRecurs(q, epsi, iterations, 0);
	}
	else if (orientationMethod == OA_MC)
	{
		std::vector<FACC> relevantQs(TrelevantQs.size());
		std::copy_n(TrelevantQs.begin(), TrelevantQs.size(), relevantQs.begin());

		unsigned long long minIter = 100ULL;
		Eigen::Array<FACC, Eigen::Dynamic, Eigen::Dynamic> resHistory;
		resHistory.resize(relevantQs.size(), 4);

		std::mt19937 rng;
		rng.seed(seed);

		std::uniform_real_distribution<FACC> ranU2(0.0, 2.0);
		std::complex<FACC> phase, im(0.0, 1.0);

		Eigen::Array<FACC, Eigen::Dynamic, 1> runningIntensitySum(relevantQs.size());
		runningIntensitySum.setZero();

		for (uint64_t i = 0; i < iterations; i++)
		{
			FACC theta, phi, u2, v2;

			u2 = ranU2(rng);
			v2 = ranU2(rng);
			phi = u2 * M_PI;
			theta = acos(v2 - 1.0);

			ArrayXcX ampSums(relevantQs.size());
			ampSums.setZero();

			for (int j = 0; j < _amps.size(); j++)
				ampSums += _amps[j]->getAmplitudesAtPoints(relevantQs, theta, phi);

			runningIntensitySum += (ampSums * ampSums.conjugate()).real();

			if (epsi > 0.0) {
				if (i % minIter == 0)
				{
					int position = (i / minIter) % resHistory.cols();
					// Convergence Place TODO FIXME
					// Convergence test
					resHistory.col(position) = runningIntensitySum / FACC(i + 1);
					if (i >= 4 * minIter)
					{
						bool converged = true;

						for (int j = 0; converged && j < resHistory.cols(); j++)
							converged &= ((1.0 - (resHistory.col(j) / resHistory.col(0))).abs() < epsi).all();

						if (converged)
						{
							ArrayT::Map(&reses[0], relevantQs.size()) = resHistory.col(position).cast<T>();
							return;
						}

					}
				}
			} // if i >=...

		} // for
		ArrayT::Map(&reses[0], relevantQs.size()) = (runningIntensitySum / iterations).cast<T>();
		return;

	} // elif orientationMethod == OA_MC

	printf("Integration of multiple points not implemented on the CPU for the chosen method");
	throw backend_exception(ERROR_GENERAL, "Integration of multiple points not implemented on the CPU for the chosen method");
}

//template <typename T> // should PolarCalculationData be template? is there any case were q is float instead of double?
void DomainModel::AverageIntensitiesBetweenLayers2D(std::vector<PolarCalculationData*> relevantQData, size_t layerInd, FACC epsi, unsigned int seed, uint64_t iterations)
{
	//std::cout << "--------- DomainModel::AverageIntensitiesBetweenLayers2D -----------" << std::endl;

	int q;
	if (orientationMethod == OA_ADAPTIVE_GK)
	{
		throw std::logic_error("The method or operation is not implemented for Gauss Kronrod.");
		//return GaussKron2DSphereRecurs(q, epsi, iterations, 0);
	}
	else if (orientationMethod == OA_MC)
	{
		unsigned long long minIter = 100ULL;
		// resHistory is a matrix of PolarCalculationData. only rIntensities are relevant
		Eigen::Array<PolarCalculationData, Eigen::Dynamic, Eigen::Dynamic> resHistory;
		resHistory.resize(relevantQData.size(), 4);

		std::mt19937 rng;
		rng.seed(seed);

		std::uniform_real_distribution<FACC> ranU2(0.0, 2.0);
		std::complex<FACC> phase, im(0.0, 1.0);

		Eigen::Array<PolarCalculationData, Eigen::Dynamic, 1> runningIntensitySum(relevantQData.size());
		for (int q = 0; q < relevantQData.size(); q++)
		{
			runningIntensitySum[q] = PolarCalculationData(relevantQData[q]->theta.size());
			std::fill(runningIntensitySum[q].rIntensities.begin(), runningIntensitySum[q].rIntensities.end(), 0);
		}
		

		for (uint64_t i = 0; i < iterations; i++)
		{
			FACC phi, u2, v2;

			u2 = ranU2(rng);
			v2 = ranU2(rng);
			phi = u2 * M_PI;
			

			std::vector<PolarCalculationData> ampSums(relevantQData.size()); 
			for (int a = 0; a < relevantQData.size(); a++)
			{
				ampSums[a] = PolarCalculationData(relevantQData[a]->theta.size());
				ampSums[a].cIntensities.setZero();
			}

			for (int j = 0; j < _amps.size(); j++)
			{
				_amps[j]->getAmplitudesAtPoints2D(relevantQData, phi);
				for (int a = 0; a < relevantQData.size(); a++)
				{
					ampSums[a].addCIntensities(*(relevantQData[a]));
				}
			}

			for (int a = 0; a < ampSums.size(); a++)
			{
				ampSums[a].SqrComplexToReal();
				runningIntensitySum[a].addRIntensities(ampSums[a]);
			}

			if (epsi > 0.0) {
				if (i % minIter == 0)
				{
					int position = (i / minIter) % resHistory.cols();
					// Convergence Place TODO FIXME
					// Convergence test
					for (int i = 0; i < resHistory.col(position).size(); i++)
					{
						resHistory.col(position)[i].rIntensities = runningIntensitySum[i].rIntensities / FACC(i + 1);
					}
					
					if (i >= 4 * minIter)
					{
						bool converged = true;

						for (int j = 0; converged && j < resHistory.cols(); j++)
							for (int q = 0; q < resHistory.col(j).size() ; q++)
								converged &= ((1.0 - (resHistory.col(j)[q].rIntensities / resHistory.col(0)[q].rIntensities)).abs() < epsi).all();

						if (converged)
						{
							//ArrayT::Map(&reses[0], relevantQs.size()) = resHistory.col(position).cast<T>();
							for (int i = 0; i < relevantQData.size(); i++)
							{
								relevantQData[i]->rIntensities = resHistory.col(position)[i].rIntensities;
							}
							return;
						}
					}
				}
			} // if i >=...

		} // for
		//ArrayT::Map(&reses[0], relevantQs.size()) = (runningIntensitySum / iterations).cast<T>();
		for (int i = 0; i < relevantQData.size(); i++)
		{
			relevantQData[i]->rIntensities = runningIntensitySum[i].rIntensities / iterations;
		}
		return;
	} // elif orientationMethod == OA_MC

	printf("Integration of multiple points not implemented on the CPU for the chosen method");
	throw backend_exception(ERROR_GENERAL, "Integration of multiple points not implemented on the CPU for the chosen method");
}

bool DomainModel::GetHasAnomalousScattering()
{
	for (const auto& child : _amps)
		if (child->GetHasAnomalousScattering()) return true;
	return false;
}

std::string DomainModel::Hash() const
{
	std::string str = "Domain model: ";
	for (const auto & child : _amps)
		str += child->Hash();
	return md5(str);
}

// double DomainModel::CubicSplineSphere( double q ) {
// 	// For math, see http://mathworld.wolfram.com/CubicSpline.html
// 
// /*
// 	for(unsigned int j = 0; j < _amps.size(); j++) {
// 	kVals[i] += _amps[j]->getAmplitude(q * st * cos(xi), q * st * sin(xi), q * ct);
// 	}
// */
// 
// 	// Make sure that all grids are aligned and of the same orientation (?)
// 	// Create a 2D layer
// 
// 	for(unsigned int j = 0; j < _amps.size(); j++) {
// 	//	kVals[i] += _amps[j]->getAmplitude(q * st * cos(xi), q * st * sin(xi), q * ct);
// 	}
// 	return -3.28168;
// }
// 
// void DomainModel::CubicSplineSphere( double qMin, double qMax, VectorXd &qPoints ) {
// 
// 	// For each radius that the grids have calculations, calculate the shell layer
// 	//_amps[0]
// 	
// }
// 
// PDB_READER_ERRS DomainModel::CubicSplineSphereVector( const std::vector<double>& Q, std::vector<FACC>& res ) {
// 	res.resize(Q.size());
// 	const double cProgMin = 0.3, cProgMax = 1.0;
// 	int prog = 0;
// 
// 	// Pre-calculate the control points on the q-axis.
// 	VectorXd qPoints;
// 	CubicSplineSphere(Q[0], Q.back(), qPoints);
// 
// 	for(unsigned int i = 0; i < Q.size(); i++)
// 
// #pragma omp parallel for schedule(dynamic, Q.size() / 50)
// 	for(int i = 0; i < Q.size(); i++) {
// 		if(pStop && *pStop)
// 			continue;
// 
// 		// FIX THIS
// 		res[i] = CubicSplineSphere(Q[i]);
// 
// #pragma omp critical
// 		{
// 			if(progFunc)
// 				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(Q.size())) + cProgMin);
// 		}
// 
// 	}
// 
// 	if(pStop && *pStop)
// 		return STOPPED;
// 
// 	return PDB_OK;
// }

#pragma endregion	// Intensity Calculator class


#pragma region Amplitude Producer interface class
Amplitude::Amplitude() {
	grid = NULL;
	bUseGrid = false;
	bUseGPU = true;
	gridStatus = AMP_UNINITIALIZED;
	status = UNINITIALIZED;
	scale = 1.0;
	SetLocationData(0.0, 0.0, 0.0, Radian(), Radian(), Radian());
}

Amplitude::~Amplitude() {
	if (grid) {
		delete grid;
		grid = NULL;
	}
	if (fs::exists(rndPath)) {
		//fs::remove(rndPath);
		//if(fs::exists(rndPath))
		//std::wcout << L"Error removing file \"" << rndPath << L"\".\n";
		std::cout << "amp files not deleted";
	}
}

void Amplitude::calculateGrid(FACC qmax, int sections, progressFunc progFunc, void *progArgs, double progMin, double progMax, int *pStop) {
	PDB_READER_ERRS dbg = getError();
	if (!bUseGrid) {
		std::cout << "Not using grid";
		return;
	}

	bool suc = InitializeGrid(qmax, sections);

	if (gridStatus == AMP_CACHED) {
		ReadAmplitudeFromCache();
	}
	else {
		// Fill/refill grid
		std::function< std::complex<FACC>(FACC, FACC, FACC)> bindCalcAmp = [=](FACC qx, FACC qy, FACC qz) {
			return this->calcAmplitude(qx, qy, qz);
		};
		grid->Fill(bindCalcAmp, (void *)progFunc, progArgs, progMin, progMax, pStop);
	}
	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;
	if (progFunc)
		progFunc(progArgs, progMax);
}


std::complex<FACC> Amplitude::getAmplitude(FACC qx, FACC qy, FACC qz) {

	std::complex<FACC> Im(0.0, 1.0);
	Eigen::Vector3d Q(qx, qy, qz), Qt;
	Eigen::Matrix3d rot;
	Eigen::Vector3d R(tx, ty, tz);
	Qt = Q.transpose() * RotMat;

	qx = Qt.x();
	qy = Qt.y();
	qz = Qt.z();

	if (!grid || !bUseGrid) {
		//printf("WHAT? %s,  %p\n", bUseGrid ? "true" : "false", grid);
		return exp(Im * (Q.dot(R))) * calcAmplitude(qx, qy, qz);
	}

	if (grid)
		return exp(Im * (Q.dot(R))) * grid->GetCart(qx, qy, qz);
	else // WTF?
		throw backend_exception(13, "Amplitude::getAmplitude illegal area of code");
}

PDB_READER_ERRS Amplitude::getError() const {
	return status;
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToStream(std::ostream & stream) {
	std::string st;
	GetHeader(0, st);
	st.append("\n");	// TODO::HEADER Determine if this newline is needed or hurts

	return WriteAmplitudeToStream(stream, st);
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToStream(std::ostream & stream, std::string header) {
	if (!grid)
		return GENERAL_ERROR;

	/* List of things that need to be written:
	* Header
	* File version number
	* Bytes per data (sizeof(std::complex<FACC>))
	* Delta Q!
	* Size of trnsfrm --> txSize
	* txSize times size of trnsfrm[i] --> tySize[txSize]
	* txSize * tySize.size() sizes of tz trnsfrm[i][j].size()
	* Contents of trnsfrm
	*
	**/
	if (header.size() == 0) {
		std::stringstream ss;
		ss << "# No header specified.\n";
		ss << "# qMax = " << grid->GetQMax() << "\n";
		ss << "# Grid step size = " << grid->GetStepSize() << "\n";
		ss << "# Grid size = " << grid->GetSize() << "\n";
		ss << "# Program revision: " << BACKEND_VERSION << "\n";
		ss << "\n";

		header = ss.str();
	}

	if (stream.good()) {
		if (grid)
			grid->ExportBinaryData(stream, header);

	}
	else {
		return NO_FILE;
	}

	return PDB_OK;


}


PDB_READER_ERRS Amplitude::WriteAmplitudeToFile(const std::string& fileName) {
	std::string st;
	GetHeader(0, st);
	st.append("\n");	// TODO::HEADER Determine if this newline is needed or hurts

	return WriteAmplitudeToFile(fileName, st);
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToFile(const std::string& zipFilename, std::string header) {
	if (fs::exists(zipFilename))
		fs::remove(zipFilename);
	StoreMethod::Ptr ctx = StoreMethod::Create();
	ZipArchive::Ptr archive = ZipFile::Open(zipFilename);
	archive->RemoveEntry("grid.dat");
	archive->RemoveEntry("criticalinfo.json");
	archive->RemoveEntry("header.json");

	ZipArchiveEntry::Ptr gridEntry = archive->CreateEntry("grid.dat");
	if (gridEntry != nullptr)
	{
		char gridContent[] = "Content to add";
		imemstream gridContentStream(gridContent);
		grid->ExportBinaryDataToStream(gridContentStream);

		gridEntry->SetCompressionStream(
			gridContentStream,
			ctx,
			ZipArchiveEntry::CompressionMode::Immediate
		);
	}

	char *infoCaution;
	ZipArchiveEntry::Ptr infoEntry = archive->CreateEntry("criticalinfo.json");
	if (infoEntry != nullptr)
	{

		JsonWriter infoWriter;
		infoWriter.StartObject();
		grid->WriteToJsonWriter(infoWriter);
		infoWriter.Key("Params");
		infoWriter.StartArray();
		for (unsigned i = 0; i < previousParameters.size(); i++)
		{
			infoWriter.Double(previousParameters[i]);
		}
		infoWriter.EndArray();

		infoWriter.EndObject();
		const char * chartest = infoWriter.GetString();
		std::string str(chartest);

		infoCaution = new char[str.length() + 1];
		str.copy(infoCaution, str.length() + 1);
		imemstream infoContentStream(infoCaution, str.length());
		infoEntry->SetCompressionStream(
			infoContentStream,
			ctx,
			ZipArchiveEntry::CompressionMode::Immediate
		);

		
	}

	char *caution2;
	ZipArchiveEntry::Ptr headerEntry = archive->CreateEntry("header.json");
	if (headerEntry != nullptr)
	{
		JsonWriter entryWriter;
		entryWriter.StartObject();
		entryWriter.Key("Old Header:");
		entryWriter.String(header.c_str());
		entryWriter.Key("Header");
		entryWriter.StartObject();
		GetHeader(0, entryWriter);
		entryWriter.EndObject();
		entryWriter.EndObject();
		const char * chartest2 = entryWriter.GetString();

		std::string str2(chartest2);
		caution2 = new char[str2.length() + 1];
		str2.copy(caution2, str2.length() + 1);
		imemstream headerContentStream(caution2, str2.length());
		headerEntry->SetCompressionStream(
			headerContentStream,
			ctx,
			ZipArchiveEntry::CompressionMode::Immediate
		);

		
	}

	ZipFile::SaveAndClose(archive, zipFilename);
	delete[] infoCaution;
	delete[] caution2;



	//ZipArchive::Ptr archive2 = ZipFile::Open(zipFilename);
	//JacobianSphereGrid testgrid(archive2);
	return PDB_OK;
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToFile(const std::wstring& fileName) {
	std::string st;
	GetHeader(0, st);
	st.append("\n");	// TODO::HEADER Determine if this newline is needed or hurts

	return WriteAmplitudeToFile(fileName, st);
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToFile(const std::wstring& fileName, std::string header) {


	if (!grid)
		return GENERAL_ERROR;

	/* List of things that need to be written:
	* Header
	* File version number
	* Bytes per data (sizeof(std::complex<FACC>))
	* Delta Q!
	* Size of trnsfrm --> txSize
	* txSize times size of trnsfrm[i] --> tySize[txSize]
	* txSize * tySize.size() sizes of tz trnsfrm[i][j].size()
	* Contents of trnsfrm
	*
	**/
	if (header.size() == 0) {
		std::stringstream ss;
		ss << "# No header specified.\n";
		ss << "# qMax = " << grid->GetQMax() << "\n";
		ss << "# Grid step size = " << grid->GetStepSize() << "\n";
		ss << "# Grid size = " << grid->GetSize() << "\n";
		ss << "# Program revision: " << BACKEND_VERSION << "\n";
		ss << "\n";

		header = ss.str();
	}

	fs::wpath pathName(fileName);
	pathName = fs::system_complete(fileName);
	std::wstring a1, a2, a3;
	a1 = pathName.leaf().wstring();
	a2 = pathName.parent_path().wstring();

	if (!fs::exists(pathName.parent_path())) {
		std::wstring strD = pathName.parent_path().wstring();
		boost::system::error_code er;
		if (!fs::create_directories(pathName.parent_path(), er)) {
			std::cout << "\nError code: " << er << "\n";
			while (!fs::exists(pathName.parent_path())) {
				pathName = pathName.parent_path();
			}
			pathName = fs::wpath(pathName.wstring() + L"ERROR_CREATING_DIR");
			{fs::ofstream f(pathName); }
			return FILE_ERROR;
		}
	}

	fs::fstream writeFile;
	writeFile.open(pathName, std::ios::binary | std::ios::out);
	if (writeFile.is_open()) {
		if (grid)
			grid->ExportBinaryData(writeFile, header);

		writeFile.close();
	}

	return PDB_OK;
}

bool Amplitude::GetUseGrid() const {
	return bUseGrid;
}

void Amplitude::SetUseGrid(bool bUse) {
	bUseGrid = bUse;
}

bool Amplitude::GridIsReadyToBeUsed() {
	if (!bUseGrid) {
		return false;	// I think...
	}
	std::cout << "GridIsReadyToBeUsed\n";
	if (gridStatus == AMP_CACHED) {
		std::cout << "AMP_CACHED\n";
		if (PDB_OK != ReadAmplitudeFromCache())
		{
			gridStatus = AMP_OUT_OF_DATE;
			return false;
		}
	}
	if (gridStatus == AMP_READY) {
		std::cout << "AMP_READY\n";
		return true;
	}
	std::cout << "Amp: " << gridStatus << "\n";
	return false;

}

void Amplitude::PreCalculate(VectorXd& p, int nLayers) {
	bool changed = false;
	previousParameters = AmplitudeCache::previousParameters(this);
	if (previousParameters.size() == p.size()) {
		for (int i = 0; i < p.size(); i++) {
			if (p(i) == 0.0 && p(i) != previousParameters(i)) {
				changed = true;
				break;
			}
			if (!isequal(previousParameters[i], p[i], 10)) {
				changed = true;
				break;
			}
		}
		if (previousParameters.size() > 0)
			if (AmplitudeCache::amp_is_cached(this))
				//previous Params is larger than zero and has a stored amplitude
				gridStatus = AMP_CACHED;
	}
	else {
		changed = true;
	}

	if (changed) {
		this->status = UNINITIALIZED;
		previousParameters = p;
		if (AmplitudeCache::amp_is_cached(this)) {
			if (fs::exists(rndPath)) {
				fs::remove(rndPath);
				if (fs::exists(rndPath))
					std::wcout << L"Error removing file \"" << rndPath << L"\".\n";
				std::cout << "amp files should be deleted\n";
			}
			gridStatus = AMP_UNINITIALIZED;
		}
	}
	else
		this->status = PDB_OK;

	if (status != PDB_OK) {
		if (gridStatus == AMP_CACHED) {
			// Re-allocate the grid
			grid->RunBeforeReadingCache();
		}
		gridStatus = AMP_OUT_OF_DATE;
	}

	RotMat = EulerD<FACC>(ra, rb, rg);
}

void Amplitude::ResetGrid() {
	gridStatus = AMP_OUT_OF_DATE;
	ampWasReset = true;

}

void Amplitude::SetLocationData(FACC x, FACC y, FACC z, Radian alpha, Radian beta, Radian gamma) {
	this->tx = x;
	this->ty = y;
	this->tz = z;

	this->ra = alpha;
	this->rb = beta;
	this->rg = gamma;
}

int Amplitude::GetGridSize() const {
	if (grid)
		return grid->GetSize();
	else
		return 0;
}

double Amplitude::GetGridStepSize() const {
	if (grid)
		return grid->GetStepSize();
	else
		return 0.0;
}

double * Amplitude::GetDataPointer() {
	return grid->GetDataPointer();
}

u64 Amplitude::GetDataPointerSize() {
	return grid->GetRealSize();
}

bool Amplitude::InitializeGrid(double qMax, int sections) {
	if (!grid || (fabs(grid->GetQMax() - qMax) > 1e-6 || grid->GetSize() != sections) || grid->GetDataPointer() == NULL) {
		if (grid)
			delete grid;
		try {
			grid = new CurGrid(sections, qMax);
		}
		catch (...) {
			grid = NULL;
			throw backend_exception(ERROR_INSUFFICIENT_MEMORY, std::string("Insufficient memory allocating a grid of size: " + std::to_string(sections) + ".").c_str());
			return false;
		}
	}

	return true;
}

#define FOR_NO_CACHE false

bool Amplitude::ampiscached()
{
	if (ampWasReset)
		return false; // when there is amp that doesn't fit to the new running params a new cash should be saved
	bool res = AmplitudeCache::amp_is_cached(this);
	return res;
}

PDB_READER_ERRS Amplitude::WriteAmplitudeToCache() {
	clock_t rBeg = clock();


	if (FOR_NO_CACHE) {
		grid->RunAfterCache();
		gridStatus = AMP_UNINITIALIZED;
		return NO_FILE;
	}

	rndPath = AmplitudeCache::getFilepath(this);
	std::string s(rndPath.begin(), rndPath.end());
	PDB_READER_ERRS err = WriteAmplitudeToFile(s);
	if (err!=PDB_OK)
		return err;
	AmplitudeCache::ampAddedtoCache(this, previousParameters);

	grid->RunAfterCache();
	ampWasReset = false;
	gridStatus = AMP_CACHED;
	std::cout << "Took " << double(clock() - rBeg) / CLOCKS_PER_SEC << " seconds to write cache file." << std::endl;
	return PDB_OK;

}

PDB_READER_ERRS Amplitude::ReadAmplitudeFromCache() {
	if (!fs::exists(rndPath)) {
		if (!fs::exists(rndPath = AmplitudeCache::getFilepath(this)))
		{
			std::cout << "\nError! Cache file " << fs::absolute(rndPath) << " does not exist.\n";
			return NO_FILE;
		}
	}
	if (FOR_NO_CACHE) {
		return NO_FILE;
	}
	std::cout << "\nUsing cache file " << fs::absolute(rndPath) << "\n";
	clock_t rBeg = clock();
	rndPath = AmplitudeCache::getFilepath(this);
	std::string s(rndPath.begin(), rndPath.end());

	ZipArchive::Ptr archive = ZipFile::Open(s);
	ZipArchiveEntry::Ptr dataentry = archive->GetEntry("grid.dat");
	ZipArchiveEntry::Ptr paramentry = archive->GetEntry("criticalinfo.json");
	if (paramentry == nullptr || dataentry == nullptr)
		throw backend_exception(ERROR_GENERAL, "Amp file is not valid");

	rapidjson::Document doc;
	std::istream * file = paramentry->GetRawStream();
	std::string ret;
	char buffer[4096];
	while (file->read(buffer, sizeof(buffer)))
		ret.append(buffer, sizeof(buffer));
	ret.append(buffer, file->gcount());

	doc.Parse(ret.c_str());
	if (doc.HasParseError())
		throw backend_exception(ERROR_GENERAL, "Amp file is not valid");

	const rapidjson::Value &v = doc.FindMember("qmax")->value;
	double qmax = v.GetDouble();

	const rapidjson::Value &v2 = doc.FindMember("gridSize")->value;
	unsigned short gridSize = v2.GetInt();





	if (grid)
	{
		grid->RunBeforeReadingCache();
		if (qmax != grid->GetQMax() || gridSize!= grid->GetSize())
			throw backend_exception(ERROR_GENERAL, "Amp file is not valid- parameter mismatch");

		try
		{
			std::istream * file = dataentry->GetRawStream();
			grid->ImportBinaryData(file);
		}
		catch (...)
		{
			std::cout << "\nError reading file from cache\n";
			return FILE_ERROR;
		}
	}

	else
	{
		std::istream * filet = dataentry->GetRawStream();
		grid= new JacobianSphereGrid(filet, gridSize, qmax);
	}

	grid->RunAfterReadingCache();
	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;
	std::cout << "Took " << double(clock() - rBeg) / CLOCKS_PER_SEC << " seconds to read cache file." << std::endl;
	return PDB_OK;

}

Grid *Amplitude::GetInternalGridPointer() {
	return grid;
}

void Amplitude::SetUseGPU(bool bUse) {
	bUseGPU = bUse;
}

bool Amplitude::GetUseGPU() const {
	return bUseGPU;
}

void Amplitude::GetTranslationRotationVariables(FACC& x, FACC& y, FACC& z, FACC& a, FACC& b, FACC& g) {
	x = tx;	y = ty; z = tz;
	a = ra;	b = rb;	g = rg;
}

bool Amplitude::GetUseGridAnyChildren() const {
	return bUseGrid;
}

bool Amplitude::GetUseGridWithChildren() const {
	return bUseGrid;
}

void Amplitude::getNewThetaPhiAndPhases(const std::vector<FACC>& relevantQs, FACC theta, FACC phi,
	double &newTheta, double &newPhi, ArrayXcX &phases)
{
	// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
	FACC st = sin(theta);
	FACC ct = cos(theta);
	FACC sp = sin(phi);
	FACC cp = cos(phi);
	const auto Q = relevantQs[0];

	Eigen::Vector3d Qcart(Q * st * cp, Q * st * sp, Q * ct), Qt;
	Eigen::Matrix3d rot;
	Eigen::Vector3d R(tx, ty, tz);
	Qt = (Qcart.transpose() * RotMat) / Q;

	newTheta = acos(Qt.z());
	newPhi = atan2(Qt.y(), Qt.x());

	if (newPhi < 0.0)
		newPhi += M_PI * 2.;

	phases = (
		std::complex<FACC>(0., 1.) *
		(Qt.dot(R) *
			Eigen::Map<const Eigen::ArrayXd>(relevantQs.data(), relevantQs.size()))
		).exp();
}

ArrayXcX Amplitude::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
{
	double newTheta, newPhi;
	ArrayXcX phases;
	getNewThetaPhiAndPhases(relevantQs, theta, phi, newTheta, newPhi, phases);

	ArrayXcX reses(relevantQs.size());
	reses.setZero();
	if (GetUseGridWithChildren())
	{
		JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

		if (jgrid)
		{
			// Get amplitudes from grid
			return jgrid->getAmplitudesAtPoints(relevantQs, newTheta, newPhi) * phases;
		}
	}

	return scale * getAmplitudesAtPointsWithoutGrid(newTheta, newPhi, relevantQs, phases);

}

void Amplitude::getAmplitudesAtPoints2D(vector<PolarCalculationData*> relevantQData, FACC phi)
{
	// This happens too many times to print //std::cout << " ---------- Amplitude::getAmplitudesAtPoints2D -------------" << std::endl;
	double newTheta, newPhi, currTheta;
	ArrayXcX phases;
	std::vector<FACC> relevantQs;
	int q, t;
	


	for (q = 0; q < relevantQData.size(); q++)
	{
		relevantQs.push_back(relevantQData[q]->q);
	}

	for (q = 0; q < relevantQData.size(); q++)
	{
		for (t = 0; t < relevantQData[q]->theta.size(); t++)
		{
			currTheta = relevantQData[q]->theta[t];
			getNewThetaPhiAndPhases(relevantQs, currTheta, phi, newTheta, newPhi, phases);
			relevantQData[q]->theta[t] = newTheta;
		}
	}

	if (GetUseGridWithChildren())
	{
		JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);
		if (jgrid)
		{
			getNewThetaPhiAndPhases(relevantQs, relevantQData[0]->theta[0], phi, newTheta, newPhi, phases);
			// Get amplitudes from grid
			jgrid->getAmplitudesAtPoints2D(relevantQData, newPhi);
			for (int i = 0; i < relevantQData.size(); i++)
			{
				relevantQData[i]->cIntensities = relevantQData[i]->cIntensities * phases[i];
			}
		}
	}
	else
	{
		// just to get newPhi and phases. Theta doesn't matter.
		getNewThetaPhiAndPhases(relevantQs, relevantQData[0]->theta[0], phi, newTheta, newPhi, phases);
		
		// pass 'scale' as an argument instead of multiplying the result vector by scale, as done in 1D
		getAmplitudesAtPointsWithoutGrid2D(relevantQData, newPhi, phases, scale);
	}

	

}

ArrayXcX Amplitude::getAmplitudesAtPointsWithoutGrid(double newTheta, double newPhi, const std::vector<FACC> &relevantQs, Eigen::Ref<ArrayXcX> phases)
{
	ArrayXcX reses(relevantQs.size());

	FACC st = sin(newTheta);
	FACC ct = cos(newTheta);
	FACC sp = sin(newPhi);
	FACC cp = cos(newPhi);

	for (size_t q = 0; q < relevantQs.size(); q++)
		reses(q) = calcAmplitude(relevantQs[q] * st*cp, relevantQs[q] * st * sp, relevantQs[q] * ct);

	return reses * phases;
}

void Amplitude::getAmplitudesAtPointsWithoutGrid2D(std::vector<PolarCalculationData*> qData, double newPhi, Eigen::Ref<ArrayXcX> phases, double scale)
{
	std::cout << " ::::::::::::::: getAmplitudesAtPointsWithoutGrid2D ::::::::::::: " << std::endl;
	FACC sp = sin(newPhi);
	FACC cp = cos(newPhi);
	FACC st, ct, currQ, currTheta;
	
	for (size_t q = 0; q < qData.size(); q++)
	{
		currQ = qData[q]->q;
		for (size_t t = 0; t < qData[q]->theta.size(); t++)
		{
			currTheta = qData[q]->theta[t];
			st = sin(currTheta);
			ct = cos(currTheta);
			qData[q]->cIntensities[t] = calcAmplitude(currQ * st * cp, currQ * st * sp, currQ * ct) * phases[t] * scale;
		}
	}
}

bool Amplitude::GetHasAnomalousScattering()
{
	return false;
}

Eigen::VectorXd Amplitude::GetPreviousParameters()
{
	return previousParameters;
}

#pragma endregion

AmpGridAmplitude::AmpGridAmplitude(string filename) {
	clock_t rBeg = clock();
	std::string str = "AMP: ";
	str += filename;
	hash = md5(str.c_str(), str.length());

	PDB_READER_ERRS err = ReadAmplitudeFromFile(filename);
	if (err != PDB_OK)
		throw backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]);

	std::cout << "Took " << double(clock() - rBeg) / CLOCKS_PER_SEC << " seconds to read amp file." << std::endl;

	JacobianSphereGrid* casted = dynamic_cast<JacobianSphereGrid*>(grid);
	if (casted)
	{
		originalGrid = new JacobianSphereGrid(*casted);
		(dynamic_cast<JacobianSphereGrid*>(originalGrid))->CalculateSplines();
		grid->Scale(scale);
	}
}

AmpGridAmplitude::AmpGridAmplitude(const char *buffer, size_t buffSize) {
	std::string str = "AMPBuff: ";
	MD5_CTX ctx;
	unsigned char dres[16] = { 0 };
	MD5_Init(&ctx);
	MD5_Update(&ctx, str.c_str(), str.length());
	MD5_Update(&ctx, buffer, buffSize);
	MD5_Final(dres, &ctx);
	hash = bin2hexstr(dres, 16);

	ReadAmplitudeFromBuffer(buffer, buffSize);

	JacobianSphereGrid* casted = dynamic_cast<JacobianSphereGrid*>(grid);
	if (casted)
	{
		originalGrid = new JacobianSphereGrid(*casted);
		(dynamic_cast<JacobianSphereGrid*>(originalGrid))->CalculateSplines();
		grid->Scale(scale);
	}
}

PDB_READER_ERRS AmpGridAmplitude::ReadAmplitudeFromFile(std::string fileName) {
	fs::path pathName(fileName);
	pathName = fs::system_complete(fileName);
	if (!fs::exists(pathName)) {
		return NO_FILE;
	}

	string extension = fs::extension(fileName);
	if (extension == ".amp")
		{
			//legacy file support
		// 	std::string a1, a2, a3;
		// 	a1 = pathName.leaf().string();
		// 	a2 = pathName.parent_path().string();

		PDB_READER_ERRS err = NO_FILE;

		fs::ifstream readFile(pathName.c_str(), std::ios::binary | std::ios::in);

		gridStatus = AMP_UNINITIALIZED;

		if (readFile.is_open()) {
			err = ReadAmplitudeFromStream(readFile);
			readFile.close();
		}

		status = err;
		return err;
	}

	return ReadAmplitudeFromAmpjFile(fileName);
}

PDB_READER_ERRS AmpGridAmplitude::ReadAmplitudeFromAmpjFile(std::string zipFileName) {
	//fs::path pathName(zipFileName);
	//pathName = fs::system_complete(zipFileName);
	ZipArchive::Ptr archive = ZipFile::Open(zipFileName);
	ZipArchiveEntry::Ptr dataentry = archive->GetEntry("grid.dat");
	if (grid) {
		// Try to use the existing grid and import the data to it
		std::istream * file = dataentry->GetRawStream();
		if (!grid->ImportBinaryData(file)) {
			delete grid;
			grid = NULL;
		}
	}



	ZipArchiveEntry::Ptr paramentry = archive->GetEntry("criticalinfo.json");
	if (dataentry == nullptr || paramentry == nullptr)
	{
		PDB_READER_ERRS err = NO_FILE;
		return err;
	}

	if (!grid)
	{
		rapidjson::Document doc;
		std::istream * file = paramentry->GetRawStream();
		std::string ret;
		char buffer[4096];
		while (file->read(buffer, sizeof(buffer)))
			ret.append(buffer, sizeof(buffer));
		ret.append(buffer, file->gcount());

		doc.Parse(ret.c_str());
		if (doc.HasParseError())
			throw backend_exception(ERROR_GENERAL, "Amp file is not valid");

		const rapidjson::Value &v = doc.FindMember("qmax")->value;
		double qmax = v.GetDouble();

		const rapidjson::Value &v2 = doc.FindMember("gridSize")->value;
		unsigned short gridSize = v2.GetInt();

		std::istream * filet = dataentry->GetRawStream();
		grid = new CurGrid(filet, gridSize, qmax);
	}
	status = PDB_OK;
	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;
	bUseGrid = true;

	return status;

}

PDB_READER_ERRS AmpGridAmplitude::ReadAmplitudeFromStream(std::istream& readFile) {
	fileHeader.resize(0);
	if (grid) {
		// Try to use the existing grid and import the data to it
		if (!grid->ImportBinaryData(readFile, fileHeader)) {
			delete grid;
			grid = NULL;
		}
	}

	if (!grid)
		grid = new CurGrid(readFile, fileHeader);

	status = PDB_OK;
	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;
	bUseGrid = true;

	return status;
}

PDB_READER_ERRS AmpGridAmplitude::ReadAmplitudeFromBuffer(const char *buffer, size_t buffSize) {
	if (!buffer || buffSize == 0) {
		gridStatus = AMP_UNINITIALIZED;

		status = NO_FILE;
		return status;
	}

	boost::iostreams::stream< boost::iostreams::basic_array_source<char> > bufstream(buffer, buffSize);

	return ReadAmplitudeFromStream(bufstream);
}

PDB_READER_ERRS AmpGridAmplitude::ReadAmplitudeHeaderFromFile(std::string fileName, std::stringstream& header) {
	fs::path pathName(fileName);
	std::string a1, a2, a3;
	pathName = fs::system_complete(fileName);

	a1 = pathName.leaf().string();
	a2 = pathName.parent_path().string();

	fs::ifstream readFile(pathName.c_str(), std::ios::binary | std::ios::in);

	if (readFile.is_open()) {
		readFile.seekg(0, std::ios::beg);
		std::string head;
		while (readFile.peek() == '#') {
			getline(readFile, head);
			header << head;
		}
	}
	else {
		readFile.close();
		return FILE_ERROR;
	}

	fileHeader = header.str();
	readFile.close();
	return PDB_OK;
}

AmpGridAmplitude::~AmpGridAmplitude() {
	if (grid) {
		delete grid;
		grid = NULL;
	}
	if (originalGrid)
	{
		delete originalGrid;
		originalGrid = nullptr;
	}
}

void AmpGridAmplitude::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String("Loaded Amplitude From File");

	writer.Key("Position");
	writer.StartArray();
	writer.Double(tx);
	writer.Double(ty);
	writer.Double(tz);
	writer.EndArray();

	writer.Key("Rotation");
	writer.StartArray();
	writer.Double(ra);
	writer.Double(rb);
	writer.Double(rg);
	writer.EndArray();

	writer.Key("Scale");
	writer.Double(scale);

	writer.Key("File Header");
	std::stringstream ss;
	std::string line;
	std::string fileheader;
	ss << fileHeader;
	while (!ss.eof()) {
		getline(ss, line);
		fileheader.append(line + "\n");
	}
	writer.String(fileheader.c_str());
}

void AmpGridAmplitude::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers, line;
	ampers.resize(depth, '#');

	if (depth == 0) {
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
	}

	header.append(ampers + " //////////////////////////////////////\n");
	header.append(ampers + "Loaded amplitude from file\n");
	std::stringstream ss;
	ss.str("");
	ss << " Position (" << tx << "," << ty << "," << tz << ")\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << " Rotation (" << ra << "," << rb << "," << rg << ")\n";
	header.append(ampers + ss.str());
	ss.str("");

	header.append(ampers + "# //////////////////////////////////////\n");

	ss << fileHeader;
	while (!ss.eof()) {
		getline(ss, line);
		header.append(ampers + line + "\n");
	}

	header.append(ampers + "Scale: " + boost::lexical_cast<std::string>(this->scale) + "\n");

}

void AmpGridAmplitude::PreCalculate(VectorXd& p, int nLayers) {
	// Amplitude::PreCalculate(p, nLayers);	// DON'T DO THIS!!! IT MIGHT CHANGE THE GRID STATUS
	// 	Amplitude::PreCalculate(p, nLayers);	// BUT I NEED TO FOR CACHING REASONS...
	// 	if(status == UNINITIALIZED)
	// 		status = PDB_OK;
	scale = p(0);

	RotMat = EulerD<FACC>(ra, rb, rg);

	//grid = new(grid)JacobianSphereGrid(*(dynamic_cast<JacobianSphereGrid*>(originalGrid)));
	if (grid)
		delete grid;
	grid = new JacobianSphereGrid(*(dynamic_cast<JacobianSphereGrid*>(originalGrid)));
	grid->Scale(scale);
	(dynamic_cast<JacobianSphereGrid*>(grid))->CalculateSplines();

}

void AmpGridAmplitude::calculateGrid(FACC qMax, int sections /*= 150*/, progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/, double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL*/) {
	if (gridStatus == AMP_CACHED)
		Amplitude::calculateGrid(qMax, sections, progFunc, progArgs, progMin, progMax, pStop);
	if (progFunc)
		progFunc(progArgs, progMax);
}

std::string AmpGridAmplitude::Hash() const
{
	return hash;
}

std::string AmpGridAmplitude::GetName() const {
	return "AMPGrid";
}

std::complex<FACC> AmpGridAmplitude::getAmplitude(FACC qx, FACC qy, FACC qz) {

	std::complex<FACC> Im(0.0, 1.0);
	Eigen::Vector3d Q(qx, qy, qz), Qt;
	Eigen::Matrix3d rot;
	Eigen::Vector3d R(tx, ty, tz);
	Qt = Q.transpose() * RotMat;

	qx = Qt.x();
	qy = Qt.y();
	qz = Qt.z();

	if (grid)
		return exp(Im * (Q.dot(R))) * originalGrid->GetCart(qx, qy, qz) * scale;
	else // WTF?
		throw backend_exception(13, "AmpGridAmplitude::getAmplitude illegal area of code");
}

bool AmpGridAmplitude::CalculateGridGPU(GridWorkspace& workspace) {
	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuAmpGridHybridAmplitude)
		gpuAmpGridHybridAmplitude = (GPUHybridCalcAmpGrid_t)GPUHybrid_AmpGridAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_AmpGridAmplitudeDLL");
	if (!gpuAmpGridHybridAmplitude)
		return false;

	return gpuAmpGridHybridAmplitude(workspace, grid->GetDataPointer());
}

bool AmpGridAmplitude::SetModel(GridWorkspace& workspace) {
	workspace.scale = scale;

	if (!g_useGPUAndAvailable)
		return false;
	if (gridStatus == AMP_CACHED)
		Amplitude::calculateGrid(workspace.qMax, 2 * (workspace.qLayers - 3/*Extras*/));

	return true;
}

bool AmpGridAmplitude::ImplementedHybridGPU() {
	return true;
}


// SolventSpcae:


SolventSpace::ScalarType& SolventSpace::operator()(size_t x, size_t y, size_t z)
{
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);

	assert(x < _x_size);
	assert(y < _y_size);
	assert(z < _z_size);

	return _solvent_space(x * _zy_plane + y * _z_size + z);
}

void SolventSpace::allocate(size_t x, size_t y, size_t z, float voxel_length)
{
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);

	// Align to 16 bytes
	const int numelements = 16 / sizeof(ScalarType);

	_z_size = z + (numelements - 1 - (z + numelements - 1) % numelements);
	_y_size = y;
	_x_size = x;

	_zy_plane = _z_size * _y_size;

	_solvent_space = array_t::Zero(_x_size, _z_size * _y_size);
	_voxel_length = voxel_length;
}


Eigen::Map<SolventSpace::array_t, Eigen::AlignmentType::Aligned> SolventSpace::SliceX(size_t x)
{
	return Eigen::Map<array_t, Eigen::AlignmentType::Aligned>(_solvent_space.data() + x * _zy_plane, _z_size, _y_size);
}

Eigen::Map<SolventSpace::array_t, 0, Eigen::Stride<Eigen::Dynamic, 1>> SolventSpace::SurroundingVoxels(size_t x, size_t y, size_t z)
{
	Eigen::Map<array_t, 0, Eigen::Stride<Eigen::Dynamic, 1>> box9(
		_solvent_space.data() + x * _zy_plane + (y - 1) * _z_size + (z - 1),
		3, 3, Eigen::Stride<Eigen::Dynamic, 1>(_z_size, 1));

	return box9;
}

SolventSpace::array_t& SolventSpace::SpaceRef()
{
	return _solvent_space;
}

SolventSpace::Dimensions SolventSpace::dimensions()
{
	return Dimensions(_x_size, _y_size, _z_size);
}
void SolventSpace::deallocate()
{
	_solvent_space.resize(0, 0);
}






