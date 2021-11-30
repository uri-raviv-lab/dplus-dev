#define AMP_EXPORTER

#include "../backend_version.h"

#include "Amplitude.h"
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

inline Eigen::ArrayXf SincArrayXf(const Eigen::ArrayXf& x)
{
	Eigen::ArrayXf res = x.sin() / x;
	for (int i = 0; i < res.size(); i++)
	{
		if (res(i) != res(i))
			res(i) = 1.f;
	}
	return res;
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
				subAmp->GetUseGridAnyChildren()
				//subAmp->GetUseGridWithChildren() // Has to use a grid and not be hybrid (for now)
				// The minimal requirement should be that all the leaves have grids
				//&
				// Has to be the JacobianSphereGrid (may also be a half grid in the future)
				//dynamic_cast<JacobianSphereGrid*>(subAmp->GetInternalGridPointer()) != nullptr;
				// This doesn't work if doing a hybrid calculation. I need to think of another test.
				;
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
void DomainModel::setPreviousValues(std::vector<T> & res, const std::vector<T> & Q)
{
	_previous_hash = Hash();
	_previous_intensity = (Eigen::Map < Eigen::Array<T, Eigen::Dynamic, 1> >(res.data(), res.size())).template cast<double>();
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

FACC DomainModel::CalculateIntensity(FACC q, FACC epsi, unsigned int seed, uint64_t iterations) {
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
			theta = thRan(rng);
			phi = phRan(rng);
#else
			// See http://mathworld.wolfram.com/SpherePointPicking.html
			u2 = ranU2(rng);
			v2 = ranU2(rng);
			phi = u2 * M_PI;
			theta = acos(v2 - 1.0);
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

bool DomainModel::AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs) {
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

		PDBAmplitude* pdbCast = dynamic_cast<PDBAmplitude*>(_amps[i]);
		if (pdbCast) {
			pdbCast->AssemblePDBFile(subLines, subLocs);

			lines.reserve(lines.size() + subLines.size());
			lines.insert(lines.end(), subLines.begin(), subLines.end());
			locs.reserve(locs.size() + subLocs.size());
			locs.insert(locs.end(), subLocs.begin(), subLocs.end());
		} // if pdbCast
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

	//exp(Im * (Q.dot(1.0 * R))) * OldGrid.getAmplitude( Qt  ) ;
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
		return std::complex<FACC>(-42.03, -1.0);
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
	if (gridStatus == AMP_CACHED) {
		if (PDB_OK != ReadAmplitudeFromCache())
		{
			gridStatus = AMP_OUT_OF_DATE;
			return false;
		}
	}
	if (gridStatus == AMP_READY) {
		return true;
	}
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

ArrayXcX Amplitude::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
{
	// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
	FACC st = sin(theta);
	FACC ct = cos(theta);
	FACC sp = sin(phi);
	FACC cp = cos(phi);
	const auto Q = relevantQs[0];

	Eigen::Vector3d Qcart(Q * st*cp, Q * st * sp, Q*ct), Qt;
	Eigen::Matrix3d rot;
	Eigen::Vector3d R(tx, ty, tz);
	Qt = (Qcart.transpose() * RotMat) / Q;

	double newTheta = acos(Qt.z());
	double newPhi = atan2(Qt.y(), Qt.x());

	if (newPhi < 0.0)
		newPhi += M_PI * 2.;

	ArrayXcX phases = (
		std::complex<FACC>(0., 1.) *
		(Qt.dot(R) *
			Eigen::Map<const Eigen::ArrayXd>(relevantQs.data(), relevantQs.size()))
		).exp();

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

	//exp(Im * (Q.dot(1.0 * R))) * OldGrid.getAmplitude( Qt  ) ;
	qx = Qt.x();
	qy = Qt.y();
	qz = Qt.z();

	if (grid)
		return exp(Im * (Q.dot(R))) * originalGrid->GetCart(qx, qy, qz) * scale;
	else // WTF?
		return std::complex<FACC>(-42.03, -1.0);
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

void DebyeCalTester::SetStop(int *stop) {
	pStop = stop;
}

void DebyeCalTester::OrganizeParameters(const VectorXd& p, int nLayers) {
	return;	// For now (during testing), there will be no parameters
	//throw std::exception("The method or operation is not implemented.");
}

void DebyeCalTester::PreCalculate(VectorXd& p, int nLayers) {
	// This "optimization" slows down the calculation by about a minute
	/*
	distances.resize(pdb->sortedX.size(), pdb->sortedX.size());
	for(int i = 0; i < pdb->sortedX.size(); i++) {
	for(int j = 0; j < i; j++) {
	distances(i,j) = sq(pdb->sortedX[i] - pdb->sortedX[j]) +
	sq(pdb->sortedY[i] - pdb->sortedY[j]) +
	sq(pdb->sortedZ[i] - pdb->sortedZ[j]);
	}
	}
	distances = distances.sqrt();
	*/
	typedef Eigen::Map<Eigen::Matrix<F_TYPE, 1, Eigen::Dynamic>> mp;
	int num = pdb->sortedX.size();
	sortedLocations.resize(Eigen::NoChange, num);
	sortedLocations.row(0) = (mp(pdb->sortedX.data(), num)).cast<float>();
	sortedLocations.row(1) = (mp(pdb->sortedY.data(), num)).cast<float>();
	sortedLocations.row(2) = (mp(pdb->sortedZ.data(), num)).cast<float>();
	if (sortedLocations.RowsAtCompileTime == 4)
		sortedLocations.row(3).setZero();

	pdb->SetRadiusType(RAD_DUMMY_ATOMS_ONLY);

	int comb = CALC_ATOMIC_FORMFACTORS;
	comb |= ((p(0) /*solED*/ != 0.0) ? CALC_DUMMY_SOLVENT : 0x00);

	//	comb |= (anomalousVals ? CALC_ANOMALOUS : 0x00);

	int numUnIons = pdb->atomsPerIon.size();

	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.size());
	Eigen::Map<Eigen::ArrayXf>(fAtmFFcoefs.data(), fAtmFFcoefs.size()) = (Eigen::Map<Eigen::ArrayXd>(pdb->sortedCoeffs.data(), fAtmFFcoefs.size())).cast<float>();

	if (_aff_calculator)
		delete _aff_calculator;
	_aff_calculator = new atomicFFCalculator(comb, num, numUnIons, fAtmFFcoefs.data(), pdb->atomsPerIon.data());

	if (comb & CALC_DUMMY_SOLVENT)
	{
		std::vector<float> rads(pdb->rad->size());
		for (int i = 0; i < pdb->rad->size(); i++)
			rads[i] = pdb->rad->at(i);

		std::vector<float> ionRads(numUnIons);
		int off = 0;
		for (int i = 0; i < numUnIons; i++) {
			ionRads[i] = rads[pdb->sortedAtmInd[off]];
			off += pdb->atomsPerIon[i];
		}

		_aff_calculator->SetSolventED(p(0), p(2), ionRads.data());
	}
}



VectorXd DebyeCalTester::CalculateVector(
	const std::vector<double>& q,
	int nLayers,
	VectorXd& p /*= VectorXd( ) */,
	progressFunc progress /*= NULL*/,
	void *progressArgs /*= NULL*/)
{
	size_t points = q.size();
	VectorXd vecCPU = VectorXd::Zero(points);
	VectorXd vecGPU = VectorXd::Zero(points);
	std::vector<double> rVec(points);

	progFunc = progress;
	progArgs = progressArgs;

	PreCalculate(p, nLayers);
	clock_t cpuBeg, gpuBeg, cpuEnd, gpuEnd;

	// Determine if we have and want to use a GPU
	{
		int devCount;
		cudaError_t t = UseGPUDevice(&devCount);
	}


	if (g_useGPUAndAvailable) {
		gpuBeg = clock();

		vecGPU = CalculateVectorGPU(q, nLayers, p, progFunc, progArgs);
		gpuEnd = clock();

		printf("\n\n Timing:\n\tGPU %f seconds",
			double(gpuEnd - gpuBeg) / CLOCKS_PER_SEC);

		return vecGPU;
	}

	const double cProgMin = 0.0, cProgMax = 1.0;
	int prog = 0;

	if (progFunc)
		progFunc(progArgs, cProgMin);
	cpuBeg = clock();

#pragma omp parallel for if(true || pdb->atomLocs.size() > 500000) schedule(dynamic, q.size() / 50)
	for (int i = 0; i < q.size(); i++) {
		if (pStop && *pStop)
			continue;

		vecCPU[i] = Calculate(q[i], nLayers, p);

#pragma omp critical
		{
			if (progFunc)
				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(q.size())) + cProgMin);
		}
	}	// for i

	cpuEnd = clock();

	printf("\n\n Timing:\n\tCPU %f seconds\n",
		double(cpuEnd - cpuBeg) / CLOCKS_PER_SEC);

	return vecCPU;
}

VectorXd DebyeCalTester::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

double DebyeCalTester::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd( ) */) {
	int pdbLen = pdb->sortedX.size();

	F_TYPE res = 0.0;
	F_TYPE fq = (F_TYPE)q;

	int atmI = -1;
	int prevIon = 255;

	// Calculate the atomic form factors for this q
	Eigen::Array<float, -1, 1> atmAmps;
	//Eigen::ArrayXf atmAmps;
	F_TYPE aff = 0.0;
	F_TYPE fq10 = q / (10.0);
	atmAmps.resize(pdbLen);
	_aff_calculator->GetAllAFFs(atmAmps.data(), q);

	if (p(1) > 0.1) // Debye-Waller
	{
		for (int i = 0; i < pdbLen; i++)
		{
			atmAmps(i) *= exp(-(pdb->sortedBFactor[i] * fq10 * fq10 / (16. * M_PI * M_PI)));
		}
	}

	// Sum the Debye contributions
	for (int i = 1; i < pdbLen; i++) {
		res += 2.0 * ((atmAmps(i) * atmAmps.head(i)).cast<F_TYPE>() *
			SincArrayXf(
			(float(fq) * (sortedLocations.leftCols(i).colwise() - sortedLocations.col(i)).colwise().norm()).array()
			).cast<F_TYPE>()
			).sum();
	}
	res += atmAmps.square().sum();

	return res;
}

VectorXd DebyeCalTester::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

PDB_READER_ERRS DebyeCalTester::LoadPDBFile(string filename, int model /*= 0*/) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

void DebyeCalTester::GetHeader(unsigned int depth, JsonWriter &writer)
{
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

void DebyeCalTester::GetHeader(unsigned int depth, std::string &header) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

// typedef int(*GPUCalculateDebye_t)(int qValues, F_TYPE *qVals, F_TYPE *outData,
// 	F_TYPE *loc, u8 *ionInd, int numAtoms, F_TYPE *coeffs, int numCoeffs, bool bSol,
// 	u8 * atmInd, F_TYPE *rad, double solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
// 
// GPUCalculateDebye_t gpuCalcDebye = NULL;

typedef int(*GPUCalculateDebyeV2_t)(int numQValues, float qMin, float qMax,
	F_TYPE *outData,
	int numAtoms, const int *atomsPerIon,
	float4 *loc, u8 *ionInd, float2 *anomalousVals,
	bool bBfactors, float *BFactors,
	float *coeffs, bool bSol,
	bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs,
	double progmin, double progmax, int *pStop);
GPUCalculateDebyeV2_t gpuCalcDebyeV2 = NULL;

//GPUCalculateDebyeV2_t gpuCalcDebyeV3MAPS = NULL;

typedef int(*GPUCalcDebyeV4MAPS_t)(
	int numQValues, float qMin, float qMax, F_TYPE *outData, int numAtoms,
	const int *atomsPerIon, float4 *atomLocations, float2* anomFactors, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
GPUCalcDebyeV4MAPS_t thisSucks = NULL;

VectorXd DebyeCalTester::CalculateVectorGPU(
	const std::vector<double>& q,
	int nLayers,
	VectorXd& p /*= VectorXd( ) */,
	progressFunc progress /*= NULL*/,
	void *progressArgs /*= NULL*/)
{

	if (g_useGPUAndAvailable) {
		if (sizeof(F_TYPE) == sizeof(double))
		{
			gpuCalcDebyeV2 = (GPUCalculateDebyeV2_t)GPUCalculateDebyeDv2;
			thisSucks = (GPUCalcDebyeV4MAPS_t)GPUCalcDebyeV4MAPSD;
		}
	}

	if (g_useGPUAndAvailable && (/*gpuCalcDebye == NULL || */gpuCalcDebyeV2 == NULL || thisSucks == NULL)) {
		printf("Error loading GPU functions\n");
		return VectorXd();
	}
	std::vector<F_TYPE> ftq(q.size());
	for (int i = 0; i < ftq.size(); i++)
		ftq[i] = (F_TYPE)(q[i]);
	// LOC
	std::vector<F_TYPE> loc(this->pdb->sortedX.size() * 3);

	int pos = 0;
	for (int i = 0; i < pdb->sortedX.size(); i++) {
		loc[pos++] = pdb->sortedX[i];
		loc[pos++] = pdb->sortedY[i];
		loc[pos++] = pdb->sortedZ[i];
	}
	Eigen::Matrix<F_TYPE, -1, 1, 0, -1, 1> res;
	res.resize(ftq.size());

	std::vector<float> rads(pdb->rad->size());
	for (int i = 0; i < pdb->rad->size(); i++) {
		rads[i] = pdb->rad->at(i);
	}

	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.size());
	Eigen::Map<Eigen::ArrayXf>(fAtmFFcoefs.data(), fAtmFFcoefs.size()) = (Eigen::Map<Eigen::ArrayXd>(pdb->sortedCoeffs.data(), fAtmFFcoefs.size())).cast<float>();

	std::vector<float2> anomfPrimesAsFloat2;

	if (pdb->haveAnomalousAtoms)
	{
		size_t sz = pdb->sortedAnomfPrimes.size();
		anomfPrimesAsFloat2.resize(sz);
		Eigen::Map<Eigen::ArrayXf>((float*)anomfPrimesAsFloat2.data(), 2 * sz) =
			(Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>((float*)pdb->sortedAnomfPrimes.data(), 2 * sz)).cast<float>();
	}
	bool bSol = (p.size() > 0 && p(0) > 0.);
	bool bDW = (p.size() > 1 && p(1) > 0.1);
	int gpuRes;
	const float c1 = (p.size() > 2 ? p(2) : 1.0f);

	switch (kernelVersion)
	{
		// 	case 1:
		// 		gpuRes = gpuCalcDebye(ftq.size(), &ftq[0], &res(0), &loc[0], (u8*)&pdb->sortedIonInd[0],
		// 			pdb->sortedIonInd.size(), &atmFFcoefs(0, 0), atmFFcoefs.size(), bSol,
		// 			NULL, NULL, 0.0,
		// 			progFunc, progArgs, 0.0, 1.0, pStop);
		// 		break;
	case 2:
		gpuRes = gpuCalcDebyeV2(ftq.size(), q.front(), q.back(), res.data(),
			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
			(float4 *)pdb->atomLocs.data(),
			(u8*)pdb->sortedCoeffIonInd.data(),
			pdb->haveAnomalousAtoms ? anomfPrimesAsFloat2.data() : NULL,
			bDW, pdb->sortedBFactor.data(),
			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
			rads.data(), (float)(p[0]), c1, progress, progressArgs, 0.0, 1.0, pStop);
		break;

	case 4:
		gpuRes = thisSucks(ftq.size(), q.front(), q.back(), res.data(),
			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
			(float4 *)pdb->atomLocs.data(), NULL,
			bDW, pdb->sortedBFactor.data(),
			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
			rads.data(), (float)(p[0]), c1, progress, progressArgs, 0.0, 1.0, pStop);
		break;
	default:

		break;
	}

	return res.cast<double>();
}


void DebyeCalTester::initialize() {
	pdb = nullptr;
	_aff_calculator = nullptr;

	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 9);
#pragma region Atomic form factor coefficients
	atmFFcoefs << 0.49300, 10.51090, 0.32290, 26.1257, 0.14020, 3.14240, 0.04080, 57.79980, 0.0030,	// H
		0.87340, 9.10370, 0.63090, 3.35680, 0.31120, 22.9276, 0.17800, 0.98210, 0.0064,	// He
		1.12820, 3.95460, 0.75080, 1.05240, 0.61750, 85.3905, 0.46530, 168.26100, 0.0377,	// Li
		0.69680, 4.62370, 0.78880, 1.95570, 0.34140, 0.63160, 0.15630, 10.09530, 0.0167,	// Li+1
		1.59190, 43.6427, 1.12780, 1.86230, 0.53910, 103.483, 0.70290, 0.54200, 0.0385,	// Be
		6.26030, 0.00270, 0.88490, 0.93130, 0.79930, 2.27580, 0.16470, 5.11460, -6.1092,	// Be+2
		2.05450, 23.2185, 1.33260, 1.02100, 1.09790, 60.3498, 0.70680, 0.14030, -0.1932,	// B
		2.31000, 20.8439, 1.02000, 10.2075, 1.58860, 0.56870, 0.86500, 51.65120, 0.2156, // Carbon
		12.2126, 0.00570, 3.13220, 9.89330, 2.01250, 28.9975, 1.16630, 0.58260, -11.5290,	// N
		3.04850, 13.2771, 2.28680, 5.70110, 1.54630, 0.32390, 0.86700, 32.90890, 0.2508,	// O
		4.19160, 12.8573, 1.63969, 4.17236, 1.52673, 47.0179, -20.307, -0.01404, 21.9412,	// O-1
		3.53920, 10.2825, 2.64120, 4.29440, 1.51700, 0.26150, 1.02430, 26.14760, 0.2776,	// F
		3.63220, 5.27756, 3.51057, 14.7353, 1.26064, 0.442258, 0.940706, 47.34370, 0.653396,	// F-1
		3.95530, 8.40420, 3.11250, 3.42620, 1.45460, 0.23060, 1.12510, 21.71840, 0.3515,	// Ne
		4.76260, 3.28500, 3.17360, 8.84220, 1.26740, 0.31360, 1.11280, 129.42400, 0.676,	// Na
		3.25650, 2.66710, 3.93620, 6.11530, 1.39980, 0.20010, 1.00320, 14.03900, 0.404,	// Na+1
		5.42040, 2.82750, 2.17350, 79.2611, 1.22690, 0.38080, 2.30730, 7.19370, 0.8584,	// Mg
		3.49880, 2.16760, 3.83780, 4.75420, 1.32840, 0.18500, 0.84970, 10.14110, 0.4853,	// Mg+2
		6.42020, 3.03870, 1.90020, 0.74260, 1.59360, 31.5472, 1.96460, 85.08860, 1.1151,	// Al
		4.17448, 1.93816, 3.38760, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786,	// Al+3
		6.29150, 2.43860, 3.03530, 32.3337, 1.98910, 0.67850, 1.54100, 81.69370, 1.1407,	// Si_v
		4.43918, 1.64167, 3.20345, 3.43757, 1.19453, 0.21490, 0.41653, 6.65365, 0.746297,	// Si+4
		6.43450, 1.90670, 4.17910, 27.1570, 1.78000, 0.52600, 1.49080, 68.16450, 1.1149,	// P
		6.90530, 1.46790, 5.20340, 22.2151, 1.43790, 0.25360, 1.58630, 56.1720, 0.86690,	// S
		11.46040, 0.01040, 7.19640, 1.16620, 6.25560, 18.5194, 1.64550, 47.77840, -9.5574,	// Cl
		18.29150, 0.00660, 7.40840, 1.17170, 6.53370, 19.5424, 2.33860, 60.44860, -16.378,	// Cl-1
		7.48450, 0.90720, 6.77230, 14.8407, 0.65390, 43.8983, 1.64420, 33.39290, 1.4445,
		8.21860, 12.79490, 7.43980, 0.77480, 1.05190, 213.187, 0.86590, 41.68410, 1.4228,
		7.95780, 12.63310, 7.49170, 0.76740, 6.35900, -0.0020, 1.19150, 31.91280, -4.9978,
		8.62660, 10.44210, 7.38730, 0.65990, 1.58990, 85.7484, 1.02110, 178.43700, 1.3751,
		15.63480, -0.00740, 7.95180, 0.60890, 8.43720, 10.3116, 0.85370, 25.99050, -14.875,
		9.18900, 9.02130, 7.36790, 0.57290, 1.64090, 136.108, 1.46800, 51.35310, 1.3329,
		13.40080, 0.29854, 8.02730, 7.96290, 1.65943, -0.28604, 1.57936, 16.06620, -6.6667,
		9.75950, 7.85080, 7.35580, 0.50000, 1.69910, 35.6338, 1.90210, 116.10500, 1.2807,
		9.11423, 7.52430, 7.62174, 0.457585, 2.27930, 19.5361, 0.087899, 61.65580, 0.897155,
		17.73440, 0.22061, 8.73816, 7.04716, 5.25691, -0.15762, 1.92134, 15.97680, -14.652,
		19.51140, 0.178847, 8.23473, 6.67018, 2.01341, -0.29263, 1.52080, 12.94640, -13.28,
		10.29710, 6.86570, 7.35110, 0.43850, 2.07030, 26.8938, 2.05710, 102.47800, 1.2199,
		10.10600, 6.88180, 7.35410, 0.44090, 2.28840, 20.3004, 0.02230, 115.12200, 1.2298,
		9.43141, 6.39535, 7.74190, 0.383349, 2.15343, 15.1908, 0.016865, 63.96900, 0.656565,
		15.68870, 0.679003, 8.14208, 5.40135, 2.03081, 9.97278, -9.57600, 0.940464, 1.7143,
		10.64060, 6.10380, 7.35370, 0.39200, 3.32400, 20.26260, 1.49220, 98.73990, 1.1832,
		9.54034, 5.66078, 7.75090, 0.344261, 3.58274, 13.30750, 0.509107, 32.42240, 0.616898,
		9.68090, 5.59463, 7.81136, 0.334393, 2.87603, 12.82880, 0.113575, 32.87610, 0.518275,
		11.28190, 5.34090, 7.35730, 0.34320, 3.01930, 17.86740, 2.24410, 83.75430, 1.0896,
		10.80610, 5.27960, 7.36200, 0.34350, 3.52680, 14.34300, 0.21840, 41.32350, 1.0874,
		9.84521, 4.91797, 7.87194, 0.294393, 3.56531, 10.81710, 0.323613, 24.12810, 0.393974,
		9.96253, 4.84850, 7.97057, 0.283303, 2.76067, 10.48520, 0.054447, 27.57300, 0.251877,
		11.76950, 4.76110, 7.35730, 0.30720, 3.52220, 15.35350, 2.30450, 76.88050, 1.0369,
		11.04240, 4.65380, 7.37400, 0.30530, 4.13460, 12.05460, 0.43990, 31.28090, 1.0097,
		11.17640, 4.61470, 7.38630, 0.30050, 3.39480, 11.67290, 0.07240, 38.55660, 0.9707,
		12.28410, 4.27910, 7.34090, 0.27840, 4.00340, 13.53590, 2.34880, 71.16920, 1.0118,
		11.22960, 4.12310, 7.38830, 0.27260, 4.73930, 10.24430, 0.71080, 25.64660, 0.9324,
		10.33800, 3.90969, 7.88173, 0.238668, 4.76795, 8.35583, 0.725591, 18.34910, 0.286667,
		12.83760, 3.87850, 7.29200, 0.25650, 4.44380, 12.17630, 2.38000, 66.34210, 1.0341,
		11.41660, 3.67660, 7.40050, 0.24490, 5.34420, 8.87300, 0.97730, 22.16260, 0.8614,
		10.78060, 3.54770, 7.75868, 0.22314, 5.22746, 7.64468, 0.847114, 16.96730, 0.386044,
		13.33800, 3.58280, 7.16760, 0.24700, 5.61580, 11.39660, 1.67350, 64.81260, 1.191,
		11.94750, 3.36690, 7.35730, 0.22740, 6.24550, 8.66250, 1.55780, 25.84870, 0.89,
		11.81680, 3.37484, 7.11181, .244078, 5.78135, 7.98760, 1.14523, 19.89700, 1.14431,
		14.07430, 3.26550, 7.03180, 0.23330, 5.16520, 10.31630, 2.41000, 58.70970, 1.3041,
		11.97190, 2.99460, 7.38620, 0.20310, 6.46680, 7.08260, 1.39400, 18.09950, 0.7807,
		15.23540, 3.06690, 6.70060, 0.24120, 4.35910, 10.78050, 2.96230, 61.41350, 1.7189,
		12.69200, 2.81262, 6.69883, 0.22789, 6.06692, 6.36441, 1.00660, 14.41220, 1.53545,
		16.08160, 2.85090, 6.37470, 0.25160, 3.70680, 11.44680, 3.68300, 54.76250, 2.1313,
		12.91720, 2.53718, 6.70003, 0.205855, 6.06791, 5.47913, 0.859041, 11.60300, 1.45572,
		16.67230, 2.63450, 6.07010, 0.26470, 3.43130, 12.94790, 4.27790, 47.79720, 2.531,
		17.00600, 2.40980, 5.81960, 0.27260, 3.97310, 15.23720, 4.35436, 43.81630, 2.8409,
		17.17890, 2.17230, 5.23580, 16.57960, 5.63770, 0.26090, 3.98510, 41.43280, 2.9557,
		17.17180, 2.20590, 6.33380, 19.33450, 5.57540, 0.28710, 3.72720, 58.15350, 3.1776,
		17.35550, 1.93840, 6.72860, 16.56230, 5.54930, 0.22610, 3.53750, 39.39720, 2.825,
		17.17840, 1.78880, 9.64350, 17.31510, 5.13990, 0.27480, 1.52920, 164.93400, 3.4873,
		17.58160, 1.71390, 7.65980, 14.79570, 5.89810, 0.16030, 2.78170, 31.20870, 2.0782,
		17.56630, 1.55640, 9.81840, 14.09880, 5.42200, 0.16640, 2.66940, 132.37600, 2.5064,
		18.08740, 1.49070, 8.13730, 12.69630, 2.56540, 24.56510, -34.19300, -0.01380, 41.4025,
		17.77600, 1.40290, 10.29460, 12.80060, 5.72629, 0.125599, 3.26588, 104.35400, 1.91213,
		17.92680, 1.35417, 9.15310, 11.21450, 1.76795, 22.65990, -33.10800, -0.01319, 40.2602,
		17.87650, 1.27618, 10.94800, 11.91600, 5.41732, 0.117622, 3.65721, 87.66270, 2.06929,
		18.16680, 1.21480, 10.05620, 10.14830, 1.01118, 21.60540, -2.64790, -0.10276, 9.41454,
		17.61420, 1.18865, 12.01440, 11.76600, 4.04183, 0.204785, 3.53346, 69.79570, 3.75591,
		19.88120, 0.019175, 18.06530, 1.13305, 11.01770, 10.16210, 1.94715, 28.33890, -12.912,
		17.91630, 1.12446, 13.34170, 0.028781, 10.79900, 9.28206, 0.337905, 25.72280, -6.3934,
		3.70250, 0.27720, 17.23560, 1.09580, 12.88760, 11.00400, 3.74290, 61.65840, 4.3875,
		21.16640, 0.014734, 18.20170, 1.03031, 11.74230, 9.53659, 2.30951, 26.63070, -14.421,
		21.01490, 0.014345, 18.09920, 1.02238, 11.46320, 8.78809, 0.740625, 23.34520, -14.316,
		17.88710, 1.03649, 11.17500, 8.48061, 6.57891, 0.058881, 0.00000, 0.00000, 0.344941,
		19.13010, 0.864132, 11.09480, 8.14487, 4.64901, 21.57070, 2.71263, 86.84720, 5.40428,
		19.26740, 0.80852, 12.91820, 8.43467, 4.86337, 24.79970, 1.56756, 94.29280, 5.37874,
		18.56380, 0.847329, 13.28850, 8.37164, 9.32602, 0.017662, 3.00964, 22.88700, -3.1892,
		18.50030, 0.844582, 13.17870, 8.12534, 4.71304, 0.036495, 2.18535, 20.85040, 1.42357,
		19.29570, 0.751536, 14.35010, 8.21758, 4.73425, 25.87490, 1.28918, 98.60620, 5.328,
		18.87850, 0.764252, 14.12590, 7.84438, 3.32515, 21.24870, -6.19890, -0.01036, 11.8678,
		18.85450, 0.760825, 13.98060, 7.62436, 2.53464, 19.33170, -5.65260, -0.01020, 11.2835,
		19.33190, 0.69866, 15.50170, 7.98939, 5.29537, 25.20520, 0.60584, 76.89860, 5.26593,
		19.17010, 0.696219, 15.20960, 7.55573, 4.32234, 22.50570, 0.00000, 0.00000, 5.2916,
		19.24930, 0.683839, 14.79000, 7.14833, 2.89289, 17.91440, -7.94920, 0.005127, 13.0174,
		19.28080, 0.64460, 16.68850, 7.47260, 4.80450, 24.66050, 1.04630, 99.81560, 5.179,
		19.18120, 0.646179, 15.97190, 7.19123, 5.27475, 21.73260, 0.357534, 66.11470, 5.21572,
		19.16430, 0.645643, 16.24560, 7.18544, 4.37090, 21.40720, 0.00000, 0.00000, 5.21404,
		19.22140, 0.59460, 17.64440, 6.90890, 4.46100, 24.70080, 1.60290, 87.48250, 5.0694,
		19.15140, 0.597922, 17.25350, 6.80639, 4.47128, 20.25210, 0.00000, 0.00000, 5.11937,
		19.16240, 0.54760, 18.55960, 6.37760, 4.29480, 25.84990, 2.03960, 92.80290, 4.9391,
		19.10450, 0.551522, 18.11080, 6.32470, 3.78897, 17.35950, 0.00000, 0.00000, 4.99635,
		19.18890, 5.83030, 19.10050, 0.50310, 4.45850, 26.89090, 2.46630, 83.95710, 4.7821,
		19.10940, 0.50360, 19.05480, 5.83780, 4.56480, 23.37520, 0.48700, 62.20610, 4.7861,
		18.93330, 5.76400, 19.71310, 0.46550, 3.41820, 14.00490, 0.01930, -0.75830, 3.9182,
		19.64180, 5.30340, 19.04550, 0.46070, 5.03710, 27.90740, 2.68270, 75.28250, 4.5909,
		18.97550, 0.467196, 18.93300, 5.22126, 5.10789, 19.59020, 0.288753, 55.51130, 4.69626,
		19.86850, 5.44853, 19.03020, 0.467973, 2.41253, 14.12590, 0.00000, 0.00000, 4.69263,
		19.96440, 4.81742, 19.01380, 0.420885, 6.14487, 28.52840, 2.52390, 70.84030, 4.352,
		20.14720, 4.34700, 18.99490, 0.23140, 7.51380, 27.76600, 2.27350, 66.87760, 4.07121,
		20.23320, 4.35790, 18.99700, 0.38150, 7.80690, 29.52590, 2.88680, 84.93040, 4.0714,
		20.29330, 3.92820, 19.02980, 0.34400, 8.97670, 26.46590, 1.99000, 64.26580, 3.7118,
		20.38920, 3.56900, 19.10620, 0.31070, 10.66200, 24.38790, 1.49530, 213.90400, 3.3352,
		20.35240, 3.55200, 19.12780, 0.30860, 10.28210, 23.71280, 0.96150, 59.45650, 3.2791,
		20.33610, 3.21600, 19.29700, 0.27560, 10.88800, 20.20730, 2.69590, 167.20200, 2.7731,
		20.18070, 3.21367, 19.11360, 0.28331, 10.90540, 20.05580, 0.77634, 51.74600, 3.02902,
		20.57800, 2.94817, 19.59900, 0.244475, 11.37270, 18.77260, 3.28719, 133.12400, 2.14678,
		20.24890, 2.92070, 19.37630, 0.250698, 11.63230, 17.82110, 0.336048, 54.94530, 2.4086,
		21.16710, 2.81219, 19.76950, 0.226836, 11.85130, 17.60830, 3.33049, 127.11300, 1.86264,
		20.80360, 2.77691, 19.55900, 0.23154, 11.93690, 16.54080, 0.612376, 43.16920, 2.09013,
		20.32350, 2.65941, 19.81860, 0.21885, 12.12330, 15.79920, 0.144583, 62.23550, 1.5918,
		22.04400, 2.77393, 19.66970, 0.222087, 12.38560, 16.76690, 2.82428, 143.64400, 2.0583,
		21.37270, 2.64520, 19.74910, 0.214299, 12.13290, 15.32300, 0.97518, 36.40650, 1.77132,
		20.94130, 2.54467, 20.05390, 0.202481, 12.46680, 14.81370, 0.296689, 45.46430, 1.24285,
		22.68450, 2.66248, 19.68470, 0.210628, 12.77400, 15.88500, 2.85137, 137.90300, 1.98486,
		21.96100, 2.52722, 19.93390, 0.199237, 12.12000, 14.17830, 1.51031, 30.87170, 1.47588,
		23.34050, 2.56270, 19.60950, 0.202088, 13.12350, 15.10090, 2.87516, 132.72100, 2.02876,
		22.55270, 2.41740, 20.11080, 0.185769, 12.06710, 13.12750, 2.07492, 27.44910, 1.19499,
		24.00420, 2.47274, 19.42580, 0.19651, 13.43960, 14.39960, 2.89604, 128.00700, 2.20963,
		23.15040, 2.31641, 20.25990, .174081, 11.92020, 12.15710, 2.71488, 24.82420, .954586,
		24.62740, 2.38790, 19.08860, 0.19420, 13.76030, 17.75460, 2.92270, 123.17400, 2.5745,
		24.00630, 2.27783, 19.95040, 0.17353, 11.80340, 11.60960, 3.87243, 26.51560, 1.36389,
		23.74970, 2.22258, 20.37450, 0.16394, 11.85090, 11.31100, 3.26503, 22.99660, 0.759344,
		25.07090, 2.25341, 19.07980, 0.181951, 13.85180, 12.93310, 3.54545, 101.39800, 2.4196,
		24.34660, 2.15530, 20.42080, 0.15552, 11.87080, 10.57820, 3.71490, 21.70290, 0.64509,
		25.89760, 2.24256, 18.21850, 0.196143, 14.31670, 12.66480, 2.95354, 115.36200, 3.58324,
		24.95590, 2.05601, 20.32710, 0.149525, 12.24710, 10.04990, 3.77300, 21.27730, 0.691967,
		26.50700, 2.18020, 17.63830, 0.202172, 14.55960, 12.18990, 2.96577, 111.87400, 4.29728,
		25.53950, 1.98040, 20.28610, 0.143384, 11.98120, 9.34972, 4.50073, 19.58100, 0.68969,
		26.90490, 2.07051, 17.29400, 0.19794, 14.55830, 11.44070, 3.63837, 92.65660, 4.56796,
		26.12960, 1.91072, 20.09940, 0.139358, 11.97880, 8.80018, 4.93676, 18.59080, 0.852795,
		27.65630, 2.07356, 16.42850, 0.223545, 14.97790, 11.36040, 2.98233, 105.70300, 5.92046,
		26.72200, 1.84659, 19.77480, 0.13729, 12.15060, 8.36225, 5.17379, 17.89740, 1.17613,
		28.18190, 2.02859, 15.88510, 0.238849, 15.15420, 10.99750, 2.98706, 102.96100, 6.75621,
		27.30830, 1.78711, 19.33200, 0.136974, 12.33390, 7.96778, 5.38348, 17.29220, 1.63929,
		28.66410, 1.98890, 15.43450, 0.257119, 15.30870, 10.66470, 2.98963, 100.41700, 7.56672,
		28.12090, 1.78503, 17.68170, 0.15997, 13.33350, 8.18304, 5.14657, 20.39000, 3.70983,
		27.89170, 1.73272, 18.76140, 0.13879, 12.60720, 7.64412, 5.47647, 16.81530, 2.26001,
		28.94760, 1.90182, 15.22080, 9.98519, 15.10000, 0.261033, 3.71601, 84.32980, 7.97628,
		28.46280, 1.68216, 18.12100, 0.142292, 12.84290, 7.33727, 5.59415, 16.35350, 2.97573,
		29.14400, 1.83262, 15.17260, 9.59990, 14.75860, 0.275116, 4.30013, 72.02900, 8.58154,
		28.81310, 1.59136, 18.46010, 0.128903, 12.72850, 6.76232, 5.59927, 14.03660, 2.39699,
		29.20240, 1.77333, 15.22930, 9.37046, 14.51350, 0.295977, 4.76492, 63.36440, 9.24354,
		29.15870, 1.50711, 18.84070, 0.116741, 12.82680, 6.31524, 5.38695, 12.42440, 1.78555,
		29.08180, 1.72029, 15.43000, 9.22590, 14.43270, 0.321703, 5.11982, 57.05600, 9.8875,
		29.49360, 1.42755, 19.37630, 0.104621, 13.05440, 5.93667, 5.06412, 11.19720, 1.01074,
		28.76210, 1.67191, 15.71890, 9.09227, 14.55640, 0.35050, 5.44174, 52.08610, 10.472,
		28.18940, 1.62903, 16.15500, 8.97948, 14.93050, 0.38266, 5.67589, 48.16470, 11.0005,
		30.41900, 1.37113, 15.26370, 6.84706, 14.74580, 0.165191, 5.06795, 18.00300, 6.49804,
		27.30490, 1.59279, 16.72960, 8.86553, 15.61150, 0.41792, 5.83377, 45.00110, 11.4722,
		30.41560, 1.34323, 15.86200, 7.10909, 13.61450, 0.204633, 5.82008, 20.32540, 8.27903,
		30.70580, 1.30923, 15.55120, 6.71983, 14.23260, 0.167252, 5.53672, 17.49110, 6.96824,
		27.00590, 1.51293, 17.76390, 8.81174, 15.71310, .424593, 5.78370, 38.61030, 11.6883,
		29.84290, 1.32927, 16.72240, 7.38979, 13.21530, 0.263297, 6.35234, 22.94260, 9.85329,
		30.96120, 1.24813, 15.98290, 6.60834, 13.73480, 0.16864, 5.92034, 16.93920, 7.39534,
		16.88190, 0.46110, 18.59130, 8.62160, 25.55820, 1.48260, 5.86000, 36.39560, 12.0658,
		28.01090, 1.35321, 17.82040, 7.73950, 14.33590, 0.356752, 6.58077, 26.40430, 11.2299,
		30.68860, 1.21990, 16.90290, 6.82872, 12.78010, 0.212867, 6.52354, 18.65900, 9.0968,
		20.68090, 0.54500, 19.04170, 8.44840, 21.65750, 1.57290, 5.96760, 38.32460, 12.6089,
		25.08530, 1.39507, 18.49730, 7.65105, 16.88830, 0.443378, 6.48216, 28.22620, 12.0205,
		29.56410, 1.21152, 18.06000, 7.05639, 12.83740, .284738, 6.89912, 20.74820, 10.6268,
		27.54460, 0.65515, 19.15840, 8.70751, 15.53800, 1.96347, 5.52593, 45.81490, 13.1746,
		21.39850, 1.47110, 20.47230, 0.517394, 18.74780, 7.43463, 6.82847, 28.84820, 12.5258,
		30.86950, 1.10080, 18.38410, 6.53852, 11.93280, 0.219074, 7.00574, 17.21140, 9.8027,
		31.06170, 0.69020, 13.06370, 2.35760, 18.44200, 8.61800, 5.96960, 47.25790, 13.4118,
		21.78860, 1.33660, 19.56820, 0.48838, 19.14060, 6.77270, 7.01107, 23.81320, 12.4734,
		32.12440, 1.00566, 18.80030, 6.10926, 12.01750, 0.147041, 6.96886, 14.71400, 8.08428,
		33.36890, 0.70400, 12.95100, 2.92380, 16.58770, 8.79370, 6.46920, 48.00930, 13.5782,
		21.80530, 1.23560, 19.50260, 6.24149, 19.10530, 0.469999, 7.10295, 20.31850, 12.4711,
		33.53640, 0.91654, 25.09460, 0.039042, 19.24970, 5.71414, 6.91555, 12.82850, -6.7994,
		34.67260, 0.700999, 15.47330, 3.55078, 13.11380, 9.55642, 7.02588, 47.00450, 13.677,
		35.31630, 0.68587, 19.02110, 3.97458, 9.49887, 11.38240, 7.42518, 45.47150, 13.7108,
		35.56310, 0.66310, 21.28160, 4.06910, 8.00370, 14.04220, 7.44330, 44.24730, 13.6905,
		35.92990, 0.646453, 23.05470, 4.17619, 12.14390, 23.10520, 2.11253, 150.64500, 13.7247,
		35.76300, 0.616341, 22.90640, 3.87135, 12.47390, 19.98870, 3.21097, 142.32500, 13.6211,
		35.21500, 0.604909, 21.67000, 3.57670, 7.91342, 12.60100, 7.65078, 29.84360, 13.5431,
		35.65970, 0.589092, 23.10320, 3.65155, 12.59770, 18.59900, 4.08655, 117.02000, 13.5266,
		35.17360, 0.579689, 22.11120, 3.41437, 8.19216, 12.91870, 7.05545, 25.94430, 13.4637,
		35.56450, 0.563359, 23.42190, 3.46204, 12.74730, 17.83090, 4.80703, 99.17220, 13.4314,
		35.10070, 0.555054, 22.44180, 3.24498, 9.78554, 13.46610, 5.29444, 23.95330, 13.376,
		35.88470, 0.547751, 23.29480, 3.41519, 14.18910, 16.92350, 4.17287, 105.25100, 13.4287,
		36.02280, 0.52930, 23.41280, 3.32530, 14.94910, 16.09270, 4.18800, 100.61300, 13.3966,
		35.57470, 0.52048, 22.52590, 3.12293, 12.21650, 12.71480, 5.37073, 26.33940, 13.3092,
		35.37150, 0.516598, 22.53260, 3.05053, 12.02910, 12.57230, 4.79840, 23.45820, 13.2671,
		34.85090, 0.507079, 22.75840, 2.89030, 14.00990, 13.17670, 1.21457, 25.20170, 13.1665,
		36.18740, 0.511929, 23.59640, 3.25396, 15.64020, 15.36220, 4.18550, 97.49080, 13.3573,
		35.70740, 0.502322, 22.61300, 3.03807, 12.98980, 12.14490, 5.43227, 25.49280, 13.2544,
		35.51030, 0.498626, 22.57870, 2.96627, 12.77660, 11.94840, 4.92159, 22.75020, 13.2116,
		35.01360, 0.48981, 22.72860, 2.81099, 14.38840, 12.33000, 1.75669, 22.65810, 13.113,
		36.52540, 0.499384, 23.80830, 3.26371, 16.77070, 14.94550, 3.47947, 105.9800, 13.3812,
		35.84000, 0.484938, 22.71690, 2.96118, 13.58070, 11.53310, 5.66016, 24.39920, 13.1991,
		35.64930, 0.481422, 22.64600, 2.89020, 13.35950, 11.31600, 5.18831, 21.83010, 13.1555,
		35.17360, 0.473204, 22.71810, 2.73848, 14.76350, 11.55300, 2.28678, 20.93030, 13.0582,
		36.67060, 0.483629, 24.09920, 3.20647, 17.34150, 14.31360, 3.49331, 102.2730, 13.3592,
		36.64880, 0.465154, 24.40960, 3.08997, 17.39900, 13.43460, 4.21665, 88.48340, 13.2887,
		36.78810, 0.451018, 24.77360, 3.04619, 17.89190, 12.89460, 4.23284, 86.00300, 13.2754,
		36.91850, 0.437533, 25.19950, 3.00775, 18.33170, 12.40440, 4.24391, 83.78810, 13.2674,
		//////////////////////////////////////////////////////////////////////////
		// Modified atomic form factors
		0.894937, 55.7145, 0.894429, 4.03158, 3.78824, 24.8323, 3.14683e-6, 956.628, 1.42149,	// CH
		1.61908, 52.1451, 2.27205, 24.6589, 2.1815, 24.6587, 0.0019254, 152.165, 1.92445,		// CH2
		12.5735, 38.7341, -0.456658, -6.28167, 5.71547, 54.955, -11.711, 47.898, 2.87762,		// CH3
		0.00506991, 108.256, 2.03147, 14.6199, 1.82122, 14.628, 2.06506, 35.4102, 2.07168,		// NH
		3.00872, 28.3717, 0.288137, 63.9637, 3.39248, 3.51866, 2.03511, 28.3675, 0.269952,		// NH2
		0.294613, 67.4408, 6.48379, 29.1576, 5.67182, 0.54735, 6.57164, 0.547493, -9.02757,		// NH3
		-2.73406, 22.1288, 0.00966263, 94.3428, 6.64439, 13.9044, 2.67949, 32.7607, 2.39981,	// OH
		-127.811, 7.19935, 62.5514, 12.1591, 160.747, 1.88979, 2.34822, 55.952, -80.836			// SH
		;
#pragma endregion

}

F_TYPE DebyeCalTester::atomicFF(F_TYPE q, int elem) {
	F_TYPE res = 0.0;
	F_TYPE sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

#pragma unroll 4
	for (int i = 0; i < 4; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	res += (atmFFcoefs(elem, 8));
	return res;
}

bool DebyeCalTester::GetHasAnomalousScattering()
{
	return pdb->getHasAnomalous();
}

DebyeCalTester::~DebyeCalTester()
{
	if (pdb) delete pdb;
	if (_aff_calculator) delete _aff_calculator;
}


double AnomDebyeCalTester::Calculate(double q, int nLayers, VectorXd& p)
{
	//////////////////////////////////////////////////////////////////////////
	// The regular atoms
	int pdbLen = pdb->sortedX.size();
	F_TYPE res = 0.0;
	F_TYPE fq = (F_TYPE)q;

	int atmI = -1;
	int prevIon = 255;

	// Calculate the atomic form factors for this q
	Eigen::ArrayXcf atmAmps;
	F_TYPE aff = 0.0;
	const F_TYPE fq10 = q / (10.0);
	atmAmps.resize(pdbLen);
	_aff_calculator->GetAllAFFs((float2*)atmAmps.data(), q);
	for (int i = 0; i < pdbLen; i++) {
		// 		if (prevIon != pdb->sortedIonInd[i]) {
		// 			aff = atomicFF(fq10, pdb->sortedIonInd[i]);
		// 			prevIon = pdb->sortedIonInd[i];
		// 			if (p[0] != 0.)	// Subtract dummy atom solvent ED
		// 			{
		// 				F_TYPE rad = pdb->rad->at(pdb->sortedAtmInd[i]);
		// #ifdef USE_FRASER
		// 				aff -= 5.5683279968317084528/*4.1887902047863909846 /*4\pi/3*/ * rad * rad * rad * exp(-(rad * rad * (q*q) / 4.)) * p(0);
		// #else
		// 				aff -= 4.1887902047863909846 * rad * rad * rad * exp(-(0.20678349696647 * sq(rad * q))) * p(0);
		// #endif
		// 			}
		// 		}

		atmAmps(i) += pdb->sortedAnomfPrimes[i];

		if (p(1) > 0.1) { // Debye-Waller
			atmAmps(i) *= exp(-(pdb->sortedBFactor[i] * fq10 * fq10 / (16. * M_PI * M_PI)));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Calculate the intensity
	double intensity = 0.;
	// Sum the Debye contributions
#pragma omp parallel for if(pdbLen >= 500000) reduction(+:intensity) schedule(dynamic, pdbLen / 500)
	for (int i = 1; i < pdbLen; i++) {
		intensity += 2.0 * (
			(atmAmps(i).real() * atmAmps.head(i).real() + atmAmps(i).imag() * atmAmps.head(i).imag()).array().cast<F_TYPE>() *
			SincArrayXf(
			(float(fq) * (sortedLocations.leftCols(i).colwise() - sortedLocations.col(i)).colwise().norm()).array()
			).cast<F_TYPE>()
			).sum();
	} // for i

	intensity += atmAmps.abs2().sum();

	return intensity;
}

VectorXd AnomDebyeCalTester::CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress /*= NULL*/, void *progressArgs /*= NULL*/)
{
	return VectorXd();
	// IF THIS COMMENT IS UNCOMMENTED, MUST REPLACE GETPROCADDRESS CALLS WITH DIRECT FUNCTION CALLS
	// 	if (!g_gpuModule) {
	// 		load_gpu_backend(g_gpuModule);
	// 		if (g_gpuModule) {
	// 			if (sizeof(F_TYPE) == sizeof(double)) {
	// 				gpuCalcAnomDebye = (GPUCalculateDebye_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeD");
	// 				gpuCalcAnomDebyeV2 = (GPUCalculateAnomDebyeV2_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeDv2");
	// 				thisSucksAnom = (GPUCalcDebyeV4MAPS_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalcDebyeV4MAPSD");
	// 			}
	// 			else if (sizeof(F_TYPE) == sizeof(float)) {
	// 				gpuCalcAnomDebye = (GPUCalculateDebye_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeF");
	// 				gpuCalcAnomDebyeV2 = (GPUCalculateAnomDebyeV2_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeFv2");
	// 				thisSucksAnom = (GPUCalcAnomDebyeV4MAPS_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalcAnomDebyeV4MAPSF");
	// 			}
	// 		} // if g_gpuModule
	// 	}
	// 
	// 	if (g_gpuModule != NULL && (gpuCalcDebye == NULL || gpuCalcDebyeV2 == NULL || thisSucks == NULL)) {
	// 		printf("Error loading GPU functions\n");
	// 		return VectorXd();
	// 	}
	// 	std::vector<F_TYPE> ftq(q.size());
	// 	for (int i = 0; i < ftq.size(); i++)
	// 		ftq[i] = (F_TYPE)(q[i]);
	// 	// LOC
	// 	std::vector<F_TYPE> loc(this->pdb->sortedX.size() * 3);
	// 
	// 	int pos = 0;
	// 	for (int i = 0; i < pdb->sortedX.size(); i++) {
	// 		loc[pos++] = pdb->sortedX[i];
	// 		loc[pos++] = pdb->sortedY[i];
	// 		loc[pos++] = pdb->sortedZ[i];
	// 	}
	// 	Eigen::Matrix<F_TYPE, -1, 1, 0, -1, 1> res;
	// 	res.resize(ftq.size());
	// 
	// 	std::vector<float> rads(pdb->rad->size());
	// 	for (int i = 0; i < pdb->rad->size(); i++) {
	// 		rads[i] = pdb->rad->at(i);
	// 	}
	// 
	// 	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.begin(), pdb->sortedCoeffs.end());
	// 
	// 	bool bSol = (p.size() > 0 && p(0) > 0.);
	// 	bool bDW = (p.size() > 1 && p(1) > 0.1);
	// 	int gpuRes;
	// 
	// 	switch (kernelVersion)
	// 	{
	// 	case 1:
	// 		if (gpuCalcDebye == NULL)
	// 		{
	// 
	// 		}
	// 		gpuRes = gpuCalcDebye(ftq.size(), &ftq[0], &res(0), &loc[0], (u8*)&pdb->sortedIonInd[0],
	// 			pdb->sortedIonInd.size(), &atmFFcoefs(0, 0), atmFFcoefs.size(), bSol,
	// 			NULL, NULL, 0.0,
	// 			progFunc, progArgs, 0.0, 1.0, pStop);
	// 		break;
	// 	case 2:
	// 		gpuRes = gpuCalcDebyeV2(ftq.size(), q.front(), q.back(), res.data(),
	// 			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
	// 			(float4 *)pdb->atomLocs.data(),
	// 			(u8*)pdb->sortedCoeffIonInd.data(), bDW, pdb->sortedBFactor.data(),
	// 			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
	// 			rads.data(), (float)(p[0]), progress, progressArgs, 0.0, 1.0, pStop);
	// 		break;
	// 
	// 	case 4:
	// 		gpuRes = thisSucks(ftq.size(), q.front(), q.back(), res.data(),
	// 			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
	// 			(float4 *)pdb->atomLocs.data(),
	// 			bDW, pdb->sortedBFactor.data(),
	// 			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
	// 			rads.data(), (float)(p[0]), progress, progressArgs, 0.0, 1.0, pStop);
	// 		break;
	// 	default:
	//
	//		break;
	// 	}
	// 
	// 	return res.cast<double>();
}

void AnomDebyeCalTester::PreCalculate(VectorXd& p, int nLayers)
{
	DebyeCalTester::PreCalculate(p, nLayers);
}

std::complex<F_TYPE> AnomDebyeCalTester::anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime)
{
	return DebyeCalTester::atomicFF(q, elem) + fPrimePrime + std::complex<F_TYPE>(0, 1) * fPrimePrime;
}

bool PDBAmplitude::GetHasAnomalousScattering()
{
	return pdb.getHasAnomalous();
}
