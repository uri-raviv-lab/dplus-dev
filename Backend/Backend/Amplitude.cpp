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

		electronPDBAmplitude* pdbCast = dynamic_cast<electronPDBAmplitude*>(_amps[i]);
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

void electronDebyeCalTester::SetStop(int *stop) {
	pStop = stop;
}

void electronDebyeCalTester::OrganizeParameters(const VectorXd& p, int nLayers) {
	return;	// For now (during testing), there will be no parameters
	//throw std::exception("The method or operation is not implemented.");
}

void electronDebyeCalTester::PreCalculate(VectorXd& p, int nLayers) {
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



VectorXd electronDebyeCalTester::CalculateVector(
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

VectorXd electronDebyeCalTester::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

double electronDebyeCalTester::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd( ) */) {
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
	_aff_calculator->electronGetAllAFFs(atmAmps.data(), q);

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

VectorXd electronDebyeCalTester::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

PDB_READER_ERRS electronDebyeCalTester::LoadPDBFile(string filename, int model /*= 0*/) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

void electronDebyeCalTester::GetHeader(unsigned int depth, JsonWriter &writer)
{
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

void electronDebyeCalTester::GetHeader(unsigned int depth, std::string &header) {

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

VectorXd electronDebyeCalTester::CalculateVectorGPU(
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


void electronDebyeCalTester::electronInitialize() {
	pdb = nullptr;
	_aff_calculator = nullptr;

	atmFFcoefs.resize(ELECTRON_NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
	// I'm almost certain we can delete this part of the code but I changed just in case.
#pragma region Atomic form factor coefficients - Peng
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.0573, 18.9525, 0.1195, 38.6269,	// H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653,	// He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337,	// Li
		0.00460, 0.0358, 0.0165, 0.239, 0.0435, 0.879, 0.0649, 2.64, 0.0270, 7.09, 			//Li+1
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273,	// Be
		0.00340, 0.0267, 0.0103, 0.162, 0.0233, 0.531, 0.0325, 1.48, 0.0120, 3.88, 			//Be+2
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314,	// B
		0.0893, 0.2465, 0.2563, 1.7100, 0.7570, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523,	// C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715, 48.1431,	// N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.6943, 0.6990, 12.7105, 0.2039, 32.4726,	// O
		0.205, 0.397, 0.628, 0.264, 1.17, 8.80, 1.03, 27.1, 0.290, 91.8, 					//O-1
		0.1083, 0.2057, 0.3175, 1.3439, 0.6487, 4.2788, 0.5846, 11.3932, 0.1421, 28.7881,	// F
		0.134, 0.228, 0.391, 1.47, 0.814, 4.68, 0.928, 13.2, 0.347, 36.0, 					//F-1
		0.1269, 0.2200, 0.3535, 1.3779, 0.5582, 4.0203, 0.4674, 9.4934, 0.1460, 23.1278,	// Ne
		0.2142, 0.334, 0.6853, 2.3446, 0.7692, 10.0830, 1.6589, 48.3037, 1.4482, 137.2700,	// Na
		0.0256, 0.0397, 0.0919, 0.287, 0.297, 1.18, 0.514, 3.75, 0.199, 10.8,				// Na+1
		0.2314, 0.3278, 0.6866, 2.2720, 0.9677, 10.9241, 2.1182, 32.2898, 1.1339, 101.9748,	// Mg
		0.0210, 0.0331, 0.0672, 0.222, 0.198, 0.838, 0.368, 2.48, 0.174, 6.75,				// Mg+2
		0.2390, 0.3138, 0.6573, 2.1063, 1.2011, 10.4163, 2.5586, 34.4552, 1.2312, 98.5344,	// Al
		0.0192, 0.0306, 0.0579, 0.198, 0.163, 0.713, 0.284, 2.04, 0.114, 5.25,				// Al+3
		0.2519, 0.3075, 0.6372, 2.0174, 1.3795, 9.6746, 2.5082, 29.3744, 1.0500, 80.4732,	// Si_v
		0.192, 0.359, 0.289, 1.96, 0.100, 9.34, -0.0728, 11.1, 0.00120, 13.4,				// Si+4
		0.2548, 0.2908, 0.6106, 1.8740, 1.4541, 8.5176, 2.3204, 24.3434, 0.8477, 63.2996,	// P
		0.2497, 0.2681, 0.5628, 1.6711, 1.3899, 7.0267, 2.1865, 19.5377, 0.7715, 50.3888,	// S
		0.2443, 0.2468, 0.5397, 1.5242, 1.3919, 6.1537, 2.0197, 16.6687, 0.6621, 42.3086,	// Cl
		0.265, 0.252, 0.596, 1.56, 1.60, 6.21, 2.69, 17.8, 1.23, 47.8,						// Cl-1
		0.2385, 0.2289, 0.5017, 1.3694, 1.3128, 5.2561, 1.8899, 14.0928, 0.6079, 35.5361,	//Ar
		0.4115, 0.3703, 1.4031, 3.3874, 2.2784, 13.1029, 2.6742, 68.9592, 2.2162, 194.4329,	//K
		0.199, 0.192, 0.396, 1.10, 0.928, 3.91, 1.45, 9.75, 0.450, 23.4,					//K+1
		0.4054, 0.3499, 1.880, 3.0991, 2.1602, 11.9608, 3.7532, 53.9353, 2.2063, 142.3892,	//Ca
		0.164, 0.157, 0.327, 0.894, 0.743, 3.15, 1.16, 7.67, 0.307, 17.7,					//Ca+2
		0.3787, 0.3133, 1.2181, 2.5856, 2.0594, 9.5813, 3.2618, 41.7688, 2.3870, 116.7282,	//Sc
		0.163, 0.157, 0.307, 0.899, 0.716, 3.06, 0.880, 7.05, 0.139, 16.1,					//Sc+3
		0.3825, 0.3040, 1.2598, 2.4863, 2.0008, 9.2783, 3.0617, 39.0751, 2.0694, 109.4583,	//Ti
		0.399, 0.376, 1.04, 2.74, 1.21, 8.10, -0.0797, 14.2, 0.352, 23.2,					//Ti+2
		0.364, 0.364, 0.919, 2.67, 1.35, 8.18, -0.933, 11.8, 0.589, 14.9,					//Ti+3
		0.116, 0.108, 0.256, 0.655, 0.565, 2.38, 0.772, 5.51, 0.32, 12.3,					//Ti+4
		0.3876, 0.2967, 1.2750, 2.3780, 1.9109, 8.7981, 2.8314, 35.9528, 1.8979, 101.7201,	//V
		0.317, 0.269, 0.939, 2.09, 1.49, 7.22, -1.31, 15.2, 1.47, 17.6,						//V+2
		0.341, 0.321, 0.805, 2.23, 0.942, 5.99, 0.0783, 13.4, 0.156, 16.9,					//V+3
		0.0367, 0.0330, 0.124, 0.222, 0.244, 0.824, 0.723, 2.8, 0.435, 6.70,				//V+5
		0.4046, 0.2986, 1.3696, 2.3958, 1.8941, 9.1406, 2.0800, 37.4701, 1.2196, 113.7121,	//Cr
		0.237, 0.177, 0.634, 1.35, 1.23, 4.30, 0.713, 12.2, 0.0859, 39.0,					//Cr+2
		0.393, 0.359, 1.05, 2.57, 1.62, 8.68, -1.15, 11.0, 0.407, 15.8,						//Cr+3
		0.3796, 0.2699, 1.2094, 2.0455, 1.7815, 7.4726, 2.5420, 31.0604, 1.5937, 91.5622,	//Mn
		0.0576, 0.0398, 0.210, 0.284, 0.604, 1.29, 1.32, 4.23, 0.659, 14.5,					//Mn+2
		0.116, 0.0117, 0.523, 0.876, 0.881, 3.06, 0.589, 6.44, 0.214, 14.3,					//Mn+3
		0.381, 0.354, 1.83, 2.72, -1.33, 3.47, 0.995, 5.47, 0.0618, 16.1,					//Mn+4
		0.3946, 0.2717, 1.2725, 2.0443, 1.7031, 7.6007, 2.3140, 29.9714, 1.4795, 86.2265,	//Fe
		0.307, 0.230, 0.838, 1.62, 1.11, 4.87, 0.280, 10.7, 0.277, 19.2,					//Fe+2
		0.198, 0.154, 0.384, 0.893, 0.889, 2.62, 0.709, 6.65, 0.117, 18.0,				 	//Fe+3
		0.4118, 0.2742, 1.3161, 2.0372, 1.6493, 7.7205, 2.1930, 29.9680, 1.2830, 84.9383,	//Co
		0.213, 0.148, 0.488, 0.939, 0.998, 2.78, 0.828, 7.31, 0.230, 20.7,					//Co+2
		0.331, 0.267, 0.487, 1.41, 0.729, 2.89, 0.608, 6.45, 0.131, 15.8,					//Co+3
		0.3860, 0.2478, 1.1765, 1.7660, 1.5451, 6.3107, 2.0730, 25.2204, 1.3814, 74.3146,	//Ni
		0.338, 0.237, 0.982, 1.67, 1.32, 5.73, -3.56, 11.4, 3.62, 12.1,						//Ni+2
		0.347, 0.260, 0.877, 1.71, 0.790, 4.75, 0.0538, 7.51, 0.192, 13.0,					//Ni+3
		0.4314, 0.2694, 1.3208, 1.9223, 1.5236, 7.3474, 1.4671, 28.9892, 0.8562, 90.6246,	//Cu
		0.312, 0.201, 0.812, 1.31, 1.11, 3.80, 0.794, 10.5, 0.257, 28.2,					//Cu+1
		0.224, 0.145, 0.544, 0.933, 0.970, 2.69, 0.727, 7.11, 0.1882, 19.4,					//Cu+2
		0.4288, 0.2593, 1.2646, 1.7998, 1.4472, 6.7500, 1.8294, 25.5860, 1.0934, 73.5284,	//Zn
		0.252, 0.161, 0.600, 1.01, 0.917, 2.76, 0.663, 7.08, 0.161, 19.0,					//Zn+2
		0.4818, 0.2825, 1.4032, 1.9785, 1.6561, 8.7546, 2.4605, 32.5238, 1.1054, 98.5523,	//Ga
		0.391, 0.264, 0.947, 1.65, 0.690, 4.82, 0.0709, 10.7, 0.0653, 15.2,					//Ga+3
		0.4655, 0.2647, 1.3014, 1.7926, 1.6088, 7.6071, 2.6998, 26.5541, 1.3003, 77.5238,	//Ge
		0.346, 0.232, 0.830, 1.45, 0.599, 4.08, 0.0949, 13.2, -0.0217, 29.5,				//Ge+4
		0.4517, 0.2493, 1.2229, 1.6436, 1.5852, 6.8154, 2.7958, 22.3681, 1.2638, 62.0390,	//As
		0.4477, 0.2405, 1.1678, 1.5442, 1.5843, 6.3231, 2.8087, 19.4610, 1.1956, 52.0233,	//Se
		0.4798, 0.2504, 1.1948, 1.5963, 1.8695, 6.9653, 2.6953, 19.8492, 0.8203, 50.3233,	//Br
		0.125, 0.0530, 0.563, 0.469, 1.43, 2.15, 3.25, 11.1, 3.22, 38.9,					//Br-1
		0.4546, 0.2309, 1.0993, 1.4279, 1.76966, 5.9449, 2.7068, 16.6752, 0.8672, 42.2243,	//Kr
		1.0160, 0.4853, 2.8528, 5.0925, 3.5466, 25.7851, -7.7804, 130.4515, 12.1148, 138.6775,//Rb
		0.368, 0.187, 0.884, 1.12, 1.12, 3.98, 2.26, 10.9, 0.881, 26.6,						//Rb+1
		0.6703, 0.3190, 1.4926, 2.2287, 3.3368, 10.3504, 4.4600, 52.3291, 3.1501, 151.2216,	//Sr
		0.346, 0.176, 0.804, 1.04, 0.988, 3.59, 1.89, 9.32, 0.609, 21.4,					//Sr+2
		0.6894, 0.3189, 1.5474, 2.2904, 3.2450, 10.0062, 4.2126, 44.0771, 2.9764, 125.0120,	//Y
		0.465, 0.240, 0.923, 1.43, 2.41, 6.45, -2.31, 9.97, 2.48, 12.2,						//Y+3
		0.6719, 0.3036, 1.4684, 2.1249, 3.1668, 8.9236, 3.9957, 36.8458, 2.8920, 108.2049,	//Zr
		0.34, 0.113, 0.642, 0.736, 0.747, 2.54, 1.47, 6.72, 0.377, 14.7,					//Zr+4
		0.6123, 0.2709, 1.2677, 1.7683, 3.0348, 7.2489, 3.3841, 27.9465, 2.3683, 98.5624,	//Nb
		0.377, 0.184, 0.749, 1.02, 1.29, 3.80, 1.61, 9.44, 0.481, 25.7,						//Nb+3
		0.0828, 0.0369, 0.271, 0.261, 0.654, 0.957, 1.24, 3.94, 0.829, 9.44,				//Nb+5
		0.6773, 0.2920, 1.4798, 2.0606, 3.1788, 8.1129, 3.0824, 30.5336, 1.8384, 100.0658,	//Mo
		0.401, 0.191, 0.756, 1.06, 1.38, 3.84, 1.58, 9.38, 0.497, 24.6,						//Mo+3
		0.479, 0.241, 0.846, 1.46, 15.6, 6.79, -15.2, 7.13, 1.60, 10.4,						//Mo+5
		0.203, 0.0971, 0.567, 0.647, 0.646, 2.28, 1.16, 5.61, 0.171, 12.4,					//Mo+6
		0.7082, 0.2976, 1.6392, 2.2106, 3.1993, 8.5246, 3.4327, 33.1456, 1.8711, 96.6377,	//Tc
		0.6735, 0.2773, 1.4934, 1.9716, 3.0966, 7.3249, 2.7254, 26.6891, 1.5597, 90.5581,	//Ru
		0.428, 0.191, 0.773, 1.09, 1.55, 3.82, 1.46, 9.08, 0.486, 21.7,						//Ru+3
		0.2882, 0.125, 0.653, 0.753, 1.14, 2.85, 1.53, 7.01, 0.418, 17.5,					//Ru+4
		0.6413, 0.2580, 1.3690, 1.7721, 2.9854, 6.3854, 2.6952, 23.2549, 1.5433, 58.1517,	//Rh
		0.352, 0.151, 0.723, 0.878, 1.50, 3.28, 1.63, 8.16, 0.499, 20.7,					//Rh+3
		0.397, 0.177, 0.725, 1.01, 1.51, 3.62, 1.19, 8.56, 0.251, 18.9,						//Rh+4
		0.5904, 0.2324, 1.1775, 1.5019, 2.6519, 5.1591, 2.2875, 15.5428, 0.8689, 46.8213,	//Pd
		0.935, 0.393, 3.11, 4.06, 24.6, 43.1, -43.6, 54.0, 21.2, 69.8,						//Pd+2
		0.348, 0.151, 0.640, 0.832, 1.22, 2.85, 1.45, 6.59, 0.427, 15.6,					//Pd+4
		0.6377, 0.2466, 1.3790, 1.6974, 2.8294, 5.7656, 2.3631, 20.0943, 1.4553, 76.7372,	//Ag
		0.503, 0.199, 0.940, 1.19, 2.17, 4.05, 1.99, 11.3, 0.726, 32.4,						//Ag+1
		0.431, 0.175, 0.756, 0.979, 1.72, 3.30, 1.78, 8.24, 0.526, 21.4,					//Ag+2
		0.6364, 0.2407, 1.4247, 1.6823, 2.7802, 5.6588, 2.5973, 20.7219, 1.7886, 69.1109,	//Cd
		0.425, 0.168, 0.745, 0.944, 1.73, 3.14, 1.74, 7.84, 0.487, 20.4,					//Cd+2
		0.6768, 0.2522, 1.6589, 1.8545, 2.7740, 6.2936, 3.1835, 25.1457, 2.1326, 84.5448,	//In
		0.417, 0.164, 0.755, 0.960, 1.59, 3.08, 1.36, 7.03, 0.451, 16.1,					//In+3
		0.7224, 0.2651, 1.9610, 2.0604, 2.7161, 7.3011, 3.5603, 27.5493, 1.8972, 81.3349,	//Sn
		0.797, 0.317, 2.13, 2.51, 2.15, 9.04, -1.64, 24.2, 2.72, 26.4,						//Sn+2
		0.261, 0.0957, 0.642, 0.625, 1.53, 2.51, 1.36, 6.31, 0.177, 15.9,					//Sn+4
		0.7106, 0.2562, 1.9247, 1.9646, 2.6149, 6.8852, 3.8322, 24.7648, 1.8899, 68.9168,	//Sb
		0.552, 0.212, 1.14, 1.42, 1.87, 4.21, 1.36, 12.5, 0.414, 29.0,						//Sb+3
		0.377, 0.151, 0.588, 0.812, 1.22, 2.40, 1.18, 5.27, 0.244, 11.9,					//Sb+5
		0.6947, 0.2459, 1.8690, 1.8542, 2.5356, 6.4411, 4.0013, 22.1730, 1.8955, 59.2206,	//Te
		0.7047, 0.2455, 1.9484, 1.8638, 2.5940, 6.7639, 4.1526, 21.8007, 1.5057, 56.4395,	//I
		0.901, 0.312, 2.80, 2.59, 5.61, 4.1, -8.69, 34.4, 12.6, 39.5,						//I-1
		0.6737, 0.2305, 1.7908, 1.6890, 2.4129, 5.8218, 4.5100, 18.3928, 1.7058, 47.2496,	//Xe
		1.2704, 0.4356, 3.8018, 4.2058, 5.6618, 23.4342, 0.9205, 136.7783, 4.8105, 171.7561,//Cs
		0.587, 0.200, 1.40, 1.38, 1.87, 4.12, 3.48, 13.0, 1.67, 31.8,						//Cs+1
		0.9049, 0.3066, 2.6076, 2.4363, 4.8498, 12.1821, 5.1603, 54.6135, 4.7388, 161.9978,	//Ba
		0.733, 0.258, 2.05, 1.96, 23.0, 11.8, -152, 14.4, 134, 14.9,						//Ba+2
		0.8405, 0.2791, 2.3863, 2.1410, 4.6139, 10.3400, 4.1514, 41.9148, 4.7949, 132.0204,	//La
		0.493, 0.167, 1.10, 1.11, 1.50, 3.11, 2.70, 9.61, 1.08, 21.2,						//La+3
		0.8551, 0.2805, 2.3915, 2.1200, 4.5772, 10.1808, 5.0278, 42.0633, 4.5118, 130.9893,	//Ce
		0.560, 0.190, 1.35, 1.30, 1.59, 3.93, 2.63, 10.7, 0.706, 23.8,						//Ce+3
		0.483, 0.165, 1.09, 1.10, 1.34, 3.02, 2.45, 8.85, 0.797, 18.8,						//Ce+4
		0.9096, 0.2939, 2.5313, 2.2471, 4.5266, 10.8266, 4.6376, 48.8842, 4.3690, 147.6020,	//Pr
		0.663, 0.226, 1.73, 1.61, 2.35, 6.33, 0.351, 11.0, 1.59, 16.9,						//Pr+3
		0.512, 0.177, 1.19, 1.17, 1.33, 3.28, 2.36, 8.94, 0.690, 19.3,						//Pr+4
		0.8807, 0.2802, 2.4183, 2.0836, 4.4448, 10.0357, 4.6858, 47.4506, 4.1725, 146.9976,	//Nd
		0.501, 0.162, 1.18, 1.08, 1.45, 3.06, 2.53, 8.80, 0.920, 19.6,						//Nd+3
		0.9471, 0.2977, 2.5463, 2.2276, 4.3523, 10.5762, 4.4789, 49.3619, 3.9080, 145.3580,	//Pm
		0.496, 0.156, 1.20, 1.05, 1.47, 3.07, 2.43, 8.56, 0.943, 19.2,						//Pm+3
		0.9699, 0.3003, 2.5837, 2.2447, 4.2778, 10.6487, 4.4575, 50.7994, 3.5985, 146.4176,	//Sm
		0.518, 0.163, 1.24, 1.08, 1.43, 3.11, 2.40, 8.52, 0.781, 19.1,						//Sm+3
		0.8694, 0.2653, 2.2413, 1.8590, 3.9196, 8.3998, 3.9694, 36.7397, 4.5498, 125.7089,	//Eu
		0.613, 0.190, 1.53, 1.27, 1.84, 4.18, 2.46, 10.7, 0.714, 26.2,						//Eu+2
		0.496, 0.152, 1.21, 1.01, 1.45, 2.95, 2.36, 8.18, 0.774, 18.5,						//Eu+3
		0.9673, 0.2909, 2.4702, 2.1014, 4.1148, 9.7067, 4.4972, 43.4270, 3.2099, 125.9474,	//Gd
		0.490, 0.148, 1.19, 0.974, 1.42, 2.81, 2.30, 7.78, 0.795, 17.7,						//Gd+3
		0.9325, 0.2761, 2.3673, 1.9511, 3.8791, 8.9296, 3.9674, 41.5937, 3.7996, 131.0122,	//Tb
		0.503, 0.150, 1.22, 0.982, 1.42, 2.86, 2.24, 7.77, 0.710, 17.7,						//Tb+3
		0.9505, 0.2773, 2.3705, 1.9469, 3.8218, 8.8862, 4.0471, 43.0938, 3.4451, 133.1396,	//Dy
		0.503, 0.148, 1.24, 0.970, 1.44, 2.88, 2.17, 7.73, 0.643, 17.6,						//Dy+3
		0.9248, 0.2660, 2.2428, 1.8183, 3.6182, 7.9655, 3.7910, 33.1129, 3.7912, 101.8139,	//Ho
		0.456, 0.129, 1.17, 0.869, 1.43, 2.61, 2.15, 7.24, 0.692, 16.7,						//Ho+3
		1.0373, 0.2944, 2.4824, 2.0797, 3.6558, 9.4156, 3.8925, 45.8056, 3.0056, 132.7720,	//Er
		0.522, 0.150, 1.28, 0.964, 1.46, 2.93, 2.05, 7.72, 0.508, 17.8,						//Er+3
		1.0075, 0.2816, 2.3787, 1.9486, 3.5440, 8.7162, 3.6932, 41.8420, 3.1759, 125.0320,	//Tm
		0.475, 0.132, 1.20, 0.864, 1.42, 2.60, 2.05, 7.09, 0.584, 16.6,						//Tm+3
		1.0347, 0.2855, 2.3911, 1.9679, 3.4619, 8.7619, 3.6556, 42.3304, 3.0052, 125.6499,	//Yb
		0.508, 0.136, 1.37, 0.922, 1.76, 3.12, 2.23, 8.722, 0.584, 23.7,					//Yb+2
		0.498, 0.138, 1.22, 0.881, 1.39, 2.63, 1.97, 6.99, 0.559, 16.3,						//Yb+3
		0.9927, 0.2701, 2.2436, 1.8073, 3.3554, 7.8112, 3.7813, 34.4849, 3.0994, 103.3526,	//Lu
		0.483, 0.131, 1.21, 0.845, 1.41, 2.57, 1.94, 6.88, 0.522, 16.2,						//Lu+3
		1.0295, 0.2761, 2.2911, 1.8625, 3.4110, 8.0961, 3.9497, 34.2712, 2.4925, 98.5295,	//Hf
		0.522, 0.145, 1.22, 0.896, 1.37, 2.74, 1.68, 6.91, 0.312, 16.1,						//Hf+4
		1.0190, 0.2694, 2.2291, 1.7962, 3.4097, 7.6944, 3.9252, 31.0942, 2.2679, 91.1089,	//Ta
		0.569, 0.161, 1.26, 0.972, 0.979, 2.76, 1.29, 5.40, 0.551, 10.9,					//Ta+5
		0.9853, 0.2569, 2.1167, 1.6745, 3.3570, 7.0098, 3.7981, 26.9234, 2.2798, 81.3910,	//W
		0.181, 0.0118, 0.873, 0.442, 1.18, 1.52, 1.48, 4.35, 0.56, 9.42,					//W+6
		0.9914, 0.2548, 2.0858, 1.6518, 3.4531, 6.8845, 3.8812, 26.7234, 1.8526, 81.7215,	//Re
		0.9813, 0.2487, 2.0322, 1.5973, 3.3665, 6.4737, 3.6235, 23.2817, 1.9741, 70.9254,	//Os
		0.586, 0.155, 1.31, 0.938, 1.63, 3.19, 1.71, 7.84, 0.540, 19.3,						//Os+4
		1.0194, 0.2554, 2.0645, 1.6475, 3.4425, 6.5966, 3.4914, 23.2269, 1.6976, 70.0272,	//Ir
		0.692, 0.182, 1.37, 1.04, 1.80, 3.47, 1.97, 8.51, 0.804, 21.2,						//Ir+3
		0.653, 0.174, 1.29, 0.992, 1.50, 3.14, 1.74, 7.22, 0.683, 17.2,						//Ir+4
		0.9148, 0.2263, 1.8096, 1.3813, 3.2134, 5.3243, 3.2953, 17.5987, 1.5754, 60.0171,	//Pt
		0.872, 0.223, 1.68, 1.35, 2.63, 4.99, 1.93, 13.6, 0.475, 33.0,						//Pt+2
		0.550, 0.142, 1.21, 0.833, 1.62, 2.81, 1.95, 7.21, 0.610, 17.7,						//Pt+4
		0.9674, 0.2358, 1.8916, 1.4712, 3.3993, 5.6758, 3.0524, 18.7119, 1.2607, 61.5286,	//Au
		0.811, 0.201, 1.57, 1.18, 2.63, 4.25, 2.68, 12.1, 0.998, 34.4,						//Au+1
		0.722, 0.184, 1.39, 1.06, 1.94, 3.58, 1.94, 8.56, 0.699, 20.4,						//Au+3
		1.0033, 0.2413, 1.9469, 1.5298, 3.4396, 5.8009, 3.1548, 19.4520, 1.4180, 60.5753,	//Hg
		0.796, 0.194, 1.56, 1.14, 2.72, 4.21, 2.76, 12.4, 1.18, 36.2,						//Hg+1
		0.773, 0.191, 1.49, 1.12, 2.45, 4.00, 2.23, 1.08, 0.570, 27.6,						//Hg+2
		1.0689, 0.2540, 2.1038, 1.6715, 3.6039, 6.3509, 3.4927, 23.1531, 1.8283, 78.7099,	//Tl
		0.820, 0.197, 1.57, 1.16, 2.78, 4.23, 2.82, 12.7, 1.31, 35.7,						//Tl+1
		0.836, 0.208, 1.43, 1.20, 0.394, 2.57, 2.51, 4.86, 1.50, 13.5,						//Tl+3
		1.0891, 0.2552, 2.1867, 1.7174, 3.6160, 6.5131, 3.8031, 23.9170, 1.8994, 74.7039,	//Pb
		0.755, 0.181, 1.44, 1.05, 2.48, 3.75, 2.45, 10.6, 1.03, 27.9,						//Pb+2
		0.583, 0.144, 1.14, 0.796, 1.60, 2.58, 2.06, 6.22, 0.662, 14.8,						//Pb+4
		1.1007, 0.2546, 2.2306, 1.7351, 3.5689, 6.4948, 4.1549, 23.6464, 2.0382, 70.3780,	//Bi
		0.708, 0.170, 1.35, 0.981, 2.28, 3.44, 2.18, 9.41, 0.797, 23.7,						//Bi+3
		0.654, 0.162, 1.18, 0.905, 1.25, 2.68, 1.66, 5.14, 0.778, 11.2,						//Bi+5
		1.1568, 0.2648, 2.4353, 1.8786, 3.6459, 7.1749, 4.4064, 25.1766, 1.7179, 69.2821,	//Po
		1.0909, 0.2466, 2.1976, 1.6707, 3.3831, 6.0197, 4.6700, 207657, 2.1277, 57.2663,	//At
		1.0756, 0.2402, 2.1630, 1.6169, 3.3178, 5.7644, 4.8852, 19.4568, 2.0489, 52.5009,	//Rn
		1.4282, 0.3183, 3.5081, 2.6889, 5.6767, 13.4816, 4.1964, 54.3866, 3.8946, 200.8321,	//Fr
		1.3127, 0.2887, 3.1243, 2.2897, 5.2988, 10.8276, 5.3891, 43.5389, 5.4133, 145.6109,	//Ra
		0.911, 0.204, 1.65, 1.26, 2.53, 1.03, 3.62, 12.6, 1.58, 30.0,						//Ra+2
		1.3128, 0.2861, 3.1021, 2.2509, 5.3385, 10.5287, 5.9611, 41.7796, 4.7562, 128.2973,	//Ac
		0.915, 0.205, 1.64, 1.28, 2.26, 3.92, 3.18, 11.3, 1.25, 25.1,						//Ac+3
		1.2553, 0.2701, 2.9178, 2.0636, 5.0862, 9.3051, 6.1206, 34.5977, 4.7122, 107.9200,	//Th
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 														//Th+4
		1.3218, 0.2827, 3.1444, 2.2250, 5.4371, 10.2454, 5.6444, 41.1162, 4.0107, 124.4449,	//Pa
		1.3382, 0.2838, 3.2043, 2.2452, 5.4558, 10.2519, 5.4839, 41.7251, 3.6342, 124.9023,	//U
		1.14, 0.250, 2.48, 1.84, 3.61, 7.39, 1.13, 18.0, 0.900, 22.7,						//U+3
		1.09, 0.243, 2.32, 1.75, 12.0, 7.79, -9.11, 8.31, 2.15, 16.5,						//U+4
		0.687, 0.154, 1.14, 0.861, 1.83, 2.58, 2.53, 7.70, 0.957, 15.9,						//U+6
		1.5193, 0.3213, 4.0053, 2.8206, 6.5327, 14.8878, -0.1402, 68.9103, 6.7489, 81.7257,	//Np
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+6
		1.3517, 0.2813, 3.2937, 2.2418, 5.3212, 9.9952, 4.6466, 42.7939, 3.5714, 132.1739,	//Pu
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+6
		1.2135, 0.2483, 2.7962, 1.8437, 4.7545, 7.5421, 4.5731, 29.3841, 4.4786, 112.4579,	//Am
		1.2937, 0.2638, 3.1100, 2.0341, 5.0393, 8.7101, 4.7546, 35.2992, 3.5031, 109.4972,	//Cm
		1.2915, 0.2611, 3.1023, 2.0023, 4.9309, 8.4377, 4.6009, 34.1559, 3.4661, 105.8911,	//Bk
		1.2089, 0.2421, 2.7391, 1.7487, 4.3482, 6.7262, 4.0047, 23.2153, 4.6497, 80.3108,	//Cf
		// Atomic groups:
		0.1796, 73.76, 0.8554, 5.399, 1.75, 27.15, 0.05001, 0.1116, 0.2037, 1.062,			//CH
		0.1575, 89.04, 0.8528, 4.637, 2.359, 30.92, 0.00496, -0.344, 0.1935, 0.6172,		// CH2
		0.4245, 4.092, 0.4256, 4.094, 0.2008, 74.32, 2.884, 33.65, 0.16, 0.4189,			// CH3
		0.1568, 64.9, 0.222, 1.017, 0.8391, 4.656, 1.469, 23.17, 0.05579, 0.11,				// NH
		1.991, 25.94, 0.2351, 74.54, 0.8575, 3.893, 5.336, 0.3422, -5.147, 0.3388,			// NH2
		-0.1646, 168.7, 0.2896, 147.3, 0.838, 3.546, 0.1736, 0.4059, 2.668, 29.57,			// NH3
		0.1597, 53.82, 0.2445, 0.7846, 0.8406, 4.042, 1.235, 20.92, 0.03234, -0.01414,		// OH
		-78.51, 9.013, 80.62, 9.014, 0.6401, 1.924, 2.665, 37.71, 0.2755, 0.2941,			// SH
		// Ions that had no x-ray form factors:
		0.0421, 0.0609, 0.210, 0.559, 0.852, 2.96, 1.82, 11.5, 1.17, 37.7,					// O-2
		0.132, 0.109, 0.292, 0.695, 0.703, 2.39, 0.692, 5.65, 0.0959, 14.7					//Cr+4
		;

#pragma endregion
}

F_TYPE electronDebyeCalTester::electronAtomicFF(F_TYPE q, int elem) {
	//This code is not being used in the DC
	F_TYPE res = 0.0;
	F_TYPE sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

#pragma unroll 5
	for (int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	return res;
}

bool electronDebyeCalTester::GetHasAnomalousScattering()
{
	return pdb->getHasAnomalous();
}

electronDebyeCalTester::~electronDebyeCalTester()
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
	_aff_calculator->electronGetAllAFFs((float2*)atmAmps.data(), q);
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
	electronDebyeCalTester::PreCalculate(p, nLayers);
}

std::complex<F_TYPE> AnomDebyeCalTester::anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime)
{
	return electronDebyeCalTester::electronAtomicFF(q, elem) + fPrime + std::complex<F_TYPE>(0, 1) * fPrimePrime;
}

bool electronPDBAmplitude::GetHasAnomalousScattering()
{
	return pdb.getHasAnomalous();
}