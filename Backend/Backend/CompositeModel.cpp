#define AMP_EXPORTER

#include "../backend_version.h"

#include "Amplitude.h"
#include "Model.h"
#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>


void CompositeModel::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String("Multiple-Domain Composite Model");

	writer.Key("Scale");
	writer.Double(scale);

	writer.Key("Constant");
	writer.Double(constant);

	writer.Key("SubModels");
	writer.StartArray();
	for (unsigned int i = 0; i < _additives.size(); i++) {
		DomainModel *dom = dynamic_cast<DomainModel *>(_additives[i]);
		if (dom)
			writer.StartObject();
			dom->GetHeader(depth + 1, writer);
			writer.EndObject();
	}
	writer.EndArray();
}

void CompositeModel::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers;
	ampers.resize(depth+1, '#');
	ampers.append(" ");

	std::stringstream ss;

	if(depth == 0)
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");

	header.append(ampers + "//////////////////////////////////////\n");
	ss << "Multiple-Domain Composite Model\n";
	header.append(ampers + ss.str());
	ss.str("");

	ss << "Scale: " << scale << "\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Constant: " << constant << "\n";
	header.append(ampers + ss.str());
	ss.str("");

	for(unsigned int i = 0; i < _additives.size(); i++) {
		DomainModel *dom = dynamic_cast<DomainModel *>(_additives[i]);
		if(dom)
			dom->GetHeader(depth + 1, header);
	}
}


void CompositeModel::OrganizeParameters(const VectorXd& p, int nLayers) {
	_weights.resize(_additives.size());
	_params.resize(_additives.size());
	_layers.resize(_additives.size());
	
	size_t nNodeParams = 0;
	const double *nodeparams = ParameterTree::GetNodeParamVec(p.data(), p.size(), nNodeParams, false);

	// We start from nodeparams[1] to skip nLayers
	scale = nodeparams[1];
	constant = nodeparams[2];
	
	std::string hashes;
	for (const auto& d : _additives)
	{
		auto dom = dynamic_cast<DomainModel*>(d);
		if (dom)
			hashes += dom->Hash();
	}

	//Eigen::Map<const Eigen::ArrayXd> node_param_map(nodeparams, nNodeParams);
	if (p.size() == _previous_parameters.size())
	{
		const int numberChanged = (_previous_parameters != p.array()).tail(p.size() - 2).count();
		if (
			_previous_hashes == hashes &&
				(
					(numberChanged == 1 && (_previous_scale != scale || _previous_constant != constant)) ||
					(numberChanged == 2 && (_previous_scale != scale && _previous_constant != constant))
				)
			)
		{
			return;
		}
	}
	_previous_hashes = hashes;
	_previous_intensity.resize(0);
	_previous_q_values.resize(0);
	_previous_parameters = p;
	_previous_scale = scale;
	_previous_constant = constant;

	// Convert average population sizes to weights
	double sumWeights = 0.0;
	for(unsigned int i = 3; i < 3 + _additives.size(); i++)
	{
		_weights[i - 3] = nodeparams[i];
		sumWeights += nodeparams[i];
	}
	for(unsigned int i = 0; i < _additives.size(); i++)
		_weights[i] /= sumWeights;

	// Set submodel parameters (additives)
	for(unsigned int i = 0; i < _additives.size(); i++)
	{
		size_t subParams = 0;
		const double *subpd	= ParameterTree::GetChildParamVec(p.data(), p.size(), i, subParams, 
															  (ParameterTree::GetChildNumChildren(p.data(), i) > 0));			

		if(ParameterTree::GetChildNumChildren(p.data(), i) > 0) {
			// Since we extracted the whole parameter vector, with info
			_layers[i] = int(ParameterTree::GetChildNLayers(p.data(), i) + 0.001);
		} else {
			// Compute the number of layers
			_layers[i] = int(*subpd++ + 0.001);
			subParams--;
		}

		_params[i] = VectorXd::Zero(subParams);
		for(unsigned int j = 0; j < subParams; j++) _params[i][j] = subpd[j];		
	}

	// Set submodel parameters (multipliers)
	VectorXd tmppar;
	for(unsigned int i = 0; i < _fullMults.size(); i++)
	{
		size_t subParams = 0;
		const double *subpd	= ParameterTree::GetChildParamVec(p.data(), p.size(), i, subParams, 
			(ParameterTree::GetChildNumChildren(p.data(), i) > 0));			

		if(ParameterTree::GetChildNumChildren(p.data(), i) > 0) {
			// Since we extracted the whole parameter vector, with info
			_multLayers[_fullMults[i]] = int(ParameterTree::GetChildNLayers(p.data(), i) + 0.001);
		} else {
			// Compute the number of layers
			_multLayers[_fullMults[i]] = int(*subpd++ + 0.001);
			subParams--;
		}

		tmppar = VectorXd::Zero(subParams);
		for(unsigned int j = 0; j < subParams; j++) tmppar[j] = subpd[j];
		_multParams[_fullMults[i]] = tmppar;
	}
}

double CompositeModel::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd()*/) {
	double result = 0.;

	for(unsigned int i = 0; i < _additives.size(); i++)
	{
		result += _weights[i] * _additives[i]->Calculate(q, _layers[i], _params[i]);

		for(auto mult : _multipliers[i])
			result *= mult->Calculate(q, _multLayers[mult], _multParams[mult]);
	}

	return scale * result + constant;
}

bool CompositeModel::GetHasAnomalousScattering() {
	for (const auto& child : _additives)
		if (child->GetHasAnomalousScattering()) return true;
	return false;
}

void CompositeModel::PreCalculate(VectorXd& p, int nLayers) {
	OrganizeParameters(p, nLayers);

	for(unsigned int i = 0; i < _additives.size(); i++)
	{
		// Leave this to the DomainModel
		//_additives[i]->PreCalculate(_params[i], _layers[i]);		
	}

	for(auto mult : _fullMults) 
		mult->PreCalculate(_multParams[mult], _multLayers[mult]);
}


struct FakeProgressArgs
{
	progressFunc realProgress;
	void *realProgressArgs;
	double minProgress;
	double maxProgress;
};
void STDCALL FakeProgressFunc(void *args, double progress) 
{
	if(args)
	{
		FakeProgressArgs *fpa = (FakeProgressArgs *)args;
		if(fpa->realProgress)
			fpa->realProgress(fpa->realProgressArgs, fpa->minProgress + (fpa->maxProgress - fpa->minProgress) * progress);
	}
}

VectorXd CompositeModel::CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */, progressFunc progress /*= NULL*/, void *progressArgs /*= NULL*/) {
	size_t points = q.size();
	VectorXd vec = VectorXd::Zero(points);
	VectorXd tmpvec;

	FakeProgressArgs fpa;
	fpa.realProgress = progress;
	fpa.realProgressArgs = progressArgs;
	fpa.minProgress = 0.0;
	fpa.maxProgress = 1.0;
	
	unsigned int totalModels = (unsigned int)_additives.size();

	PreCalculate(p, nLayers);

	if (_previous_intensity.size() > 1 &&
		_previous_q_values.size() == q.size() &&
		(_previous_q_values == Eigen::Map<const Eigen::ArrayXd>(q.data(), q.size())).all() // The q-values haven't changed
		)
		return scale * _previous_intensity + constant;

	for(unsigned int i = 0; i < totalModels; i++) 
	{
		fpa.minProgress = (double)i / (double)totalModels;
		fpa.maxProgress = (double)(i + 1) / (double)totalModels;
		if(progress && progressArgs)
			progress(progressArgs, fpa.minProgress);
		if(pStop && *pStop)
			break;

		tmpvec = _weights[i] * _additives[i]->CalculateVector(q, _layers[i], _params[i], FakeProgressFunc, &fpa);

		// Multiply by all multipliers
		for(auto mult : _multipliers[i]) 
		{
			tmpvec *= mult->CalculateVector(q, _multLayers[mult], _multParams[mult]);			
		}

		vec += tmpvec;
	}

	if(progress && progressArgs)
		progress(progressArgs, fpa.maxProgress);

	_previous_intensity = vec;
	_previous_q_values = Eigen::Map<const Eigen::ArrayXd>(q.data(), q.size());

	return (scale * vec).array() + constant;
}

MatrixXd CompositeModel::CalculateMatrix(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */, progressFunc progress /*= NULL*/, void* progressArgs /*= NULL*/) 
{
	size_t points = q.size();
	MatrixXd mat = MatrixXd::Zero(points, points);
	MatrixXd tmpmat;

	FakeProgressArgs fpa;
	fpa.realProgress = progress;
	fpa.realProgressArgs = progressArgs;
	fpa.minProgress = 0.0;
	fpa.maxProgress = 1.0;

	unsigned int totalModels = (unsigned int)_additives.size();

	PreCalculate(p, nLayers);
	if (_previous_intensity_2D.rows() > 1 &&
		_previous_q_values.size() == q.size() &&
		(_previous_q_values == Eigen::Map<const Eigen::ArrayXd>(q.data(), q.size())).all() // The q-values haven't changed
		)
		return scale * _previous_intensity + constant;
	for (unsigned int i = 0; i < totalModels; i++)
	{
		fpa.minProgress = (double)i / (double)totalModels;
		fpa.maxProgress = (double)(i + 1) / (double)totalModels;
		if (progress && progressArgs)
			progress(progressArgs, fpa.minProgress);
		if (pStop && *pStop)
			break;

		
		MatrixXd additive_i = _additives[i]->CalculateMatrix(q, _layers[i], _params[i], FakeProgressFunc, &fpa);
		tmpmat = _weights[i] * additive_i;

		// Multiply by all multipliers
		for (auto mult : _multipliers[i])
		{
			MatrixXd calculated = mult->CalculateMatrix(q, _multLayers[mult], _multParams[mult]);
			tmpmat *= calculated;
		}

		mat = mat + tmpmat;
		
	
	}

	if (progress && progressArgs)
		progress(progressArgs, fpa.maxProgress);

	_previous_intensity_2D = mat;
	_previous_q_values = Eigen::Map<const Eigen::ArrayXd>(q.data(), q.size());


	return (scale * mat).array() + constant;
}

VectorXd CompositeModel::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	// TODO::Fit: Figure out which indexes refer to translation and rotation, use special derivatives
	// h is so "large" because grids do not support that kind of small modifications
	return NumericalDerivative(this, x, param, nLayers, ai, 1.0e-2);
}

VectorXd CompositeModel::GPUCalculate(const std::vector<double>& q,int nLayers, VectorXd& p /*= VectorXd( ) */) {
	// Not applicable
	return VectorXd::Zero(0);
}

void CompositeModel::SetStop(int *stop) {
	pStop = stop; 
}

void CompositeModel::AddMultiplier(IModel *mult, const std::vector<unsigned int>& indices) {
	_fullMults.push_back(mult);
	for(unsigned int index : indices)
	{
		if(index >= _multipliers.size() && index < MAX_COMPOSITE_ADDITIVES)
			_multipliers.resize(index + 1);

		_multipliers[index].push_back(mult);		
	}

	// Reset layers and params
	_multLayers[mult] = 0;
	_multParams[mult] = VectorXd();
}


bool CompositeModel::CalculateVectorForCeres(const double* qs, double const* const* p, double* residual, int points ) {
	for(auto par = mutParams.begin(); par != mutParams.end(); par++)
	{
		*(par->second) = p[0][par->first];
	}
	std::vector<double> q(qs, qs + points);

	VectorXd vec = CalculateVector(q, 0, *pVecCopy);

	memcpy(residual, vec.data(), points * sizeof(double));

	return true;
}

void CompositeModel::SetInitialParamVecForCeres( VectorXd* p , std::vector<int> &mutIndices) {
	pVecCopy = p;
	mutParams.clear();
	for(int i = 0; i < mutIndices.size(); i++) {
		mutParams[i] = &((*p)[mutIndices[i]]);
	}
}

void CompositeModel::GetMutatedParamVecForCeres( VectorXd* p) {
	p->resize(mutParams.size());
	for(auto it = mutParams.begin(); it != mutParams.end(); ++it) {
		(*p)[it->first] = *(it->second);
	}
}

void CompositeModel::SetMutatedParamVecForCeres(const VectorXd& p) {
	for(auto it = mutParams.begin(); it != mutParams.end(); ++it) {
		*(it->second) = p[it->first];
	}
}