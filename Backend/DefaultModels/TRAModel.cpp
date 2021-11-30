#include "TRAModel.h"

#pragma region Time Resolved Analysis Model

TRAModel::TRAModel(std::string st, int extras, int nlp, int minlayers,
				   int maxlayers, EDProfile edp) :
				Geometry(st, extras, nlp, minlayers, maxlayers, edp) {
	numModels = 0;
}

double TRAModel::Calculate(double q, int nLayers, VectorXd &p) {
	double res = 0.0;
	int ind = qindex;
	if(!(fabs(Qi[ind] - q) < 1.0e-9))
		ind = FindQ(ind, q);

	for(int i = 0 ; i < numModels; i++)
		res += (*parameters)(i) * (Ii[i])(ind);

	return res * scale + BG;
}

double TRAModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
	return 1.0;
} 


std::string TRAModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
	if(index == 0)
		return "Weight";
	return Geometry::GetLayerParamName(index, edpfunc);
}

std::string TRAModel::GetLayerName(int layer) {
	if(layer < 0)
		return "N/A";

	return "Geometry %d";
}

void TRAModel::AddModel(FFModel *mod) {
	models.push_back(mod);
	numModels++;
}

void TRAModel::RemoveModel(int index) {
	if(models[index])
		delete models[index];

	models[index] = NULL;
	models.erase(models.begin() + index);
	
	Ii.erase(Ii.begin() + index);
	numModels--;
	
}

int TRAModel::GetModelIndex(FFModel *mod){
	for(int i = 0; i < numModels; i++)
		if(mod == models[i])
			return i;
	
	return -1;
}

void TRAModel::CalculateModel(int index, int mnLayers, VectorXd& p) {
	if(index < 0 || index >= numModels)
		return;

	Ii[index] = models[index]->CalculateVector(Qi, mnLayers, p);
}

VectorXd TRAModel::CalculateVector(const std::vector<double> &q, int nLayers, Eigen::VectorXd &p) {
	VectorXd res = VectorXd::Zero(q.size());
	PreCalculate(p, nLayers);

	for(int i = 1; i < numModels; i++) {
		// One of the models was not calculated over the right q-range
		if((Ii[i]).size() != (Ii[i]).size())
			return res;
	}

	for(int i = 0; i < int(q.size()); i++) {
		for(int j = 0; j < numModels; j++)
                  res[i] += (*parameters)(0,j) * (Ii[j])(i);
	}

	return res;
}

int TRAModel::FindQ(int iNindex, double q) {
	if(iNindex > 0 || iNindex < int(Qi.size()) - 1) {
		if(iNindex > 0 && fabs(Qi[iNindex - 1] - q) < 1.0e-9)
			return iNindex - 1;
		if(iNindex < int(Qi.size()) - 1 && fabs(Qi[iNindex + 1] - q) < 1.0e-9)
			return iNindex + 1;
	}

	int indMin = 0, indMax = (int)Qi.size() - 1;
	while(fabs(Qi[iNindex] - q) > 1.0e-9 && indMin != indMax) {
		if(Qi[iNindex] < q)
			indMin = iNindex;
		if(Qi[iNindex] > q)
			indMax = iNindex;
		iNindex = (indMax - indMin) / 2 + indMin;
	}

	return iNindex;
}

void TRAModel::PreCalculate(VectorXd &p, int nLayers) {
	OrganizeParameters(p, nLayers);

	qindex	= 0;
	scale	= (*extraParams)(0);
	BG		= (*extraParams)(1);
}


TRAModel::~TRAModel() {
	while(models.size() > 0)
		RemoveModel(0);
}

#pragma endregion

#pragma region Predefined Geometry

PredefinedModel::PredefinedModel(std::vector<double> qVector, 
								 std::vector<double> IVector,
								 std::string st, int extras, int nlp,
								 int minlayers, int maxlayers, 
								 EDProfile edp) : 
Q(qVector), I(IVector), FFModel(st, extras, nlp, minlayers, 
								maxlayers, edp) {
}

void PredefinedModel::PreCalculate(VectorXd &p, int nLayers) {
	qindex = 0;
}

int PredefinedModel::FindQ(int iNindex, double q) {
	if(iNindex > 0 || iNindex < int(Q.size()) - 1) {
		if(iNindex > 0 && fabs(Q[iNindex - 1] - q) < 1.0e-9)
			return iNindex - 1;
		if(iNindex < int(Q.size()) - 1 && fabs(Q[iNindex + 1] - q) < 1.0e-9)
			return iNindex + 1;
	}

	int indMin = 0, indMax = (int)Q.size() - 1;
	while(fabs(Q[iNindex] - q) > 1.0e-9 && indMin != indMax) {
		if(Q[iNindex] < q)
			indMin = iNindex;
		if(Q[iNindex] > q)
			indMax = iNindex;
		iNindex = (indMax - indMin) / 2 + indMin;
	}

	return iNindex;
}

double PredefinedModel::Calculate(double q, int nLayers, VectorXd &p) {
	double res = 0.0;
	int ind = qindex;
	if(!(fabs(Q[ind] - q) < 1.0e-9))
		ind = FindQ(ind, q);

	return I[ind] * (*extraParams)(0) + (*extraParams)(1);
}

std::complex<double> PredefinedModel::CalculateFF(Vector3d qvec, 
												  int nLayers, double w, double precision, VectorXd* p) {
	return std::complex<double> (0.0, 1.0);
}


#pragma endregion
