#include "SampleModel.h"

// OpenGL includes
#ifdef _WIN32
#include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>
// END of OpenGL includes

int GetNumCategories() {
	return 2;
}

ModelCategory GetCategoryInformation(int catInd) {
	switch(catInd) {
		default:
		{
			ModelCategory mc = { "N/A", MT_FORMFACTOR, {-1} };
			return mc;
		}

		case 0:
		{
			ModelCategory mc = { "Sample Models", MT_FORMFACTOR, {0, -1} };
			return mc;
		}

		case 1:
		{
			ModelCategory mc = { "Waveforms", MT_FORMFACTOR, {1, 2, -1} };
			return mc;
		}
	}
}

// Returns the number of models in this container
int GetNumModels() {
	return 3;	
}

// Returns the model's display name from the index. Supposed to return "N/A"
// for indices that are out of bounds
ModelInformation GetModelInformation(int index) {
	switch(index) {
		default:
			return ModelInformation("N/A");
		case 0:
			return ModelInformation("Sample 1", 0, index, true, 2, 1, 5, 4, 1);
		case 1:
			return ModelInformation("Square Wave", 1, index, true, 5, 1, -1, 0, 0);
		case 2:
			return ModelInformation("Sine Wave", 1, index, true, 4, 1, -1, 0, 0);
	}
}

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
Geometry *GetModel(int index) {
	switch(index) {
		default:
			return NULL;
		case 0:
			return new SampleModel();
		case 1:
			return new SquareWaveModel();
		case 2:
			return new SineWaveModel();
	}
}

template<typename T>
static void *GetMIProc(InformationProcedure type) {
	if(type == IP_LAYERPARAMNAME)         return &T::GetLayerParamName;
	else if(type == IP_LAYERNAME)         return &T::GetLayerName;
	else if(type == IP_EXTRAPARAMETER)    return &T::GetExtraParameter;
	else if(type == IP_DEFAULTPARAMVALUE) return &T::GetDefaultParamValue;
	else if(type == IP_ISPARAMAPPLICABLE) return &T::IsParamApplicable;
	else if(type == IP_DISPLAYPARAMNAME)  return &T::GetDisplayParamName;
	else if(type == IP_DISPLAYPARAMVALUE) return &T::GetDisplayParamValue;

	return NULL;
}

EXPORTED void *GetModelInformationProcedure(int index, InformationProcedure type) {
	if(type <= IP_UNKNOWN || type >= IP_UNKNOWN2)
		return NULL;

	// This serves as an abstract factory of sorts
	switch(index) {
		default:
			return NULL;
		case 0:
			return GetMIProc<SampleModel>(type);
		case 1:
			return GetMIProc<SquareWaveModel>(type);
		case 2:
			return GetMIProc<SineWaveModel>(type);
	}
}


// SampleModel functions

SampleModel::SampleModel() : FFModel("Sample 1", 4, 2, 1, 5) {}

// A simple sine wave
double SampleModel::Calculate(double q, int nLayers, VectorXd& p) {
	return sin(q * 10.0);
}

std::complex<double> SampleModel::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

// Sample extra parameters
ExtraParam SampleModel::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Infinity", 0.0, true);
	case 1:
		return ExtraParam("Ranged", 3.0, false, false, true, 1.0, 5.0, false);
	case 2:
		return ExtraParam("AbsIntegral", 5, true, true, false, 0.0, 0.0, true);
	case 3:
		return ExtraParam("Precise", 1.0, false, false, false, 0.0, 0.0, false, 8);
	}
}

int SampleModel::GetNumDisplayParams() {
	return 1;
}

std::string SampleModel::GetDisplayParamName(int index) {
	if(index == 0)
		return "Solvent Mult";
	return "N/A";
}

double SampleModel::GetDisplayParamValue(int index, const paramStruct *p) {
	if(index == 0) {
		// Returns solvent radius * ED
		return p->params[0][0].value * p->params[1][0].value;
	}
	return 0.0;
}

void DrawPreviewScene() {
	int numc = 32, numt = 32;
	int i, j, k;
	double s, t, x, y, z, twopi;

	double scale = 0.4;

	glEnable(GL_LIGHTING);

	GLfloat LightAmbient[]  = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat LightDiffuse[]  = { 0.9f, 0.9f, 0.9f, 1.0f };
	GLfloat LightPosition[] = { 0.0f, 0.0f, 2.0f, 1.0f };
	glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
	glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);	// Position The Light
	glEnable(GL_LIGHT1);								// Enable Light One

	
	GLfloat mat[] = { 0.7f, 0.5f, 0.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
		mat);

	twopi = 2 * 3.1415926;
	for (i = 0; i < numc; i++) {
		glBegin(GL_QUAD_STRIP);
		for (j = 0; j <= numt; j++) {
			for (k = 1; k >= 0; k--) {
				s = (i + k) % numc + 0.5;
				t = j % numt;

				x = (1+scale*cos(s*twopi/numc))*cos(t*twopi/numt);
				y = (1+scale*cos(s*twopi/numc))*sin(t*twopi/numt);
				z = scale * sin(s * twopi / numc);
				glVertex3d(x, y, z);
			}
		}
		glEnd();
	}

	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHTING);
}

void DrawOpenGLPreview(const paramStruct& p, const EDProfile& profile) {
	DrawPreviewScene();
}

// Returns the number of models in this container
int GetNumModelRenderers() {
	return 1;
}

// Returns the model's preview rendering procedure (i.e., the one shown in the 
// opening window). Returns NULL if index is incorrect.
previewRenderFunc GetModelPreview(int index) {
	if(index != 0)
		return NULL;
	return (previewRenderFunc)&DrawOpenGLPreview;
}

// Returns the model's rendering procedure (i.e., the 3D preview in the fitter 
// window). Returns NULL if index is incorrect.
renderFunc GetModelRenderer(int index) {
	if(index != 0)
		return NULL;
	return (renderFunc)&DrawPreviewScene;
}


SquareWaveModel::SquareWaveModel() : FFModel("Square Wave", 0, 5, 1, -1, EDProfile(NONE)) {}

std::complex<double> SquareWaveModel::CalculateFF(Vector3d qvec, int nLayers, double w, double precision, VectorXd& p /*= VectorXd() */) {
	return std::complex<double> (0.0, 1.0);
}

inline double Clamp(double val, double min, double max) {
	return (val < min ? 
				min : 
				(val > max ? max : val));
}

void SquareWaveModel::PreCalculate(VectorXd &p, int nLayers) {
	OrganizeParameters(p, nLayers);
}

double SquareWaveModel::Calculate(double x, int nLayers, VectorXd& p) {
	double res = 0.0;

	// One second is the length of the waveform
	double length = 1000.0;
#define PERIOD_LENGTH (2.0)	
#define GLP(ind, layer) ((*parameters)(layer, ind))

	for(int i = 0; i < nLayers; i++) {
		double offset = GLP(0, i), sVol = GLP(1, i), eVol = GLP(1, i), 
			   sPitch = GLP(2, i), ePitch = GLP(2, i), height = GLP(3, i), 
			   onepercentage = GLP(4, i);

		// Some parameters
		double interpval = 1.0 - (x / length);
		double curVol = interpval * sVol + (1.0 - interpval) * eVol;
		double curPitch = interpval * sPitch + (1.0 - interpval) * ePitch;

		/*
		// Compute the offset in the period that we are in
		double pitchedX = x * curPitch + offset;

		int period = (int)(pitchedX / PERIOD_LENGTH);
		double periodLoc = (pitchedX - (period * PERIOD_LENGTH)) / PERIOD_LENGTH;
		// ^^ This should give us a number between 0.0 and 1.0 that indicates our 
		//    position in the period

		// Next, compute the value in the period
		double val = (periodLoc <= onepercentage ? 1.0 : -1.0);		

		res += Clamp(height + curVol * val, -1.0, 1.0);
		*/
		// Compute the offset in the period that we are in
		double pitchedX = x * curPitch + offset;

		/*int period = (int)(pitchedX / PERIOD_LENGTH);
		double periodLoc = (pitchedX - (period * PERIOD_LENGTH)) / PERIOD_LENGTH;
		// ^^ This should give us a number between 0.0 and 1.0 that indicates our 
		//    position in the period

		// Next, compute the value in the period
		double val = (periodLoc <= onepercentage ? curVol : height);		

		res += Clamp(val, -1.0, 1.0);*/

#define SMOOTH 50.0
		res += curVol * (tanh(SMOOTH*(sin(pitchedX)+onepercentage*2.0))-tanh(SMOOTH*(sin(pitchedX)-onepercentage*2.0))) + height;		
	}

	return res;
	//return Clamp(res, -1.0, 1.0);
}

std::string SquareWaveModel::GetLayerParamName(int index) {
	switch(index) {
		default:
			return "N/A";
		case 0:
			return "Offset";
		case 1:
			return "Volume";
		case 2:
			return "Pitch";
		case 3:
			return "Height";
		case 4:
			return "Ratio";
	}
}

double SquareWaveModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
	switch(paramIndex) {
		default:
			return Geometry::GetDefaultParamValue(paramIndex, layer, edpfunc);

		case 0: // Offset
			return 0.0;
		case 1: // Volume
			return 0.5;
		case 2: // Pitch
			return 1.0;
		case 3: // Height
			return 0.0;
		case 4: // Ratio
			return 0.2;			
	}
}

std::string SquareWaveModel::GetLayerName(int layer) {
	std::stringstream ss;
	ss << "Wave " << (layer + 1);

	return ss.str();
}

SineWaveModel::SineWaveModel() : FFModel("Sine Wave", 0, 4, 1, -1, EDProfile(NONE)) {}

std::complex<double> SineWaveModel::CalculateFF(Vector3d qvec, int nLayers, double w, double precision, VectorXd& p /*= VectorXd() */) {
	return std::complex<double> (0.0, 1.0);
}

void SineWaveModel::PreCalculate(VectorXd &p, int nLayers) {
	OrganizeParameters(p, nLayers);
}

double SineWaveModel::Calculate(double x, int nLayers, VectorXd& p) {
	double res = 0.0;

#define GLP(ind, layer) ((*parameters)(layer, ind))

	for(int i = 0; i < nLayers; i++) {
		double offset = GLP(0, i), sVol = GLP(1, i), eVol = GLP(1, i), 
			   sPitch = GLP(2, i), ePitch = GLP(2, i), height = GLP(3, i);
		
		double curPitch = sPitch, curVol = sVol;

		res += curVol * ((sin(x * curPitch + offset) + 1.0) / 2.0) + height;
	}

	return res;
}

std::string SineWaveModel::GetLayerParamName(int index) {
	switch(index) {
		default:
			return "N/A";
		case 0:
			return "Offset";
		case 1:
			return "Volume";
		case 2:
			return "Pitch";
		case 3:
			return "Height";
	}
}

double SineWaveModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
	switch(paramIndex) {
		default:
			return Geometry::GetDefaultParamValue(paramIndex, layer, edpfunc);

		case 0: // Offset
			return 0.0;
		case 1: // Volume
			return 0.5;
		case 2: // Pitch
			return 1.0;
		case 3: // Height
			return 0.0;		
	}
}

std::string SineWaveModel::GetLayerName(int layer) {
	std::stringstream ss;
	ss << "Wave " << (layer + 1);

	return ss.str();
}