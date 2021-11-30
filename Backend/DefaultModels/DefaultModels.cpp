#define EXPORTER

// Specific model headers
#include "SphericalModels.h"
#include "CylindricalModels.h"
#include "HelicalModels.h"
#include "SlabModels.h"
#include "Symmetries.h"

#include "ModelContainer.h" // After SphericalModels.h to minimize redefinition warnings

// The model container (contained within the backend DLL) that
// contains the default models included with the program

enum ModelIndices {UNIFORM_HOLLOW_CYLINDER, CYLINDROID, SPHERE, CUBOID, SYMMETRIC_LAYERED_SLABS,
				  ASYMMETRIC_LAYERED_SLABS, HELIX, DELIX, GAUSSIAN_SLABS, MEMBRANE, GAUSSIAN_SPHERE,
				  GAUSSIAN_HOLLOW_CYLINDER, SMOOTH_SPHERE, MICROEMULSION, GAUSSIAN_HOLLOW_CYLINDER_W_HEX_SF,
				  UNIFORM_HOLLOW_CYLINDER_W_HEX_SF, HELIX_W_HEX_SF, GAUSSIAN_DELIX, CYLINDROID_W_VAR_ECC,
				  GAUSSIAN_SIGMA_PEAKS, LORENTZIAN_PEAKS, LORENTZIAN_SQUARED_PEAKS,
				  CAILLE_SF, ADDITIVE_BG, MIXED_PEAK_SF, GRID_SYMMETRY, MANUAL_SYMMETRY,
				  MIND_SIZE};

int GetNumCategories() {
	return 10;
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
			ModelCategory mc = { "Cylindrical Models", MT_FORMFACTOR,
								{UNIFORM_HOLLOW_CYLINDER, /*GAUSSIAN_HOLLOW_CYLINDER, 
								GAUSSIAN_HOLLOW_CYLINDER_W_HEX_SF, UNIFORM_HOLLOW_CYLINDER_W_HEX_SF,*/ -1} };
			return mc;
		}

		case 1:	
		{
			ModelCategory mc = { "Spherical Models", MT_FORMFACTOR,
								{SPHERE, /*GAUSSIAN_SPHERE, SMOOTH_SPHERE,*/ -1} };
			return mc;
		}

		case 2:	
		{
			ModelCategory mc = { "Slab Models", MT_FORMFACTOR,
								{SYMMETRIC_LAYERED_SLABS, ASYMMETRIC_LAYERED_SLABS, /*GAUSSIAN_SLABS, MEMBRANE,*/ -1} };
			return mc;
		}

		case 3:	
		{
			ModelCategory mc = { "Helical Models", MT_FORMFACTOR, 
								{HELIX, /*DELIX, HELIX_W_HEX_SF, GAUSSIAN_DELIX,*/ -1} };
			return mc;
		}

		case 4:	
		{
			ModelCategory mc = { "Cylindroids", MT_FORMFACTOR,
								{/*CYLINDROID, CYLINDROID_W_VAR_ECC,*/ -1} };
			return mc;
		}

		case 5:	
		{
			ModelCategory mc = { "Microemulsions", MT_FORMFACTOR,
								{/*MICROEMULSION,*/ -1} };
			return mc;
		}

		case 6:	
		{
			ModelCategory mc = { "Cuboids", MT_FORMFACTOR,
								{/*CUBOID,*/ -1} };
			return mc;
		}

		case 7:
		{
			ModelCategory mc = { "Structure Factors", MT_STRUCTUREFACTOR,
								{/*GAUSSIAN_SIGMA_PEAKS, LORENTZIAN_PEAKS,
								LORENTZIAN_SQUARED_PEAKS, CAILLE_SF, MIXED_PEAK_SF, */-1} };
			return mc;
		}

		case 8:
		{
			ModelCategory mc = { "Backgrounds", MT_BACKGROUND,
								{/*ADDITIVE_BG,*/ -1} };
			return mc;
		}

		case 9:
		{
			ModelCategory mc = { "Symmetries", MT_SYMMETRY,
			{ GRID_SYMMETRY, MANUAL_SYMMETRY, -1 } };
			return mc;

		}

	}
}

int GetNumModels() {
	return MIND_SIZE;
}

ModelInformation GetModelInformation(int index) {
	bool bGPU = false;

	switch(index) {
		default:		
			return ModelInformation("N/A", -1, -1, true, 0, -1, 0, 0, 0, false, false, false, false);

		case UNIFORM_HOLLOW_CYLINDER:
			return ModelInformation("Uniform Hollow Cylinder", 0, index, true, 2, 2, -1, 3, 0, false, false, true);

		//case GAUSSIAN_HOLLOW_CYLINDER: //being added
		//	return ModelInformation("Gaussian Hollow Cylinder", 0, index, false, 3, 2, -1, 3, 0, bGPU, false, true);

		case SPHERE:
			return ModelInformation("Sphere", 1, index, true, 2, 2, -1, 2, 0, bGPU, false, true);

		case SYMMETRIC_LAYERED_SLABS:
			return ModelInformation("Symmetric Layered Slabs", 2, index, true, 2, 2, -1, 4, 0, bGPU, false, true);

		case ASYMMETRIC_LAYERED_SLABS:
			return ModelInformation("Asymmetric Layered Slabs", 2, index, true, 2, 2, -1, 4, 0, bGPU, false, true);

		case HELIX:
			return ModelInformation("Helix", 3, index, false, 3, 2, 2, 5, 0, bGPU, false, true);

		case GRID_SYMMETRY:
			return ModelInformation("Space-filling Symmetry", 9, index, true, 3, 3, 3, 1);

		case MANUAL_SYMMETRY:
			return ModelInformation("Manual Symmetry", 9, index, true, 6, 0, -1, 1);
	}
}

IModel *GetModel(int index) {
	// This serves as an abstract factory of sorts
	switch(index) {
		default:
			return NULL;
		case UNIFORM_HOLLOW_CYLINDER:
			return new UniformHCModel();
		//case GAUSSIAN_HOLLOW_CYLINDER:
		//	return new GaussianHCModel();
		case SPHERE:
			return new UniformSphereModel();
		case SYMMETRIC_LAYERED_SLABS:
			return new UniformSlabModel();
		case ASYMMETRIC_LAYERED_SLABS:
			return new UniformSlabModel("Asymmetric Uniform Slabs", ASYMMETRIC);
		case HELIX:
			return new HelixModel();
		case GAUSSIAN_SPHERE:
			return new GaussianSphereModel();

		// This is dirty but had to be done (these are actually ISymmetry*s)
		case GRID_SYMMETRY:
			return (IModel *)new GridSymmetry();
		case MANUAL_SYMMETRY:
			return (IModel *)new ManualSymmetry();
	}

	return NULL;
}

template<typename T>
static void *GetMIProc(InformationProcedure type) {
        if(type == IP_LAYERPARAMNAME)         return (void *)&T::GetLayerParamNameStatic;
	else if(type == IP_LAYERNAME)         return (void *)&T::GetLayerNameStatic;
	else if(type == IP_EXTRAPARAMETER)    return (void *)&T::GetExtraParameterStatic;
	else if(type == IP_DEFAULTPARAMVALUE) return (void *)&T::GetDefaultParamValue;
	else if(type == IP_ISPARAMAPPLICABLE) return (void *)&T::IsParamApplicable;
	else if(type == IP_DISPLAYPARAMNAME)  return (void *)&T::GetDisplayParamName;
	else if(type == IP_DISPLAYPARAMVALUE) return (void *)&T::GetDisplayParamValue;
	
	return NULL;
}

EXPORTED void *GetModelInformationProcedure(int index, InformationProcedure type) {
	if(type <= IP_UNKNOWN || type >= IP_UNKNOWN2)
		return NULL;

	// This serves as an abstract factory of sorts
	switch(index) {
		default:
			return NULL;
		case UNIFORM_HOLLOW_CYLINDER:
			return GetMIProc<UniformHCModel>(type);
		case SPHERE:
			return GetMIProc<UniformSphereModel>(type);
		case SYMMETRIC_LAYERED_SLABS:
		case ASYMMETRIC_LAYERED_SLABS:
			return GetMIProc<UniformSlabModel>(type);
		case HELIX:
			return GetMIProc<HelixModel>(type);
		//case GAUSSIAN_HOLLOW_CYLINDER:
		//	return GetMIProc<GaussianHCModel>(type);

		case GRID_SYMMETRY:
			return GetMIProc<GridSymmetry>(type);
		case MANUAL_SYMMETRY:
			return GetMIProc<ManualSymmetry>(type);
	}
}
