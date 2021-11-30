#include "Prep.h"

#include "Backend/Amplitude.h"
#include "DefaultModels/Symmetries.h"
#include "PDBReaderLib.h"
#include "Backend/Symmetry.h"

#include "vector_functions.h"


bool PrepareAllParameters(
	std::string inFilename,
	std::vector<int>	&atomsPerIon,
	std::vector<float4>	&loc, 
	std::vector<char>	&atmInd, 
	std::vector<u8>		&ionInd, 
	std::vector<float>	&BFactors, 
	std::vector<float>	&coeffs, 
	std::vector<float>	&atmRad
)
{

	PDBReader::PDBReaderOb<float> pdb(inFilename, false);

	pdb.SetRadiusType(RAD_DUMMY_ATOMS_ONLY);


	loc.resize(pdb.sortedX.size());

	int pos = 0;
	for(int i = 0; i < pdb.sortedX.size(); i++) {
		loc[i] = make_float4(pdb.sortedX[i], pdb.sortedY[i], pdb.sortedZ[i], 0.f);
	}

	atmRad.resize(pdb.rad->size());
	for(int i = 0; i < pdb.rad->size(); i++) {
		atmRad[i]   = pdb.rad->at(i);
	}
	atmInd.resize(pdb.sortedAtmInd.size());
	for(int i = 0; i < pdb.sortedAtmInd.size(); i++) {
		atmInd[i]	= pdb.sortedAtmInd[i];
	}

	std::vector<float> fAtmFFcoefs (pdb.sortedCoeffs.begin(), pdb.sortedCoeffs.end());
	
	coeffs		= pdb.sortedCoeffs;

	atomsPerIon	= pdb.atomsPerIon;
	ionInd		= pdb.sortedCoeffIonInd;
	BFactors	= pdb.sortedBFactor;

	return true;
}
