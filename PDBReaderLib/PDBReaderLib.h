#ifndef __PDBREADERLIB_H
#define __PDBREADERLIB_H

#include "GeneralPDBReaderLib.h"


// For GPUs
namespace PDBReader {

struct float4
{
	float x, y, z, w;
};

template<class FLOAT_TYPE = double>
struct IonEntry {
	u8 ionInd, atmInd;
	FLOAT_TYPE x, y, z, rad, BFactor, fPrime, fPrimePrime;
};
template<class FLOAT_TYPE>
bool SortIonEntry(const IonEntry<FLOAT_TYPE>& a,const IonEntry<FLOAT_TYPE>& b) { return ((int)a.ionInd < (int)b.ionInd); }

template<class FLOAT_TYPE>
class EXPORTED_PDBREADER PDBReaderOb {
	typedef std::pair<FLOAT_TYPE, FLOAT_TYPE> fpair;

public:
	vector<FLOAT_TYPE> x, y, z, BFactor, occupancy, sortedX, sortedY, sortedZ;
	vector<float> sortedBFactor;
	vector<PDBReader::float4> atomLocs; ///< The locations of all the atoms, sorted by ion type
	string pdbPreBla;
	vector<string> pdbAtomSerNo;
	vector<string> pdbAtomName;
	vector<string> pdbResName;
	vector<char>   pdbChain;
	vector<string> pdbResNo;
	vector<string> pdbSegID;
	vector<string> atom;
	vector<string> implicitAtom;
	vector<unsigned long int> mdl, ter;
	unsigned long int numberOfAA;
	int number_of_implicit_atoms;
	int number_of_implicit_amino_acids;
	// A list of 118 (I think) atomic weights
	vector<FLOAT_TYPE> atmWt;
	vector<u8> atmInd, ionInd, sortedAtmInd, sortedIonInd;
	vector<FLOAT_TYPE> *rad;
	vector<FLOAT_TYPE> vdwRad;
	vector<FLOAT_TYPE> empRad;
	vector<FLOAT_TYPE> calcRad;
	vector<FLOAT_TYPE> svgRad;
	vector<FLOAT_TYPE> sortedCoeffs;
	vector<unsigned char> sortedCoeffIonInd;
	vector<int> atomsPerIon; ///< The number of atoms of the i'th type, where 'i' is the index given after sorting. For H_20 the vector would be [2,1] assuming H was first.

	vector<FLOAT_TYPE> atmFFcoefs;

	// Parameters related to anomalous scattering
	string energyStr;	///< The text that was after " # Energy " in the anomalous file
	std::map<fpair, vector<int>> anomGroups;
	std::map<int, fpair> anomIndex;
	std::map<string, fpair> anomTypes;
	bool haveAnomalousAtoms;
	vector<std::complex<float>> anomfPrimes, sortedAnomfPrimes;

	bool bMoveToCOM;
	
	// Filename of the PDB
	string fn;
	string anomalousfn;
	PDB_READER_ERRS status;

	bool bOutputSlices;
	bool bOnlySolvent;
	bool bFillHoles;
	std::string slicesBasePathSt;

	ATOM_RADIUS_TYPE atmRadType;

public:
	virtual ~PDBReaderOb();
	virtual PDB_READER_ERRS readPDBfile(string filename, bool bCenter, int model = 0);
	virtual PDB_READER_ERRS readPDBbuffer(const char *buffer, size_t buffSize, bool bCenter, int model = 0);
	virtual PDB_READER_ERRS readAnomalousfile(string filename);
	virtual PDB_READER_ERRS readAnomalousbuffer(const char *buffer, size_t buffSize);
	virtual void AlignPDB();
	virtual PDB_READER_ERRS getAtomsAndCoords(std::vector<float>& xOut, std::vector<float>& yOut, std::vector<float>& zOut, std::vector<u8>& atomInd) const;
	virtual PDB_READER_ERRS getAtomsAndCoords(std::vector<float>& xOut, std::vector<float>& yOut, std::vector<float>& zOut, std::vector<short>& atomInd) const;
	virtual PDB_READER_ERRS writeCondensedFile(string filename);
	virtual PDB_READER_ERRS readCondensedFile(string filename);
	virtual void SetRadiusType(ATOM_RADIUS_TYPE type);
	virtual ATOM_RADIUS_TYPE GetRadiusType();
	virtual PDB_READER_ERRS CopyPDBData(const PDBReaderOb& src);

	virtual void BuildPDBFromList(std::vector<std::vector<FLOAT_TYPE> > &FormListData);
	virtual PDB_READER_ERRS WritePDBToFile(std::string fileName, const std::stringstream& header);

	virtual bool getHasAnomalous();
	virtual bool getBOnlySolvent();
	virtual void setBOnlySolvent(bool bSolv);

	virtual void moveCOMToOrigin();
	virtual void moveGeometricCenterToOrigin();

protected:
	virtual void initialize() = 0;
	PDB_READER_ERRS readPDBstream(std::istream& inFile, bool bCenter, int model);
	virtual PDB_READER_ERRS readAnomalousstream(std::istream& inFile);
	PDB_READER_ERRS ionIndToatmInd();
	virtual void getAtomIonIndices(string atm, u8& atmInd, u8& ionInd);
	virtual void ExtractRelevantCoeffs(std::vector< IonEntry<FLOAT_TYPE> > &entries,
		u64 vecSize, std::vector<u8>& sIonInd, std::vector<u8>& sAtmInd, std::vector<int>& atmsPerIon,
		std::vector<unsigned char>& sortedCoefIonInd, std::vector<FLOAT_TYPE>& sX,
		std::vector<FLOAT_TYPE>& sY, std::vector<FLOAT_TYPE>& sZ, std::vector<PDBReader::float4>& atmLocs,
		std::vector<float> sBFactor, std::vector<FLOAT_TYPE>& sCoeffs, std::vector<std::complex<float>>* fPrimes = NULL);
	
	void ChangeResidueGroupsToImplicit(std::string amino_acid_name, const int i, const int aa_size);
	std::string removeSpaces(std::string source); //we had to create these two functions because of a gcc bug
	std::string removeDigits(std::string source);
};

template<class FLOAT_TYPE>
class EXPORTED_PDBREADER XRayPDBReaderOb : public PDBReaderOb<FLOAT_TYPE>
{
public:
	XRayPDBReaderOb();
	XRayPDBReaderOb(string filename, bool moveToCOM, int model = 0, string anomalousFName = "");

protected:
	void initialize();
};

#ifndef _GLIBCXX_NOEXCEPT 
#define _GLIBCXX_NOEXCEPT // This is so we can use GNU libstdc++ and VS's versions of std::exception
#endif
class EXPORTED_PDBREADER pdbReader_exception : public std::exception {
public:
	virtual ~pdbReader_exception() throw() {}

	explicit pdbReader_exception(PDB_READER_ERRS error_code, const char *error_message = "");

	PDB_READER_ERRS GetErrorCode() const;

	std::string GetErrorMessage() const;

	virtual const char* what() const _GLIBCXX_NOEXCEPT;
private:
	PDB_READER_ERRS _errorCode;
	std::string _errorMessage;
};


};

#endif


