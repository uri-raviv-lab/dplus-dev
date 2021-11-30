#ifndef __GRID_H
#define __GRID_H

#pragma once
#include "Common.h"
#include <complex>
#include <vector>
#include "Eigen/Core"
#include <omp.h>
#include <functional>
#include "../../Common/ZipLib/Source/streams/memstream.h"
#include "../../Common/ZipLib/Source/ZipFile.h"

typedef Eigen::Array<std::complex<FACC>, Eigen::Dynamic, 1> ArrayXcX;

class JsonWriter;

// Grid interface
class EXPORTED_BE Grid {
protected:
	double qmax;	/**< The effective maximum q-value the grid can be used for. Units of \f$nm^{-1}\f$. */
	
	unsigned short gridSize, actualGridSize;
	unsigned short Extras;
	unsigned int versionNum, bytesPerElement;

	//Eigen::Matrix3d existingRotation;

	virtual void SetCart(double x, double y, double z, std::complex<double> val) = 0;
	virtual void SetSphr(double x, double y, double z, std::complex<double> val) = 0;

public:
	Grid() : qmax(0.0), gridSize(0), actualGridSize(0), stepSize(0) {
//		existingRotation.setIdentity(3,3);
	}
	double stepSize;	/**< The size of a step between two adjacent q-values. Units of \f$nm^{-1}\f$. */
	Grid(unsigned short gridsz, double qMax) : qmax(qMax), gridSize(gridsz) {
		Extras = 11;	// Up until file version 10 (incl) this was 7
		actualGridSize = gridsz + Extras;

		// Since grid is in the range [-qmax, qmax], stepsize = 2qmax / gridsize
		stepSize = 2.0 * qMax / double(gridSize); 
//		existingRotation.setIdentity(3,3);
	}

	virtual ~Grid() {}

	// Not today...
	// Only returns required size if data == NULL
	/*virtual unsigned long long CopyTo(double *data) = 0;
	virtual bool CopyFrom(const Grid *other) = 0;*/
	
	virtual unsigned short GetDimX() const = 0;
	virtual unsigned short GetDimY(unsigned short x) const = 0;
	virtual unsigned short GetDimZ(unsigned short x, unsigned short y) const = 0;

	virtual double* GetDataPointer() = 0;

	virtual std::complex<double> GetCart(double x, double y, double z) const = 0;	
	virtual std::complex<double> GetSphr(double x, double y, double z) const = 0;	
	
	virtual void Fill(std::function< std::complex<FACC>(FACC, FACC, FACC)>calcAmplitude, void *progFunc, void *progArgs, double progMin, double progMax, int *pStop) = 0;

	// TODO: We can refactor import and export binary data, given the "InnerImport"/"InnerExport"
	//       methods implemented in every class, in conjunction with bytesPerElement and versionNum

	virtual bool ExportBinaryData(std::ostream& stream, const std::string& header) const = 0;
	virtual void ExportBinaryDataToStream(imemstream &strm) const = 0;
	// Fails if file is not of the same gridsize or stepsize
	virtual bool ImportBinaryData(std::istream& stream, std::string &header) = 0;
	virtual bool ImportBinaryData(std::istream * file) = 0;

	virtual unsigned short GetSize() const { return gridSize; }
	virtual unsigned short GetActualSize() const { return actualGridSize; }
	virtual unsigned short GetExtras() const { return Extras; }
	virtual u64 GetRealSize() const = 0;
	virtual double GetQMax() const { return qmax; }
	virtual double GetStepSize() const { return stepSize; }

	virtual void Add(Grid* other, double scale) = 0;
	virtual void Scale(double scale) = 0;
	virtual bool Validate() const = 0;

	virtual bool IndicesToVectors(unsigned short xi, unsigned short yi, unsigned short zi, OUT double &x, OUT double &y, OUT double &z) = 0;

	// Returns a pointer to the data, filled with the x,y,z (dimy, dimz) indices
	virtual double *GetPointerWithIndices() = 0;

	virtual bool RunAfterCache() = 0;
	virtual bool RunAfterReadingCache() = 0;
	virtual bool RunBeforeReadingCache() = 0;

	virtual void WriteToJsonWriter(JsonWriter &writer) = 0;
	virtual std::string GetParamJsonString() = 0;
};


#define URIS_GRID
#ifdef URIS_GRID
/************************************************************************/
/* JacobianSphereGrid: Each shell is a square on the theta-phi plane    */
/************************************************************************/
class EXPORTED_BE JacobianSphereGrid : public Grid {
protected:
	Eigen::ArrayXd data;
	// DO NOT MODIFY TO u64, OPENMP (v2.0) DOES NOT SUPPORT UNSIGNED TYPES (and MS doesn't seem to want to upgrade)
	long long totalsz; // 64-bit value (number of double elements, 2 times number of complex elements)

	Eigen::ArrayXd interpolantCoeffs;

	
	//double rotatedTheta, rotatedPhi;


	virtual void SetCart(double x, double y, double z, std::complex<double> val);
	virtual void SetSphr(double q, double th, double ph, std::complex<double> val);

	virtual void InitializeGrid();

	virtual bool InnerImportBinaryData(std::istream& file);

	

public:
	inline std::complex<double> InterpolateThetaPhiPlane(const u16 ri, const double theta, const double phi) const;
	char thetaDivisions, phiDivisions;	// 4*i+1 --> 4 and 8*i --> 8
	//	JacobianSphereGrid(const JacobianSphereGrid& other);
	JacobianSphereGrid() {};
	JacobianSphereGrid(unsigned short gridSize, double qMax);
	JacobianSphereGrid(std::istream& stream, std::string &header);
	JacobianSphereGrid(std::istream * file, unsigned short gridsize, double qMax);

	~JacobianSphereGrid();

	virtual unsigned short GetDimX() const;                    // DimR
	virtual unsigned short GetDimY(unsigned short x) const;    // DimPhi
	virtual unsigned short GetDimZ(unsigned short x, unsigned short y) const; // DimTheta

	virtual std::complex<double> GetCart(double x, double y, double z) const;	
	virtual std::complex<double> GetSphr(double r, double th, double ph) const;	

	virtual int GetSplineBetweenPlanes(FACC q, FACC theta, FACC phi, OUT std::complex<double>& pl1, OUT std::complex<double>& pl2, OUT std::complex<double>& d1, OUT std::complex<double>& d2);
	virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi);

	inline long long IndexFromIndices(int qi, long long ti, long long pi) const;
	inline void IndicesFromIndex(long long index, int &qi, long long &ti, long long &pi);
	inline void IndicesFromRadians(const u16 ri, const double theta, const double phi,
		long long &ti, long long &pi, long long &base, double &tTh, double &tPh) const;

	virtual double* GetDataPointer();
	virtual void Fill(std::function< std::complex<FACC>(FACC, FACC, FACC)>calcAmplitude, void *progFunc, void *progArgs, double progMin, double progMax, int *pStop);

	void CalculateSplines();

	// TODO: We can refactor import and export binary data, given the "InnerImport"/"InnerExport"
	//       methods implemented in every class, in conjunction with bytesPerElement and versionNum
	virtual bool ExportBinaryData(std::ostream& stream, const std::string& header) const;
	virtual void ExportBinaryDataToStream(imemstream &strm) const;
	// Fails if file is not of the same gridsize or stepsize
	virtual bool ImportBinaryData(std::istream& stream, std::string &header);
	virtual bool ImportBinaryData(std::istream * file);

	virtual u64 GetRealSize() const { return totalsz * sizeof(double); }

	virtual void Add(Grid* other, double scale);
	virtual void Scale(double scale);
	virtual bool Validate() const;

	virtual bool IndicesToVectors(unsigned short xi, unsigned short yi, unsigned short zi, OUT double &x, OUT double &y, OUT double &z);

	// Returns a pointer to the data, filled with the x,y,z (dimy, dimz) indices
	virtual double *GetPointerWithIndices();

	virtual double *GetInterpolantPointer();

	virtual bool RunAfterCache();
	virtual bool RunAfterReadingCache();
	virtual bool RunBeforeReadingCache();

	virtual bool RotateGrid( double thet, double ph, double psi );

	//virtual bool ExportThetaPhiPlane(std::string outFileName, long long index);

	virtual void DebugMethod();

	virtual void WriteToJsonWriter(JsonWriter &writer);

	virtual std::string GetParamJsonString();
	
};
#endif	//URIS_GRID

#define USE_JACOBIAN_SPHERE_GRID
#define USE_SPHERE_GRID
#ifdef USE_SPHERE_GRID
	#ifdef USE_JACOBIAN_SPHERE_GRID
		typedef JacobianSphereGrid CurGrid;
		#define CURGRIDNAME   "JacobianSphereGrid"
		#define WCURGRIDNAME L"JacobianSphereGrid"
	#else
		typedef SphereGrid CurGrid;
		#define CURGRIDNAME   "SphereGrid"
		#define WCURGRIDNAME L"SphereGrid"
	#endif
#else
	typedef GridB CurGrid;
	#define CURGRIDNAME   "GridB"
	#define WCURGRIDNAME L"GridB"
#endif	// USE_SPHERE_GRID
#endif	// __GRID_H
