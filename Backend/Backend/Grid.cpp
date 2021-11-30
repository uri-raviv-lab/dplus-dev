

#include "Grid.h"
//#include "boost/filesystem/fstream.hpp"
//#include "boost/filesystem.hpp"
#include "backend_exception.h"
#include "PeriodicSplineSolver.h"

#include <Eigen/Eigenvalues>

#include <chrono>
//#include <boost/timer/timer.hpp>
//#include <boost/chrono/chrono.hpp>

#include <rapidjson/document.h>
#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>

#include <iostream>
using std::ios;

//namespace fs = boost::filesystem;
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_2PI
#define M_2PI 6.28318530717958647692528676656
#endif

typedef uint32_t U32;
typedef uint64_t U64;

//calculate third root of int
U32 icbrt64(U64 x) {
	int s;
	U32 y;
	U64 b;

	y = 0;
	for (s = 63; s >= 0; s -= 3) {
		y += y;
		b = 3*y*((U64) y + 1) + 1;
		if ((x >> s) >= b) {
			x -= b << s;
			y++;
		}
	}
	return y;
}


/************************************************************************/
/* JacobianSphereGrid: Each shell is a square on the theta-phi plane    */
/************************************************************************/
std::complex<double> JacobianSphereGrid::InterpolateThetaPhiPlane( const u16 ri, const double theta, const double phi ) const {
	long long base, tI, pI;
	double tPh, tTh;

	IndicesFromRadians(ri, theta, phi, tI, pI, base, tTh, tPh);

	// Use the pre-calculated splines on phi to setup the points to interpolate on theta
	Eigen::VectorXcd nodes;
	Eigen::VectorXcd ds;
	int points = 4;
	int mInd = 1;	// The number of points below the lowest neighbor we would like to take
	nodes.resize(points);

	int phiPoints = phiDivisions * ri;
	int thePoints = thetaDivisions * ri;

	// The index of the point above the value in the phi axis
	int pI1 = (pI + 1) % phiPoints;

	std::complex<double> p1, p2;
	std::complex<double> d1, d2;

	for(int i = -mInd; i < points - mInd; i++) {
		// This is actually incorrect at the edges.
		// If we deviate from [0,pi], we have to add pi to phi
		// See addition in 6 lines that deals with this
		int theInd = (thePoints + 1 + tI + i) % (thePoints + 1);
		long long pos1, pos2;
		// Calculate using spline
		if(theInd == tI + i) {

			pos1 = 2*(base + phiPoints * theInd + pI );
			pos2 = 2*(base + phiPoints * theInd + pI1);

		} else {
			// There is a "2*" and a "/2" that have been removed (canceled out)
			// Assumes phiPoints is even
			if(tI + i > thePoints)
				theInd = (tI + i) - (thePoints+1);
			pos1 = 2*(base + phiPoints * theInd + pI );
			pos2 = 2*(base + phiPoints * theInd + pI1);

			pos1 += ((2*pI  >= phiPoints) ? -phiPoints : phiPoints);
			pos2 += ((2*pI1 >= phiPoints) ? -phiPoints : phiPoints);
		}


		// TODO: Deal with the case where the theta points loop around [0,pi)
		//			The index for phi then has to be moved by pi.

		d1 = *(std::complex<double>*)(&interpolantCoeffs[pos1]);
		d2 = *(std::complex<double>*)(&interpolantCoeffs[pos2]);

		p1 = *(std::complex<double> *)(&data[pos1]);
		p2 = *(std::complex<double> *)(&data[pos2]);

		nodes[i+mInd] = p1 + std::complex<double>(d1) * tPh + 
						(3.0 * (p2 - p1) - 2.0 * std::complex<double>(d1) - std::complex<double>(d2)) * (tPh*tPh) + 
						(2.0 * (p1 - p2) + std::complex<double>(d1 + d2)) * (tPh*tPh*tPh);
	}

	// Calculate the spline for theta using 'points' nodes
	if(points != 4) {
		ds = EvenlySpacedSplineSolver(&nodes[0], nodes.size());
	} else {
		ds.resize(4);
		EvenlySpacedFourPointSpline(nodes[0], nodes[1], nodes[2], nodes[3], &ds[mInd], &ds[mInd+1]);
	}

	// Return the value at (theta,phi)
	return nodes[mInd] + ds[mInd] * tTh + 
		(3.0 * (nodes[mInd+1] - nodes[mInd]) - 2.0 * ds[mInd] - ds[mInd+1]) * (tTh*tTh) + 
		(2.0 * (nodes[mInd] - nodes[mInd+1]) + ds[mInd] + ds[mInd+1]) * (tTh*tTh*tTh);
}

void JacobianSphereGrid::SetCart( double x, double y, double z, std::complex<double> val ) {
	double qq = sqrt(x*x + y*y + z*z);
	if(fabs(z) > qq) qq = fabs(z);	// Correct if (z/r) is greater than 1.0 by a few bits
	double th = acos(z / qq);
	double ph = atan2(y, x);
	if(ph < 0.0)
		ph += M_2PI;

	SetSphr(qq, th, ph, val);
}

void JacobianSphereGrid::SetSphr( double q, double th, double ph, std::complex<double> val ) {
	// Not sure this is necessary, trying to make sure we don't miss a level
	double *fptr = &q;
	(*((long long *)fptr)) += 8;

	long long ind = (long long)(q / stepSize);
	long long thI, phI;

	if(ind == 0) {
		data[0] = val.real();
		data[1] = val.imag();
		return;
	}

	thI  = (long long)(double(thetaDivisions * ind + 1) * (th / M_PI ));
	phI  = (long long)(double(phiDivisions   * ind) * (ph / M_2PI));

	long long pos = IndexFromIndices(ind, thI, phI);
	data[pos  ] = val.real();
	data[pos+1] = val.imag();
}

void JacobianSphereGrid::InitializeGrid() {
	thetaDivisions = 3;// approx pi
	phiDivisions = 6;//approx 2*pi

	//interpolantCoeffs = NULL;
	interpolantCoeffs = Eigen::ArrayXd();

	// Since grid is in the range [-qmax, qmax], stepsize = 2qmax / gridsize
	stepSize = qmax / double(gridSize / 2); 
	actualGridSize = gridSize / 2 + Extras;

	long long i = actualGridSize;
	/*This is the result of the equation: sigma from 1 to n on (6i*(3i+1))
	when 6i*(3i+1) is the number of point for each layer (according to radius- q), when the distance of point on the circle smaller than q (radius of the circle)
	*/
	totalsz = (phiDivisions * i * (i+1) * (3 + thetaDivisions + 2 * thetaDivisions * i)) / 6;
	totalsz++;	// Add the origin
	totalsz *= 2;	// Complex

	try {
		//data = new double[totalsz];		
		data.resize(totalsz);
		data.setZero();
	} catch(...) {
		//data = NULL;
		throw backend_exception(ERROR_INSUFFICIENT_MEMORY);
	}


	printf("Total number of cells: %lld\n\tcells on outer layer: %lld\n",
		totalsz/2, (thetaDivisions*i+1)*(phiDivisions*i));
}

bool JacobianSphereGrid::ImportBinaryData(std::istream * file)
{
	//std::istream * file = entry1->GetRawStream();
	file->read((char *)(data.data()), (size_t)(sizeof(double) * totalsz));
	if (!file->good())
		//data = NULL;
		data = Eigen::ArrayXd();
	return file->good();
}

bool JacobianSphereGrid::InnerImportBinaryData( std::istream& file ) {
	file.read((char *)(data.data()), sizeof(double) * totalsz);
	if (!file.good())
		//data = NULL;
		data = Eigen::ArrayXd();
	return file.good();
}

JacobianSphereGrid::JacobianSphereGrid( unsigned short gridsize, double qMax ) {
	Extras = 3;
	gridSize = gridsize;
	qmax = qMax;

	InitializeGrid();
}

JacobianSphereGrid::JacobianSphereGrid(std::istream * file, unsigned short gridsize, double qMax)
{
	Extras = 3;
	gridSize = gridsize;
	qmax = qMax;
	InitializeGrid();
	ImportBinaryData(file);

}

void JacobianSphereGrid::WriteToJsonWriter(JsonWriter &writer)
{
	unsigned short actualGridSize;
	unsigned short Extras;
	long long totalsz; // 64-bit value (number of double elements, 2 times number of complex elements)
	// char thetaDivisions, phiDivisions;//???

	//the bare minimum needed for a grid: without this, no grid can be created
	writer.Key("qmax");
	writer.Double(qmax);
	writer.Key("qmin"); //not currently used, but adding for the future
	writer.Double(0); //not currently used, but adding for the future
	writer.Key("gridSize");
	writer.Int(gridSize);

	writer.Key("stepSize");
	writer.Double(stepSize);

}
std::string JacobianSphereGrid::GetParamJsonString()
{
	JsonWriter infoWriter;
	infoWriter.StartObject();
	WriteToJsonWriter(infoWriter);
	infoWriter.Key("Params");
	infoWriter.StartArray();
	infoWriter.EndArray();
	infoWriter.EndObject();
	const char * chartest = infoWriter.GetString();
	std::string str(chartest);
	return str;
}


JacobianSphereGrid::JacobianSphereGrid( std::istream& stream, std::string &header ) {
	versionNum = 0;
	unsigned int numBytes = 0;
//	data = NULL;
	data = Eigen::ArrayXd();
//	interpolantCoeffs = NULL;
	interpolantCoeffs = Eigen::ArrayXd();

	char desc[2] = {0};
	unsigned int offset = 0;

	if(stream.peek() == '#') {
		stream.read(desc, 2);
		if(desc[1] == '@') { // We have an offset
			stream.read((char *)&offset, sizeof(unsigned int));
			std::string del;
			getline(stream, del);
		} else if(desc[1] != '\n') {
			std::string tmphead;
			getline(stream, tmphead);
			header.append(tmphead + "\n");
		}
	}

	// Read the header
	std::string head;
	while(stream.peek() == '#') {
		getline(stream, head);
		header.append(head + "\n");
	}

	if(offset > 0) {
		// Skip the header
		stream.seekg(offset, ios::beg);
	} else {
	}

	// Version
	stream >> versionNum;

	// Check if current binary file - no backward compatibility
	if(versionNum < 13) {
		std::cout << "The file version does not match the grid... " << versionNum << std::endl;
		return;	// We should change it to use GridB or something else
	}

	stream >> numBytes;
	if(numBytes != sizeof(std::complex<double>))
		return;

	// Fill grid parameters
	// stream >> stepSize;
	stream >> actualGridSize;
	stream >> Extras;

	gridSize = (actualGridSize - Extras) * 2;

	// Ignore the newline
	stream.ignore(100, '\n');
	stream.read((char *)&(stepSize), sizeof(double));
//	stream.read((char *)&(existingRotation), sizeof(double)*9);
//	std::cout << existingRotation << std::endl;
	qmax = stepSize * (gridSize) / 2.0;
	if (qmax < 0 || stepSize < 0 || gridSize < 0 || actualGridSize < 0 )
		throw backend_exception(ERROR_GENERAL, "Amp file is not valid");
	InitializeGrid();

	if(!InnerImportBinaryData(stream)) {
		gridSize = 0;
		actualGridSize = 0;
		stepSize = 0.0;
		qmax = 0.0;
		return;
	}
	CalculateSplines();
}

// JacobianSphereGrid::JacobianSphereGrid(const JacobianSphereGrid& other)
// {
// 	*this = other;
// 	InitializeGrid();
// 	memcpy(data.data(), other.data.data(), other.GetRealSize());
// 
// }

JacobianSphereGrid::~JacobianSphereGrid() {
// 	if(data)
// 		delete [] data;
// 	data = NULL;
// 	if(interpolantCoeffs)
// 		delete [] interpolantCoeffs;
// 	interpolantCoeffs = NULL;
}

unsigned short JacobianSphereGrid::GetDimX() const {
	return u16(actualGridSize);
}

unsigned short JacobianSphereGrid::GetDimY( unsigned short x ) const {
	if(x == 0) {
		return 1;
	}
	return (thetaDivisions * x + 1);
}

unsigned short JacobianSphereGrid::GetDimZ( unsigned short x, unsigned short y ) const {
	if(x == 0) {
		return 1;
	}
	return (phiDivisions * x);
}

std::complex<double> JacobianSphereGrid::GetCart( double x, double y, double z ) const {
	double r  = sqrt(x*x + y*y + z*z);
	if(fabs(z) > r) r = fabs(z);	// Correct if (z/r) is greater than 1.0 by a few bits
	double th = acos(z / r);
	double ph = atan2(y, x);
	if(ph < 0.0)
		ph += M_2PI;
	return GetSphr(r, th, ph);
}

std::complex<double> JacobianSphereGrid::GetSphr( double rr, double th, double ph ) const {

	std::complex<double> res, plane1, plane2, plane3, plane4;
	double frac;

	// Origin
	if(rr == 0.0) {
		return std::complex<double>(data[0], data[1]);
	}

	double frc = (rr / stepSize);
	// Make sure there aren't any floating point rounding issues
	//  (e.g. int(0.99999999999999989) --> 0 instead of 1)
	double *fptr = &frc;
	(*((long long *)fptr)) += 8;

	int fInd = int(*fptr);
	frac = frc - double(fInd);

	plane2 = (fInd > 0) ? InterpolateThetaPhiPlane(fInd, th, ph) : std::complex<double>(data[0], data[1]);

 	if(frac < 1.e-9)
  		return plane2;
	plane3 = InterpolateThetaPhiPlane(fInd + 1, th, ph);
	double new_th = 0;
	double new_phi = 0;
	switch (fInd)
	{
	case 0:
		new_th = M_PI - th;
		new_phi = ph + M_PI;
		// save new_ph in range [0, 2pi]
		if (new_phi < 0)
			new_phi = M_2PI + (new_phi - int(new_phi / M_2PI)* M_2PI);
		if (new_phi >= M_2PI)
			new_phi = new_phi - int(new_phi / M_2PI)*M_2PI;
		// save new_th in range [0, pi]
		if (new_th < 0)
			new_th = 0.;
		if (new_th > M_PI)
			new_th = M_PI;

		plane1 = InterpolateThetaPhiPlane(fInd + 1, new_th, new_phi);
		break;
	case 1:
		plane1 = std::complex<double>(data[0], data[1]);
		break;
	default:
		plane1 = InterpolateThetaPhiPlane(fInd - 1, th, ph);
		break;
	}

	// For the last plane, assume it continues straight (this can be corrected later in life)
	plane4 = (fInd + 2 < actualGridSize) ? InterpolateThetaPhiPlane(fInd + 2, th, ph) : plane3;

	std::complex<double> d1,d2;
	EvenlySpacedFourPointSpline(plane1, plane2, plane3, plane4, &d1, &d2);

	return plane2 + d1 * frac + 
		(3.0 * (plane3 - plane2) - 2.0 * d1 - d2) * (frac*frac) + 
		(2.0 * (plane2 - plane3) + d1 + d2) * (frac*frac*frac);
}


int JacobianSphereGrid::GetSplineBetweenPlanes(FACC q, FACC th, FACC ph, OUT std::complex<double>& pl1, OUT std::complex<double>& pl2, OUT std::complex<double>& d1, OUT std::complex<double>& d2)
{
	std::complex<double> res, plane1, plane2, plane3, plane4;
	double frac;

	double frc = (q / stepSize);
	// Make sure there aren't any floating point rounding issues
	//  (e.g. int(0.99999999999999989) --> 0 instead of 1)
	double *fptr = &frc;
	(*((long long *)fptr)) += 8;

	int fInd = int(*fptr);
	frac = frc - double(fInd);


	switch (fInd)
	{
	case 0:
	{
		double new_th = M_PI - th;
		double new_phi = ph + M_PI;
		// save new_ph in range [0, 2pi]
		if (new_phi < 0)
			new_phi = M_2PI + (new_phi - int(new_phi / M_2PI)* M_2PI);
		if (new_phi >= M_2PI)
			new_phi = new_phi - int(new_phi / M_2PI)*M_2PI;
		// save new_th in range [0, pi]
		if (new_th < 0)
			new_th = 0.;
		if (new_th > M_PI)
			new_th = M_PI;

		plane1 = InterpolateThetaPhiPlane(fInd + 1, new_th, new_phi);
		break;
	}
	case 1:
		plane1 = std::complex<double>(data[0], data[1]);
		break;
	default:
		plane1 = InterpolateThetaPhiPlane(fInd - 1, th, ph);
		break;
	}

	plane2 = (fInd > 0) ? InterpolateThetaPhiPlane(fInd, th, ph) : std::complex<double>(data[0], data[1]);

	plane3 = InterpolateThetaPhiPlane(fInd + 1, th, ph);

	// For the last plane, assume it continues straight (this can be corrected later in life)
	plane4 = (fInd + 2 < actualGridSize) ? InterpolateThetaPhiPlane(fInd + 2, th, ph) : plane3;

	EvenlySpacedFourPointSpline(plane1, plane2, plane3, plane4, &d1, &d2);

	pl1 = plane2;
	pl2 = plane3;

	return std::max(fInd, 0);
}

ArrayXcX JacobianSphereGrid::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
{
	ArrayXcX reses(relevantQs.size());
	size_t i = 0;

	std::complex<double> pl1, pl2;
	std::complex<double> d1, d2;
	
	double frac;

	auto  qForSpline = relevantQs.begin();
	while (qForSpline != relevantQs.end())
	{
		int planeIndex = GetSplineBetweenPlanes(*qForSpline, theta, phi, pl1, pl2, d1, d2);
		auto curQ = qForSpline;
		while (curQ != relevantQs.end() && *curQ < *qForSpline + stepSize)
		{
			double frc = (*curQ / stepSize);
			// Make sure there aren't any floating point rounding issues
			//  (e.g. int(0.99999999999999989) --> 0 instead of 1)
			double *fptr = &frc;
			(*((long long *)fptr)) += 8;

			frac = frc - double(planeIndex);

			auto myres = pl1 + d1 * frac +
				(3.0 * (pl2 - pl1) - 2.0 * d1 - d2) * (frac*frac) +
				(2.0 * (pl1 - pl2) + d1 + d2) * (frac*frac*frac);
			
			reses(i) = myres;

			i++;
			curQ++;
		} // while *curQ
		
		qForSpline = curQ;

	} // while qForSpline
	
	return reses;
} //getAmplitudesAtPoints

double* JacobianSphereGrid::GetDataPointer() {
	return data.data();

}

void JacobianSphereGrid::Fill(std::function< std::complex<FACC>(FACC, FACC, FACC)>calcAmplitude, void *prog, void *progArgs, double progMin, double progMax, int *pStop ) {
	progressFunc progFunc = (progressFunc)prog;

	const long long dims = totalsz / 2;

	int iprog = 0;

	// This is an inefficient form for the CPU. If using nested loops, sin/cos(theta) gets calculated a lot less.
	// It's implemented this way (for now) as a template for the GPU
#pragma omp parallel for schedule(guided, dims / (omp_get_num_procs() * 16))
	for(long long ind = 0; ind < dims; ind++) {
		std::complex<double> rs(0.0,0.0);
		int qi;
		long long thi, phi;
		
		// Determine q_vec as a function of the single index
		IndicesFromIndex(ind, qi, thi, phi);

		//const int dimy = int(thetaDivisions) * qi + 1;	// (4*i+1)
		const int dimz = int(phiDivisions) * qi;

		// Stop check		
		if(pStop && *pStop)
			continue;

		// Progress report
		if(progFunc) {
#pragma omp atomic
			++iprog;
			if((100 * iprog) % dims < 100) {
#pragma omp critical
				{			
					progFunc(progArgs, (progMax - progMin) * (double(iprog) / double(dims)) + progMin);
				}
			}
		}

		double qI = double(qi) * stepSize;
		double tI = M_PI  * double(thi) / double(int(thetaDivisions) * qi);	// dimy - 1
		double pI = M_2PI * double(phi) / double(dimz);

		double cst = cos(tI),
			   snt = sin(tI);

		double csp = cos(pI),
			   snp = sin(pI);

		if(qi == 0) {
			cst = snt = csp = snp = 0.0;
		}
#ifdef _DEBUG5
		if(ind < 1024) {
			std::cout << ind << ":\t(" << qi << ",\t" << thi << ",\t" << phi << ")";
			std::cout << "\t(" << qI << ",\t" << tI << ",\t" << pI << ")" <<std::endl;
		}
#endif
		/*The points on the sphere with radius qI can be parameterized via:
		x= x0 + qI*cos(phi)*sin(theta)
		y = y0 + qI*sin(phi)*sin(theta)
		z= z0 + qI*cos(theta)
		calcAmplitude - calculate the fourier transform of this vec(x,y,z) - (in q space)
		*/
		rs = calcAmplitude(qI * snt * csp, qI * snt * snp, qI * cst);

		data[2*ind]   = rs.real();
 		data[2*ind+1] = rs.imag();
	}
	//the interpolation calculation- more accurate than polynomial interpolation
	CalculateSplines();
/*
	ExportThetaPhiPlane("S:\\Basic Dimer\\Grid Tests\\debug\\Plane 1.dat" , 1);
	ExportThetaPhiPlane("S:\\Basic Dimer\\Grid Tests\\debug\\Plane 5.dat" , 5);
	ExportThetaPhiPlane("S:\\Basic Dimer\\Grid Tests\\debug\\Plane 10.dat" , 10);
	ExportThetaPhiPlane("S:\\Basic Dimer\\Grid Tests\\debug\\Plane 50.dat" , 50);
*/
	/************************************************************************/
	/* Debug section                                                        */
	/* Check the validity of the IndexFromIndices and IndicesFromIndex      */
	/************************************************************************/

}

void JacobianSphereGrid::ExportBinaryDataToStream(imemstream &strm) const
{
	strm = imemstream((char *)data.data(), (size_t)(sizeof(double) * totalsz));
}

bool JacobianSphereGrid::ExportBinaryData( std::ostream& stream, const std::string& header ) const {
	if(!stream)
		return false;

	// Versions 0 and 1 are text
	// Version 1 has "\n" in the data region
	// Versions >= 10 contain the data as binary
	// Version 4 is reserved for files converted from formated to binary
	// Version 2 is reserved for files converted from binary to formatted
	// Version 11 uses the new form of grid and removed other crap
	// Version 12 uses the new form of spherical grid and removed other crap
	// Version 13 uses Uri's quasi-spherical grid

	unsigned int versionNum = 13;

	char DESCRIPTOR[3] = "#@";
	unsigned int headlen = (unsigned int)(2 * sizeof(char) + sizeof(unsigned int) + sizeof(char) + 
		                   header.length() * sizeof(char) + sizeof(char)); // descriptor + \n + header length + \n

	stream.write(DESCRIPTOR, 2 * sizeof(char));
	stream.write((const char *)&headlen, sizeof(unsigned int)); // 32-bit, always
	
	stream << "\n" << header << "\n";

	// File version number
	stream << versionNum << "\n";
	// Bytes per element
	stream << sizeof(std::complex<double>) << "\n";
	// Size of trnsfm --> txSize
	unsigned short dimx = GetDimX();
	stream << dimx << "\n";	
	stream << Extras << "\n";
	stream.write((const char *)&(stepSize), sizeof(double));
//	stream.write((const char *)&(existingRotation), sizeof(double)*9);

	// Later, for file-based grids
	/*
	headlen = (unsigned int)stream.tellp(); // Say we're 32-bit
	stream.seekp(2, ios::beg);
	stream.write((const char *)&headlen, sizeof(unsigned int)); // 32-bit, always
	stream.seekp(headlen, ios::beg);
	*/

	stream.write((const char *)(data.data()), sizeof(double) * totalsz);

	return stream.good();

}

bool JacobianSphereGrid::ImportBinaryData( std::istream& stream, std::string &header ) {
	unsigned int versionNum = 0;
	unsigned int numBytes = 0;

	// Disabled, unnecessary
	//stream.seekg(0, ios::beg);

	char desc[2] = {0};
	unsigned int offset = 0;

	if(stream.peek() == '#') {
		stream.read(desc, 2);
		if(desc[1] == '@') { // We have an offset
			stream.read((char *)&offset, sizeof(unsigned int));
			std::string del;
			getline(stream, del);
		} else if(desc[1] != '\n') {
			std::string tmphead;
			getline(stream, tmphead);
			header.append(tmphead + "\n");
		}
	}

	// Read the header
	std::string head;
	while(stream.peek() == '#') {
		getline(stream, head);
		header.append(head + "\n");
	}

	if(offset > 0) {
		// Skip the header
		stream.seekg(offset, ios::beg);
	} else {
	}

	// Version
	stream >> versionNum;

	// Check if a binary file
	if(/*versionNum != 4 ||*/ versionNum < 10)
		return false;

	if(versionNum < 13)	// Decide what to do later
		return false;

	stream >> numBytes;
	if(numBytes != sizeof(std::complex<double>))
		return false;

	// Delta Q
	unsigned int tmpGridSize, tmpExtras;
	stream >> tmpGridSize;
	if(versionNum > 10)
		stream >> tmpExtras;

	if(actualGridSize != tmpGridSize)
		return false;
	if(versionNum > 10 && Extras != tmpExtras)
		return false;

	gridSize = (actualGridSize - Extras) * 2;

	// Ignore the newline
	stream.ignore(100, '\n');
	stream.read((char *)&(stepSize), sizeof(double));
//	stream.read((char *)&(existingRotation), sizeof(double)*9);
	return InnerImportBinaryData(stream);

}

void JacobianSphereGrid::Add( Grid* other, double scale ) {
	JacobianSphereGrid *rhs = dynamic_cast<JacobianSphereGrid*>(other);
	if(!rhs)
		return;

	if(GetSize() != rhs->GetSize())
		return;

#pragma omp parallel for
	for(long long pos = 0; pos < totalsz; pos++) {
		data[pos] += scale * rhs->data[pos];
	}

}

void JacobianSphereGrid::Scale( double scale ) {
	for(long long pos = 0; pos < totalsz; pos++)
		data[pos] *= scale;
}

bool JacobianSphereGrid::Validate() const {
	volatile bool res = true;

#pragma omp parallel for schedule(dynamic, totalsz / (omp_get_num_procs() * 8))
	for(long long pos = 0; pos < totalsz; pos++) {
		if(!res)
			continue;
		if(data[pos] != data[pos] || 
			data[pos] == std::numeric_limits<double>::infinity() ||
			data[pos] == -std::numeric_limits<double>::infinity() )
		{
			res = false;
			std::cout << "BAD! data[" << pos << "] = " << data[pos] << std::endl;
		}
	}
	if(!res) {
		std::cout << "Grid contains invalid numbers!" << std::endl;
		throw backend_exception(ERROR_GENERAL, "Grid contains invalid numbers!");
	}
	return res;
}

bool JacobianSphereGrid::IndicesToVectors( unsigned short xi, unsigned short yi, unsigned short zi, OUT double &r, OUT double &thet, OUT double &phi ) {
	r = double(xi) * stepSize;
	thet = M_PI  * double(yi) / double(thetaDivisions * xi);	// +1 -1
	phi  = M_2PI * double(zi) / double(phiDivisions * xi);
	return true;

}

double * JacobianSphereGrid::GetPointerWithIndices() {
	// Input (in hex form)	// Numbers are indices and max indices
	// real part = [ THTH DYDY PHPH DZDZ ]
	// imag part = [ 0000 0000 0000 RRRR ]

	u64 *dataAsULL = (u64 *)data.data();

	size_t indexCtr = 0;

	const unsigned short dimx = GetDimX();
	for(unsigned short ri = 0; ri < dimx; ri++) {
		const unsigned short dimy = GetDimY(ri);

		for(unsigned short ti = 0; ti < dimy; ti++) {
			const unsigned short dimz = GetDimZ(ri, ti);

			for(unsigned short pi = 0; pi < dimz; pi++) {
				dataAsULL[indexCtr++] = (((u64)ti << 48) | ((u64)dimy << 32) | ((u64)pi << 16) | (u64)dimz);
				dataAsULL[indexCtr++] = (u64)ri;
			}
		}
	}

	return data.data();
}

bool JacobianSphereGrid::RunAfterCache() {
	data = Eigen::ArrayXd();
	interpolantCoeffs = Eigen::ArrayXd();
// 	if(data.size()) {
// // 		delete data;
// // 		data = NULL;
// 	}
// 	if(interpolantCoeffs) {
// 		delete interpolantCoeffs;
// 		interpolantCoeffs = NULL;
// 	}
	return true;
}

bool JacobianSphereGrid::RunAfterReadingCache() {
	CalculateSplines();
	return true;
}

bool JacobianSphereGrid::RunBeforeReadingCache() {
	try {
		data.resize(totalsz);
		//data = new double[totalsz];		
	} catch(...) {
		throw backend_exception(ERROR_INSUFFICIENT_MEMORY);
	}

	return true;
}

bool JacobianSphereGrid::RotateGrid( double thet, double ph, double psi ) {
	// TODO::JacobianSphereGrid
	return false;
}

long long JacobianSphereGrid::IndexFromIndices( int qi, long long ti, long long pi ) const {
	if(qi == 0)
		return 0;
	qi--;
	long long base = ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;
	return (base + ti * phiDivisions * (qi+1) + pi + 1);	// The +1 is for the origin
}

//This function find for each index (cell) in the grid what should be the values of (q,phi,theta) in this cell
void JacobianSphereGrid::IndicesFromIndex( long long index, int &qi, long long &ti, long long &pi ) {
	// Check the origin
	if(index == 0) {
		qi = 0;
		ti = 0;
		pi = 0;
		return;
	}

	long long bot, rem;
	// Find the q-radius
	long long lqi = icbrt64((3*index)/(thetaDivisions*phiDivisions));
	bot = (lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	if(index > bot )
		lqi++;
	lqi--;
	bot =(lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	lqi++;
	qi = (int)lqi;
	rem = index - bot - 1;
	// Find the theta and phi radii
	ti = rem / (phiDivisions*qi);
	pi = rem % (phiDivisions*qi);

}

void JacobianSphereGrid::CalculateSplines() {
// 	if(interpolantCoeffs)
// 		delete interpolantCoeffs;
	try {
//		interpolantCoeffs = new double[totalsz];
		interpolantCoeffs.resize(totalsz);
	} catch (...) {

		throw backend_exception(ERROR_INSUFFICIENT_MEMORY);
	}


	std::cout << "Starting splines...";

    auto t1 = std::chrono::high_resolution_clock::now();

#define ORIGINAL_METHOD
#ifdef ORIGINAL_METHOD
	//cpu.start();
	//start_time.wall = cpu.elapsed().wall;
	interpolantCoeffs[0] = interpolantCoeffs[1] = 0.0f;
	// The [0] is the origin and doesn't need interpolation
#pragma omp parallel for schedule(dynamic, actualGridSize / (omp_get_num_procs() * 4))
	for(int i = 1; i <= actualGridSize; i++) {
		long long thetaRange = thetaDivisions * i + 1;
		for(long long j = 0; j < thetaRange; j++) {
			long long pos = IndexFromIndices(i, j, 0);
#ifdef _DEBUG
			//std::cout << pos << ":\t(" << i << ",\t" << j << ")\t" << i * phiDivisions <<std::endl;
#endif

			pos *= 2;
			EvenlySpacedPeriodicSplineSolver((std::complex<double>*)(&data[pos]), (std::complex<double>*)(&interpolantCoeffs[pos]), i * phiDivisions);
		}
	}
    auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << " Done! Took "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}

//bool JacobianSphereGrid::ExportThetaPhiPlane( std::string outFileName, long long index ) {
//	fs::path pathName(outFileName);
//	pathName = fs::system_complete(outFileName);
//	std::wstring a1, a2, a3;
//	a1 = pathName.leaf().wstring();
//	a2 = pathName.parent_path().wstring();
//
//	if(!fs::exists(pathName.parent_path())) {
//		std::string strD = pathName.parent_path().string();
//		boost::system::error_code er;
//		if(!fs::create_directories(pathName.parent_path(), er) ) {
//			std::cout << "\nError code: " << er << "\n";
//			while(!fs::exists(pathName.parent_path())) {
//				pathName = pathName.parent_path();
//			}
//			pathName = fs::wpath(pathName.string() + "ERROR_CREATING_DIR");
//			{fs::ofstream f(pathName);}
//			return false;
//		}
//	}
//
//	long long base;
//	long long lqi = index-1;
//	base = 1 + (lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
//// 	if( index > base )
//// 		lqi++;
//// 	lqi--;
//// 	base = 1 + (lqi*(28 + lqi*(60 + 32*lqi))) / 3;
//	lqi++;
//	long long len;
//	if(index == 0) {
//		base = 0;
//		len = 2 * 1;
//	} else {
//		len = 2 * ((thetaDivisions * lqi + 1) * phiDivisions * lqi);
//	}
//	fs::fstream writeFile;
//	writeFile.open(pathName, /*ios::binary |*/ ios::out);
//	if(writeFile.is_open()) {
//		if(data.size() > 1) {
//			//writeFile.write((const char *)(data + 2*base), sizeof(double) * len);
//			for(int i = 0; i < thetaDivisions * lqi + 1; i++) {
//				for(int j = 0; j < phiDivisions * lqi; j++) {	// Just the real part for now
//					writeFile << *(data.data() + 2*(base + i * phiDivisions * lqi + j)) << " ";
//				}
//				writeFile << std::endl;
//			}
//		}
//		writeFile.close();
//	}
//
//	return true;
//}

double *JacobianSphereGrid::GetInterpolantPointer() {
	return interpolantCoeffs.data();
}

void JacobianSphereGrid::DebugMethod() {
#ifdef _DEBUG__875
	long long pp = 1350;
	std::cout << "data["<< pp << "] = " << data[pp] << " data["<< pp+1 << "] = " << data[pp+1] << std::endl;
	std::cout << "interpolantCoeffs["<< pp << "] = " << interpolantCoeffs[pp] << 
		" interpolantCoeffs["<< pp+1 << "] = " << interpolantCoeffs[pp+1] << std::endl;
#endif
}

void JacobianSphereGrid::IndicesFromRadians( const u16 ri, const double theta, const double phi,
											long long &tI, long long &pI, long long &base, double &tTh, double &tPh ) const {
	//this function receives index i plus phi and theta and returns indices j, k, plus base, tTh, and tPh. in avi's words:
	//base is the index of the first element in the theta-phi plane
	//IIRC, `tTh` and `tPh`are the fractions(`[0..1)`) along each axis to be used for interpolation
	
	// Determine the first cell using ri
	int qi = ri - 1;
	base = 1 + ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;

	// Determine the lowest neighbors coordinates within the plane
	int phiPoints = phiDivisions * ri;
	int thePoints = thetaDivisions * ri;
	double edge = M_2PI / double(phiPoints);

	tI = (theta / M_PI) * double(thePoints);
	pI = (phi  / M_2PI) * double(phiPoints);

	// The value [0, 1] representing the location ratio between the two points
	tTh = (theta / edge) - tI; //fmod(theta, edge) / edge;
	tPh = (phi   / edge) - pI; //fmod(phi, edge) / edge;

	if(fabs(tTh) < 1.0e-10)
		tTh = 0.0;
	if(fabs(tPh) < 1.0e-10)
		tPh = 0.0;
	assert(tTh >= 0.0);
	assert(tPh >= 0.0);
	assert(tTh <= 1.0000000001);
	assert(tPh <= 1.0000000001);

	//pI = (pI == phiPoints) ? 0 : pI;
	if(pI == phiPoints) {
		assert(tPh <= 0.000001);
		pI = 0;
	}
}

#endif
