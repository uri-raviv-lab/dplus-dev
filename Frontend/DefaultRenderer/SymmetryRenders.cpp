#include "SymmetryRenders.h"
#include "Eigen/Core"

#define _USE_MATH_DEFINES
#include  <math.h>

using namespace Eigen;

std::vector<LocationRotation> RenderGridSymmetry(const paramStruct& p) {	
	double da = p.params[0][0].value, db = p.params[0][1].value, dc = p.params[0][2].value; // Distance
	double alpha = p.params[1][0].value, beta = p.params[1][1].value, gamma = p.params[1][2].value; // Angle
	int Na = (int)(p.params[2][0].value + 0.1), Nb = (int)(p.params[2][1].value + 0.1), Nc = (int)(p.params[2][2].value + 0.1); // Repetitions

	double sa, sb, sc, ca, cb, cc, cat;

	sa = sin(alpha * M_PI / 180.0);
	sb = sin(beta * M_PI / 180.0);
	sc = sin(gamma * M_PI / 180.0);
	ca = cos(alpha * M_PI / 180.0);
	cb = cos(beta * M_PI / 180.0);
	cc = cos(gamma * M_PI / 180.0);
	
	cat = (ca - cb * cc)/(sb * sc);
	//calculation of Cartesian coordinates
	Eigen::Vector3d va(da,			0.0,			0.0),
					vb(db * cc,		db * sc,		0.0), 
					vc(dc * cb,		dc * sb * cat,	dc * sb * sin(acos(cat)));

	// DEBUG: CHECK NORMS
	double nor1 = va.norm(),
		nor2 = vb.norm(),
		nor3 = vc.norm();

	nor3 = vc.x();

	
	std::vector<LocationRotation> result;

	result.reserve(Na * Nb * Nc);

	for(int i = 0; i < Na; i++) {
		for(int j = 0; j < Nb; j++) {
			for(int k = 0; k < Nc; k++) {
// 				result.push_back(LocationRotation(double(i) * va.x() + double(j) * va.y() + double(k) * va.z(),
// 								 double(i) * vb.x() + double(j) * vb.y() + double(k) * vb.z(),
// 								 double(i) * vc.x() + double(j) * vc.y() + double(k) * vc.z() )
 				result.push_back(LocationRotation(
					double(i) * va.x() + double(j) * vb.x() + double(k) * vc.x(),
					double(i) * va.y() + double(j) * vb.y() + double(k) * vc.y(),
					double(i) * va.z() + double(j) * vb.z() + double(k) * vc.z() )
					);
			}
		}
	}

	return result;
}

std::vector<LocationRotation> RenderManualSymmetry(const paramStruct& p) {
	std::vector<LocationRotation> result (p.layers);

	for(int i = 0; i < p.layers; i++) {
		result[i] = LocationRotation(p.params[0][i].value, p.params[1][i].value, p.params[2][i].value,
									 p.params[3][i].value, p.params[4][i].value, p.params[5][i].value);
	}

	return result;
}

