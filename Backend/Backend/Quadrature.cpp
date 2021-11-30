#include "Quadrature.h"

QuadFunc Quadrature = GaussLegendre2D;

int defaultQuadRes = DEF_QUAD_RES;

EXPORTED_BE void SetQuadResolution(int res) {
	defaultQuadRes = res;
}

void ClassifyQuadratureMethod(QuadratureMethod method) {
	switch(method) {
	default:
	case QUAD_GAUSSLEGENDRE:
		Quadrature = GaussLegendre2D;
		break;
	case QUAD_MONTECARLO:
		Quadrature = MonteCarlo2D;
		break;
	case QUAD_SIMPSON:
		Quadrature = SimpsonRule2D;
		break;
	}
}

