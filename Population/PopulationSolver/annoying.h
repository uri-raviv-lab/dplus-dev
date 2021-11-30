#include "Eigen/Core"
using namespace Eigen;

#undef EXPORTED
#ifdef _WIN32
	#ifdef EXPORTER
		#define EXPORTED __declspec(dllexport)
	#else
		#define EXPORTED __declspec(dllimport)
	#endif
#else
	#define EXPORTED extern "C"
#endif

#ifndef __CONSSTRUCT
#define __CONSSTRUCT
typedef struct ConsStruct {
	ArrayXi index, link;
	ArrayXd num;

	ConsStruct(int m) : index(ArrayXi::Constant(m, -1)), link(ArrayXi::Constant(m, -1)), num(ArrayXd::Zero(m)) {}
	ConsStruct()  {}
} cons;
#endif


