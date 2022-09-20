#ifndef __GENERALPDBREADERLIB_H
#define __GENERALPDBREADERLIB_H


#define BOOST_SYSTEM_NO_DEPRECATED
#pragma warning( disable : 4251 )

#pragma once
#include <vector>
#include <map>       
#include <complex>
#include <string>

#include "Common.h"

#define NUMBER_OF_ATOMIC_FORM_FACTORS (208 + 8)
#define ELECTRON_NUMBER_OF_ATOMIC_FORM_FACTORS (208 + 8 + 2)

using std::vector;
using std::string;

#if !(defined __MATHFUNCS_H) && !(defined MATH_SQ)
#define MATH_SQ
template <typename T> inline T sq(T x) { return x * x; }
#endif

#ifdef _WIN32
#ifdef PDBOB_INNER_TEMPLATE
#define EXPORTED_PDBREADER __declspec(dllexport)
#else
#define EXPORTED_PDBREADER __declspec(dllimport)
#endif
#else
#ifdef PDBOB_INNER_TEMPLATE
#define EXPORTED_PDBREADER __attribute__ ((visibility ("default")))
#else
#define EXPORTED_PDBREADER
#endif
#endif

#endif