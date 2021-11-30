#ifndef __RENDERER_CONTAINER_H
#define __RENDERER_CONTAINER_H

////////////////////////////////////////////////////////////////////////
// Prelude:                                                           //
// This .h file has to be implemented by every DLL renderer container //
////////////////////////////////////////////////////////////////////////

#include "Common.h" // For Parameter and function definitions

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

#ifdef _WIN32
	// Since std::string is a C++ type and we are exporting C-type declarations,
	// we disable the "C++ type in C declaration" warning (so, so hacky)
	#pragma warning(push)
	#pragma warning(disable: 4190)

	#ifdef __cplusplus    // If used by C++ code, 
	extern "C" {          // we need to export the C interface
	#endif
#endif

// Returns the number of models in this container
EXPORTED int GetNumModelRenderers();

// Returns the model's preview rendering procedure (i.e., the one shown in the 
// opening window). Returns NULL if index is incorrect.
EXPORTED previewRenderFunc GetModelPreview(int index);

// Returns the model's rendering procedure (i.e., the 3D preview in the fitter 
// window). Returns NULL if index is incorrect.
EXPORTED renderFunc GetModelRenderer(int index);

// Returns the symmetry's rendering procedure, which generates the points
// and orientations for drawing the sub-amplitudes in the symmetry.
EXPORTED symmetryRenderFunc GetSymmetryRenderer(int index);

#ifdef _WIN32
	#ifdef __cplusplus
	}
	#endif

	#pragma warning(pop)
#endif

#endif
