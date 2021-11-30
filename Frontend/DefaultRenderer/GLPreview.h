#ifndef __GLPREVIEW_H
#define __GLPREVIEW_H

#include "Common.h" // For LevelOfDetail

void DrawGLNLHollowSphere(float rad, float ed, LevelOfDetail lod, bool bNoColor);

void DrawGLSphere(float ed, LevelOfDetail lod, bool bNoColor);

void DrawGLCylindroid(float innerRadius, float outerRadius, float height, LevelOfDetail lod, bool bNoColor);

void DrawGLNLayeredCylindroid(double *rad, double height, double *ed, int n, LevelOfDetail lod, bool bNoColor);

void DrawGLNLayeredHC(double *rad, double height, double *ed, int n, LevelOfDetail lod, bool bNoColor);

void DrawGLMicrotubule(float r, int totalsize, LevelOfDetail lod, bool bNoColor);

void DrawGLMembrane(float r, int height, int size, float headED, LevelOfDetail lod, bool bNoColor);

void DrawGLNHelix(const double* offset, const double* ed, double* cross_section, const int n, const double radius, const double height, const double pitch, const LevelOfDetail lod, const bool bNoColor);

void DrawGLRectangular(float ed, LevelOfDetail lod, bool bNoColor);

void DrawGLNLayeredAsymSlabs(float *rad, float *ed, float height, float x_width, float y_width,
							 int n, LevelOfDetail lod, bool bNoColor);

void DrawGLNLayeredSlabs(float *rad, float *ed, float height, float x_width, float y_width, int n, LevelOfDetail lod, bool bNoColor);

#endif