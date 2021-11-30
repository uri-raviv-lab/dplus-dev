#ifndef __MODEL_RENDERS_H
#define __MODEL_RENDERS_H

#include "Common.h" // For paramStruct

void RenderSphericalModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderSlabModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderGaussianSlabModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderHelixModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderDelixModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderCylindricalModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);
void RenderCylindroid(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor);

#endif
