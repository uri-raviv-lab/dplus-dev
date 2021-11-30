
#include "ModelRenders.h"
#include "GLPreview.h"

// TODO::Render3D: Make these drawings physically accurate (and in scale with the PDBs and the units)

void RenderSphericalModel(const paramStruct &p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	std::vector<float> rad (p.layers);
	std::vector<float> ed (p.layers);

	float cumRad = 0.0f;
	for(int i = 0; i < p.layers; i++)
		cumRad += float(p.params[0][i].value);


	DrawGLNLHollowSphere(cumRad, float(p.params[1][p.layers - 1].value), lod, bNoColor);
}

void RenderSlabModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor){
	std::vector<float> rad(p.layers), ed(p.layers);

	for(int i = 0; i < p.layers; i++) {
		rad[i] = (float)p.params[0][i].value;
		ed[i]  = (float)p.params[1][i].value;
	}

	float x_width = (float)p.extraParams[2].value;
	float y_width = (float)p.extraParams[3].value;
	if(profile.type == SYMMETRIC)
		DrawGLNLayeredSlabs(rad.data(), ed.data(), 0.0f, x_width, y_width, p.layers, lod, bNoColor);
	if (profile.type == ASYMMETRIC)
	{

		float height = 0;
		if (true)
		{	// This should be enabled if/when we decide to move the drawing to the center.
			for (const auto &r : rad)
				height += r;
//			height *= 0.5; // This is not needed as DrawGLNLayeredAsymSlabs eventually halves the height
		}
		DrawGLNLayeredAsymSlabs(rad.data(), ed.data(), height, x_width, y_width, p.layers, lod, bNoColor);
	}
}

void RenderGaussianSlabModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	// TODO::OneStepAhead Draw a better (Gaussian) profile
	RenderSlabModel(p, profile, lod, bNoColor);
}
void RenderCuboidModel(const paramStruct &p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	DrawGLRectangular((float)p.params[1][0].value, lod, bNoColor);
}

void RenderHelixModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	std::vector<double> offset, ed, cross_section;
	for (int i = 0; i < p.params[1].size(); i++)
	{
		offset.push_back(p.params[0][i].value);
		ed.push_back(p.params[1][i].value);
		cross_section.push_back(p.params[2][i].value);
	}
	
	DrawGLNHelix(offset.data(), ed.data(), cross_section.data(), offset.size(),
		p.extraParams[3].value, p.extraParams[2].value, p.extraParams[4].value,
		lod, bNoColor);
}

void RenderDelixModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	// Get the electron density of the first Delix and display that
	if (p.params[1].size() > 1)
		;// DrawGLHelix(p.params[1][1].value, lod, bNoColor);
	// We can draw a better one: multiple helices; spheres
}

void RenderCylindricalModel(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	std::vector<double> rad (p.layers);
	std::vector<double> ed (p.layers);

	for(int i = 0; i < p.layers; i++) {
		rad[i] = p.params[0][i].value;
		ed[i]  = p.params[1][i].value;
	}

	DrawGLNLayeredHC(&rad[0], p.extraParams[2].value, &ed[0], p.layers, lod, bNoColor);
}

void RenderCylindroid(const paramStruct &p, const EDProfile& profile, LevelOfDetail lod, bool bNoColor) {
	std::vector<double> rad (p.layers);
	std::vector<double> ed (p.layers);

	for(int i = 0; i < p.layers; i++) {
		rad[i] = p.params[0][i].value;
		ed[i]  = p.params[1][i].value;
	}

	DrawGLNLayeredCylindroid(&rad[0], p.extraParams[2].value, &ed[0], p.layers, lod, bNoColor);
}
