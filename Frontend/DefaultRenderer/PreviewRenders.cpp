
#include "PreviewRenders.h"
#include "GLPreview.h"

void CylindricalPreviewScene() {
	// Draw a default HC
	double rad[2] = { 0.9, 0.5 };
	double ed[2] = { 0.0, 0.0 };
	DrawGLNLayeredHC(rad, 3.0, ed, 2, LOD_MEDIUM, false);	
}

void CylindroidPreviewScene() {
	DrawGLCylindroid(0.9f, 0.5f, 3.0f, LOD_MEDIUM, false);
}

void SphericalPreviewScene() {
	DrawGLSphere(6.28f, LOD_MEDIUM, false);	
}

void SlabPreviewScene() // So far, they're all the same
{
	DrawGLMembrane(0.2f, 6, 10, 1.0f, LOD_MEDIUM, false);
}

void CuboidPreviewScene() {
	DrawGLRectangular(-1.0, LOD_MEDIUM, false);
}

void HelixPreviewScene() {
	//DrawGLHelix(-1.0, LOD_MEDIUM, false);
}

void DelixPreviewScene() {
	DrawGLMicrotubule(0.1f, 15, LOD_MEDIUM, false);
}
