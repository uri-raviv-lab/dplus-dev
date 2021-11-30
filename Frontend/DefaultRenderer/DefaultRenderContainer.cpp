#define EXPORTER
#include "RendererContainer.h"

#include "PreviewRenders.h"
#include "ModelRenders.h"
#include "SymmetryRenders.h"

int GetNumModelRenderers() {
	return 27;
}

previewRenderFunc GetModelPreview(int index) {
	switch(index) {
		default:
			return NULL;

		case 0:
			return &CylindricalPreviewScene;
		case 1:
			return &CylindroidPreviewScene;
		case 2:
			return &SphericalPreviewScene;
		case 3:
			return &CuboidPreviewScene;
		case 4:
			return &SlabPreviewScene;
		case 5:
			return &SlabPreviewScene;
		case 6:
			return &HelixPreviewScene;
		case 7:
			return &DelixPreviewScene;
		// From here on, the models are not reachable from within the OpeningWindow
		case 8:
			return &SlabPreviewScene;
		case 9:
			return &SlabPreviewScene;
		case 10:
			return &SphericalPreviewScene;
		case 11:
			return &CylindricalPreviewScene;
		case 12:
			return &SphericalPreviewScene;
		case 13:
			return NULL;
		case 14:
			return &CylindricalPreviewScene;
		case 15:
			return &CylindricalPreviewScene;
		case 16:
			return &HelixPreviewScene;
		case 17:
			return &DelixPreviewScene;
		case 18:
			return &CylindroidPreviewScene;
		// The rest are non-FF models
	}
}

renderFunc GetModelRenderer(int index) {
	// This serves as an abstract factory of sorts
	switch(index) {
		default:
			return NULL;
		case 0:
			return &RenderCylindricalModel;
		case 1:
			return &RenderCylindroid;
		case 2:
			return &RenderSphericalModel;
		case 3:
			return NULL;
		case 4:
			return &RenderSlabModel;
		case 5:
			return &RenderSlabModel;
		case 6:
			return &RenderHelixModel;
		case 7:
			return &RenderDelixModel;
		// From here on, the models are not reachable from within the OpeningWindow
		case 8:
			return &RenderSlabModel;
		case 9:
			return &RenderSlabModel;
		case 10:
			return &RenderSphericalModel;
		case 11:
			return &RenderCylindricalModel;
		case 12:
			return &RenderSphericalModel;
		case 13:
			return NULL;
		case 14:
			return &RenderCylindricalModel;
		case 15:
			return &RenderCylindricalModel;
		case 16:
			return &RenderHelixModel;
		case 17:
			return &RenderDelixModel;
		case 18:
			return &RenderCylindroid;

		// The rest are non-FF models
	}

	return NULL;
}

symmetryRenderFunc GetSymmetryRenderer(int index) {
	switch(index) {
	default:
		return NULL;
	case 25:
		return &RenderGridSymmetry;
	case 26:
		return &RenderManualSymmetry;
	}
}
