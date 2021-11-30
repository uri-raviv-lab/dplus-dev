//

#include "ModelUI.h"

#include <map>
#include <string>

#include <windows.h>

#define DEFAULT_CONTAINER L"xplusmodels"

#ifdef _WIN32
#define CONTAINER_SUFFIX L".dll"
#else
#define CONTAINER_SUFFIX L".so"
#endif

typedef int (*qCountFunc)();
typedef renderFunc (*getRenderFunc)(int ind);
typedef previewRenderFunc (*getPreviewFunc)(int ind);
typedef symmetryRenderFunc (*getSymmetryFunc)(int ind);


// GLOBAL containing a temporary array of opened containers
static std::map<std::wstring, HMODULE> s_containers;

// Helper function
static HMODULE GetRenderContainer(const wchar_t *container) {
	std::wstring scon, scon2;
	if(!container)
		scon2 = scon = DEFAULT_CONTAINER;
	else
		scon2 = scon = container;

	scon += CONTAINER_SUFFIX;
	scon2 += L".xrn";

	HMODULE hMod = NULL;
	if(s_containers.find(scon) != s_containers.end()) {
		// Module already loaded
		hMod = s_containers[scon];
	} else if(s_containers.find(scon2) != s_containers.end()) {
		// Module already loaded
		hMod = s_containers[scon2];
	} else {
		// Module not loaded
		hMod = LoadLibraryW(scon.c_str());
		if(hMod) {
			if(GetProcAddress(hMod, "GetNumModelRenderers") != NULL)
				s_containers[scon] = hMod;
			else {
				FreeLibrary(hMod);
				hMod = NULL;
			}
		}

		if(!hMod) { // Try second type
			hMod = LoadLibraryW(scon2.c_str());	
			if(hMod) {
				if(GetProcAddress(hMod, "GetNumModelRenderers") != NULL)
					s_containers[scon2] = hMod;
				else {
					FreeLibrary(hMod);
					hMod = NULL;
				}
			}
		}
	}

	return hMod;
}

ModelRenderer::ModelRenderer(const wchar_t *container, int index) : 
		renderFunction(NULL), previewFunction(NULL) {
	if(index < 0)
		return;
	HMODULE hMod = GetRenderContainer(container);
	if(!hMod)
		return;

	// Retrieve number of models and verify index
	qCountFunc qRenCount = (qCountFunc)GetProcAddress(hMod, "GetNumModelRenderers");
	if(!qRenCount)
		return;
	int numRenderers = qRenCount();
	if(index >= numRenderers)
		return;

	// Retrieve renderer and preview functions
	getRenderFunc  gRenderer = (getRenderFunc)GetProcAddress(hMod, "GetModelRenderer");
	getPreviewFunc gPreview  = (getPreviewFunc)GetProcAddress(hMod, "GetModelPreview");
	getSymmetryFunc gSymmetry  = (getSymmetryFunc)GetProcAddress(hMod, "GetSymmetryRenderer");
	if(!gRenderer || !gPreview || !gSymmetry)
		return;

	renderFunction = gRenderer(index);
	previewFunction = gPreview(index);
	symmetryFunction = gSymmetry(index);
}

renderFunc ModelRenderer::GetRenderer() {
	return renderFunction;
}

previewRenderFunc ModelRenderer::GetPreview() {
	return previewFunction;
}

symmetryRenderFunc ModelRenderer::GetSymmetryRenderer() {
	return symmetryFunction;
}
