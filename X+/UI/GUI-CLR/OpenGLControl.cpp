#include "OpenGLControl.h"

namespace GUICLR {
	System::Void OpenGLWidget::Render(System::Void) {
		MakeCurrent();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear screen and depth buffer
		glLoadIdentity();									// Reset the current modelview matrix
		
		RenderGLScene();
	}
};

