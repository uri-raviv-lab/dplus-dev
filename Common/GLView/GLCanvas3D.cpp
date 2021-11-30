#include "StdAfx.h"
#include "GLCanvas3D.h"
#include "GLGraphics3D.h"
#include "Utility.h"
#include "Camera.h"

namespace GLView {

	GLCanvas3D::GLCanvas3D()
	{
		// Event handlers
		if(!this->DesignMode)
		{
			this->Resize += gcnew System::EventHandler(this, &GLCanvas3D::ControlResize);
			this->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas3D::ControlMouseDown);
			this->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas3D::ControlMouseMove);
			this->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas3D::ControlMouseUp);
			this->MouseWheel += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas3D::ControlMouseWheel);
			this->MouseDoubleClick += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas3D::ControlMouseDoubleClick);
		}

		// Set control styles
		this->SuspendLayout();
		this->Name = L"GLCanvas3D";
		this->Size = System::Drawing::Size(300, 300);
		this->SetStyle(ControlStyles::DoubleBuffer, false);
		this->SetStyle(ControlStyles::AllPaintingInWmPaint, true);
		this->SetStyle(ControlStyles::UserPaint, true);
		this->SetStyle(ControlStyles::Opaque, true);
		//this->SetStyle(ControlStyles::ResizeRedraw, true);
		this->Cursor = Windows::Forms::Cursors::Default;
		this->ResumeLayout(false);

		// Set property defaults
		AllowZoomAndPan = true;
		BackColor = Drawing::Color::White;
		mPanning = false;
		mOrbiting = false;
		mSelecting = false;
		mCamera = gcnew Camera(Point3D(-5, -5, 5), Point3D(0, 0, 0));
		DrawFloor = true;
		mFloorColor = Drawing::Color::Beige;
		mGridColor = Drawing::Color::Bisque;
		mSelectionRectColor = Drawing::Color::LightSkyBlue;
		ShowAxis = true;
		ShowConstantSizedAxis = false;
		ShowCornerAxis = true;
		mAntiAlias = true;

		mOrigin = Point3D(0, 0, 0);
		mSize = 1.0f;

		// Set selected objects
		mSelectedObjects = gcnew List<Object ^>();

		if(!this->DesignMode)
		{
			// Get the device context
			mhDC = GetDC((HWND)this->Handle.ToPointer());

			// Choose a pixel format
			PIXELFORMATDESCRIPTOR pfd = {
				sizeof(PIXELFORMATDESCRIPTOR),	// size
				1,								// version
				PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,	// flags
				PFD_TYPE_RGBA,					// pixel type
				32,								// color bits
				0, 0, 0, 0, 0, 0, 0, 0,			// RGBA bits and shifts
				0,								// accumulation buffer bits
				0, 0, 0, 0,						// accumulation buffer RGBA bits
				32,								// depth bits
				24,								// stencil bits
				0,								// aux bits
				PFD_MAIN_PLANE,					// layer type
				0,								// reserved
				0, 0, 0							// layer masks
			};

			// Set the format
			int iPixelFormat = ChoosePixelFormat(mhDC, &pfd);
			SetPixelFormat(mhDC, iPixelFormat, &pfd);
			mIsAccelerated = !(pfd.dwFlags & PFD_GENERIC_FORMAT);

			// Create the render context
			mhGLRC = wglCreateContext(mhDC);
			wglMakeCurrent(mhDC, mhGLRC);

			// Set the viewport
			glViewport(0, 0, 300, 300);

			// Set OpenGL parameters
			glDisable(GL_LIGHTING);
			glShadeModel(GL_SMOOTH);
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_TRUE);
			glDepthFunc(GL_LEQUAL);
			glEnable(GL_POLYGON_OFFSET_FILL);
			glPolygonOffset(0.0f, 0.5f);
			glEnable(GL_LINE_SMOOTH);

			// Enable lighting
			mLighting = true;
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
			glEnable(GL_COLOR_MATERIAL);
			GLfloat pos[] = { 1, 1, 1 };
			GLfloat amb[] = { 0, 0, 0, 1 };
			GLfloat dif[] = { 1, 1, 1, 1 };
			glLightfv(GL_LIGHT0, GL_POSITION, pos);
			glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
			glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
			glLightfv(GL_LIGHT0, GL_SPECULAR, dif);
			glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, amb);
			GLfloat mspec[] = { 0, 0, 0, 1 };
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mspec);

			// Create the font display lists
			SelectObject(mhDC, (HGDIOBJ)this->Font->ToHfont());
			base = glGenLists(256);
			rasterbase = glGenLists(256);

			wglUseFontOutlines(mhDC, 0, 256, base, 0.0f, 0.0f, WGL_FONT_POLYGONS, NULL);
			wglUseFontBitmaps(mhDC, 0, 256, rasterbase);
		}
	}

	GLCanvas3D::~GLCanvas3D()
	{
		if(!this->DesignMode)
		{
			wglMakeCurrent(NULL, NULL);
			wglDeleteContext(mhGLRC);
			ReleaseDC((HWND)this->Handle.ToPointer(), mhDC);

			// Delete font display lists
			glDeleteLists(base, 256);
			glDeleteLists(rasterbase, 256);
		}
	}

	void GLCanvas3D::OnPaint(System::Windows::Forms::PaintEventArgs^ e) 
	{
		if(this->DesignMode) 
		{
			e->Graphics->FillRectangle(gcnew System::Drawing::SolidBrush(this->BackColor), this->ClientRectangle);
			e->Graphics->DrawString("GLCanvas3D", this->Font, gcnew SolidBrush(this->ForeColor), 10, 10);
			return;
		}

		// Save previous context and make our context current
		bool contextDifferent = (wglGetCurrentContext() != mhGLRC);
		HDC mhOldDC;
		HGLRC mhOldGLRC;
		if(contextDifferent)
		{
			mhOldDC = wglGetCurrentDC();
			mhOldGLRC = wglGetCurrentContext();
			wglMakeCurrent(mhDC, mhGLRC);
		}
	
		// Set the view frustrum
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		int cheight = Math::Max(1, this->ClientRectangle.Height);
		float fwidth = 1.0f * (float)this->ClientRectangle.Width / (float)cheight;		
		glFrustum(-fwidth, fwidth, -1, 1, 1.0f, ViewDistance);

		// Set the camera
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(mCamera->Position.X, mCamera->Position.Y, mCamera->Position.Z, mCamera->Target.X, mCamera->Target.Y, mCamera->Target.Z, mCamera->Up.X, mCamera->Up.Y, mCamera->Up.Z);

		// Create the GLGraphics object
		GLView::GLGraphics3D ^ graphics = gcnew GLView::GLGraphics3D(this, e->Graphics, false);

		// Clear screen		
		glClearColor(((float)BackColor.R) / 255, ((float)BackColor.G) / 255, ((float)BackColor.B) / 255, ((float)BackColor.A) / 255);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(this->DrawFog) {
			GLfloat fogColor[4] = {((float)BackColor.R) / 255, ((float)BackColor.G) / 255, ((float)BackColor.B) / 255, ((float)BackColor.A) / 255};
			glFogi(GL_FOG_MODE, GL_LINEAR);
			glHint(GL_FOG_HINT, GL_DONT_CARE);
			glFogf(GL_FOG_DENSITY, 0.5f);			
			glFogfv(GL_FOG_COLOR, fogColor);
			glFogf(GL_FOG_START, ViewDistance - 5.0f);
			glFogf(GL_FOG_END, ViewDistance);
			glEnable(GL_FOG);
		}

		// Draw the floor
		if(this->DrawFloor)
		{
			glColor4f((float)mFloorColor.R / 256.0f, (float)mFloorColor.G / 256.0f, (float)mFloorColor.B / 256.0f, (float)mFloorColor.A / 256.0f);
			glBegin(GL_QUADS);
			float floorSize = 5.0f * mSize;
			glVertex3f(-floorSize, -floorSize, 0);
			glVertex3f(-floorSize, floorSize, 0);
			glVertex3f(floorSize, floorSize, 0);
			glVertex3f(floorSize, -floorSize, 0);
			glEnd();

			// Draw the grid
			glColor4f((float)mGridColor.R / 256.0f, (float)mGridColor.G / 256.0f, (float)mGridColor.B / 256.0f, (float)mGridColor.A / 256.0f);
			float spacing = Math::Max(mSize / 20.0f, 1.0f);
			glBegin(GL_LINES);
			for(int i = -5; i <= 5; i++)
			{
				glVertex3f(-floorSize, (float)i * spacing, 0);
				glVertex3f(floorSize, (float)i * spacing, 0);
				glVertex3f((float)i * spacing, -floorSize, 0);
				glVertex3f((float)i * spacing, floorSize, 0);
			}
			glEnd();
		}

		// Raise the custom draw event
		Render(this, graphics);

		// Get view properties
		mOrigin = graphics->ModelOrigin();
		mSize = graphics->ModelSize();

		// Draw the axis
		if(ShowAxis)
		{
			float length = Math::Min(1.0f, mSize / 2.0f);

			if (ShowConstantSizedAxis)
				length = mCamera->Distance / 4.f;

			graphics->FillBox(0, 0, 0, length, 0, 0, length / 10.0f, length / 10.0f, Color::Red);
			graphics->FillBox(0, 0, 0, 0, length, 0, length / 10.0f, length / 10.0f, Color::Green);
			graphics->FillBox(0, 0, 0, 0, 0, length, length / 10.0f, length / 10.0f, Color::Blue);

		}

		if(ShowCornerAxis)
		{	
			float length = Math::Min(1.0f, mSize / 2.0f);
			glPushMatrix();

			// No lighting
			bool enLight = glIsEnabled(GL_LIGHTING) ? true : false;
			if(enLight)
				glDisable(GL_LIGHTING);

			// Reset the viewport
			glClear(GL_DEPTH_BUFFER_BIT);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glViewport(ClientSize.Width - 100, 0, 100, 100);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glViewport(ClientSize.Width - 100, 0, 100, 100);		

			// Set the camera rotation
			glRotatef(mCamera->Roll,  0.0, 0.0, 1.0);
			glRotatef(mCamera->Pitch, 1.0, 0.0, 0.0);
			glRotatef(mCamera->Yaw,   0.0, -1.0, 0.0);			

			// Draw the axes
			graphics->FillBox(0, 0, 0, length, 0, 0, length / 10.0f, length / 10.0f, Color::Green);
			graphics->FillBox(0, 0, 0, 0, length, 0, length / 10.0f, length / 10.0f, Color::Blue);
			graphics->FillBox(0, 0, 0, 0, 0, length, length / 10.0f, length / 10.0f, Color::Red);

			//glRasterPos3f(length * 1.1f, 0, 0);
			const auto textColor = Color::White;
			const auto	tenth = length * 0.1f;
			const auto thickness = length * 0.05f;
			// X
			graphics->FillBox(0, -tenth, 1.1*length, 0, tenth, 1.4*length, thickness, thickness, textColor);
			graphics->FillBox(0, tenth, 1.1*length, 0, -tenth, 1.4*length, thickness, thickness, textColor);
			// Y
			graphics->FillBox(1.1*length, 0, 0, 1.25*length, 0, 0,      thickness, thickness, textColor);
			graphics->FillBox(1.25*length, 0, 0, 1.4*length, 0, -tenth, thickness, thickness, textColor);
			graphics->FillBox(1.25*length, 0, 0, 1.4*length, 0, +tenth, thickness, thickness, textColor);
			// Z
			graphics->FillBox(-tenth, 1.4*length, 0, tenth, 1.4*length, 0, thickness, thickness, textColor);
			graphics->FillBox(-tenth, 1.1*length, 0, tenth, 1.4*length, 0, thickness, thickness, textColor);
			graphics->FillBox(-tenth, 1.1*length, 0, tenth, 1.1*length, 0, thickness, thickness, textColor);

			// Set back the viewport and lighting
			glViewport(0, 0, ClientSize.Width, ClientSize.Height);
			if(enLight)
				glEnable(GL_LIGHTING);

			glPopMatrix();
		}

		if(this->DrawFog) {
			glDisable(GL_FOG);
		}
		
		// Draw selection rectangle, if required
		if(mSelecting) {			
			int top = Math::Min(mMouseDown.Y, mLastMouse.Y);
			int bottom = Math::Max(mMouseDown.Y, mLastMouse.Y);
			int left = Math::Min(mMouseDown.X, mLastMouse.X);
			int right = Math::Max(mMouseDown.X, mLastMouse.X);

			// Move to 2D mode			
			glClear(GL_DEPTH_BUFFER_BIT);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, Width, Height, 0, 0, 1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			// Draw selection rectangle
			glBegin(GL_QUADS);
			
			glColor4f(mSelectionRectColor.R / 255.0f, mSelectionRectColor.G / 255.0f, mSelectionRectColor.B / 255.0f, 0.6f);
			glVertex2i(left, top);
			glVertex2i(right, top);
			glVertex2i(right, bottom);
			glVertex2i(left, bottom);

			glEnd();
			glDisable(GL_BLEND);
		}

		// Finish
		glFinish();

		// Swap buffers
		SwapBuffers(mhDC);

		// Restore previous context
		if(contextDifferent)
		{
			wglMakeCurrent(mhOldDC, mhOldGLRC);
		}

		// Raise the render done event
		RenderDone(this);
	}

	void GLCanvas3D::OnPaintBackground(System::Windows::Forms::PaintEventArgs^ e) 
	{
	}

	System::Void GLCanvas3D::ControlResize(System::Object^ sender, System::EventArgs^ e)
	{
		ResetViewport();
		Invalidate();
	}

	System::Void GLCanvas3D::ResetViewport()
	{
		// Save previous context and make our context current
		bool contextDifferent = (wglGetCurrentContext() != mhGLRC);
		HDC mhOldDC;
		HGLRC mhOldGLRC;
		if(contextDifferent)
		{
			mhOldDC = wglGetCurrentDC();
			mhOldGLRC = wglGetCurrentContext();
			wglMakeCurrent(mhDC, mhGLRC);
		}

		// Reset the current viewport
		glViewport(0, 0, ClientSize.Width, ClientSize.Height);

		// Restore previous context
		if(contextDifferent)
		{
			wglMakeCurrent(mhOldDC, mhOldGLRC);
		}
	}

	System::Void GLCanvas3D::ControlMouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Right) && AllowZoomAndPan && !(mOrbiting))
		{
			mOrbiting = true;
			mLastMouse = e->Location;
			this->Cursor = Windows::Forms::Cursors::NoMove2D;
		} 
		else if ((e->Button == Windows::Forms::MouseButtons::Middle) && AllowZoomAndPan && !(mPanning))
		{
			mPanning = true;
			mLastMouse = e->Location;
			this->Cursor = Windows::Forms::Cursors::SizeAll;
		}		
		else if ((e->Button == Windows::Forms::MouseButtons::Left) && MouseSelect && !(mSelecting))
		{
			mSelecting = true;
			mLastMouse = e->Location;
			mMouseDown = e->Location;
			this->Cursor = Windows::Forms::Cursors::Cross;
		}
	}

	System::Void GLCanvas3D::ControlMouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Right) && (mOrbiting))
		{
			float deltax = (float)mLastMouse.X - (float)e->Location.X;
			float maxdeltax = Math::Max(1.0f, (float)this->ClientRectangle.Width / 2.0f);
			float deltaanglex = deltax / maxdeltax * 180.0f / 2.0f;
			mCamera->Yaw += deltaanglex;
			float deltay = (float)mLastMouse.Y - (float)e->Location.Y;
			float maxdeltay = Math::Max(1.0f, (float)this->ClientRectangle.Height / 2.0f);
			float deltaangley = deltay / maxdeltay * 90.0f / 2.0f;
			mCamera->Pitch -= deltaangley;
			mLastMouse = e->Location;
			Invalidate();
		} 
		else if ((e->Button == Windows::Forms::MouseButtons::Middle) && (mPanning))
		{
			float deltax = (mLastMouse.X - e->Location.X) / 10.0f;			
			float deltay = (mLastMouse.Y - e->Location.Y) / 10.0f;

			// Compute vertical pan
			Point3D up = mCamera->Up;
			up.X *= deltay; up.Y *= deltay; up.Z *= deltay;
			mCamera->Pan(-up.X, -up.Y, -up.Z);

			// Compute horizontal pan
			Point3D right = mCamera->Right;
			right.X *= deltax; right.Y *= deltax; right.Z *= deltax;
			mCamera->Pan(-right.X, -right.Y, -right.Z);

			mLastMouse = e->Location;
			Invalidate();
		}
		else if ((e->Button == Windows::Forms::MouseButtons::Left) && (mSelecting))
		{
			mLastMouse = e->Location;
			Invalidate();
		}
	}

	System::Void GLCanvas3D::ControlMouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Right) && mOrbiting)
		{
			mOrbiting = false;
			this->Cursor = Windows::Forms::Cursors::Default;
			Invalidate();
		}
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && mPanning)
		{
			mPanning = false;
			this->Cursor = Windows::Forms::Cursors::Default;
			Invalidate();
		}
		if ((e->Button == Windows::Forms::MouseButtons::Left) && mSelecting)
		{
			mSelecting = false;
			this->Cursor = Windows::Forms::Cursors::Default;
			EndSelection(mMouseDown, e->Location, (System::Windows::Forms::Control::ModifierKeys & Keys::Control) == Keys::Control);
			// Call the event
			SelectionChanged(this, gcnew EventArgs());
			Invalidate();
		}
	}

	System::Void GLCanvas3D::ControlMouseWheel(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if (AllowZoomAndPan)
		{
			mCamera->Distance = Math::Max(mSize, mCamera->Distance - (float)Math::Sign(e->Delta) * mSize / 10.0f);
			Invalidate();
		}
	}

	System::Void GLCanvas3D::ResetView()
	{
		mCamera->Pan(mCamera->Target, Point3D(0, 0, 0) );
		//mCamera->Pan(mCamera->Target, mOrigin);	// Original Code
		mCamera->Distance = Math::Max(4.0f, mSize * 2.0f);
		Invalidate();
	}

	System::Void GLCanvas3D::ControlMouseDoubleClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && AllowZoomAndPan)
			ResetView();
	}

	System::Void GLCanvas3D::Deselect() {
		// Clear selection
		mSelectedObjects->Clear();

		// Trigger the event
		SelectionChanged(this, gcnew EventArgs());
		Invalidate();
	}

	GLView::GLGraphics3D ^GLCanvas3D::RenderForSelection() {
		glDisable (GL_DITHER);
		glDisable (GL_LIGHTING);
		
		// Set the view frustum
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		int cheight = Math::Max(1, this->ClientRectangle.Height);
		float fwidth = 1.0f * (float)this->ClientRectangle.Width / (float)cheight;
		glFrustum(-fwidth, fwidth, -1, 1, 1.0f, ViewDistance);

		// Set the camera
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(mCamera->Position.X, mCamera->Position.Y, mCamera->Position.Z, mCamera->Target.X, mCamera->Target.Y, mCamera->Target.Z, mCamera->Up.X, mCamera->Up.Y, mCamera->Up.Z);

		// Create the GLGraphics object
		GLView::GLGraphics3D ^ graphics = gcnew GLView::GLGraphics3D(this, this->CreateGraphics(), true);

		// Clear screen
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Raise the custom draw event
		ColorCodedRender(this, graphics);

		glEnable (GL_DITHER);
		glEnable (GL_LIGHTING);	

		return graphics;
	}

	System::Void GLCanvas3D::EndSelection(Point startLoc, Point endLoc, bool isCtrlClick) {
		int top    = Math::Min(startLoc.Y, endLoc.Y);
		int bottom = Math::Max(startLoc.Y, endLoc.Y);
		int left   = Math::Min(startLoc.X, endLoc.X);
		int right  = Math::Max(startLoc.X, endLoc.X);
		int wid = right - left;
		int hgt = bottom - top;

		if(wid == 0)
			wid = 1;
		if(hgt == 0)
			hgt = 1;

		// Save previous context and make our context current
		bool contextDifferent = (wglGetCurrentContext() != mhGLRC);
		HDC mhOldDC;
		HGLRC mhOldGLRC;
		if(contextDifferent)
		{
			mhOldDC = wglGetCurrentDC();
			mhOldGLRC = wglGetCurrentContext();
			wglMakeCurrent(mhDC, mhGLRC);
		}
		
		// Render the special color-coded version for selection
		GLView::GLGraphics3D ^graphics = RenderForSelection();
				
		// If the control button is held during the click, keep the previous selected objects
		if(!isCtrlClick)
			mSelectedObjects->Clear();


		// Get the size of the window
		GLint viewport[4];		
		glGetIntegerv(GL_VIEWPORT,viewport);

		// Create a buffer for the pixels
		unsigned int *pixels = new unsigned int[wid * hgt];

		// Read the pixels of the rectangle
		glReadPixels(left, viewport[3] - bottom, wid, hgt,
					 GL_RGBA, GL_UNSIGNED_BYTE, (void *)pixels);

		// For each pixel, check its color value
		for(int i = 0; i < wid; i++) {
			for(int j = 0; j < hgt; j++) {	
				unsigned char *pix = (unsigned char *)&pixels[i * hgt + j];
				int glid = (pix[0] / 25)
						 +((pix[1] / 25) * 10)
						 +((pix[2] / 25) * 100);

				if(glid > 0) {
					// If the object is not already on the list, add it
					Object ^obj = graphics->ObjectFromGLID(glid);
					if(mSelectedObjects->IndexOf(obj) < 0)
						mSelectedObjects->Add(obj);
					else if(wid == 1 && hgt == 1 && isCtrlClick)
						// If we clicked (not dragged) with control and the object is selected, deselect it
						mSelectedObjects->Remove(obj);

				}
			}
		}

		delete[] pixels;
		// Selection is DONE

		// Restore previous context
		if(contextDifferent)
		{
			wglMakeCurrent(mhOldDC, mhOldGLRC);
		}
	}

}