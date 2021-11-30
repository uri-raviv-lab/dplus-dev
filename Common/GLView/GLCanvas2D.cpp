#include "StdAfx.h"
#include "GLCanvas2D.h"
#include "GLGraphics.h"
#include "GLVertexArray.h"
#include "GLPerformanceTimer.h"

namespace GLView {

	GLCanvas2D::GLCanvas2D()
	{
		// Event handlers
		if(!this->DesignMode)
		{
			this->Resize += gcnew System::EventHandler(this, &GLCanvas2D::ControlResize);
			this->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas2D::ControlMouseDown);
			this->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas2D::ControlMouseMove);
			this->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas2D::ControlMouseUp);
			this->MouseWheel += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas2D::ControlMouseWheel);
			this->MouseDoubleClick += gcnew System::Windows::Forms::MouseEventHandler(this, &GLCanvas2D::ControlMouseDoubleClick);
		}

		// Set control styles
		this->SuspendLayout();
		this->Name = L"GLCanvas2D";
		this->Size = System::Drawing::Size(300, 300);
		this->SetStyle(ControlStyles::DoubleBuffer, false);
		this->SetStyle(ControlStyles::AllPaintingInWmPaint, true);
		this->SetStyle(ControlStyles::UserPaint, true);
		this->SetStyle(ControlStyles::Opaque, true);
		//this->SetStyle(ControlStyles::ResizeRedraw, true);
		this->Cursor = Windows::Forms::Cursors::Cross;
		this->ResumeLayout(false);

		// Set property defaults
		mLimits = Drawing::RectangleF::Empty;
		mZoomFactor = 5.0f / 3.0f;
		AllowZoomAndPan = true;
		mShowGrid = true;
		mShowAxes = true;
		mGridSpacing = 10;
		mMinorGridColor = Drawing::Color::Beige;
		mMajorGridColor = Drawing::Color::Bisque;
		mAxisColor = Drawing::Color::BurlyWood;
		BackColor = Drawing::Color::White;
		SelectionMode = SelectMode::None;
		SelectionColor = Drawing::SystemColors::Highlight;
		ReverseSelectionColor = Drawing::SystemColors::Highlight;
		mDynamicGrid = true;
		mPanning = false;
		mSelecting = false;
		mCameraPosition = PointF(0, 0);
		mAntiAlias = false;

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
			glShadeModel(GL_FLAT);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_TRUE);
			glDepthFunc(GL_LESS);
			glEnable(GL_POLYGON_OFFSET_FILL);
			glPolygonOffset(0.0f, 0.5f);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			glLineStipple(1, 61680);

			// Create the font display lists
			SelectObject(mhDC, (HGDIOBJ)this->Font->ToHfont());
			base = glGenLists(256);
			rasterbase = glGenLists(256);

			wglUseFontOutlines(mhDC, 0, 256, base, 0.0f, 0.0f, WGL_FONT_POLYGONS, NULL);
			wglUseFontBitmaps(mhDC, 0, 256, rasterbase);
		}
	}

	GLCanvas2D::~GLCanvas2D()
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

	void GLCanvas2D::OnPaint(System::Windows::Forms::PaintEventArgs^ e) 
	{
		if(this->DesignMode) 
		{
			e->Graphics->FillRectangle(gcnew System::Drawing::SolidBrush(this->BackColor), this->ClientRectangle);
			e->Graphics->DrawString("GLCanvas2D", this->Font, gcnew SolidBrush(this->ForeColor), 10, 10);
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

		// Set an orthogonal projection matrix
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(mCameraPosition.X - ((float)ClientRectangle.Width) * mZoomFactor / 2, mCameraPosition.X + ((float)ClientRectangle.Width) * mZoomFactor / 2, mCameraPosition.Y - ((float)ClientRectangle.Height) * mZoomFactor / 2, mCameraPosition.Y + ((float)ClientRectangle.Height) * mZoomFactor / 2, -1.0f, 1.0f);

		// Set the model matrix as the current matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Get view bounds
		Drawing::RectangleF bounds = GetViewPort();

		// Create the GLGraphics object
		GLView::GLGraphics ^ graphics = gcnew GLView::GLGraphics(this, e->Graphics);

		// Clear screen
		glClearColor(((float)BackColor.R) / 255, ((float)BackColor.G) / 255, ((float)BackColor.B) / 255, ((float)BackColor.A) / 255);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Draw the grid
		glLoadIdentity();
		if (mShowGrid)
		{
			float mSpacing = mGridSpacing;
			if (mDynamicGrid)
			{
				while (WorldToScreen(SizeF(mSpacing, 0.0f)).Width > 12.0f)
					mSpacing /= 10.0f;

				while (WorldToScreen(SizeF(mSpacing, 0.0f)).Width < 4.0f)
					mSpacing *= 10.0f;
			}
			if (WorldToScreen(SizeF(mSpacing, 0.0f)).Width >= 4.0f)
			{
				int k = 0;
				for (float i = 0; i > bounds.Left; i -= mSpacing)
				{
					if (k <= 1) glColor4ub(mMinorGridColor.R, mMinorGridColor.G, mMinorGridColor.B, mMinorGridColor.A);
					if (k == 10) { glColor4ub(mMajorGridColor.R, mMajorGridColor.G, mMajorGridColor.B, mMajorGridColor.A); k = 0; }
					k++;
					glBegin(GL_LINES);
					glVertex3f(i, bounds.Top, -0.95f);
					glVertex3f(i, bounds.Bottom, -0.95f);
					glEnd();
				}
				k = 0;
				for (float i = 0; i < bounds.Right; i += mSpacing)
				{
					if (k <= 1) glColor4ub(mMinorGridColor.R, mMinorGridColor.G, mMinorGridColor.B, mMinorGridColor.A);
					if (k == 10) { glColor4ub(mMajorGridColor.R, mMajorGridColor.G, mMajorGridColor.B, mMajorGridColor.A); k = 0; }
					k++;
					glBegin(GL_LINES);
					glVertex3f(i, bounds.Top, -0.95f);
					glVertex3f(i, bounds.Bottom, -0.95f);
					glEnd();
				}
				k = 0;
				for (float i = 0; i > bounds.Top; i -= mSpacing)
				{
					if (k <= 1) glColor4ub(mMinorGridColor.R, mMinorGridColor.G, mMinorGridColor.B, mMinorGridColor.A);
					if (k == 10) { glColor4ub(mMajorGridColor.R, mMajorGridColor.G, mMajorGridColor.B, mMajorGridColor.A); k = 0; }
					k++;
					glBegin(GL_LINES);
					glVertex3f(bounds.Left, i, -0.95f);
					glVertex3f(bounds.Right, i, -0.95f);
					glEnd();
				}
				k = 0;
				for (float i = 0; i < bounds.Bottom; i += mSpacing)
				{
					if (k <= 1) glColor4ub(mMinorGridColor.R, mMinorGridColor.G, mMinorGridColor.B, mMinorGridColor.A);
					if (k == 10) { glColor4ub(mMajorGridColor.R, mMajorGridColor.G, mMajorGridColor.B, mMajorGridColor.A); k = 0; }
					k++;
					glBegin(GL_LINES);
					glVertex3f(bounds.Left, i, -0.95f);
					glVertex3f(bounds.Right, i, -0.95f);
					glEnd();
				}
			}
		}

		// Axes
		glLoadIdentity();
        if (mShowAxes)
		{
			glColor4ub(mAxisColor.R, mAxisColor.G, mAxisColor.B, mAxisColor.A);
			glBegin(GL_LINES);
			glVertex3f(0.0f, bounds.Top, -0.94f);
			glVertex3f(0.0f, bounds.Bottom, -0.94f);
			glVertex3f(bounds.Left, 0.0f, -0.94f);
			glVertex3f(bounds.Right, 0.0f, -0.94f);
			glEnd();
		}

		// Raise the custom draw event
		glLoadIdentity();
		Render(this, graphics);

		// Render drawing objects
		glLoadIdentity();
		mLimits = graphics->Render();

		// Draw selection rectangle if in selection mode
		float r;
		glLoadIdentity();
		if (mSelecting)
		{
			Drawing::PointF p1 = ScreenToWorld(mSelPt1.X, mSelPt1.Y);
			Drawing::PointF p2 = ScreenToWorld(mSelPt2.X, mSelPt2.Y);
			switch (SelectionMode)
			{
			case SelectMode::Line:
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINES);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glEnd();
				glDisable(GL_LINE_STIPPLE);
				break;
			case SelectMode::Circle:
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_LOOP);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
				r = (float)Math::Sqrt((p2.Y - p1.Y) * (p2.Y - p1.Y) + (p2.X - p1.X) * (p2.X - p1.X));
				for (float a = 0.0f; a < 2 * (float)Math::PI; a += (float)Math::PI / 40.0f)
				{
					glVertex3f(p1.X + r * (float)Math::Cos(a), 
							   p1.Y + r * (float)Math::Sin(a), 0.0f);
				}
				glEnd();
				glDisable(GL_LINE_STIPPLE);
				break;
			case SelectMode::Rectangle:
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_LOOP);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
				glDisable(GL_LINE_STIPPLE);
				break;
			case SelectMode::ShadedRectangle:
				glBegin(GL_QUADS);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, 32);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
				glBegin(GL_LINE_LOOP);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
				break;
            case SelectMode::ReversableShadedRectangle:
				glBegin(GL_QUADS);
                if(p1.X < p2.X)
				    glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, 32);
                else
				    glColor4ub(ReverseSelectionColor.R, ReverseSelectionColor.G, ReverseSelectionColor.B, 32);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
                if(p1.X > p2.X)	glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_LOOP);
                if(p1.X < p2.X)
    				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
                else
				    glColor4ub(ReverseSelectionColor.R, ReverseSelectionColor.G, ReverseSelectionColor.B, ReverseSelectionColor.A);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
                if(p1.X > p2.X)	glDisable(GL_LINE_STIPPLE);
				break;
            case SelectMode::InvertedShadedRectangle:
				glBegin(GL_QUADS);

				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, 32);
				glVertex3f(bounds.Left, bounds.Top, 0.0f);
				glVertex3f(bounds.Right, bounds.Top, 0.0f);
                glVertex3f(bounds.Right, Math::Min(p1.Y, p2.Y), 0.0f);
				glVertex3f(bounds.Left, Math::Min(p1.Y, p2.Y), 0.0f);

				glVertex3f(bounds.Left, bounds.Bottom, 0.0f);
				glVertex3f(bounds.Right, bounds.Bottom, 0.0f);
                glVertex3f(bounds.Right, Math::Max(p1.Y, p2.Y), 0.0f);
				glVertex3f(bounds.Left, Math::Max(p1.Y, p2.Y), 0.0f);

				glVertex3f(bounds.Left, Math::Min(p1.Y, p2.Y), 0.0f);
				glVertex3f(Math::Min(p1.X, p2.X), Math::Min(p1.Y, p2.Y), 0.0f);
				glVertex3f(Math::Min(p1.X, p2.X), Math::Max(p1.Y, p2.Y), 0.0f);
				glVertex3f(bounds.Left, Math::Max(p1.Y, p2.Y), 0.0f);

				glVertex3f(Math::Max(p1.X, p2.X), Math::Min(p1.Y, p2.Y), 0.0f);
				glVertex3f(bounds.Right, Math::Min(p1.Y, p2.Y), 0.0f);
				glVertex3f(bounds.Right, Math::Max(p1.Y, p2.Y), 0.0f);
				glVertex3f(Math::Max(p1.X, p2.X), Math::Max(p1.Y, p2.Y), 0.0f);
                glEnd();

				glBegin(GL_LINE_LOOP);
				glColor4ub(SelectionColor.R, SelectionColor.G, SelectionColor.B, SelectionColor.A);
				glVertex3f(p1.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p1.Y, 0.0f);
				glVertex3f(p2.X, p2.Y, 0.0f);
				glVertex3f(p1.X, p2.Y, 0.0f);
				glEnd();
				break;
			}	
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

	void GLCanvas2D::OnPaintBackground(System::Windows::Forms::PaintEventArgs^ e) 
	{
	}

	System::Void GLCanvas2D::ControlResize(System::Object^ sender, System::EventArgs^ e)
	{
		ResetViewport();
		Invalidate();
	}

	System::Void GLCanvas2D::ResetViewport()
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

	System::Void GLCanvas2D::ControlMouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && AllowZoomAndPan && !(mSelecting))
		{
			mPanning = true;
			mLastMouse = e->Location;
			this->Cursor = Windows::Forms::Cursors::NoMove2D;
		}
		else if (SelectionMode != SelectMode::None)
		{
            mSelecting = true;
            mSelPt1.X = e->X;
            mSelPt1.Y = e->Y;
			this->Cursor = Windows::Forms::Cursors::Arrow;
		}
	}

	System::Void GLCanvas2D::ControlMouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && (mPanning))
		{
			// Relative mouse movement
			PointF cloc = ScreenToWorld(e->Location);
			PointF ploc = ScreenToWorld(mLastMouse);
			SizeF delta(cloc.X - ploc.X, cloc.Y - ploc.Y);
			mCameraPosition -= delta;
			mLastMouse = e->Location;
			Invalidate();
		}
		else if (mSelecting)
		{
			mSelPt2.X = Math::Min(ClientRectangle.Right - 1, Math::Max(1, e->X));
			mSelPt2.Y = Math::Min(ClientRectangle.Bottom - 1, Math::Max(1, e->Y));
            Invalidate();
		}
	}

	System::Void GLCanvas2D::ControlMouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && mPanning)
		{
			mPanning = false;
			Invalidate();
		}
		else if (mSelecting)
		{
			mSelPt2.X = Math::Min(ClientRectangle.Right - 1, Math::Max(1, e->X));
			mSelPt2.Y = Math::Min(ClientRectangle.Bottom - 1, Math::Max(1, e->Y));
            mSelecting = false;
            Invalidate();
			if (((Math::Abs(mSelPt1.X - mSelPt2.X) < 2) && (Math::Abs(mSelPt1.Y - mSelPt2.Y) < 2)) || (SelectionMode == SelectMode::PointPick))
				MousePick(this, mSelPt2, e->Button);
			else
				MouseSelect(this, mSelPt1, mSelPt2, e->Button);
		}

		this->Cursor = Windows::Forms::Cursors::Cross;
	}

	System::Void GLCanvas2D::ControlMouseWheel(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if (AllowZoomAndPan)
		{
			Drawing::Point pt = e->Location;
			Drawing::PointF ptw = ScreenToWorld(pt);

			if (e->Delta > 0)
			{
				if (mZoomFactor > float::Epsilon * 1000.0f)
				{
					mZoomFactor /= 1.1f;
					mCameraPosition.X = mCameraPosition.X + (ptw.X - mCameraPosition.X) * 0.1F;
					mCameraPosition.Y = mCameraPosition.Y + (ptw.Y - mCameraPosition.Y) * 0.1F;
				}
			}
			else 
			{
				if (mZoomFactor < float::MaxValue / 1000.0f) 
				{
					mZoomFactor *= 1.1f;
					mCameraPosition.X = mCameraPosition.X - (ptw.X - mCameraPosition.X) * 0.1F;
					mCameraPosition.Y = mCameraPosition.Y - (ptw.Y - mCameraPosition.Y) * 0.1F;
				}
			}
			Invalidate();
		}
	}

	System::Void GLCanvas2D::ControlMouseDoubleClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
	{
		if ((e->Button == Windows::Forms::MouseButtons::Middle) && AllowZoomAndPan)
		{
			if (mLimits.IsEmpty)
				SetView(-250, -250, 250, 250);
			else
			{
				Drawing::RectangleF vp = Drawing::RectangleF::Inflate(mLimits, mLimits.Width / 20, mLimits.Height / 20);
				SetView(vp.X, vp.Y, vp.X + vp.Width, vp.Y + vp.Height);
			}
			Invalidate();
		}
	}
}
