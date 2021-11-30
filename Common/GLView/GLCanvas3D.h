#pragma once

#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "Point3D.h"
#include "Camera.h"

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Drawing;
using namespace System::ComponentModel;
using namespace System::Windows::Forms;

namespace GLView 
{
	// Forward class declarations
	ref class GLGraphics3D;
	ref class Camera;

	/// <summary>
	/// Represents a 2D drawing canvas utilizing OpenGL.
	/// </summary>
	[ToolboxBitmap(GLView::GLCanvas3D::typeid)]
    [DefaultEvent("Render")]
	public ref class GLCanvas3D : public System::Windows::Forms::UserControl
	{
	// Constructor/destructor
	public:
		GLCanvas3D();

	protected:
		~GLCanvas3D();

	public:
		ref class GLGraphics3D;

	// Member variables
	private:
		HDC mhDC;
		HGLRC mhGLRC;
		bool mIsAccelerated;
		Camera ^ mCamera;
	    bool mPanning, mOrbiting, mSelecting;
		Point mLastMouse, mMouseDown;
		Point3D mOrigin;
		float mSize;
		bool mAntiAlias;
		GLuint base, rasterbase;
		System::Drawing::Color mFloorColor;
		System::Drawing::Color mGridColor;
		System::Drawing::Color mSelectionRectColor;
		bool mLighting;
		List<Object ^> ^mSelectedObjects;

	public:
		/// <summary>
		/// Gets or sets the camera target.
		/// </summary>
		property Point3D camTarget
		{
			virtual Point3D get(void) { return mCamera->Target; }
			virtual void set(Point3D value) { mCamera->Target = value; }
		}

		/// <summary>
		/// Gets or sets the cameras pitch.
		/// </summary>
		property float camRoll
		{
			virtual float get(void) { return mCamera->Roll; }
			virtual void set(float value) { mCamera->Roll = value; }
		}

		/// <summary>
		/// Gets or sets the cameras pitch.
		/// </summary>
		property float camPitch
		{
			virtual float get(void) { return mCamera->Pitch; }
			virtual void set(float value) { mCamera->Pitch = value; }
		}

		
		/// <summary>
		/// Gets or sets the camera position.
		/// </summary>
		property Point3D camPosition
		{
			virtual Point3D get(void) { return mCamera->Position; }
			virtual void set(Point3D value) { mCamera->Position = value; }
		}

		/// <summary>
		/// Gets or sets the camera pitch angle in degrees.
		/// </summary>
		property float Pitch
		{
			virtual float get(void) { return mCamera->Pitch; }
			virtual void set(float value) { mCamera->Pitch = value; }
		}
		/// <summary>
		/// Gets or sets the camera yaw angle in degrees.
		/// </summary>
		property float Yaw
		{
			virtual float get(void) { return mCamera->Yaw; }
			virtual void set(float value) { mCamera->Yaw = value; }
		}
		/// <summary>
		/// Gets or sets the camera roll angle in degrees.
		/// </summary>
		property float Roll
		{
			virtual float get(void) { return mCamera->Roll; }
			virtual void set(float value) { mCamera->Roll = value; }
		}
		/// <summary>
		/// Gets or sets the distance to camera target.
		/// </summary>
		property float Distance
		{
			virtual float get(void) { return mCamera->Distance; }
			virtual void set(float value) { mCamera->Distance = value; }
		}
		/// <summary>
		/// Determines whether hardware acceleration is enabled.
		/// </summary>
		[Category("Behavior"), Browsable(false), Description("Determines whether hardware acceleration is enabled.")] 
		property bool IsAccelerated
		{
			virtual bool get(void) { return mIsAccelerated; }
		}
		/// <summary>
		/// Determines whether zooming and panning with the mouse is allowed.
		/// </summary>
		[Category("Behavior"), Browsable(true), DefaultValue(true), Description("Determines whether zooming and panning with the mouse is allowed.")] 
		property bool AllowZoomAndPan;
		/// <summary>
		/// Determines whether object selection with the left mouse button is enabled.
		/// </summary>
		[Category("Behavior"), Browsable(true), DefaultValue(false), Description("Determines whether object selection with the left mouse button is enabled.")] 
		property bool MouseSelect;
		/// <summary>
		/// Determines the maximal number of selected objects.
		/// </summary>
		[Category("Behavior"), Browsable(true), DefaultValue(512), Description("Determines the maximal number of selected objects.")] 
		property unsigned int MaxSelectedObjects;
		/// <summary>
		/// Determines the maximal view distance.
		/// </summary>
		[Category("Behavior"), Browsable(true), DefaultValue(50.0f), Description("Determines the maximal view distance.")] 
		property float ViewDistance;
		/// <summary>
		/// Draws fog at end of view distance.
		/// </summary>
		[Category("Behavior"), Browsable(true), DefaultValue(true), Description("Draws fog at end of view distance.")] 
		property bool DrawFog;
		/// <summary>
		/// Gets or sets the array of selected objects.
		/// </summary>
		[Category("Behavior"), Browsable(false), Description("Gets the array of selected objects.")] 
		property List<Object ^> ^SelectedObjects {
			List<Object ^> ^get() { return mSelectedObjects; }

			void set(List<Object ^> ^value) { 
				mSelectedObjects = value;
				SelectionChanged(this, gcnew EventArgs());
				Invalidate(); 
			}
		}
		/// <summary>
		/// The cursor that appears when the pointer moves over the control.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(Windows::Forms::Cursor::typeid, "Cross"), Description("The cursor that appears when the pointer moves over the control.")]
        property Windows::Forms::Cursor ^ Cursor
		{
			virtual Windows::Forms::Cursor ^ get(void) override { return Control::Cursor; }
			virtual void set(Windows::Forms::Cursor ^ value) override { Control::Cursor = value; }
		}
		/// <summary>
		/// Determines whether the floor is drawn.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines whether the floor is drawn.")] 
		property bool DrawFloor;
		/// <summary>
		/// Determines whether the axis is drawn.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines whether the axis is drawn.")] 
		property bool ShowAxis;
		/// <summary>
		/// Determines how the axis at the origin is drawn.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines how the axis at the origin is drawn.")] 
		property bool ShowConstantSizedAxis;
		/// <summary>
		/// Determines whether the axis is drawn on the bottom-right corner of the screen.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines whether the axis is drawn on the bottom-right corner of the screen.")] 
		property bool ShowCornerAxis;
		/// <summary>
		/// Gets or sets the color of floor.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(System::Drawing::Color::typeid, "Beige"), Description("Gets or sets the color of floor.")]
		property Drawing::Color FloorColor
		{
			virtual Drawing::Color get(void) { return mFloorColor; }
			virtual void set(Drawing::Color value) { mFloorColor = value; Invalidate(); }
		}
		/// <summary>
		/// Gets or sets the color of the selection rectangle.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(System::Drawing::Color::typeid, "LightSkyBlue"), Description("Gets or sets the color of the selection rectangle.")]
		property Drawing::Color SelectionRectColor
		{
			virtual Drawing::Color get(void) { return mSelectionRectColor; }
			virtual void set(Drawing::Color value) { mSelectionRectColor = value; Invalidate(); }
		}		
		/// <summary>
		/// Gets or sets the color of grid lines.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(System::Drawing::Color::typeid, "Bisque"), Description("Gets or sets the color of grid lines.")]
		property Drawing::Color GridColor
		{
			virtual Drawing::Color get(void) { return mGridColor; }
			virtual void set(Drawing::Color value) { mGridColor = value; Invalidate(); }
		}
		/// <summary>
		/// Gets or sets the background color of the control.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(System::Drawing::Color::typeid, "White"), Description("Gets or sets the background color of the control.")]
		property Drawing::Color BackColor
		{
			virtual Drawing::Color get(void) override { return Control::BackColor; }
			virtual void set(Drawing::Color value) override { Control::BackColor = value; Invalidate(); }
		}
		/// <summary>
		/// Determines whether lines are anti-aliased.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines whether lines are anti-aliased.")]
		property bool AntiAlias
		{
			virtual bool get(void) { return mAntiAlias; }
			virtual void set(bool value) 
			{ 
				if (value)
					glEnable(GL_LINE_SMOOTH);
				else
					glDisable(GL_LINE_SMOOTH);
				mAntiAlias = value; 
				Invalidate(); 
			}
		}
		/// <summary>
		/// Determines whether simple lighting model is active.
		/// </summary>
		[Category("Appearance"), Browsable(true), DefaultValue(true), Description("Determines whether simple lighting model is active.")]
		property bool Lighting
		{
			virtual bool get(void) { return mLighting; }
			virtual void set(bool value) 
			{ 
				if (value)
				{
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
				}
				else
				{
					glDisable(GL_LIGHTING);
					glDisable(GL_LIGHT0);
					glDisable(GL_COLOR_MATERIAL);
				}
				mLighting = value; 
				Invalidate(); 
			}
		}
		/// <summary>
		/// Returns the display list used to draw vector text.
		/// </summary>
		[Category("Appearance"), Browsable(false), Description("Returns the display list used to draw vector text.")] 
		property GLuint VectorListBase
		{
			virtual GLuint get(void) { return base; }
		}
		/// <summary>
		/// Returns the display list used to draw raster text.
		/// </summary>
		[Category("Appearance"), Browsable(false), Description("Returns the display list used to draw raster text.")] 
		property GLuint RasterListBase
		{
			virtual GLuint get(void) { return rasterbase; }
		}
		/// <summary>
		/// Gets or sets the font used to display text in the control.
		/// </summary>
		[Category("Appearance"), Browsable(true), Description("Gets or sets the font used to display text in the control.")] 
		property System::Drawing::Font ^ Font
		{
			virtual void set(System::Drawing::Font ^ value) override
			{
				Control::Font::set(value);

				if(!this->DesignMode)
				{
					// Save previous context and make our context current
					HDC mhOldDC = wglGetCurrentDC();
					HGLRC mhOldGLRC = wglGetCurrentContext();
					wglMakeCurrent(mhDC, mhGLRC);

					// Delete old display lists
					glDeleteLists(base, 256);
					glDeleteLists(rasterbase, 256);
					
					// Create the font display lists
					SelectObject(mhDC, (HGDIOBJ)value->ToHfont());
					base = glGenLists(256);
					rasterbase = glGenLists(256);
					
					wglUseFontOutlines(mhDC, 0, 256, base, 0.0f, 0.0f, WGL_FONT_POLYGONS, NULL);
					wglUseFontBitmaps(mhDC, 0, 256, rasterbase);
					
					// Restore previous context
					wglMakeCurrent(mhOldDC, mhOldGLRC);

					Invalidate(); 
				}
			}
		}

	// Public methods.
	public:
		/// <summary>
		/// Resets the camera.
		/// </summary>
		System::Void ResetView();

		/// <summary>
		/// Deselects all selected objects.
		/// </summary>
		System::Void Deselect();


	private:
		System::Void ControlResize(System::Object^  sender, System::EventArgs^  e);
		System::Void ControlMouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
		System::Void ControlMouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
		System::Void ControlMouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
		System::Void ControlMouseWheel(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
		System::Void ControlMouseDoubleClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);

		/// <summary>
		/// Does all the selection-related computing, in the rectangle between <i>startLoc</i> and <i>endLoc</i>.
		/// </summary>
		System::Void EndSelection(Point startLoc, Point endLoc, bool isCtrlClick);

		/// <summary>
		/// Helper function for <i>EndSelection</i>. Performs all the drawing for selection mode.
		/// Returns the graphics used.
		/// </summary>
		GLView::GLGraphics3D ^RenderForSelection();

	protected:
		virtual property System::Windows::Forms::CreateParams^ CreateParams
		{
			System::Windows::Forms::CreateParams ^ get(void) override sealed
			{ 
				System::Windows::Forms::CreateParams^ cp = UserControl::CreateParams;
				cp->ClassStyle = cp->ClassStyle | CS_VREDRAW | CS_HREDRAW | CS_OWNDC;

				return cp; 
			}
		}
		virtual void OnPaint(System::Windows::Forms::PaintEventArgs^  e) override sealed;
		virtual void OnPaintBackground(System::Windows::Forms::PaintEventArgs^  e) override sealed;

	// Events
	public:
		delegate void RenderHandler(System::Object ^ sender, GLView::GLGraphics3D ^ Graphics);
		delegate void RenderDoneHandler(System::Object ^ sender);
		/// <summary>
		/// Occurs when the control is redrawn.
		/// </summary>
		[Category("Appearance"), Browsable(true)] 
		event RenderHandler ^ Render;
		/// <summary>
		/// Occurs after the control is redrawn.
		/// </summary>
		[Category("Appearance"), Browsable(true)] 
		event RenderDoneHandler ^ RenderDone;
		/// <summary>
		/// Must be implemented if object selection is used. This event must 
		/// not call glColor* or glEnable(GL_{LIGHTING,DITHERING,TEXTURE}).
		/// </summary>
		[Category("Appearance"), Browsable(true)] 
		event RenderHandler ^ ColorCodedRender;
		/// <summary>
		/// Occurs when the selected objects are changed.
		/// </summary>
		[Category("Behavior"), Browsable(true)] 
		event EventHandler ^ SelectionChanged;

	private: 
		System::Void InitializeComponent()
		{
			 this->Name = L"GLCanvas3D";
		}

		System::Void ResetViewport();

		Point3D GetOGLPos(int x, int y)
		{
			GLint viewport[4];
			GLdouble modelview[16];
			GLdouble projection[16];
			GLfloat winX, winY, winZ;
			GLdouble posX, posY, posZ;

			glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
			glGetDoublev(GL_PROJECTION_MATRIX, projection);
			glGetIntegerv(GL_VIEWPORT, viewport);

			winX = (float)x;
			winY = (float)viewport[3] - (float)y;
			glReadPixels( x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );

			gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

			return Point3D((float)posX, (float)posY, (float)posZ);
		}
	};
}
