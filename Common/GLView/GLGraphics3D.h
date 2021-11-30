#pragma once

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;

namespace GLView 
{
	// Forward class declarations
	ref class GLCanvas3D;
	value class Point3D;

	/// <summary>
	/// Contains methods for drawing on the canvas.
	/// </summary>
	public ref class GLGraphics3D
	{
	// Constructor/destructor
	public:
		GLGraphics3D(GLCanvas3D ^ Canvas, Drawing::Graphics ^ GDIGraphics, bool bColorCoded);

	protected:
		~GLGraphics3D() { }

	private:
		float mLineWidth;
		Drawing::RectangleF mView;
		System::Drawing::Graphics ^ mGDIGraphics;
		float xmin, xmax, ymin, ymax, zmin, zmax;
		Dictionary<unsigned int, Object ^> glSelIDToObject;
		Dictionary<Object ^, unsigned int> glObjectToSelID;
		unsigned int objCounter;
		bool bUseColors; // Specifies that this is a color-coded rendering, so no glColor* will be called

	private:
		GLCanvas3D ^ mCanvas;

	public:
		/// <summary>
		/// Gets or sets the current line width.
		/// </summary>
		property float LineWidth
		{
			virtual float get(void) 
			{ 
				return mLineWidth; 
			}
			virtual void set(float value) 
			{ 
				mLineWidth = value;
				glLineWidth(value);
			}
		}
		
	private:
		/// <summary>
		/// Updates model limits.
		/// </summary>
		System::Void UpdateLimits(float x, float y, float z);
		/// <summary>
		/// Sets the raster position to given window coordinates.
		/// </summary>
		System::Void glWindowPos2f(GLfloat x, GLfloat y);

	public:
		/// <summary>
		/// Gets the origin of the model.
		/// </summary>
		Point3D ModelOrigin();
		/// <summary>
		/// Gets the size of the model.
		/// </summary>
		float ModelSize();
		/// <summary>
		/// Draws a line connecting the given points.
		/// </summary>
		System::Void DrawLine(float x1, float y1, float z1, float x2, float y2, float z2, Drawing::Color color);
		/// <summary>
		/// Draws a triangle specified by the given corner points.
		/// </summary>
		System::Void DrawTriangle(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, Drawing::Color color);
		/// <summary>
		/// Draws a quad specified by the given corner points.
		/// </summary>
		System::Void DrawQuad(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, Drawing::Color color);
		/// <summary>
		/// Draws a filled triangle specified by the given corner points.
		/// </summary>
		System::Void FillTriangle(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, Drawing::Color color);
		/// <summary>
		/// Draws a filled quad specified by the given corner points.
		/// </summary>
		System::Void FillQuad(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, Drawing::Color color);
		/// <summary>
		/// Draws a box.
		/// </summary>
		System::Void DrawBox(float x1, float y1, float z1, float x2, float y2, float z2, float width, float height, Drawing::Color color);
		/// <summary>
		/// Draws a filled box.
		/// </summary>
		System::Void FillBox(float x1, float y1, float z1, float x2, float y2, float z2, float width, float height, Drawing::Color color);
		/// <summary>
		/// Draws a sphere.
		/// </summary>
		System::Void DrawSphere(float x, float y, float z,float radius, Drawing::Color color);
		/// <summary>
		/// Draws a filled sphere.
		/// </summary>
		System::Void FillSphere(float x, float y, float z,float radius, Drawing::Color color);
		/// <summary>
		/// Draws a cylinder.
		/// </summary>
		System::Void DrawCylinder(float x1, float y1, float z1, float x2, float y2, float z2, float radius, Drawing::Color color);
		/// <summary>
		/// Draws a filled cylinder.
		/// </summary>
		System::Void FillCylinder(float x1, float y1, float z1, float x2, float y2, float z2, float radius, Drawing::Color color);
		/// <summary>
		/// Draws text at the given coordinates.
		/// </summary>
		System::Void DrawRasterText(float x, float y, float z, System::String ^ text, Drawing::Color color);
		/// <summary>
		/// Draws text at the given window coordinates.
		/// </summary>
		System::Void DrawRasterTextWindow(float x, float y, System::String ^ text, Drawing::Color color);
		/// <summary>
		/// Draws text at the given coordinates.
		/// </summary>
		System::Void DrawVectorText(float x, float y, float z, float height, System::String ^ text, Drawing::Color color);
		/// <summary>
		/// Marks all subsequent drawing calls as a single, selectable object. The reference to the selected object is <i>obj</i>.
		/// Returns the resulting unique OpenGL object ID. This value can be used by the program but it is not necessary.
		/// </summary>
		unsigned int BeginSelectableObject(Object ^obj);
		/// <summary>
		/// Ends the marking of <i>BeginSelectableObject</i>.
		/// </summary>
		void EndSelectableObject();
		/// <summary>
		/// Returns an object from its OpenGL selection ID.
		/// </summary>
		Object ^ObjectFromGLID(unsigned int glid);
	};

}
