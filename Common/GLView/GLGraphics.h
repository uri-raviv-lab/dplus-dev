#pragma once

#include <windows.h>
#include <GL/gl.h>
#include "GLVertexArray.h"

using namespace System;

namespace GLView {

	// Forward class declarations
	ref class GLCanvas2D;

	/// <summary>
	/// Contains methods for drawing on the canvas.
	/// </summary>
	public ref class GLGraphics
	{
	// Constructor/destructor
	public:
		GLGraphics(GLCanvas2D ^ Canvas, Drawing::Graphics ^ GDIGraphics);

	protected:
		~GLGraphics() { }

	private:
		value class GLTextParam
		{
		public:
			float x, y, height;
			System::String ^ text;
			Drawing::Color color;
			bool vectortext;

			GLTextParam(float X, float Y, float Height, System::String ^ Text, Drawing::Color TextColor, bool VectorText)
			{
				x = X; y = Y;
				height = Height;
				text = Text;
				color = TextColor;
				vectortext = VectorText;	
			}
		};

	private:
		float mLineWidth;
		float mZ;
		Drawing::RectangleF mView;
		System::Drawing::Graphics ^ mGDIGraphics;

	private:
		GLCanvas2D ^ mCanvas;
		/// <summary>
		/// Returns the number of lines required to properly approximate a curve with the given feature size.
		/// </summary>
		int GetCirclePrecision(float FeatureSize);
		GLVertexArray ^ mTriangles;
		GLVertexArray ^ mLines;
		System::Void UpdateDepth() { mZ += 0.000001f; };
		Drawing::PointF mBL, mTR;
		bool mInit;
		System::Void UpdateLimits(float x, float y);
		System::Collections::Generic::List<GLTextParam> ^ mTexts;

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
		
	internal:
		/// <summary>
		/// The GLGraphics class collects drawing objects in arrays. No drawing is actually 
		/// performed until Render() is called. Render() is automatically called by the containing 
		/// canvas class. Do not call Render() manually from your code.
		/// </summary>
		Drawing::RectangleF Render();
	public:
		/// <summary>
		/// Draws text at the given coordinates.
		/// </summary>
		System::Void DrawRasterText(float x, float y, System::String ^ text, Drawing::Color color);
		System::Void DrawRasterText(Drawing::PointF ptf, System::String ^ text, Drawing::Color color)
		{
			DrawRasterText(ptf.X, ptf.Y, text, color);
		}
		/// <summary>
		/// Draws text at the given coordinates.
		/// </summary>
		System::Void DrawVectorText(float x, float y, float height, System::String ^ text, Drawing::Color color);
		System::Void DrawVectorText(Drawing::PointF ptf, float height, System::String ^ text, Drawing::Color color)
		{
			DrawVectorText(ptf.X, ptf.Y, height, text, color);
		}
		/// <summary>
		/// Draws a line connecting the given points.
		/// </summary>
		System::Void DrawLine(float x1, float y1, float x2, float y2, Drawing::Color color);
		System::Void DrawLine(Drawing::PointF pt1, Drawing::PointF pt2, Drawing::Color color)
		{
			DrawLine(pt1.X, pt1.Y, pt2.X, pt2.Y, color);
		}
		/// <summary>
		/// Draws a line with the given thickness in world units.
		/// </summary>
		System::Void DrawLine(float x1, float y1, float x2, float y2, float thickness, Drawing::Color color);
		System::Void DrawLine(Drawing::PointF pt1, Drawing::PointF pt2, float thickness, Drawing::Color color)
		{
			DrawLine(pt1.X, pt1.Y, pt2.X, pt2.Y, thickness, color);
		}
		System::Void DrawLine(float x1, float y1, float x2, float y2, float startthickness, float endthickness, Drawing::Color color);
		System::Void DrawLine(Drawing::PointF pt1, Drawing::PointF pt2, float startthickness, float endthickness, Drawing::Color color)
		{
			DrawLine(pt1.X, pt1.Y, pt2.X, pt2.Y, startthickness, endthickness, color);
		}
		/// <summary>
		/// Draws an elliptic arc specified by center coordinates, a width, and a height.
		/// </summary>
		System::Void DrawArc(float x, float y, float width, float height, float startAngle, float sweepAngle, Drawing::Color color);
		System::Void DrawArc(Drawing::PointF pt, Drawing::SizeF sz, float startAngle, float sweepAngle, Drawing::Color color)
		{
			DrawArc(pt.X, pt.Y, sz.Width, sz.Height, startAngle, sweepAngle, color);
		}
		/// <summary>
		/// Draws a triangle specified by the given corner points.
		/// </summary>
		System::Void DrawTriangle(float x1, float y1, float x2, float y2,float x3,float y3, Drawing::Color color);
		System::Void DrawTriangle(Drawing::PointF pt1, Drawing::PointF pt2, Drawing::PointF pt3, Drawing::Color color)
		{
			DrawTriangle(pt1.X, pt1.Y, pt2.X, pt2.Y, pt3.X, pt3.Y, color);
		}
		/// <summary>
		/// Draws a rectangle specified by the given corner points.
		/// </summary>
		System::Void DrawRectangle(float x1, float y1, float x2, float y2, Drawing::Color color);
		System::Void DrawRectangle(Drawing::RectangleF rec, Drawing::Color color)
		{
			DrawRectangle(rec.Left, rec.Bottom, rec.Right, rec.Top, color);
		}
		/// <summary>
		/// Draws an ellipse specified by center coordinates, a width, and a height.
		/// </summary>
		System::Void DrawEllipse(float x, float y, float width, float height, Drawing::Color color);
		System::Void DrawEllipse(Drawing::PointF pt, Drawing::SizeF sz, Drawing::Color color)
		{
			DrawEllipse(pt.X, pt.Y, sz.Width, sz.Height, color);
		}
		/// <summary>
		/// Draws a polygon specified by the given point coordinates.
		/// </summary>
		System::Void DrawPolygon(array<Drawing::PointF, 1> ^ points, Drawing::Color color);
		/// <summary>
		/// Draws an elliptic pie specified by center coordinates, a width, and a height.
		/// </summary>
		System::Void FillPie(float x, float y, float width, float height, float startAngle, float sweepAngle, Drawing::Color color);
		System::Void FillPie(Drawing::PointF pt, Drawing::SizeF sz, float startAngle, float sweepAngle, Drawing::Color color)
		{
			FillPie(pt.X, pt.Y, sz.Width, sz.Height, startAngle, sweepAngle, color);
		}
		/// <summary>
		/// Draws a triangle specified by the given corner points.
		/// </summary>
		System::Void FillTriangle(float x1, float y1, float x2, float y2,float x3,float y3, Drawing::Color color);
		System::Void FillTriangle(Drawing::PointF pt1, Drawing::PointF pt2, Drawing::PointF pt3, Drawing::Color color)
		{
			FillTriangle(pt1.X, pt1.Y, pt2.X, pt2.Y, pt3.X, pt3.Y, color);
		}
		/// <summary>
		/// Draws a rectangle specified by the given corner points.
		/// </summary>
		System::Void FillRectangle(float x1, float y1, float x2, float y2, Drawing::Color color);
		System::Void FillRectangle(Drawing::RectangleF rec, Drawing::Color color)
		{
			FillRectangle(rec.Left, rec.Bottom, rec.Right, rec.Top, color);
		}
		/// <summary>
		/// Draws an ellipse specified by center coordinates, a width, and a height.
		/// </summary>
		System::Void FillEllipse(float x, float y, float width, float height, Drawing::Color color);
		System::Void FillEllipse(Drawing::PointF pt, Drawing::SizeF sz, Drawing::Color color)
		{
			FillEllipse(pt.X, pt.Y, sz.Width, sz.Height, color);
		}
		/// <summary>
		/// Draws a polygon specified by the given point coordinates.
		/// </summary>
		System::Void FillPolygon(array<Drawing::PointF, 1> ^ points, Drawing::Color color);
		/// <summary>
		/// Measures the given string.
		/// </summary>
		Drawing::SizeF MeasureString(System::String ^ text);
	};

}