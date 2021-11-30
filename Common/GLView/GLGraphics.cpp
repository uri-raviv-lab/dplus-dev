#include "stdafx.h"
#include "GLGraphics.h"
#include "GLCanvas2D.h"
#include <Vcclr.h>

namespace GLView {

	////////////////////////////////////////////////////////////////////////////////
	//
	// Class GLGraphics
    //
	////////////////////////////////////////////////////////////////////////////////

	GLGraphics::GLGraphics(GLCanvas2D ^ Canvas, Drawing::Graphics ^ GDIGraphics)
	{
		mCanvas = Canvas; 
		mGDIGraphics = GDIGraphics;
		LineWidth = 1.0f;
		mZ = -0.9f;
		mInit = false;
		mTriangles = gcnew GLVertexArray(GL_TRIANGLES);
		mLines = gcnew GLVertexArray(GL_LINES);
		mTexts = gcnew System::Collections::Generic::List<GLTextParam>;
		mView = Canvas->GetViewPort();
	}

	Drawing::RectangleF GLGraphics::Render()
	{		
		// Render drawing objects
		mTriangles->Render();
		mLines->Render();

		// Draw text objects
		for each (GLTextParam tp in mTexts)
		{
			// Position the text
			glLoadIdentity();
			glColor4ub(tp.color.R, tp.color.G, tp.color.B, tp.color.A);
			if (tp.vectortext)
			{
				glListBase(mCanvas->VectorListBase);
				glTranslatef(tp.x, tp.y, mZ);
				glScalef(tp.height, tp.height, tp.height);
			}
			else
			{
				glListBase(mCanvas->RasterListBase);
				glRasterPos2f(tp.x, tp.y);
			}
			// Draw the text
			System::IntPtr str = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(tp.text);
			glCallLists(tp.text->Length, GL_UNSIGNED_BYTE, 
				(GLvoid*)str.ToPointer());
			System::Runtime::InteropServices::Marshal::FreeHGlobal(str);
			UpdateDepth();
		}

		// Clear arrays
		mTriangles->Clear();
		mLines->Clear();
		// Set depth
		mZ = -0.9f;

		return Drawing::RectangleF(mBL.X, mBL.Y, mTR.X - mBL.X, mTR.Y - mBL.Y);
	}

	System::Void GLGraphics::UpdateLimits(float x, float y)
	{
		if (mInit)
		{
			mBL.X = Math::Min(mBL.X, x);
			mBL.Y = Math::Min(mBL.Y, y);
			mTR.X = Math::Max(mTR.X, x);
			mTR.Y = Math::Max(mTR.Y, y);
		}
		else
		{
			mBL = Drawing::PointF(x, y);
			mTR = Drawing::PointF(x, y);
			mInit = true;
		}
		return;
	}

	int GLGraphics::GetCirclePrecision(float FeatureSize)
	{
		// Measure the pixel size of the feature
		int pts = mCanvas->WorldToScreen(Drawing::SizeF(FeatureSize, 0.0f)).Width;				
		// Try to represent curved features by at most 4 pixels.
		int cp = pts / 4;
		// Impose some limits on precision
		if (cp < 8) cp = 8;
		if (cp > 400) cp = 400;
		return cp;
	}

	System::Void GLGraphics::DrawRasterText(float x, float y, System::String ^ text, Drawing::Color color)
	{
		mTexts->Add(GLTextParam(x, y, 0.0f, text, color, false));
	}

	System::Void GLGraphics::DrawVectorText(float x, float y, float height, System::String ^ text, Drawing::Color color)
	{
		mTexts->Add(GLTextParam(x, y, height, text, color, true));
	}

	System::Void GLGraphics::DrawLine(float x1, float y1, float x2, float y2, Drawing::Color color)
	{
		// Check intersections
		Drawing::RectangleF lRect(Math::Min(x1, x2), Math::Min(y1, y2), Math::Abs(x1 - x2), Math::Abs(y1 - y2));
		if (mView.IntersectsWith(lRect))
		{
			mLines->AddVertex(x1, y1, mZ, color);
			mLines->AddVertex(x2, y2, mZ, color);
			UpdateDepth();
		}
		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
	}

	System::Void GLGraphics::DrawLine(float x1, float y1, float x2, float y2, float thickness, Drawing::Color color)
	{
		// Check intersections
		Drawing::RectangleF lRect(Math::Min(x1, x2), Math::Min(y1, y2), Math::Abs(x1 - x2), Math::Abs(y1 - y2));
		if (mView.IntersectsWith(lRect))
		{
			float angle = (float)Math::Atan2(y2 - y1, x2 - x1);
			float t2sina = thickness / 2 * (float)Math::Sin(angle);
			float t2cosa = thickness / 2 * (float)Math::Cos(angle);
			mTriangles->AddVertex(x1 + t2sina, y1 - t2cosa, mZ, color);
			mTriangles->AddVertex(x2 + t2sina, y2 - t2cosa, mZ, color);
			mTriangles->AddVertex(x2 - t2sina, y2 + t2cosa, mZ, color);
			mTriangles->AddVertex(x2 - t2sina, y2 + t2cosa, mZ, color);
			mTriangles->AddVertex(x1 - t2sina, y1 + t2cosa, mZ, color);
			mTriangles->AddVertex(x1 + t2sina, y1 - t2cosa, mZ, color);
			UpdateDepth();
		}
		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
	}

	System::Void GLGraphics::DrawLine(float x1, float y1, float x2, float y2, float startthickness, float endthickness, Drawing::Color color)
	{
		// Check intersections
		Drawing::RectangleF lRect(Math::Min(x1, x2), Math::Min(y1, y2), Math::Abs(x1 - x2), Math::Abs(y1 - y2));
		if (mView.IntersectsWith(lRect))
		{
			float angle = (float)Math::Atan2(y2 - y1, x2 - x1);
			float t2sina1 = startthickness / 2 * (float)Math::Sin(angle);
			float t2cosa1 = startthickness / 2 * (float)Math::Cos(angle);
			float t2sina2 = endthickness / 2 * (float)Math::Sin(angle);
			float t2cosa2 = endthickness / 2 * (float)Math::Cos(angle);
			mTriangles->AddVertex(x1 + t2sina1, y1 - t2cosa1, mZ, color);
			mTriangles->AddVertex(x2 + t2sina2, y2 - t2cosa2, mZ, color);
			mTriangles->AddVertex(x2 - t2sina2, y2 + t2cosa2, mZ, color);
			mTriangles->AddVertex(x2 - t2sina2, y2 + t2cosa2, mZ, color);
			mTriangles->AddVertex(x1 - t2sina1, y1 + t2cosa1, mZ, color);
			mTriangles->AddVertex(x1 + t2sina1, y1 - t2cosa1, mZ, color);
			UpdateDepth();
		}
		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
	}

	System::Void GLGraphics::DrawArc(float x, float y, float width, float height, float startAngle, float sweepAngle, Drawing::Color color) 
	{
		bool check = mView.IntersectsWith(Drawing::RectangleF(x - width / 2, y - height / 2, width, height));
		float da = sweepAngle / (float)GetCirclePrecision(Math::Max(width, height));
		for (float a = startAngle; a < startAngle + sweepAngle; a += da)
		{
			float xv = x + width / 2 * (float)Math::Cos(a);
			float yv = y + height / 2 * (float)Math::Sin(a);
			float xend = x + width / 2 * (float)Math::Cos(a + da);
			float yend = y + height / 2 * (float)Math::Sin(a + da);
			UpdateLimits(xv, yv);
			UpdateLimits(xend, yend);

			if (check)
			{
				mLines->AddVertex(xv, yv, mZ, color);
				mLines->AddVertex(xend, yend, mZ, color);
			}
		}

		UpdateDepth();
	}

	System::Void GLGraphics::FillPie(float x, float y, float width, float height, float startAngle, float sweepAngle, Drawing::Color color) 
	{ 
		bool check = mView.IntersectsWith(Drawing::RectangleF(x - width / 2, y - height / 2, width, height));
		float da = sweepAngle / (float)GetCirclePrecision(Math::Max(width, height));
		for (float a = startAngle; a < startAngle + sweepAngle; a += da)
		{
			float xv = x + width / 2 * (float)Math::Cos(a);
			float yv = y + height / 2 * (float)Math::Sin(a);
			float xend = x + width / 2 * (float)Math::Cos(a + da);
			float yend = y + height / 2 * (float)Math::Sin(a + da);
			UpdateLimits(xv, yv);
			UpdateLimits(xend, yend);

			if (check)
			{
				mTriangles->AddVertex(x, y, mZ, color);
				mTriangles->AddVertex(xv, yv, mZ, color);
				mTriangles->AddVertex(xend, yend, mZ, color);
			}
		}

		UpdateDepth();
	}

	System::Void GLGraphics::DrawTriangle(float x1, float y1, float x2, float y2,float x3,float y3, Drawing::Color color)
	{
		float xmin = Math::Min(Math::Min(x1, x2), x3);
		float ymin = Math::Min(Math::Min(y1, y2), y3);
		float xmax = Math::Max(Math::Max(x1, x2), x3);
		float ymax = Math::Max(Math::Max(y1, y2), y3);
		Drawing::RectangleF lRect(xmin, ymin, xmax - xmin, ymax - ymin);
		if (mView.IntersectsWith(lRect))
		{
			mLines->AddVertex(x1, y1, mZ, color);
			mLines->AddVertex(x2, y2, mZ, color);

			mLines->AddVertex(x2, y2, mZ, color);
			mLines->AddVertex(x3, y3, mZ, color);

			mLines->AddVertex(x3, y3, mZ, color);
			mLines->AddVertex(x1, y1, mZ, color);

			UpdateDepth();
		}

		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
		UpdateLimits(x3, y3);
	}

	System::Void GLGraphics::DrawRectangle(float x1, float y1, float x2, float y2, Drawing::Color color) 
	{ 
		bool check = mView.IntersectsWith(Drawing::RectangleF(Math::Min(x1, x2), Math::Min(y1, y2), Math::Abs(x1 - x2), Math::Abs(y1 - y2)));

		if (check)
		{
			mLines->AddVertex(x1, y1, mZ, color);
			mLines->AddVertex(x2, y1, mZ, color);

			mLines->AddVertex(x2, y1, mZ, color);
			mLines->AddVertex(x2, y2, mZ, color);

			mLines->AddVertex(x2, y2, mZ, color);
			mLines->AddVertex(x1, y2, mZ, color);

			mLines->AddVertex(x1, y2, mZ, color);
			mLines->AddVertex(x1, y1, mZ, color);

			UpdateDepth();
		}

		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
	}

	System::Void GLGraphics::FillTriangle(float x1, float y1, float x2, float y2,float x3,float y3, Drawing::Color color)
	{
		float xmin = Math::Min(Math::Min(x1, x2), x3);
		float ymin = Math::Min(Math::Min(y1, y2), y3);
		float xmax = Math::Max(Math::Max(x1, x2), x3);
		float ymax = Math::Max(Math::Max(y1, y2), y3);
		Drawing::RectangleF lRect(xmin, ymin, xmax - xmin, ymax - ymin);
		if (mView.IntersectsWith(lRect))
		{
			mTriangles->AddVertex(x1, y1, mZ, color);
			mTriangles->AddVertex(x2, y2, mZ, color);
			mTriangles->AddVertex(x3, y3, mZ, color);

			UpdateDepth();
		}

		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
		UpdateLimits(x3, y3);
	}

	System::Void GLGraphics::FillRectangle(float x1, float y1, float x2, float y2, Drawing::Color color) 
	{ 
		bool check = mView.IntersectsWith(Drawing::RectangleF(Math::Min(x1, x2), Math::Min(y1, y2), Math::Abs(x1 - x2), Math::Abs(y1 - y2)));

		if (check)
		{
			mTriangles->AddVertex(x1, y1, mZ, color);
			mTriangles->AddVertex(x2, y1, mZ, color);
			mTriangles->AddVertex(x2, y2, mZ, color);

			mTriangles->AddVertex(x2, y2, mZ, color);
			mTriangles->AddVertex(x1, y2, mZ, color);
			mTriangles->AddVertex(x1, y1, mZ, color);

			UpdateDepth();
		}

		UpdateLimits(x1, y1);
		UpdateLimits(x2, y2);
	}

	System::Void GLGraphics::DrawEllipse(float x, float y, float width, float height, Drawing::Color color) 
	{ 
		bool check = mView.IntersectsWith(Drawing::RectangleF(x - width / 2, y - height / 2, width, height));
		if (check)
		{
			float da = 2.0f * (float)Math::PI / (float)GetCirclePrecision(Math::Max(width, height));
			for (float a = 0; a <= 2.0f * (float)Math::PI; a += da)
			{
				float xv = x + width / 2 * (float)Math::Cos(a);
				float yv = y + height / 2 * (float)Math::Sin(a);
				float xend = x + width / 2 * (float)Math::Cos(a + da);
				float yend = y + height / 2 * (float)Math::Sin(a + da);

				mLines->AddVertex(xv, yv, mZ, color);
				mLines->AddVertex(xend, yend, mZ, color);
			}
			UpdateDepth();
		}

		UpdateLimits(x - width / 2.0f, y - height / 2.0f);
		UpdateLimits(x + width / 2.0f, y + height / 2.0f);
	}

	System::Void GLGraphics::FillEllipse(float x, float y, float width, float height, Drawing::Color color) 
	{ 
		bool check = mView.IntersectsWith(Drawing::RectangleF(x - width / 2, y - height / 2, width, height));
		if (check)
		{
			float da = 2.0f * (float)Math::PI / (float)GetCirclePrecision(Math::Max(width, height));
			for (float a = 0; a <= 2.0f * (float)Math::PI; a += 2.0f * (float)Math::PI / (float)GetCirclePrecision(Math::Max(width, height)))
			{
				float xv = x + width / 2 * (float)Math::Cos(a);
				float yv = y + height / 2 * (float)Math::Sin(a);
				float xend = x + width / 2 * (float)Math::Cos(a + da);
				float yend = y + height / 2 * (float)Math::Sin(a + da);

				mTriangles->AddVertex(x, y, mZ, color);
				mTriangles->AddVertex(xv, yv, mZ, color);
				mTriangles->AddVertex(xend, yend, mZ, color);
			}
			UpdateDepth();
		}

		UpdateLimits(x - width / 2.0f, y - height / 2.0f);
		UpdateLimits(x + width / 2.0f, y + height / 2.0f);
	}

	System::Void GLGraphics::DrawPolygon(array<Drawing::PointF, 1> ^ points, Drawing::Color color) 
	{
		if (points->Length < 2) return;

		for (int i = 0; i < points->Length - 1; i++)
		{
			mLines->AddVertex(points[i].X, points[i].Y, mZ, color);
			mLines->AddVertex(points[i + 1].X, points[i + 1].Y, mZ, color);
			UpdateLimits(points[i].X, points[i].Y);
			UpdateLimits(points[i + 1].X, points[i + 1].Y);
		}
		mLines->AddVertex(points[points->Length - 1].X, points[points->Length - 1].Y, mZ, color);
		mLines->AddVertex(points[0].X, points[0].Y, mZ, color);
		UpdateDepth();
	}

	System::Void GLGraphics::FillPolygon(array<Drawing::PointF, 1> ^ points, Drawing::Color color) 
	{ 
		if (points->Length < 3) return;

		// Calculate center coordinates. Polygons are drawn as triangle fans sharing this point.
		// This is why only concave polygons are supported.
		float x = 0;
		float y = 0;
		for (int i = 0; i < points->Length; i++)
		{
			x += points[i].X;
			y += points[i].Y;
			glVertex2f(points[i].X, points[i].Y);
			UpdateLimits(points[i].X, points[i].Y);
		}
		x /= (float)points->Length;
		y /= (float)points->Length;

		for (int i = 0; i < points->Length - 1; i++)
		{
			mTriangles->AddVertex(x, y, mZ, color);
			mTriangles->AddVertex(points[i].X, points[i].Y, mZ, color);
			mTriangles->AddVertex(points[i + 1].X, points[i + 1].Y, mZ, color);
		}
		mTriangles->AddVertex(x, y, mZ, color);
		mTriangles->AddVertex(points[points->Length - 1].X, points[points->Length - 1].Y, mZ, color);
		mTriangles->AddVertex(points[0].X, points[0].Y, mZ, color);
		UpdateDepth();
	}

	Drawing::SizeF GLGraphics::MeasureString(System::String ^ text)
	{
		Drawing::SizeF szf = mGDIGraphics->MeasureString(text, mCanvas->Font);
		Drawing::SizeF sz = mCanvas->ScreenToWorld(Drawing::Size((int)szf.Width, (int)szf.Height));
		if (sz.Width < 0) sz.Width = -sz.Width;
		if (sz.Height < 0) sz.Height = -sz.Height;
		return sz;
	}

}