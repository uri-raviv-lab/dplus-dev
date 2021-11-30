#include "stdafx.h"

#include <Vcclr.h>
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "GLGraphics3D.h"
#include "GLCanvas3D.h"
#include "UnManaged.h"
#include "Point3D.h"
#include "Utility.h"

namespace GLView 
{
	GLGraphics3D::GLGraphics3D(GLCanvas3D ^ Canvas, Drawing::Graphics ^ GDIGraphics,
							   bool bColorCoded)
	{
		mCanvas = Canvas; 
		mGDIGraphics = GDIGraphics;
		LineWidth = 1.0f;
		objCounter = 1;
		bUseColors = !bColorCoded;
	}

	System::Void GLGraphics3D::UpdateLimits(float x, float y, float z)
	{
		xmin = Math::Min(xmin, x);
		xmax = Math::Max(xmax, x);
		ymin = Math::Min(ymin, y);
		ymax = Math::Max(ymax, y);
		zmin = Math::Min(zmin, z);
		zmax = Math::Max(zmax, z);
	}

	Point3D GLGraphics3D::ModelOrigin()
	{
		return Point3D((xmin + xmax) / 2.0f, (ymin + ymax) / 2.0f, (zmin + zmax) / 2.0f);
	}

	float GLGraphics3D::ModelSize()
	{
		return (float)Math::Max(Math::Sqrt((xmin - xmax) * (xmin - xmax) + (ymin - ymax) * (ymin - ymax) + (zmin - zmax) * (zmin - zmax)), 1.0);
	}

	System::Void GLGraphics3D::DrawLine(float x1, float y1, float z1, float x2, float y2, float z2, Drawing::Color color)
	{
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_LINES);

		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);

		glEnd();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
	}

	System::Void GLGraphics3D::DrawTriangle(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, Drawing::Color color)
	{
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_LINES);

		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);

		glVertex3f(x2, y2, z2);
		glVertex3f(x3, y3, z3);

		glVertex3f(x3, y3, z3);
		glVertex3f(x1, y1, z1);

		glEnd();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
		UpdateLimits(x3, y3, z3);
	}

	System::Void GLGraphics3D::DrawQuad(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, Drawing::Color color)
	{ 
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_LINES);

		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);

		glVertex3f(x2, y2, z2);
		glVertex3f(x3, y3, z3);

		glVertex3f(x3, y3, z3);
		glVertex3f(x4, z4, z4);

		glVertex3f(x4, y4, z4);
		glVertex3f(x1, y1, z1);

		glEnd();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
		UpdateLimits(x3, y3, z3);
		UpdateLimits(x4, y4, z4);
	}

	System::Void GLGraphics3D::FillTriangle(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, Drawing::Color color)
	{
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_TRIANGLES);

		float n[3];
		Utility::CrossProduct(x1 - x2, y1 - y2, z1 - z2, x3 - x2, y3 - y2, z3 - z2, n);
		glNormal3f(n[0], n[1], n[2]);

		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);
		glVertex3f(x3, y3, z3);

		glEnd();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
		UpdateLimits(x3, y3, z3);
	}

	System::Void GLGraphics3D::FillQuad(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, Drawing::Color color)
	{
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_QUADS);

		float n[3];
		Utility::CrossProduct(x1 - x2, y1 - y2, z1 - z2, x3 - x2, y3 - y2, z3 - z2, n);
		glNormal3f(n[0], n[1], n[2]);

		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);
		glVertex3f(x3, y3, z3);
		glVertex3f(x4, y4, z4);

		glEnd();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
		UpdateLimits(x3, y3, z3);
		UpdateLimits(x4, y4, z4);
	}

	System::Void GLGraphics3D::DrawBox(float x1, float y1, float z1, float x2, float y2, float z2, float width, float height, Drawing::Color color)
	{
		float len = (float)Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		float zrot = (float)(Math::Atan2(y2 - y1, x2 - x1) * 180.0 / Math::PI);
		float yrot = (float)(Math::Atan2(Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)), z2 - z1) * 180.0 / Math::PI);

		glPushMatrix();
		glTranslatef(x1, y1, z1);
		glRotatef(zrot, 0, 0, 1);
		glRotatef(yrot, 0, 1, 0);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_LINES);

		glVertex3f(-width / 2, -height / 2, 0);
		glVertex3f(-width / 2, -height / 2, len);
		glVertex3f(width / 2, -height / 2, 0);
		glVertex3f(width / 2, -height / 2, len);
		glVertex3f(-width / 2, height / 2, 0);
		glVertex3f(-width / 2, height / 2, len);
		glVertex3f(width / 2, height / 2, 0);
		glVertex3f(width / 2, height / 2, len);

		glEnd();

		// Bottom
		glBegin(GL_LINE_LOOP);
		glVertex3f(-width / 2, -height / 2, 0);
		glVertex3f(-width / 2, height / 2, 0);
		glVertex3f(width / 2, height / 2, 0);
		glVertex3f(width / 2, -height / 2, 0);
		glEnd();
		// Top
		glBegin(GL_LINE_LOOP);
		glVertex3f(-width / 2, -height / 2, len);
		glVertex3f(-width / 2, height / 2, len);
		glVertex3f(width / 2, height / 2, len);
		glVertex3f(width / 2, -height / 2, len);
		glEnd();

		glPopMatrix();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
	}

	System::Void GLGraphics3D::FillBox(float x1, float y1, float z1, float x2, float y2, float z2, float width, float height, Drawing::Color color)
	{
		float len = (float)Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		float zrot = (float)(Math::Atan2(y2 - y1, x2 - x1) * 180.0 / Math::PI);
		float yrot = (float)(Math::Atan2(Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)), z2 - z1) * 180.0 / Math::PI);

		glPushMatrix();
		glTranslatef(x1, y1, z1);
		glRotatef(zrot, 0, 0, 1);
		glRotatef(yrot, 0, 1, 0);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glBegin(GL_QUADS);

		// Front
		glNormal3f(0, -1, 0);
		glVertex3f(-width / 2, -height / 2, 0);
		glVertex3f(-width / 2, -height / 2, len);
		glVertex3f(width / 2, -height / 2, len);
		glVertex3f(width / 2, -height / 2, 0);

		// Right
		glNormal3f(1, 0, 0);
		glVertex3f(width / 2, -height / 2, 0);
		glVertex3f(width / 2, -height / 2, len);
		glVertex3f(width / 2, height / 2, len);
		glVertex3f(width / 2, height / 2, 0);

		// Back
		glNormal3f(0, 1, 0);
		glVertex3f(width / 2, height / 2, 0);
		glVertex3f(width / 2, height / 2, len);
		glVertex3f(-width / 2, height / 2, len);
		glVertex3f(-width / 2, height / 2, 0);

		// Left
		glNormal3f(-1, 0, 0);
		glVertex3f(-width / 2, height / 2, 0);
		glVertex3f(-width / 2, height / 2, len);
		glVertex3f(-width / 2, -height / 2, len);
		glVertex3f(-width / 2, -height / 2, 0);

		// Bottom
		glNormal3f(0, 0, -1);
		glVertex3f(-width / 2, -height / 2, 0);
		glVertex3f(-width / 2, height / 2, 0);
		glVertex3f(width / 2, height / 2, 0);
		glVertex3f(width / 2, -height / 2, 0);

		// Top
		glNormal3f(0, 0, 1);
		glVertex3f(-width / 2, -height / 2, len);
		glVertex3f(-width / 2, height / 2, len);
		glVertex3f(width / 2, height / 2, len);
		glVertex3f(width / 2, -height / 2, len);

		glEnd();
		
		glPopMatrix();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
	}

	System::Void GLGraphics3D::DrawCylinder(float x1, float y1, float z1, float x2, float y2, float z2, float radius, Drawing::Color color)
	{
		float len = (float)Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		float zrot = (float)(Math::Atan2(y2 - y1, x2 - x1) * 180.0 / Math::PI);
		float yrot = (float)(Math::Atan2(Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)), z2 - z1) * 180.0 / Math::PI);

		glPushMatrix();
		glTranslatef(x1, y1, z1);
		glRotatef(zrot, 0, 0, 1);
		glRotatef(yrot, 0, 1, 0);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);

		_DrawCylinder(radius, len, true);

		glPopMatrix();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
	}

	System::Void GLGraphics3D::FillCylinder(float x1, float y1, float z1, float x2, float y2, float z2, float radius, Drawing::Color color)
	{
		float len = (float)Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		float zrot = (float)(Math::Atan2(y2 - y1, x2 - x1) * 180.0 / Math::PI);
		float yrot = (float)(Math::Atan2(Math::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)), z2 - z1) * 180.0 / Math::PI);

		glPushMatrix();
		glTranslatef(x1, y1, z1);
		glRotatef(zrot, 0, 0, 1);
		glRotatef(yrot, 0, 1, 0);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		_DrawCylinder(radius, len, false);

		glPopMatrix();

		UpdateLimits(x1, y1, z1);
		UpdateLimits(x2, y2, z2);
	}

	System::Void GLGraphics3D::DrawSphere(float x, float y, float z,float radius, Drawing::Color color)
	{
		glPushMatrix();
		glTranslatef(x, y, z);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		_DrawSphere(radius, true);

		glPopMatrix();

		UpdateLimits(x - radius, y - radius, z - radius);
		UpdateLimits(x + radius, y + radius, z + radius);
	}

	System::Void GLGraphics3D::FillSphere(float x, float y, float z,float radius, Drawing::Color color)
	{
		glPushMatrix();
		glTranslatef(x, y, z);

		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		_DrawSphere(radius, false);

		glPopMatrix();

		UpdateLimits(x - radius, y - radius, z - radius);
		UpdateLimits(x + radius, y + radius, z + radius);
	}

	System::Void GLGraphics3D::DrawRasterText(float x, float y, float z, System::String ^ text, Drawing::Color color)
	{
		glDisable(GL_LIGHTING);
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glListBase(mCanvas->RasterListBase);
		glRasterPos3f(x, y, z);
		// Draw the text
		glCallLists(text->Length, GL_UNSIGNED_BYTE, 
			(GLvoid*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(text).ToPointer());
		glEnable(GL_LIGHTING);
	}

	System::Void GLGraphics3D::DrawRasterTextWindow(float x, float y, System::String ^ text, Drawing::Color color)
	{
		glDisable(GL_LIGHTING);
		if(bUseColors)
			glColor4f((float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		glListBase(mCanvas->RasterListBase);
		this->glWindowPos2f(x, y);
		// Draw the text
		glCallLists(text->Length, GL_UNSIGNED_BYTE, 
			(GLvoid*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(text).ToPointer());
		glEnable(GL_LIGHTING);
	}

	System::Void GLGraphics3D::DrawVectorText(float x, float y, float z, float height, System::String ^ text, Drawing::Color color)
	{
		glPushMatrix();
		glListBase(mCanvas->VectorListBase);
		glTranslatef(x, y, z);
		glScalef(height, height, height);
		// Draw the text
		glCallLists(text->Length, GL_UNSIGNED_BYTE, 
			(GLvoid*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(text).ToPointer());
		glPopMatrix();
	}

	System::Void GLGraphics3D::glWindowPos2f(GLfloat x, GLfloat y)
	{
		GLfloat z = 0;
		;GLfloat w = 1;
		GLfloat fx, fy;

		// Push current matrix mode and viewport attributes
		glPushAttrib(GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);

		// Setup projection parameters
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glDepthRange(z, z);
		glViewport((int)x - 1, (int)y - 1, 2, 2);

		// Set the raster (window) position
		fx = x - (int)x;
		fy = y - (int)y;
		glRasterPos4f(fx, fy, 0.0, w);

	    // Restore matrices, viewport and matrix mode
  	    glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
	    glPopAttrib();
	}

	unsigned int GLGraphics3D::BeginSelectableObject(Object ^obj) {
		unsigned int curName = 0;
		if(glObjectToSelID.ContainsKey(obj))
			curName = glObjectToSelID[obj];
		else // Doesn't exist, add new object
			curName = objCounter++;

		// Allowing up to 999 objects on screen
		unsigned char redComponent   = curName % 10;
		unsigned char greenComponent = (curName / 10) % 10;
		unsigned char blueComponent  = (curName / 100) % 10;

		glSelIDToObject[curName] = obj;
		glObjectToSelID[obj] = curName;
		//glPushName(curName);
		// [25, 250]
		if(!bUseColors)
			glColor3ub(redComponent * 25, greenComponent * 25, blueComponent * 25);
		
		/*GLenum err = glGetError();
		if(err)
			MessageBox::Show(gcnew String("Error calling glPushName: ") + err);*/

		return curName;
	}

	void GLGraphics3D::EndSelectableObject() {
		//glPopName();
		if(!bUseColors)
			glColor3ub(0, 0, 0);

		/*GLenum err = glGetError();
		if(err)
			MessageBox::Show(gcnew String("Error calling glPopName: ") + err);*/
	}

	Object ^ GLGraphics3D::ObjectFromGLID(unsigned int glid) {
		if(!glSelIDToObject.ContainsKey(glid))
			return nullptr;

		return glSelIDToObject[glid];
	}

}
