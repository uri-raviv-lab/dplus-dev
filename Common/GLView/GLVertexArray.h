#pragma once

#include <windows.h>
#include <GL/gl.h>

using namespace System;

namespace GLView {

	/// <summary>
	/// Represents a vertex array.
	/// </summary>
	ref class GLVertexArray
	{
	private:
		// Custom vertex class
		value class GLColorVertex {
		public:
			float x, y, z;
			float r, g, b, a;
		};

	// Constructor/destructor
	public:
		GLVertexArray(GLenum Type) 
		{ 
			mVertices = gcnew System::Collections::Generic::List<GLColorVertex>;
			mType = Type;
		}

	protected:
		~GLVertexArray() { }

	// Member variables
	private:
		System::Collections::Generic::List<GLColorVertex> ^ mVertices;
		GLenum mType;

	// Implementation
	public:
		System::Void Clear()
		{
			mVertices->Clear();
		}

		System::Void AddVertex(float x, float y, float z, float r, float g, float b, float a)
		{
			GLColorVertex v;
			v.x = x;
			v.y = y;
			v.z = z;
			v.r = r;
			v.g = g;
			v.b = b;
			v.a = a;

			mVertices->Add(v);
		}

		System::Void AddVertex(float x, float y, float z, Drawing::Color color)
		{
			AddVertex(x, y, z, (float)color.R / 256.0f, (float)color.G / 256.0f, (float)color.B / 256.0f, (float)color.A / 256.0f);
		}

		System::Void Render()
		{
			float * vp = new float[mVertices->Count * 3];
			float * cp = new float[mVertices->Count * 4];

			for (int j = 0; j < mVertices->Count; j++)
			{
				vp[j * 3] = mVertices[j].x;
				vp[j * 3 + 1] = mVertices[j].y;
				vp[j * 3 + 2] = mVertices[j].z;
				cp[j * 4] = mVertices[j].r;
				cp[j * 4 + 1] = mVertices[j].g;
				cp[j * 4 + 2] = mVertices[j].b;
				cp[j * 4 + 3] = mVertices[j].a;
			}

			glVertexPointer(3, GL_FLOAT, 3 * sizeof(float), vp);
			glColorPointer(4, GL_FLOAT, 4 * sizeof(float), cp);

			glDrawArrays(mType, 0, mVertices->Count);

			delete[] vp;
			vp = 0;
			delete[] cp;
			cp = 0;
		}

	// Properties
	public:
		property int Count
		{
			virtual int get(void) { return mVertices->Count; }
		}
		property GLColorVertex Vertex[int]
		{
			virtual GLColorVertex get(int index) { return mVertices[index]; }
		}

	};

}
