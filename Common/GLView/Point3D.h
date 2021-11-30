#pragma once

using namespace System;

namespace GLView 
{
	public value class Point3D
	{
	public:
	    float X;
		float Y;
		float Z;

		Point3D(float x, float y, float z)
		{
			X = x;
			Y = y;
			Z = z;
		}

		virtual System::String ^ ToString() override
		{
			return String::Format("{0:0.00}, {1:0.00}, {2:0.00}", X, Y, Z);
		}
	};
}