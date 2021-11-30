#pragma once

namespace GLView 
{
	public ref class Utility abstract sealed
	{
	public:
		/// <summary>
		/// Calculates the cross-product of the given vectors.
		/// </summary>		
		static void CrossProduct(float v1[3], float v2[3], float v[3])
		{
			v[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		/// <summary>
		/// Calculates the cross-product of the given vectors.
		/// </summary>		
		static void CrossProduct(float x1, float y1, float z1, float x2, float y2, float z2, float v[3])
		{
			v[0] = y1 * z2 - z1 * y2;
			v[1] = z1 * x2 - x1 * z2;
			v[2] = x1 * y2 - y1 * x1;
		}
	};
}