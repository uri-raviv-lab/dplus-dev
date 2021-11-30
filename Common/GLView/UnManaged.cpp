#include "stdafx.h"

#include <windows.h>
#include <GL/glu.h>
#include "UnManaged.h"

#pragma unmanaged
void _DrawCylinder(float radius, float length, bool wire)
{
	GLUquadric* q = gluNewQuadric();
	if(wire)
		gluQuadricDrawStyle(q, GLU_LINE); // GLU_SILHOUETTE
	gluCylinder(q, radius, radius, length, 16, 16);
	gluDeleteQuadric(q);
}

void _DrawSphere(float radius, bool wire)
{
	GLUquadric* q = gluNewQuadric();
	if(wire)
		gluQuadricDrawStyle(q, GLU_LINE); // GLU_SILHOUETTE
	gluSphere(q, radius, 16, 16);
	gluDeleteQuadric(q);
}
#pragma managed