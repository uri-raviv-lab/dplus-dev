#include "GLPreview.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>

#include <Eigen/Core>

#include <vector>
#include <cmath>
#include <algorithm>

using std::vector;

static vector<GLfloat> mtsins, mtcoses;

GLUquadricObj *quad = NULL;

static const int LODToRes[] = {
	0,
	3,  // LOD_NONE
	4,  // LOD_VERYLOW
	8,  // LOD_LOW
	16, // LOD_MEDIUM
	32, // LOD_HIGH
};

/************************************************************************/
/* Draws an N-Layered Hollow Sphere                                     */
/************************************************************************/
void DrawGLNLHollowSphere(GLfloat rad, GLfloat ed, LevelOfDetail lod, bool bNoColor) {
	if(!quad)
		quad = gluNewQuadric();

	GLfloat matGreen[] = { 0.2f, 0.8f, 0.0f, 1.0f };
	GLfloat matRed[] = { 0.8f, 0.2f, 0.0f, 1.0f };

	// You only see the outer layer anyway
	GLfloat color[] = { 
		(ed * matRed[0] + (1000.0f - ed) * matGreen[0]) / 1000.0f,
		(ed * matRed[1] + (1000.0f - ed) * matGreen[1]) / 1000.0f,
		0.0f
	};

	if(!bNoColor)
		glColor3fv(color);

	gluSphere(quad, rad, LODToRes[lod], LODToRes[lod]);
}

/************************************************************************/
/* Draws a Sphere														*/
/************************************************************************/
void DrawGLSphere(float ed, LevelOfDetail lod, bool bNoColor) {
	if(!bNoColor) {
		glEnable(GL_LIGHTING); glEnable(GL_LIGHT0);
		{
			GLfloat mat[] = { 0.7f, 0.0f, 0.1f, 1.0f };

			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
						 mat);
		}
	}
	
	gluSphere(quad, 1.5f, LODToRes[lod], LODToRes[lod]);
	
	if(!bNoColor) {
		glDisable(GL_LIGHT0); glDisable(GL_LIGHTING);
	}
}

/************************************************************************/
/* Draws a Hollow Cylindroid                                            */
/************************************************************************/
void DrawGLCylindroid(float innerRadius, float outerRadius, 
					  float height, LevelOfDetail lod, bool bNoColor) {
   double rad[2] = { innerRadius, outerRadius };
   double ed[2] = { 0.0, 0.0 };   
   glScaled(1.3, 0.7, 1.0);
   DrawGLNLayeredHC(rad, height, ed, 2, lod, bNoColor);
}

/************************************************************************/
/* Draws an N-Layered Cylindroid                                        */
/************************************************************************/
void DrawGLNLayeredCylindroid(double *rad, double height, 
							  double *ed, int n, LevelOfDetail lod, bool bNoColor) {
	// TODO: Scale is determined by eccentricity
	glScaled(1.0, 0.7, 1.0);
	DrawGLNLayeredHC(rad, height, ed, n, lod, bNoColor);
}

/************************************************************************/
/* Draws an N-Layered Hollow Cylinder                                   */
/************************************************************************/
void DrawGLNLayeredHC(GLdouble *rad, GLdouble height, 
					  GLdouble *ed, int n, LevelOfDetail lod, bool bNoColor) {
	if(!quad) {
	  quad = gluNewQuadric();
	  gluQuadricNormals(quad, GLU_SMOOTH);
	}

	GLdouble totalRad = 0.0, cumRad;
	
	GLdouble matGreen[] = { 0.2f, 0.8f, 0.0f, 1.0f };
	GLdouble matRed[] = { 0.8f, 0.2f, 0.0f, 1.0f };	

	glTranslated(0.0, 0.0, -1 * height / 2); // Center

	if(!bNoColor)
		glColor3d(0.0, 0.1, 0.8);


	for(int i = 0; i < n; i++)
		totalRad += rad[i];

	if (n > 1 && ed[1] != ed[0])
		gluCylinder(quad, rad[0], rad[0], height, LODToRes[lod], LODToRes[lod]);

	cumRad = rad[0];

	totalRad = rad[0];

	for (int i = 1; i < n; i++) {
		totalRad += rad[i];
		// Front face
		GLdouble matDensity[] = { (ed[i] * matRed[0] +
			(1000.0f - ed[i]) * matGreen[0]) / 1000.0f,
			(ed[i] * matRed[1] +
			(1000.0f - ed[i]) * matGreen[1]) / 1000.0f,
			0.0f };
		if (!bNoColor)
			glColor3dv(matDensity);


		if (ed[i] != ed[0])
		{
			gluCylinder(quad, cumRad, cumRad, height, LODToRes[lod], LODToRes[lod]);

			gluDisk(quad, cumRad, cumRad + rad[i], LODToRes[lod], LODToRes[lod]);

			glTranslated(0.0, 0.0, height); // Back face

			if (!bNoColor)
				glColor3dv(matDensity);

			gluDisk(quad, cumRad, cumRad + rad[i], LODToRes[lod], LODToRes[lod]);

			glTranslated(0.0, 0.0, -height); // Back face

			gluCylinder(quad, totalRad, totalRad, height, LODToRes[lod], LODToRes[lod]);
		}
		cumRad += rad[i];
	}
}

/************************************************************************/
/* Draws an Discrete Helix			                                    */
/************************************************************************/
void DrawGLMicrotubule(GLfloat r, int totalsize, LevelOfDetail lod, bool bNoColor) {
	// TODO: MAKE THIS CORRECT
	if(!quad) {
		quad = gluNewQuadric();
		gluQuadricNormals(quad, GLU_SMOOTH);
	}

	GLfloat x, y;

	if(mtsins.size() == 0) {
		float vv;
		for(int i = 0; i <= 360; i += 20) {
			vv=(i/180.0f*3.142f);
			mtsins.push_back(sin(vv));
			mtcoses.push_back(cos(vv));
		}
	}	

	GLfloat matBlue[]  = { 0.0f, 0.1f, 0.8f };
	GLfloat matGreen[] = { 0.0f, 0.8f, 0.1f };

	glTranslatef(0.0f, -r * (float)totalsize / 5.0f, (float)totalsize * (-1) / 10.0f);

	bool bGreen = true;
	for(int rad=0; rad < totalsize; rad++)			
	{
		if(!bNoColor) {
			if(!bGreen) {
				glColor3fv(matGreen); bGreen = true;
			} else {
				glColor3fv(matBlue); bGreen = false;
			}
		}

		for(int theta=0; theta < 360 / 20; theta++)
		{
			x = float(mtcoses.at(theta) * 2) * r;					
			y = float(mtsins.at(theta) * 2) * r;					

			glTranslatef(x, y, r / 10.0f);

			gluSphere(quad, r, LODToRes[lod], LODToRes[lod]);
		}
	}
}

void DrawGLMembrane(GLfloat r, int height, int size, GLfloat headED, LevelOfDetail lod, bool bNoColor) {
	// TODO: MAKE THIS CORRECT
	if(!quad) {
		quad = gluNewQuadric();
		gluQuadricNormals(quad, GLU_SMOOTH);
	}

	GLfloat matRed[] = { 0.9f, 0.1f, 0.0f, 1.0f };
	GLfloat matInside[] = { 2 * headED, 1.0f, 0.0f, 1.0f };

	if(!bNoColor) {
		glEnable(GL_LIGHTING); glEnable(GL_LIGHT0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,	matRed);

		glFogi(GL_FOG_MODE, GL_LINEAR);
		glHint(GL_FOG_HINT, GL_DONT_CARE);
		glFogf(GL_FOG_DENSITY, 0.5f);
		glFogf(GL_FOG_START, 3.0f);
		glFogf(GL_FOG_END, 6.0f);
		glEnable(GL_FOG);
	}

	GLfloat h = float(height) / 10.0f;
	
	glRotatef(90.0f, 1.0f, 0.0f, 0.0f);

	glTranslatef(-2*r * size / 2.0f, -2*r * size / 2.0f, 0.0f);

	for(int x = 0; x < size; x++) {
		glTranslatef(2*r, 0, 0);
		for(int y = 0; y < size; y++) {
			glTranslatef(0, 2*r, h);
			gluSphere(quad, r, LODToRes[lod], LODToRes[lod]);

			if(!bNoColor)
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,	matInside);

			glBegin(GL_LINES);

			glVertex3f(0, 0, 0);
			glVertex3f(r/2.0f, 0, -h);

			glVertex3f(r/2.0f, 0, -h);
			glVertex3f(0, 0, -2*h);

			glVertex3f(0, 0, 0);
			glVertex3f(-r/2.0f, 0, -h);

			glVertex3f(-r/2.0f, 0, -h);
			glVertex3f(0, 0, -2*h);
			glEnd();
			if(!bNoColor)
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,	matRed);

			glTranslatef(0, 0.0f, -2*h);
			gluSphere(quad, r, 12, 12);
			glTranslatef(0, 0, h);
		}
		glTranslatef(0, -2*size*r, 0.0f);
	}

	if(!bNoColor) {
		glDisable(GL_LIGHT0); glDisable(GL_LIGHTING);
		glDisable(GL_FOG);
	}
}

void DrawGLNHelix(const double* offset, const double* ed, double* cross_section, const int n, const double radius, const double height, const double pitch, const LevelOfDetail lod, const bool bNoColor) {
	// TODO: MAKE THIS CORRECT
	// TODO: MAKE THIS CORRECT
	// TODO: MAKE THIS CORRECT

	GLfloat x;													// Helix x Coordinate
	GLfloat y;													// Helix y Coordinate
	GLfloat z;													// Helix z Coordinate
	int phi;													// Angle
	int theta;													// Angle
	float		vertexes[4][3];	
																// Angles
	GLfloat r;													// Radius Of Twist
	float twists = height / pitch;		// 2r Twists

	float step_size = 2.*M_PI * twists / std::max(19, int(ceil(twists * 18.) + 0.5));
	const int extra_vals = 6;
	Eigen::ArrayXf thetaa, coses, sins;
	thetaa.setLinSpaced(std::max(19, int(ceil(twists * 18.)+0.5)), 0.f, 2.*M_PI * twists);

	sins = thetaa.sin();
	coses = thetaa.cos();

//	glScaled(0.5, 0.5, 0.5);
	for (int i = 1; i < n; i++)
	{
		GLfloat matDensity[] = { (GLfloat)(ed[i] * 0.0008 + (1000.0f - ed[i]) * 0.0),
			(GLfloat)(ed[i] * 0.0 + (1000.0f - ed[i]) * 0.0008),
			0.0f };
		if (!bNoColor) {
			if (ed[i] < 0)
				glColor3f(0.0f, 0.3f, 1.0f);
			else
				glColor3f(matDensity[0], matDensity[1], matDensity[2]);
		}

		glTranslatef(0, 0, offset[i]);
		//glRotatef(90.0f, 1.0f, 0.0f, 0.0f);

		r = radius;														// Radius

#define WHATS_THIS 0.0f//0*(cross_section[i]) // 2.0f

		glBegin(GL_QUADS);											// Begin Drawing Quads
		for (phi = 0; phi < 18; phi++)							// 360 Degrees In Steps Of 20
		{
			for (theta = 0; theta < thetaa.size()-1; theta++)			// 360 Degrees * Number Of Twists In Steps Of 20
			{
#ifndef OLD_ANNOYING_LOOKING_CODE_w4857362_THAT_I_DONT_KNOW_WHERE_IT_CAME_FROM
				int vertex = 0;
				for (int c = 0; c < 4; c++)
				{
					const int tt = c == 1 || c == 2;
					const int pp = c > 1;
					x = float(coses[theta + tt] * (WHATS_THIS + coses[phi + pp]))*cross_section[i] + r;					// Calculate x Position (1st Point)
					y = float(sins[theta + tt] * (WHATS_THIS + coses[phi + pp])) * cross_section[i] + r;					// Calculate y Position (1st Point)
					z = float(((thetaa[theta + tt] - (WHATS_THIS*M_PI)) + sins[phi + pp]) * cross_section[i] + r);		// Calculate z Position (1st Point)

					x = float(r * coses[theta + tt] + coses[phi + pp] * cross_section[i]);
					y = float(r * sins [theta + tt] + coses[phi + pp] * cross_section[i]);
					z = float(height * thetaa[theta + tt] / thetaa.tail(1)(0) -sins[phi + pp] * cross_section[i]);

					vertexes[vertex][0] = x;									// Set x Value Of First Vertex
					vertexes[vertex][1] = y;									// Set y Value Of First Vertex
					vertexes[vertex][2] = z;									// Set z Value Of First Vertex
					vertex++;
				}
#else
				x=float(coses[theta]*(2.0f+coses[phi] ))*r;					// Calculate x Position (1st Point)
				y=float(sins[theta]*(2.0f+coses[phi] ))*r;					// Calculate y Position (1st Point)
				z=float((( thetaa[theta]-(2.0f*3.142f)) + sins[phi] ) * r);		// Calculate z Position (1st Point)

				vertexes[0][0]=x;									// Set x Value Of First Vertex
				vertexes[0][1]=y;									// Set y Value Of First Vertex
				vertexes[0][2]=z;									// Set z Value Of First Vertex

				x=float(coses[theta+1]*(2.0f+coses[phi] ))*r;					// Calculate x Position (1st Point)
				y=float(sins[theta+1]*(2.0f+coses[phi] ))*r;					// Calculate y Position (1st Point)
				z=float((( thetaa[theta+1]-(2.0f*3.142f)) + sins[phi] ) * r);		// Calculate z Position (1st Point)

				vertexes[1][0]=x;									// Set x Value Of Second Vertex
				vertexes[1][1]=y;									// Set y Value Of Second Vertex
				vertexes[1][2]=z;									// Set z Value Of Second Vertex

				x=float(coses[theta+1]*(2.0f+coses[phi+1] ))*r;					// Calculate x Position (1st Point)
				y=float(sins[theta+1]*(2.0f+coses[phi+1] ))*r;					// Calculate y Position (1st Point)
				z=float((( thetaa[theta+1]-(2.0f*3.142f)) + sins[phi+1] ) * r);		// Calculate z Position (1st Point)

				vertexes[2][0]=x;									// Set x Value Of Third Vertex
				vertexes[2][1]=y;									// Set y Value Of Third Vertex
				vertexes[2][2]=z;									// Set z Value Of Third Vertex

				x=float(coses[theta]*(2.0f+coses[phi+1] ))*r;					// Calculate x Position (1st Point)
				y=float(sins[theta]*(2.0f+coses[phi+1] ))*r;					// Calculate y Position (1st Point)
				z=float((( thetaa[theta]-(2.0f*3.142f)) + sins[phi+1] ) * r);		// Calculate z Position (1st Point)

				vertexes[3][0]=x;									// Set x Value Of Fourth Vertex
				vertexes[3][1]=y;									// Set y Value Of Fourth Vertex
				vertexes[3][2]=z;									// Set z Value Of Fourth Vertex
#endif
				// Render The Quad
				glVertex3f(vertexes[0][0], vertexes[0][1], vertexes[0][2]);
				glVertex3f(vertexes[1][0], vertexes[1][1], vertexes[1][2]);
				glVertex3f(vertexes[2][0], vertexes[2][1], vertexes[2][2]);
				glVertex3f(vertexes[3][0], vertexes[3][1], vertexes[3][2]);
			}
		}
		glEnd();													// Done Rendering Quads
		glTranslatef(0, 0, -offset[i]);
	}
}

void DrawGLRectangular(GLfloat ed, LevelOfDetail lod, bool bNoColor) {
	GLfloat matDensity[] = { (GLfloat)(ed * 0.0008 + (1000.0f - ed) * 0.0),
							 (GLfloat)(ed * 0.0 + (1000.0f - ed) * 0.0008),
							 0.0f	};
	if(!bNoColor) {
		if(ed < 0)
			glColor3f(0.0f, 0.3f, 1.0f);
		else
			glColor3f(matDensity[0], matDensity[1], matDensity[2]);
	}

	glTranslatef(0, -1.0, 0);

	glBegin(GL_QUADS);

	// Front
      glVertex3f( -1,  0,  1 );
      glVertex3f( -1,  2,  1 );
      glVertex3f(  1,  2,  1 );
      glVertex3f(  1,  0,  1 );

	// Back
	  glVertex3f( -1,  0, -1 );
	  glVertex3f(  1,  0, -1 );
	  glVertex3f(  1,  2, -1 );
	  glVertex3f( -1,  2, -1 );

	// Left side
	  glVertex3f( -1,  0,  1 );
	  glVertex3f( -1,  2,  1 );
	  glVertex3f( -1,  2, -1 );
	  glVertex3f( -1,  0, -1 );

	// Right side
	  glVertex3f(  1,  0,  1 );
	  glVertex3f(  1,  0, -1 );
	  glVertex3f(  1,  2, -1 );
	  glVertex3f(  1,  2,  1 );
	glEnd();

}

void DrawGLNLayeredAsymSlabs(GLfloat *rad, GLfloat *ed, GLfloat height, GLfloat x_width,
			GLfloat y_width, int n, LevelOfDetail lod, bool bNoColor) {
	// TODO: MAKE THIS CORRECT
	GLfloat totalRad = 0.0, cumRad = 0.0;
	if(!bNoColor)
		glColor3f(0.0f, 0.1f, 0.8f);

	for(int i = 1; i < n; i++)
		totalRad += rad[i];

	glTranslatef(-0.5 * x_width, -0.5 * y_width ,- height / 2.0f);

	glBegin(GL_TRIANGLES);

	for(int i = 1; i < n; i++) {
		GLfloat matDensity[] = { (GLfloat)(ed[i] * 0.0008 + (1000.0f - ed[i]) * 0.0),
								 (GLfloat)(ed[i] * 0.0 + (1000.0f - ed[i]) * 0.0008),
								 0.0f	};
		if(!bNoColor)
			glColor3fv(matDensity);

		// Lower plane
		glVertex3f(0.f, 0.f, cumRad);
		glVertex3f(0.f, y_width, cumRad);
		glVertex3f(x_width, 0.f, cumRad);
		glVertex3f(x_width, y_width, cumRad);
		glVertex3f(0.f, y_width, cumRad);
		glVertex3f(x_width, 0.f, cumRad);

		// Side plane YZ (x=0)
		glVertex3f(0.f, 0.f, cumRad);
		glVertex3f(0.f, y_width, cumRad);
		glVertex3f(0.f, 0.f, cumRad + rad[i]);
		glVertex3f(0.f, y_width, cumRad + rad[i]);
		glVertex3f(0.f, y_width, cumRad);
		glVertex3f(0.f, 0.f, cumRad + rad[i]);

		// Side plane YZ (x=x_width)
		glVertex3f(x_width, 0.f, cumRad);
		glVertex3f(x_width, y_width, cumRad);
		glVertex3f(x_width, 0.f, cumRad + rad[i]);
		glVertex3f(x_width, y_width, cumRad + rad[i]);
		glVertex3f(x_width, y_width, cumRad);
		glVertex3f(x_width, 0.f, cumRad + rad[i]);

		// Side plane XZ (y=0)
		glVertex3f(0.f,     0.f,    cumRad);
		glVertex3f(x_width, 0.f,    cumRad);
		glVertex3f(0.f,     0.f,    cumRad + rad[i]);
		glVertex3f(x_width, 0.f,    cumRad + rad[i]);
		glVertex3f(x_width, 0.f,    cumRad);
		glVertex3f(0.f,     0.f,    cumRad + rad[i]);

		// Side plane XZ (y=y_width)
		glVertex3f(0.f,     y_width,    cumRad);
		glVertex3f(x_width, y_width,    cumRad);
		glVertex3f(0.f,     y_width,    cumRad + rad[i]);
		glVertex3f(x_width, y_width,    cumRad + rad[i]);
		glVertex3f(x_width, y_width,    cumRad);
		glVertex3f(0.f,     y_width,    cumRad + rad[i]);

		// Upper plane
		glVertex3f(0.f, 0.f, cumRad + rad[i]);
		glVertex3f(0.f, y_width, cumRad + rad[i]);
		glVertex3f(x_width, 0.f, cumRad + rad[i]);
		glVertex3f(x_width, y_width, cumRad + rad[i]);
		glVertex3f(0.f, y_width, cumRad + rad[i]);
		glVertex3f(x_width, 0.f, cumRad + rad[i]);


		cumRad += rad[i];

	}

	glEnd();
}

void DrawGLNLayeredSlabs(GLfloat *rad, GLfloat *ed, GLfloat height, GLfloat x_width, GLfloat y_width,
	int n, LevelOfDetail lod, bool bNoColor) {
	// TODO: MAKE THIS CORRECT
	if(n <= 1)
		return;
	// 2n - 1 layers of asymmetric slabs
	GLfloat *newRad = new GLfloat[(2 * n) - 2];
	GLfloat *newED = new GLfloat[(2 * n) - 2];

	newRad[0] = rad[0];
	newED[0] = ed[0];
	height = 0.f;

	rad[1] *= 2.f; // Double the height of the inner layer

	for(int i = 1; i <= (n - 1); i++) {
		height += newRad[i] = rad[n - i];
		newED[i] = ed[n - i];
	}

	for(int i = n; i < ((2 * n) - 2); i++) {
		height += newRad[i] = rad[i - n + 2];
		newED[i] = ed[i - n + 2];
	}

	DrawGLNLayeredAsymSlabs(newRad, newED, height, x_width, y_width, (2 * n) - 2, lod, bNoColor);

	delete[] newRad;
	delete[] newED;
}
