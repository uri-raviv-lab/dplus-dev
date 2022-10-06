
#define _USE_MATH_DEFINES
#include <cmath>
#include <windows.h> // For OpenGL

#include "GraphPane3D.h"

#include "SymmetryView.h"

// So matrix4f can work
#define EIGEN_DONT_ALIGN
#include <Eigen/Core>
#include <Eigen/Geometry>


#include <gl/GL.h>
#include <gl/GLU.h>
#include "clrfunctionality.h"
#include <iostream>

#include "PDBReaderLib.h"
#include "ElectronPDBReaderLib.h"
#include "LuaBinding.h"



class GLUquadric {};

using System::IO::File;

namespace DPlus {

array<unsigned char> ^GraphPane3D::FileToBuffer(String ^filename) {
	array<unsigned char> ^data = nullptr;
	size_t sz = 0;
	try {
		data = System::IO::File::ReadAllBytes(filename);
		sz = data->Length;		
	} catch(Exception ^ex) {
		MessageBox::Show("Cannot open " + filename + " for reading: " + ex->Message);
		return nullptr;
	}

	if(sz == 0)
		return nullptr;

	return data;
}

Entity ^GraphPane3D::RegisterAMPGrid(String ^filename, LevelOfDetail lod, bool bCentered) {
	/* // Will throw out-of-memory exceptions if files are large enough
	// Read the Amplitude Grid to a buffer
	array<unsigned char> ^data = FileToBuffer(filename);	
	if(data == nullptr)
		return nullptr;
	size_t sz = data->Length;

	// .net buffer->C buffer
	cli::pin_ptr<unsigned char> vdata = &data[0];
	const char *buff = (const char *)vdata;	
	*/

	Entity ^result = gcnew Entity();
	result->type = EntityType::TYPE_AMPGRID;
	result->modelName = result->displayName= filename->Substring(filename->LastIndexOf("\\") + 1) + gcnew System::String(" (AMP)");
	result->filename = filename;
	result->bCentered = bCentered;
	result->renderDList = NULL;
	result->colorCodedDList = NULL;
	result->frontend = parentForm->frontend;
	result->job = parentForm->job;
	result->modelUI = new AMPModelUI(clrToString(result->modelName).c_str());

	ModelInformation mi = result->modelUI->GetModelInformation();
	paramStruct ps (mi);
	ps.params.resize(mi.nlp);

	// Set extra parameters' default values
	ps.extraParams.resize(mi.nExtraParams);
	for(int i = 0; i < mi.nExtraParams; i++) {			
		ExtraParam ep = result->modelUI->GetExtraParameter(i);
		ps.extraParams[i] = Parameter(ep.defaultVal, false, false, ep.rangeMin, ep.rangeMax);
	}

	// Finally, update the parameters in the entity
	result->SetParameters(ps, parentForm->GetLevelOfDetail());

	// Find the appropriate model file (PDB)
	bool bPDBFound = false;
	String ^pdbFile = "";

	if(filename->LastIndexOf(".") >= 0) {
		String ^fileNoExt = filename->Substring(0, filename->LastIndexOf("."));
		String ^fileWithPDBSuffix = "";
		
		if(filename->LastIndexOf("_") >= 0)
			fileWithPDBSuffix = filename->Substring(0, filename->LastIndexOf("_")) + "_pdb.pdb";

		if(File::Exists(fileNoExt + ".pdb")) {
			bPDBFound = true;
			pdbFile = fileNoExt + ".pdb";
		} else if(File::Exists(fileWithPDBSuffix)) {
			bPDBFound = true;
			pdbFile = fileWithPDBSuffix;
		}
	}

	if(bPDBFound) {
		result->modelfile = pdbFile;
		array<unsigned char> ^data = FileToBuffer(pdbFile);
		// We don't care if it fails, it will show a message box from within ReadPDBFile
		unsigned int tempDList = NULL, tempCCDList = NULL;
		ReadAndSetPDBFile(data, tempDList, tempCCDList, lod, bCentered, false);
		result->currentLOD = lod;
		result->renderDList = tempDList;
		result->colorCodedDList = tempCCDList;
	} else {
// 		System::Threading::Thread ^t = gcnew System::Threading::Thread(gcnew System::Threading::ThreadStart(&MessageBox::Show("No appropriate PDB file found. Amplitude grid "
// 			 						 "will not be presented in the 3D graph", "Warning", 
// 			 						 MessageBoxButtons::OK, MessageBoxIcon::Warning)));
// 		t->Start();
		//t->Join();
		if(DPlus::Scripting::IsConsoleOpen()) {
			std::cout << "No appropriate PDB file found. Amplitude grid will not be presented in the 3D graph.\n";
		} else {
 			MessageBox::Show("No appropriate PDB file found. Amplitude grid "
 							 "will not be presented in the 3D graph", "Warning", 
 							 MessageBoxButtons::OK, MessageBoxIcon::Warning);
		}
	}

	// Create the backend amplitude
	std::wstring wfname = clrToWstring(filename);
	std::vector<const wchar_t*> filenamePointers { wfname.c_str() };
	result->BackendModel = result->frontend->CreateFileAmplitude(parentForm->job, AF_AMPGRID, filenamePointers.data(), int(filenamePointers.size()), bCentered);

	return result;
}

Entity ^GraphPane3D::RegisterPDB(String ^filename, String ^anomfilename, LevelOfDetail lod, bool bCentered, bool electron) {
	
	// Read the PDB file to buffer
	array<unsigned char> ^data = FileToBuffer(filename);
	if(data == nullptr)
		return nullptr;
	size_t sz = data->Length;

	// Get the PDB buffer as C buffer
	cli::pin_ptr<unsigned char> vdata = &data[0];
	const char *buff = (const char *)vdata;	


	array<unsigned char> ^anomdata;
	cli::pin_ptr<unsigned char> anomvdata;
	const char *anombuff;
	size_t anomsz;
	if (anomfilename->Length > 0)
	{
		anomdata = FileToBuffer(anomfilename);
		if (anomdata == nullptr)
			return nullptr;
		anomsz = anomdata->Length;

		// Get the PDB buffer as C buffer
		anomvdata = &data[0];
		buff = (const char *)anomvdata;

		// Get the anomalous file buffer as C buffer
		anomvdata = &anomdata[0];
		anombuff = (const char *)anomvdata;
	}


	// Commit the entity
	Entity ^result = gcnew Entity();
	System::String ^pdbType("");
	if (electron)
	{
		result->type = EntityType::TYPE_EPDB;
		pdbType = pdbType + gcnew System::String("(electron PDB)");
	}
	else
	{
		result->type = EntityType::TYPE_PDB;
		pdbType = "(PDB)";
	}
	
	result->modelName=result->displayName = filename->Substring(filename->LastIndexOf("\\") + 1) + gcnew System::String(pdbType);
	result->filename = filename;
	result->anomfilename = anomfilename;
	result->modelfile = filename;
	result->bCentered = bCentered;
	result->frontend = parentForm->frontend;
	result->job = parentForm->job;

	unsigned int tempDList = NULL, tempCCDList = NULL;

	if (!ReadPDBFile(data, bCentered, result->xs, result->ys,
		result->zs, result->atoms, electron)
		||
		!SetPDBFile(tempDList, tempCCDList, result->xs, result->ys,
					result->zs, result->atoms, lod)
		) {
		return nullptr;
	}

	result->currentLOD = lod;
	result->renderDList = tempDList;
	result->colorCodedDList = tempCCDList;

	result->modelUI = new PDBModelUI(clrToString(result->modelName).c_str());

	ModelInformation mi = result->modelUI->GetModelInformation();
	paramStruct ps (mi);
	ps.params.resize(mi.nlp);

	// Set extra parameters' default values
	ps.extraParams.resize(mi.nExtraParams);
	for(int i = 0; i < mi.nExtraParams; i++) {			
		ExtraParam ep = result->modelUI->GetExtraParameter(i);
		ps.extraParams[i] = Parameter(ep.defaultVal, false, false, ep.rangeMin, ep.rangeMax);
	}

	// Finally, update the parameters in the entity
	result->SetParameters(ps, parentForm->GetLevelOfDetail());

	// Consolidate pointers
	std::vector<const char*> buffers { buff };
	std::vector<unsigned int> bufSizes{ unsigned int(sz) };

	std::string fnStr(clrToString(filename));
	std::string afnStr;
	std::vector<const char*> filenames{ fnStr.c_str() };

	std::vector<unsigned int> fnLens{ unsigned int(filename->Length) };

	if (anomfilename->Length)
	{
		buffers.push_back(anombuff);
		bufSizes.push_back(unsigned int(anomsz));
		afnStr = clrToString(anomfilename);
		filenames.push_back(afnStr.c_str());
		fnLens.push_back(anomfilename->Length);
	}

	// Create the backend amplitude
	//adds the filename
	result->BackendModel = result->frontend->CreateFileAmplitude(parentForm->job, AF_PDB, buffers.data(),
												bufSizes.data(), filenames.data(), fnLens.data(), int(fnLens.size()), bCentered);

	return result;
}

bool GraphPane3D::ReadAndSetPDBFile(array<unsigned char> ^data, unsigned int& dlist, unsigned int& CCDList,
							  LevelOfDetail lod, bool bCenterPDB, bool electron)
{
	std::vector<float> x, y, z;
	std::vector<u8> atoms;

	return 
		GraphPane3D::ReadPDBFile(data, bCenterPDB, &x, &y, &z, &atoms, electron) &&
		GraphPane3D::SetPDBFile(dlist, CCDList, &x, &y, &z, &atoms, lod);
}

bool GraphPane3D::ReadPDBFile(array<unsigned char> ^data, bool bCenterPDB,
	std::vector<float>* x, std::vector<float>* y, std::vector<float>* z, std::vector<u8>* atoms, bool electron)
{
	if (data == nullptr)
		return false;

	size_t sz = data->Length;

	// Get the PDB buffer as C buffer
	cli::pin_ptr<unsigned char> vdata = &data[0];
	const char *buff = (const char *)vdata;

	if (electron)
	{
		// THIS IF/ELSE WILL BE DELETED ONCE electronPDBReaderOb INHERITS PDBReaderOb
		ElectronPDBReader::electronPDBReaderOb<double> pdbTest;
		try
		{
			PDB_READER_ERRS err = pdbTest.readPDBbuffer(buff, sz, bCenterPDB);
			if (err != PDB_OK) {
				char a[256] = { 0 };
				sprintf_s(a, "Invalid PDB! (%d)", err);
				MessageBox::Show(gcnew String(a), "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return false;
			}
		}
		catch (ElectronPDBReader::pdbReader_exception& e)
		{
			char a[256] = { 0 };
			sprintf_s(a, "Invalid PDB! (%s)", e.GetErrorMessage().c_str());
			MessageBox::Show(gcnew String(a), "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return false;
		}
		pdbTest.getAtomsAndCoords(*x, *y, *z, *atoms);
		return true;
	}
	else
	{
		PDBReader::XRayPDBReaderOb<double> pdbTest;
		try
		{
			PDB_READER_ERRS err = pdbTest.readPDBbuffer(buff, sz, bCenterPDB);
			if (err != PDB_OK) {
				char a[256] = { 0 };
				sprintf_s(a, "Invalid PDB! (%d)", err);
				MessageBox::Show(gcnew String(a), "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return false;
			}
		}
		catch (PDBReader::pdbReader_exception& e)
		{
			char a[256] = { 0 };
			sprintf_s(a, "Invalid PDB! (%s)", e.GetErrorMessage().c_str());
			MessageBox::Show(gcnew String(a), "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return false;
		}
		pdbTest.getAtomsAndCoords(*x, *y, *z, *atoms);
		return true;
	}
	
}

bool GraphPane3D::SetPDBFile(unsigned int& dlist, unsigned int& CCDList, 
	std::vector<float>* x, std::vector<float>* y, std::vector<float>* z,
	std::vector<u8>* atoms, LevelOfDetail lod)
{
	unsigned int tempDList = glGenLists(1);
	glNewList(tempDList, GL_COMPILE);	
	GeneratePDBSpheres(*x, *y, *z, *atoms, lod, true);
	glEndList();

	unsigned int tempCCDList = glGenLists(1);
	glNewList(tempCCDList, GL_COMPILE);	
	GeneratePDBSpheres(*x, *y, *z, *atoms, lod, false);
	glEndList();

	dlist = tempDList;
	CCDList = tempCCDList;

	return true;
}

Eigen::MatrixXf EulerD(Radian theta, Radian phi, Radian psi) {
	float ax, ay, az, c1, c2, c3, s1, s2, s3;
	ax = theta;
	ay = phi  ;
	az = psi  ;
	c1 = cos(ax); s1 = sin(ax);
	c2 = cos(ay); s2 = sin(ay);
	c3 = cos(az); s3 = sin(az);
	Eigen::MatrixXf rot = Eigen::MatrixXf::Identity(4, 4);

	// Tait-Bryan angles X1Y2Z3 (x-alpha, y-beta, z-gamma)
	rot(0,0) = c2*c3;			rot(0,1) = -c2*s3;			rot(0,2) = s2;
	rot(1,0) = c1*s3+c3*s1*s2;	rot(1,1) = c1*c3-s1*s2*s3;	rot(1,2) = -c2*s1;
	rot(2,0) = s1*s3-c1*c3*s2;	rot(2,1) = c3*s1+c1*s2*s3;	rot(2,2) = c1*c2;
	return rot;
}

void Flatten(Entity ^root, Eigen::MatrixXf& mat,
			 System::Collections::Generic::LinkedList<Entity ^> ^outList)
{
	// Multiply new matrix from the left => Appends transformation
	Eigen::MatrixXf T = Eigen::MatrixXf::Identity(4, 4);				
	Eigen::MatrixXf R = EulerD(root->alpha(), root->beta(), root->gamma());

	T(0, 3) = float(root->x()); T(1, 3) = float(root->y()); T(2, 3) = float(root->z());

	mat = mat * T * R;

	// If symmetry, get root positions
	if(root->type == EntityType::TYPE_SYMMETRY) 
	{
		for each (LocationRotationCLI locrot in root->locs)
		{
			T(0, 3) = float(locrot.x); T(1, 3) = float(locrot.y); T(2, 3) = float(locrot.z);
			R = EulerD(Radian(locrot.radAlpha), Radian(locrot.radBeta), Radian(locrot.radGamma));

			// First rotate, then translate
			Eigen::MatrixXf newmat = mat * T * R;

			for each (Entity ^child in root->Nodes)
			{				
				// Multiply new matrix from the left - Append transformation
				Flatten(child, newmat,	outList);
			}
		}
		return;
	}

	Entity ^newent = gcnew Entity();
	newent->renderDList = root->renderDList;

	// Obtain final entity position (by transforming (0,0,0[,1]) with the matrix)
	Eigen::VectorXf origin = Eigen::VectorXf::Zero(4);
	origin[3] = 1.0f; // (0,0,0,1)
	origin = mat * origin;

	// For rotation, transform (1,0,0[,1]), (0,1,0[,1]) and (0,0,1[,1]).
	// The [,1] is for homogeneous coordinates
	// From (0,0,0) derive position
	// From the rest of the vectors, derive orientation	
	Eigen::VectorXf xbase = Eigen::VectorXf::Zero(4), xbasenew;
	Eigen::VectorXf ybase = Eigen::VectorXf::Zero(4), ybasenew;
	Eigen::VectorXf zbase = Eigen::VectorXf::Zero(4), zbasenew;
	xbase[0] = 1.0f; xbase[3] = 1.0f; // (1,0,0,1)
	ybase[1] = 1.0f; ybase[3] = 1.0f; // (0,1,0,1)
	zbase[2] = 1.0f; zbase[3] = 1.0f; // (0,0,1,1)
	xbasenew = (mat * xbase) - origin;
	ybasenew = (mat * ybase) - origin;
	zbasenew = (mat * zbase) - origin;

	/*System::Console::Clear();
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", xbasenew[0], ybasenew[0], zbasenew[0]);
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", xbasenew[1], ybasenew[1], zbasenew[1]);
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", xbasenew[2], ybasenew[2], zbasenew[2]);*/

	bool betaInRange = false;
	if(zbasenew[0] >= 0.0f)
		betaInRange = true; // 0 <= beta <= 180 (the first two quadrants)

	Radian betac1 =  Radian(  asin(zbasenew[0])				);
	Radian alphac1 = Radian(-atan2(zbasenew[1], zbasenew[2]));
	Radian gammac1 = Radian(-atan2(ybasenew[0], xbasenew[0]));

	Radian alpha, beta, gamma;
	alpha = (alphac1 < 0.0f) ? Radian(alphac1 + 2.0 * M_PI) : alphac1;
	beta  = (betac1  < 0.0f) ? Radian(betac1  + 2.0 * M_PI) : betac1;
	gamma = (gammac1 < 0.0f) ? Radian(gammac1 + 2.0 * M_PI) : gammac1;

	// If beta is in the 2nd or 3rd quadrants, we should add 180 modulo 360 to GAMMA and ALPHA
	// but we cannot know because of how rotation matrices work with Euler/Tait-Bryan angles.

	/*System::Console::WriteLine("Alpha: " + alpha);
	System::Console::WriteLine("Beta: " + beta);
	System::Console::WriteLine("Gamma: " + gamma);

	Eigen::MatrixXf mat1 = EulerD(alpha, beta, gamma);
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", mat1(0,0), mat1(0,1), mat1(0,2));
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", mat1(1,0), mat1(1,1), mat1(1,2));
	System::Console::WriteLine("({0:F6})   ({1:F6})   ({2:F6})", mat1(2,0), mat1(2,1), mat1(2,2));
*/


	newent->SetX(origin.x());// + originX);
	newent->SetY(origin.y());// + originY);
	newent->SetZ(origin.z());// + originZ);
	newent->SetAlpha(alpha);// + originA);
	newent->SetBeta (beta );// + originB);
	newent->SetGamma(gamma);// + originG);

	outList->AddLast(newent);
}


void GraphPane3D::glCanvas3D1_Render(System::Object^ /*unused*/, GLView::GLGraphics3D ^graphics) {
	// DEBUG: Show debugging info color-coded render as real-world render
	if(bColorCodedRender) {
		glDisable (GL_DITHER);
		glDisable (GL_LIGHTING);
		
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glCanvas3D1_ColorCodedRender(this, 
									 gcnew GLView::GLGraphics3D(glCanvas3D1, this->CreateGraphics(), true));

		glEnable (GL_DITHER);
		glEnable (GL_LIGHTING);

		return;
	}

	if(bFlatRender)
	{
		System::Collections::Generic::LinkedList<Entity ^> ^flatList = gcnew System::Collections::Generic::LinkedList<Entity ^>();
		
		// Traverse the primitive/symmetry tree for drawing
		for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		{
			// Flatten the tree
			Eigen::MatrixXf mat = Eigen::MatrixXf::Identity(4, 4);
			

			Flatten((Entity ^)parentForm->entityTree->Nodes[i], mat, flatList);
		}
		DrawEntitiesFlat(flatList, graphics);
		return;
	}

	// Traverse the primitive/symmetry tree for drawing
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		DrawEntities((Entity ^)parentForm->entityTree->Nodes[i], graphics);
}

static void InnerInvalidateEntities(Entity ^root, LevelOfDetail lod) {
	for each (Entity ^ent in root->Nodes)
		InnerInvalidateEntities(ent, lod);

	root->Invalidate(lod, false);
}

void GraphPane3D::InvalidateEntities(LevelOfDetail lod) {
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		InnerInvalidateEntities((Entity ^)parentForm->entityTree->Nodes[i], lod);
}

void ExtractEulerAngles(float mat[], float& alpha, float& beta, float& gamma)
{
	const float RADIANS = 180.0f / 3.1415926f;

	float C, D, trx, ry;

	beta = D =  asin( mat[2]);        /* Calculate Y-axis angle */
	C           =  cos( beta );
	beta    *=  RADIANS;

	if ( fabs( C ) > 0.005 )             /* Gimball lock? */
	{
		trx      =  mat[10] / C;           /* No, so get X-axis angle */
		ry      = -mat[6]  / C;

		alpha  = atan2( ry, trx ) * RADIANS;

		trx      =  mat[0] / C;            /* Get Z-axis angle */
		ry      = -mat[1] / C;

		gamma  = atan2( ry, trx ) * RADIANS;
	}
	else                                 /* Gimball lock has occurred */
	{
		alpha  = 0;                      /* Set X-axis angle to zero */

		trx      =  mat[5];                 /* And calculate Z-axis angle */
		ry      =  mat[4];

		gamma  = atan2( ry, trx ) * RADIANS;
	}

	/* return only positive angles in [0,360] */
	if (alpha < 0) alpha += 360;
	if (beta < 0) beta += 360;
	if (gamma < 0) gamma += 360;
	/*
	alpha = 360 - alpha;
	beta = 360 - beta;
	gamma = 360 - gamma;
	*/
}

void GraphPane3D::DrawEntitiesFlat(System::Collections::Generic::LinkedList<Entity ^> ^flatList, GLView::GLGraphics3D ^graphics) 
{
	
	glEnable(GL_BLEND);

	// In the end
	for each(Entity ^ent in flatList) 
	{
		// Save last rendering position
		glPushMatrix();

		// Translate and rotate
		glTranslated(ent->x(), ent->y(), ent->z());
		glRotated(Degree(ent->alpha()).deg, 1.0, 0.0, 0.0);
		glRotated(Degree(ent->beta() ).deg,  0.0, 1.0, 0.0);
		glRotated(Degree(ent->gamma()).deg, 0.0, 0.0, 1.0);

		/*float mat[16] = {0};
		glGetFloatv(GL_MODELVIEW_MATRIX, mat);
		System::Console::Clear();
		System::Console::WriteLine("X " + ent->x() + " Y " + ent->y() + " Z " + ent->z());
		System::Console::WriteLine("A " + ent->alpha() + " B " + ent->beta() + " G " + ent->gamma());

		float alpha, beta, gamma;
		ExtractEulerAngles(mat, alpha, beta, gamma);

		System::Console::WriteLine("Extracted: X " + mat[12] + " Y " + mat[13] + " Z " + mat[14]);
		System::Console::WriteLine("Extracted: A " + alpha + " B " + beta + " G " + gamma);*/

		
		if(ent->renderDList)
			glCallList(ent->renderDList);

		glPopMatrix();
	}
}

void GraphPane3D::DrawEntities(Entity ^root, GLView::GLGraphics3D ^graphics) {
	// Save last rendering position
	glPushMatrix();

	glEnable(GL_BLEND);

	// Translate and rotate
	glTranslated(root->x(), root->y(), root->z());
	glRotated(Degree(root->alpha()).deg, 1.0, 0.0, 0.0);
	glRotated(Degree(root->beta() ).deg,  0.0, 1.0, 0.0);
	glRotated(Degree(root->gamma()).deg, 0.0, 0.0, 1.0);

	// Render the entity's children
	//////////////////////////////////////////////////////////////////////////
	// If symmetry, get children positions
	if(root->type == EntityType::TYPE_SYMMETRY) {
		/*if(root->renderDList) {
			GLint oldsrc, olddst;
			glGetIntegerv(GL_BLEND_SRC, &oldsrc);
			glGetIntegerv(GL_BLEND_DST, &olddst);
			if(root->selected)		
				glBlendFunc(GL_DST_COLOR, GL_ONE);		

			glCallList(root->renderDList);
	
			if(root->selected)
				glBlendFunc(oldsrc, olddst);
		} else {
			unsigned int tempDList = glGenLists(1);
			glNewList(tempDList, GL_COMPILE);*/

			for(int i = 0; i < root->Nodes->Count; i++) {
				Entity ^child = (Entity ^)root->Nodes[i];

				bool oldSelected = child->selected;
				if(root->selected)
					child->selected = true;

				// Translate and rotate to the location
				for each (LocationRotationCLI locrot in root->locs) {
					glPushMatrix();

					glTranslated(locrot.x, locrot.y, locrot.z);
					glRotated(Degree(float(locrot.radAlpha)).deg, 1.0, 0.0, 0.0);
					glRotated(Degree(float(locrot.radBeta )).deg, 0.0, 1.0, 0.0);
					glRotated(Degree(float(locrot.radGamma)).deg, 0.0, 0.0, 1.0);

					DrawEntities(child, graphics);		

					glPopMatrix();
				}

				child->selected = oldSelected;
			}

			/*glEndList();
			root->renderDList = tempDList;
		}*/
	}

	// Render the entity itself
	//////////////////////////////////////////////////////////////////////////	
	
	// Selected object identifier in 3D (makes objects highlighted)
	GLint oldsrc, olddst;
	glGetIntegerv(GL_BLEND_SRC, &oldsrc);
	glGetIntegerv(GL_BLEND_DST, &olddst);

	if(root->selected)		
		glBlendFunc(GL_DST_COLOR, GL_ONE);

	if(root->renderDList)
		glCallList(root->renderDList);
	
	if(root->selected)
		glBlendFunc(oldsrc, olddst);

	glDisable(GL_BLEND);

	// Pop back the last rendering position
	glPopMatrix();
}

void GraphPane3D::DrawEntitiesColorCoded(Entity ^root, GLView::GLGraphics3D ^graphics) {
	// Every object has a DList
	// Every symmetry draws its inside DLists multiple times in different locations and rotations
	// Every DList has its own color
	

	// Save last rendering position
	glPushMatrix();

	// Translate and rotate
	glTranslated(root->x(), root->y(), root->z());
	glRotated(Degree(root->alpha()).deg, 1.0, 0.0, 0.0);
	glRotated(Degree(root->beta() ).deg,  0.0, 1.0, 0.0);
	glRotated(Degree(root->gamma()).deg, 0.0, 0.0, 1.0);

	// Render the entity's children
	//////////////////////////////////////////////////////////////////////////
	// If symmetry, get children positions
	if(root->type == EntityType::TYPE_SYMMETRY) {
		for(int i = 0; i < root->Nodes->Count; i++) {
			Entity ^child = (Entity ^)root->Nodes[i];

			// Translate and rotate to the location
			for each (LocationRotationCLI locrot in root->locs) {
				glPushMatrix();

				glTranslated(locrot.x, locrot.y, locrot.z);
				glRotated(Degree(float(locrot.radAlpha)).deg, 1.0, 0.0, 0.0);
				glRotated(Degree(float(locrot.radBeta )).deg, 0.0, 1.0, 0.0);
				glRotated(Degree(float(locrot.radGamma)).deg, 0.0, 0.0, 1.0);

				DrawEntitiesColorCoded(child, graphics);

				glPopMatrix();
			}
		}
	}

	// Render the entity itself
	//////////////////////////////////////////////////////////////////////////

	// Register the following OpenGL calls as a separate object (for selection)
	graphics->BeginSelectableObject((Object ^)root);
	
	if(root->colorCodedDList)
		glCallList(root->colorCodedDList);

	graphics->EndSelectableObject();

	// Pop back the last rendering position
	glPopMatrix();
}

System::Void GraphPane3D::glCanvas3D1_ColorCodedRender(System::Object^ sender, GLView::GLGraphics3D^ graphics) {
	// Traverse the primitive/symmetry tree for drawing
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		DrawEntitiesColorCoded((Entity ^)parentForm->entityTree->Nodes[i], graphics);
}

static void ClearEntitySelection(Entity ^ent) {
	ent->selected = false;

	for(int i = 0; i < ent->Nodes->Count; i++)
		ClearEntitySelection((Entity ^)ent->Nodes[i]);
}

System::Void GraphPane3D::glCanvas3D1_SelectionChanged(System::Object^ sender, System::EventArgs^ e) {
	SymmetryView^ sv = nullptr;
	sv  = dynamic_cast<SymmetryView^>(parentForm->PaneList[SYMMETRY_VIEWER]);

	// Only select the objects that were selected in the 3D GUI
	sv->treeViewAdv1->ClearSelection();
		
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		ClearEntitySelection((Entity ^)parentForm->entityTree->Nodes[i]);

	for(int i = 0; i < glCanvas3D1->SelectedObjects->Count; i++) {		
		if(glCanvas3D1->SelectedObjects[i] != nullptr) {
			Entity ^ent = (Entity ^)glCanvas3D1->SelectedObjects[i];
			ent->selected = true;

			sv->treeViewAdv1->FindNodeByTag(ent)->IsSelected = true;
		}
	}
}


System::Void GraphPane3D::GeneratePDBSpheres(std::vector<float> x, std::vector<float> y, std::vector<float> z,
										     std::vector<u8> atoms, LevelOfDetail lod, bool bColored) {
	if(x.size() != y.size() ||
		x.size() != z.size() ||
		x.size() != atoms.size())
		return;

	GLUquadric* q = NULL;
	
	if(lod != LOD_NONE)
		q = gluNewQuadric();	
	else
		glBegin(GL_TRIANGLE_STRIP);


	int pdbSize = (int)x.size();
	float rad = 0.0;
	Drawing::Color defCol = Drawing::Color::Pink, col;

	for(int i = 0; i < pdbSize; i++) {
		col = defCol;
		rad = 300.0;
		switch(atoms[i]) {
			//number 	Used	symbol 	name 	empirical	Calculated 	van der Waals 	Covalent	Metallic
				default:
// 					rad = 120.0;
// 					col = Drawing::Color::Black;
					break;			
				case 1:	//H 	hydrogen 	25	53	120	38	
					rad = 120.0;
					col = Drawing::Color::White;
					break;
				case 2:	//He 	helium 	no data 	31	140	32	
					rad = 140.0;
					col = Drawing::Color::Cyan;
					break;
				case 3:	//Li 	lithium 	145	167	182	134	152
					rad = 182.0;
					col = Drawing::Color::Violet;
					break;
				case 4:	//Be 	beryllium 	105	112	153	90	112
					rad = 153.0;
					col = Drawing::Color::DarkGreen;
					break;
				case 5:	//B 	boron 	85	87	192	82	
					rad = 192.0;
					col = Drawing::Color::Salmon;
					break;
				case 6:	//C 	carbon 	70	67	170	77	
				case 119:
				case 120:
				case 121:
					rad = 170.0;
					col = Drawing::Color::Black;
					break;
				case 7:	//N 	nitrogen 	65	56	155	75	
				case 122:
				case 123:
				case 124:
					rad = 155.0;
					col = Drawing::Color::DarkBlue;
					break;
				case 8:	//O 	oxygen 	60	48	152	73	
				case 125:
					rad = 152.0;
					col = Drawing::Color::Red;
					break;
				case 9:	//F 	fluorine 	50	42	147	71	
					rad = 147.0;
					col = Drawing::Color::Green;
					break;
				case 10:	//Ne 	neon 	no data 	38	154	69	
					rad = 154.0;
					col = Drawing::Color::Cyan;
					break;
				case 11:	//Na 	sodium 	180	190	227	154	186
					rad = 227.0;
//					col = Drawing::Color::Gray;
					col = Drawing::Color::Violet;
					break;
				case 12:	//Mg 	magnesium 	150	145	173	130	160
					rad = 173.0;
					col = Drawing::Color::DarkGreen;
					break;
				case 13:	//Al 	aluminium 	125	118	184	118	143
					rad = 184.0;
					break;
				case 14:	//Si 	silicon 	110	111	210	111	
					rad = 210.0;
					break;
				case 15:	//P 	phosphorus 	100	98	180	106	
					rad = 180.0;
					col = Drawing::Color::Orange;
					break;
				case 16:	//S 	sulfur 	100	88	180	102	
				case 126:
					rad = 180.0;
					col = Drawing::Color::Yellow;
					break;
				case 17:	//Cl 	chlorine 	100	79	175	99	
					rad = 175.0;
					col = Drawing::Color::Green;
					break;
				case 18:	//Ar 	argon 	71	71	188	97	
					rad = 188.0;
					col = Drawing::Color::Cyan;
					break;
				case 19:	//K 	potassium 	220	243	275	196	227
					rad = 275.0;
					col = Drawing::Color::Violet;
					break;
				case 20:	//Ca 	calcium 	180	194	231	174	197
					rad = 231.0;
					col = Drawing::Color::NavajoWhite;
					col = Drawing::Color::DarkGreen;
					break;
				case 21:	//Sc 	scandium 	160	184	211	144	162
					rad = 211.0;
					break;
				case 22:	//Ti 	titanium 	140	176	no data 	136	147
					rad = 201.0;
					col = Drawing::Color::Gray;
					break;
				case 23:	//V 	vanadium 	135	171	no data 	125	134
					rad = 196.0;
					break;
				case 24:	//Cr 	chromium 	140	166	no data 	127	128
					rad = 191.0;
					break;
				case 25:	//Mn 	manganese 	140	161	no data 	139	127
					rad = 186.0;
					break;
				case 26:	//Fe 	iron 	140	156	no data 	125	126
					rad = 181.0;
					col = Drawing::Color::Orange;
					break;
				case 27:	//Co 	cobalt 	135	152	no data 	126	125
					rad = 177.0;
					break;
				case 28:	//Ni 	nickel 	135	149	163	121	124
					rad = 163.0;
					break;
				case 29:	//Cu 	copper 	135	145	140	138	128
					rad = 140.0;
					break;
				case 30:	//Zn 	zinc 	135	142	139	131	134
					rad = 139.0;
					break;
				case 31:	//Ga 	gallium 	130	136	187	126	135
					rad = 187.0;
					break;
				case 32:	//Ge 	germanium 	125	125	211	122	
					rad = 211.0;
					break;
				case 33:	//As 	arsenic 	115	114	185	119	
					rad = 185.0;
					break;
				case 34:	//Se 	selenium 	115	103	190	116	
					rad = 190.0;
					break;
				case 35:	//Br 	bromine 	115	94	185	114	
					rad = 185.0;
					col = Drawing::Color::DarkRed;
					break;
				case 36:	//Kr 	krypton 	no data 	88	202	110	
					rad = 202.0;
					col = Drawing::Color::Cyan;
					break;
				case 37:	//Rb 	rubidium 	235	265	303	211	248
					rad = 303.0;
					col = Drawing::Color::Violet;
					break;
				case 38:	//Sr 	strontium 	200	219	249	192	215
					rad = 249.0;
					col = Drawing::Color::DarkGreen;
					break;
				case 39:	//Y 	yttrium 	180	212	no data 	162	180
					rad = 237.0;
					break;
				case 40:	//Zr 	zirconium 	155	206	no data 	148	160
					rad = 231.0;
					break;
				case 41:	//Nb 	niobium 	145	198	no data 	137	146
					rad = 223.0;
					break;
				case 42:	//Mo 	molybdenum 	145	190	no data 	145	139
					rad = 215.0;
					break;
				case 43:	//Tc 	technetium 	135	183	no data 	156	136
					rad = 208.0;
					break;
				case 44:	//Ru 	ruthenium 	130	178	no data 	126	134
					rad = 203.0;
					break;
				case 45:	//Rh 	rhodium 	135	173	no data 	135	134
					rad = 198.0;
					break;
				case 46:	//Pd 	palladium 	140	169	163	131	137
					rad = 163.0;
					break;
				case 47:	//Ag 	silver 	160	165	172	153	144
					rad = 172.0;
					break;
				case 48:	//Cd 	cadmium 	155	161	158	148	151
					rad = 158.0;
					break;
				case 49:	//In 	indium 	155	156	193	144	167
					rad = 193.0;
					break;
				case 50:	//Sn 	tin 	145	145	217	141	
					rad = 217.0;
					break;
				case 51:	//Sb 	antimony 	145	133	206	138	
					rad = 206.0;
					break;
				case 52:	//Te 	tellurium 	140	123	206	135	
					rad = 206.0;
					break;
				case 53:	//I 	iodine 	140	115	198	133	
					rad = 198.0;
					col = Drawing::Color::DarkViolet;
					break;
				case 54:	//Xe 	xenon 	no data 	108	216	130	
					rad = 216.0;
					col = Drawing::Color::Cyan;
					break;
				case 55:	//Cs 	cesium 	260	298	343	225	265
					rad = 343.0;
					col = Drawing::Color::Violet;
					break;
				case 56:	//Ba 	barium 	215	253	268	198	222
					rad = 268.0;
					col = Drawing::Color::DarkGreen;
					break;
				case 57:	//La 	lanthanum 	195	no data 	no data 	169	187
					rad = 250.0;
					break;
				case 58:	//Ce 	cerium 	185	no data 	no data 	no data 	181.8
					rad = 250.0;
					break;
				case 59:	//Pr 	praseodymium 	185	247	no data 	no data 	182.4
					rad = 250.0;
					break;
				case 60:	//Nd 	neodymium 	185	206	no data 	no data 	181.4
					rad = 250.0;
					break;
				case 61:	//Pm 	promethium 	185	205	no data 	no data 	183.4
					rad = 250.0;
					break;
				case 62:	//Sm 	samarium 	185	238	no data 	no data 	180.4
					rad = 250.0;
					break;
				case 63:	//Eu 	europium 	185	231	no data 	no data 	180.4
					rad = 250.0;
					break;
				case 64:	//Gd 	gadolinium 	180	233	no data 	no data 	180.4
					rad = 250.0;
					break;
				case 65:	//Tb 	terbium 	175	225	no data 	no data 	177.3
					rad = 250.0;
					break;
				case 66:	//Dy 	dysprosium 	175	228	no data 	no data 	178.1
					rad = 250.0;
					break;
				case 67:	//Ho 	holmium 	175	no data 	no data 	no data 	176.2
					rad = 250.0;
					break;
				case 68:	//Er 	erbium 	175	226	no data 	no data 	176.1
					rad = 250.0;
					break;
				case 69:	//Tm 	thulium 	175	222	no data 	no data 	175.9
					rad = 250.0;
					break;
				case 70:	//Yb 	ytterbium 	175	222	no data 	no data 	176
					rad = 250.0;
					break;
				case 71:	//Lu 	lutetium 	175	217	no data 	160	173.8
					rad = 250.0;
					break;
				case 72:	//Hf 	hafnium 	155	208	no data 	150	159
					rad = 250.0;
					break;
				case 73:	//Ta 	tantalum 	145	200	no data 	138	146
					rad = 250.0;
					break;
				case 74:	//W 	tungsten 	135	193	no data 	146	139
					rad = 250.0;
					break;
				case 75:	//Re 	rhenium 	135	188	no data 	159	137
					rad = 250.0;
					break;
				case 76:	//Os 	osmium 	130	185	no data 	128	135
					rad = 250.0;
					break;
				case 77:	//Ir 	iridium 	135	180	no data 	137	135.5
					rad = 250.0;
					break;
				case 78:	//Pt 	platinum 	135	177	175	128	138.5
					rad = 175.0;
					break;
				case 79:	//Au 	gold 	135	174	166	144	144
					rad = 166.0;
					break;
				case 80:	//Hg 	mercury 	150	171	155	149	151
					rad = 155.0;
					break;
				case 81:	//Tl 	thallium 	190	156	196	148	170
					rad = 196.0;
					break;
				case 82:	//Pb 	lead 	180	154	202	147	
					rad = 202.0;
					break;
				case 83:	//Bi 	bismuth 	160	143	207	146	
					rad = 207.0;
					break;
				case 84:	//Po 	polonium 	190	135	197	no data 	
					rad = 197.0;
					break;
				case 85:	//At 	astatine 	no data 	no data 	202	no data 	
					rad = 202.0;
					break;
				case 86:	//Rn 	radon 	no data 	120	220	145	
					rad = 220.0;
					break;
				case 87:	//Fr 	francium 	no data 	no data 	348	no data 	no data
					rad = 348.0;
					break;
				case 88:	//Ra 	radium 	215	no data 	283	no data 	no data
					rad = 283.0;
					col = Drawing::Color::DarkGreen;
					break;
				case 89:	//Ac 	actinium 	195	no data 	no data 	no data 	
					rad = 250.0;
					break;
				case 90:	//Th 	thorium 	180	no data 	no data 	no data 	179
					rad = 250.0;
					break;
				case 91:	//Pa 	protactinium 	180	no data 	no data 	no data 	163 d
					rad = 250.0;
					break;
				case 92:	//U 	uranium 	175	no data 	186	no data 	156 e
					rad = 186.0;
					break;
				case 93:	//Np 	neptunium 	175	no data 	no data 	no data 	155 e
					rad = 225.0;
					break;
				case 94:	//Pu 	plutonium 	175	no data 	no data 	no data 	159 e
					rad = 250.0;
					break;
				case 95:	//Am 	americium 	175	no data 	no data 	no data 	173
					rad = 225.0;
					break;
				case 96:	//Cm 	curium 	no data 	no data 	no data 	no data 	174
					rad = 250.0;
					break;
				case 97:	//Bk 	berkelium 	no data 	no data 	no data 	no data 	170
					rad = 250.0;
					break;
				case 98:	//Cf 	californium 	no data 	no data 	no data 	no data 	186+/- 2
					rad = 250.0;
					break;
				case 99:	//Es 	einsteinium 	no data 	no data 	no data 	no data 	186+/- 2
					rad = 250.0;
					break;
		}

		glTranslatef(x[i], y[i], z[i]);

		if(bColored)
			glColor4f((float)col.R / 256.0f, (float)col.G / 256.0f, (float)col.B / 256.0f, (float)col.A / 256.0f);
		
		switch(lod)
		{		
		default:
		case LOD_NONE:
			glVertex3d(x[i], y[i], z[i]);
			break;
		
		case LOD_VERYLOW:
			gluSphere(q, (float)(rad / 1000.0f), 3, 3);
			break;
		case LOD_LOW:
			gluSphere(q, (float)(rad / 1000.0f), 4, 4);
			break;
		case LOD_MEDIUM:
			gluSphere(q, (float)(rad / 1000.0f), 8, 8);
		case LOD_HIGH:
			gluSphere(q, (float)(rad / 1000.0f), 16, 16);
			break;
		}

		glTranslatef(-x[i], -y[i], -z[i]);
	}		

	if(lod == LOD_NONE)
		glEnd();
	else
		gluDeleteQuadric(q);
}

System::Void GraphPane3D::glCanvas3D1_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back) {
		((SymmetryView ^)this->parentForm->PaneList[SYMMETRY_VIEWER])->RemoveSelectedNodes();
		e->Handled = true;
	}
}

void GraphPane3D::SaveViewGraphicToFile( System::Object^ sender, GLView::GLGraphics3D^ Graphics )
{
/*
	GLView::GLCanvas3D^ can = gcnew GLView::GLCanvas3D();
	Drawing::Graphics ^gr;
	Panel ^pn = gcnew Panel();
	can->ViewDistance = glCanvas3D1->ViewDistance;
	//gr = gcnew System::Drawing::Graphics::CopyFromScreen(0,0,0,0,gcnew Drawing::Size(2,2),Drawing::CopyPixelOperation::Blackness);
	gr = pn->CreateGraphics();
	GLView::GLGraphics3D^ toRender = gcnew GLView::GLGraphics3D(can, gr, false);
//	glRenderForFile(toRender);
	glRenderForFile(gcnew GLView::GLGraphics3D(glCanvas3D1, gr, false));
*/
	
//	Bitmap^ bm = gcnew Bitmap(glCanvas3D1->Bounds.Width, glCanvas3D1->Bounds.Height, gr);
/*
	Bitmap^ bm = gcnew Bitmap(glCanvas3D1->Bounds.Width, glCanvas3D1->Bounds.Height, glCanvas3D1->CreateGraphics());
	this->glCanvas3D1->DrawToBitmap(bm, glCanvas3D1->Bounds);
	bm->Save("C:\\Delete\\TEST_IMAGE.png", System::Drawing::Imaging::ImageFormat::Png);

	Bitmap^ bm2 = gcnew Bitmap(glCanvas3D1->Bounds.Width, glCanvas3D1->Bounds.Height, this->CreateGraphics());
	GLView::GLGraphics3D^ gr2 = gcnew GLView::GLGraphics3D(glCanvas3D1, this->CreateGraphics(), false);
	glRenderForFile(gr2);
	this->glCanvas3D1->DrawToBitmap(bm, glCanvas3D1->Bounds);
	bm->Save("C:\\Delete\\TEST_IMAGE2.png", System::Drawing::Imaging::ImageFormat::Png);


	GLView::GLGraphics3D ^ graphics = gcnew GLView::GLGraphics3D(gcnew GLView::GLCanvas3D(), CreateGraphics(), false);
	graphics->Render(this, graphics);
*/

}

void GraphPane3D::glRenderForFile( GLView::GLGraphics3D^ graphics ) {
	// Traverse the primitive/symmetry tree for drawing
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		DrawEntities((Entity ^)parentForm->entityTree->Nodes[i], graphics);
}


};
