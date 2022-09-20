#pragma once

#include "MainWindow.h"
#include "Entity.h"
#include <vector>

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;

namespace DPlus {

	/// <summary>
	/// Summary for GraphPane3D
	/// </summary>
	public ref class GraphPane3D : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
		unsigned int pdbDList, ccpdbDList;
	public:
		
		GraphPane3D(MainWindow ^pform)
		{
			InitializeComponent();
			
			parentForm = pform;
			ccpdbDList = pdbDList = 0;
			bColorCodedRender = false;
			bFlatRender = false;
		}

		// Add a new PDB or Amplitude grid to the parameter tree
		Entity ^RegisterPDB(String ^filename, String ^anomfilename, LevelOfDetail lod, bool bCentered, bool electron);
		Entity ^RegisterAMPGrid(String ^filename, LevelOfDetail lod, bool bCentered);

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~GraphPane3D()
		{
			if (components)
			{
				delete components;
			}
		}
	public: GLView::GLCanvas3D^  glCanvas3D1;
			bool bColorCodedRender, bFlatRender;
	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(GraphPane3D::typeid));
			this->glCanvas3D1 = (gcnew GLView::GLCanvas3D());
			this->SuspendLayout();
			// 
			// glCanvas3D1
			// 
			this->glCanvas3D1->BackColor = System::Drawing::Color::DimGray;
			this->glCanvas3D1->Cursor = System::Windows::Forms::Cursors::Default;
			this->glCanvas3D1->Distance = 8.660254F;
			this->glCanvas3D1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->glCanvas3D1->DrawFloor = false;
			this->glCanvas3D1->Location = System::Drawing::Point(0, 0);
			this->glCanvas3D1->MaxSelectedObjects = static_cast<System::UInt32>(0);
			this->glCanvas3D1->MouseSelect = true;
			this->glCanvas3D1->Name = L"glCanvas3D1";
			this->glCanvas3D1->Pitch = 35.26439F;
			this->glCanvas3D1->Roll = 0;
			this->glCanvas3D1->SelectedObjects = (cli::safe_cast<System::Collections::Generic::List<System::Object^ >^  >(resources->GetObject(L"glCanvas3D1.SelectedObjects")));
			this->glCanvas3D1->Size = System::Drawing::Size(431, 452);
			this->glCanvas3D1->TabIndex = 0;
			this->glCanvas3D1->Yaw = 225;
			this->glCanvas3D1->SelectionChanged += gcnew System::EventHandler(this, &GraphPane3D::glCanvas3D1_SelectionChanged);
			this->glCanvas3D1->ColorCodedRender += gcnew GLView::GLCanvas3D::RenderHandler(this, &GraphPane3D::glCanvas3D1_ColorCodedRender);
			this->glCanvas3D1->Render += gcnew GLView::GLCanvas3D::RenderHandler(this, &GraphPane3D::glCanvas3D1_Render);
			this->glCanvas3D1->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &GraphPane3D::glCanvas3D1_KeyDown);
			// 
			// GraphPane3D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(431, 452);
			this->Controls->Add(this->glCanvas3D1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"GraphPane3D";
			this->ShowIcon = false;
			this->Text = L"3D Graph";
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void glCanvas3D1_Render(System::Object^  sender, GLView::GLGraphics3D^  Graphics);			 
	private: System::Void glCanvas3D1_SelectionChanged(System::Object^  sender, System::EventArgs^  e);
	private: System::Void glCanvas3D1_ColorCodedRender(System::Object^  sender, GLView::GLGraphics3D^  Graphics);
			 void DrawEntities(Entity ^root, GLView::GLGraphics3D ^graphics);
			 void DrawEntitiesColorCoded(Entity ^root, GLView::GLGraphics3D ^graphics);
			 void DrawEntitiesFlat(System::Collections::Generic::LinkedList<Entity ^> ^flatList, GLView::GLGraphics3D ^graphics);
	public:	 static array<unsigned char> ^GraphPane3D::FileToBuffer(String ^filename);
			 static bool GraphPane3D::ReadAndSetPDBFile(array<unsigned char> ^data, unsigned int& dlist, unsigned int& CCDList,
										   LevelOfDetail lod, bool bCenterPDB, bool electron);
			 static bool GraphPane3D::ReadPDBFile(array<unsigned char> ^data, bool bCenterPDB,
				 std::vector<float>* x, std::vector<float>* y, std::vector<float>* z, std::vector<u8>* atoms, bool electron);
			 static bool GraphPane3D::SetPDBFile(unsigned int& dlist, unsigned int& CCDList,
				 std::vector<float>* x, std::vector<float>* y, std::vector<float>* z,
				 std::vector<u8>* atoms, LevelOfDetail lod);
			 static System::Void GeneratePDBSpheres(std::vector<float> x, std::vector<float> y,
													std::vector<float> z, std::vector<u8> atoms, LevelOfDetail lod,
													bool bColored);
	private: System::Void glCanvas3D1_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e);
	public:  void InvalidateEntities(LevelOfDetail lod);
	public:  void SaveViewGraphicToFile(System::Object^  sender, GLView::GLGraphics3D^  Graphics);
	public:	 void glRenderForFile(GLView::GLGraphics3D^  Graphics);
	};
}
