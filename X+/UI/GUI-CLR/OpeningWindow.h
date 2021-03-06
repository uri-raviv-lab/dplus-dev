#pragma once

#include "OpenGLControl.h"
#include "UIsettings.h"
#include "ModelUI.h"
#include "ExternalModelDialog.h"
#include "GUIHelperClasses.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

using namespace GUICLR;

// Forward declaration
class FFModel;

namespace GUICLR {

	/// <summary>
	/// Summary for OpeningWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class OpeningWindow : public System::Windows::Forms::Form
	{
	public:
		//FFModel *_currentModel;
		array<System::String^>^ files ;
		ModelInformation *_currentModel;
		renderFunc renderScene;
		previewRenderFunc previewScene;
		ExternalModelDialog ^emd;
		FrontendComm *lf;
		std::vector<ModelInformation> *FFModelInfos;
	private: System::Windows::Forms::FlowLayoutPanel^  radioButtonsFlowLayoutPanel;
	private: OpenGLWidget ^OpenGL;
			 void RenderOpeningGLScene();
			 bool bMousing;
			 bool bManualDLL;
			 bool _bRedrawGL;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
	private: System::Windows::Forms::ToolStripMenuItem^  plotElectronDensityProfileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  fileManipulationA1Nm1ToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  backendToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  cPUToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  gPUToolStripMenuItem;


	private: System::Windows::Forms::ToolStripMenuItem^  consolidateAnalysesToolStripMenuItem;
			 modelInfoRadioButton ^selectedRadio;

	private: System::Windows::Forms::ToolStripMenuItem^  signalSeriesToolStripMenuItem;

			 float crx;
	public:
	
		OpeningWindow(void)
		{
			InitializeComponent();
			
			bMousing = false;
			bManualDLL = false;
			crx = 0.0f;
			_bRedrawGL = true;
			
			previewScene = NULL;
			renderScene = NULL;

			lf = new LocalFrontend();
			FFModelInfos = new std::vector<ModelInformation>();

			_currentModel = NULL;
			
			emd = gcnew ExternalModelDialog();

			OpenGL = gcnew OpenGLWidget(oglPanel, gcnew pRenderScene(this, &GUICLR::OpeningWindow::RenderOpeningGLScene));
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~OpeningWindow()
		{
			delete lf;
			delete FFModelInfos;

			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel1;
	protected: 
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^  actionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  fitExistingModelToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  generateModelToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  plotToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  settingsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  helpToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  aboutToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  dataAgainstBackgroundToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  formFactorAgainstModelToolStripMenuItem;


	private: System::Windows::Forms::ToolStripMenuItem^  dragToFitToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
	private: System::Windows::Forms::ToolStripMenuItem^  smoothDataToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  extractBackgroundieAirDataToolStripMenuItem;
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel2;
	private: System::Windows::Forms::Button^  aboutButton;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::Panel^  panel1;
	private: System::Windows::Forms::GroupBox^  groupBox2;
	private: System::Windows::Forms::Button^  closeButton;

	private: System::Windows::Forms::Button^  fitButton;


















	private: System::Windows::Forms::Panel^  oglPanel;
	private: System::Windows::Forms::Timer^  timer1;
	private: System::ComponentModel::IContainer^  components;



	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(OpeningWindow::typeid));
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->fitButton = (gcnew System::Windows::Forms::Button());
			this->closeButton = (gcnew System::Windows::Forms::Button());
			this->aboutButton = (gcnew System::Windows::Forms::Button());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButtonsFlowLayoutPanel = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->oglPanel = (gcnew System::Windows::Forms::Panel());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->actionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fitExistingModelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->generateModelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->smoothDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->extractBackgroundieAirDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fileManipulationA1Nm1ToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->consolidateAnalysesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->signalSeriesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->plotToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->dataAgainstBackgroundToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->formFactorAgainstModelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->plotElectronDensityProfileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->settingsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->dragToFitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->backendToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->cPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->gPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->helpToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->aboutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->tableLayoutPanel1->SuspendLayout();
			this->tableLayoutPanel2->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->panel1->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->menuStrip1->SuspendLayout();
			this->SuspendLayout();
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 2;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				44.81659F)));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				55.18341F)));
			this->tableLayoutPanel1->Controls->Add(this->tableLayoutPanel2, 1, 1);
			this->tableLayoutPanel1->Controls->Add(this->aboutButton, 0, 1);
			this->tableLayoutPanel1->Controls->Add(this->groupBox1, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->panel1, 1, 0);
			this->tableLayoutPanel1->Dock = System::Windows::Forms::DockStyle::Bottom;
			this->tableLayoutPanel1->Location = System::Drawing::Point(0, 27);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 2;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 88.23529F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 11.76471F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 20)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(627, 306);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->ColumnCount = 2;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				100)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				100)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				20)));
			this->tableLayoutPanel2->Controls->Add(this->fitButton, 0, 0);
			this->tableLayoutPanel2->Controls->Add(this->closeButton, 1, 0);
			this->tableLayoutPanel2->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel2->Location = System::Drawing::Point(284, 272);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 1;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(340, 31);
			this->tableLayoutPanel2->TabIndex = 0;
			// 
			// fitButton
			// 
			this->fitButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->fitButton->Location = System::Drawing::Point(146, 3);
			this->fitButton->Name = L"fitButton";
			this->fitButton->Size = System::Drawing::Size(91, 25);
			this->fitButton->TabIndex = 1;
			this->fitButton->Text = L"&Fit to Data >";
			this->fitButton->UseVisualStyleBackColor = true;
			this->fitButton->Click += gcnew System::EventHandler(this, &OpeningWindow::fitButton_Click);
			// 
			// closeButton
			// 
			this->closeButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->closeButton->Location = System::Drawing::Point(262, 3);
			this->closeButton->Name = L"closeButton";
			this->closeButton->Size = System::Drawing::Size(75, 25);
			this->closeButton->TabIndex = 3;
			this->closeButton->Text = L"&Close";
			this->closeButton->UseVisualStyleBackColor = true;
			this->closeButton->Click += gcnew System::EventHandler(this, &OpeningWindow::closeButton_Click);
			// 
			// aboutButton
			// 
			this->aboutButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->aboutButton->Location = System::Drawing::Point(3, 278);
			this->aboutButton->Name = L"aboutButton";
			this->aboutButton->Size = System::Drawing::Size(75, 25);
			this->aboutButton->TabIndex = 0;
			this->aboutButton->Text = L"About...";
			this->aboutButton->UseVisualStyleBackColor = true;
			this->aboutButton->Click += gcnew System::EventHandler(this, &OpeningWindow::aboutButton_Click);
			// 
			// groupBox1
			// 
			this->groupBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left));
			this->groupBox1->Controls->Add(this->radioButtonsFlowLayoutPanel);
			this->groupBox1->Location = System::Drawing::Point(3, 3);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(275, 263);
			this->groupBox1->TabIndex = 2;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Choose the model that you assume to work with:";
			// 
			// radioButtonsFlowLayoutPanel
			// 
			this->radioButtonsFlowLayoutPanel->AutoScroll = true;
			this->radioButtonsFlowLayoutPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->radioButtonsFlowLayoutPanel->Location = System::Drawing::Point(3, 16);
			this->radioButtonsFlowLayoutPanel->Name = L"radioButtonsFlowLayoutPanel";
			this->radioButtonsFlowLayoutPanel->Size = System::Drawing::Size(269, 244);
			this->radioButtonsFlowLayoutPanel->TabIndex = 0;
			// 
			// panel1
			// 
			this->panel1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left));
			this->panel1->Controls->Add(this->groupBox2);
			this->panel1->Location = System::Drawing::Point(284, 3);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(340, 263);
			this->panel1->TabIndex = 3;
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->oglPanel);
			this->groupBox2->Location = System::Drawing::Point(41, 44);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(259, 189);
			this->groupBox2->TabIndex = 4;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Render Window:";
			// 
			// oglPanel
			// 
			this->oglPanel->Cursor = System::Windows::Forms::Cursors::SizeAll;
			this->oglPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->oglPanel->Location = System::Drawing::Point(3, 16);
			this->oglPanel->Name = L"oglPanel";
			this->oglPanel->Size = System::Drawing::Size(253, 170);
			this->oglPanel->TabIndex = 0;
			this->oglPanel->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &OpeningWindow::oglPanel_MouseDown);
			this->oglPanel->Resize += gcnew System::EventHandler(this, &OpeningWindow::oglPanel_Resize);
			this->oglPanel->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &OpeningWindow::oglPanel_MouseUp);
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {this->actionToolStripMenuItem, 
				this->plotToolStripMenuItem, this->settingsToolStripMenuItem, this->helpToolStripMenuItem});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->RenderMode = System::Windows::Forms::ToolStripRenderMode::System;
			this->menuStrip1->Size = System::Drawing::Size(627, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// actionToolStripMenuItem
			// 
			this->actionToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(8) {this->fitExistingModelToolStripMenuItem, 
				this->generateModelToolStripMenuItem, this->toolStripSeparator2, this->smoothDataToolStripMenuItem, this->extractBackgroundieAirDataToolStripMenuItem, 
				this->fileManipulationA1Nm1ToolStripMenuItem, this->consolidateAnalysesToolStripMenuItem, this->signalSeriesToolStripMenuItem});
			this->actionToolStripMenuItem->Name = L"actionToolStripMenuItem";
			this->actionToolStripMenuItem->Size = System::Drawing::Size(54, 20);
			this->actionToolStripMenuItem->Text = L"&Action";
			// 
			// fitExistingModelToolStripMenuItem
			// 
			this->fitExistingModelToolStripMenuItem->Checked = true;
			this->fitExistingModelToolStripMenuItem->CheckOnClick = true;
			this->fitExistingModelToolStripMenuItem->CheckState = System::Windows::Forms::CheckState::Checked;
			this->fitExistingModelToolStripMenuItem->Name = L"fitExistingModelToolStripMenuItem";
			this->fitExistingModelToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->fitExistingModelToolStripMenuItem->Text = L"Fi&t Existing Model";
			this->fitExistingModelToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::fitExistingModelToolStripMenuItem_CheckedChanged);
			// 
			// generateModelToolStripMenuItem
			// 
			this->generateModelToolStripMenuItem->CheckOnClick = true;
			this->generateModelToolStripMenuItem->Name = L"generateModelToolStripMenuItem";
			this->generateModelToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->generateModelToolStripMenuItem->Text = L"&Generate Model";
			this->generateModelToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::generateModelToolStripMenuItem_CheckedChanged);
			// 
			// toolStripSeparator2
			// 
			this->toolStripSeparator2->Name = L"toolStripSeparator2";
			this->toolStripSeparator2->Size = System::Drawing::Size(266, 6);
			// 
			// smoothDataToolStripMenuItem
			// 
			this->smoothDataToolStripMenuItem->Name = L"smoothDataToolStripMenuItem";
			this->smoothDataToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->smoothDataToolStripMenuItem->Text = L"S&mooth Data...";
			this->smoothDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::smoothDataToolStripMenuItem_Click);
			// 
			// extractBackgroundieAirDataToolStripMenuItem
			// 
			this->extractBackgroundieAirDataToolStripMenuItem->Name = L"extractBackgroundieAirDataToolStripMenuItem";
			this->extractBackgroundieAirDataToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->extractBackgroundieAirDataToolStripMenuItem->Text = L"&Extract Background... (i.e. Air Data)";
			this->extractBackgroundieAirDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::extractBackgroundieAirDataToolStripMenuItem_Click);
			// 
			// fileManipulationA1Nm1ToolStripMenuItem
			// 
			this->fileManipulationA1Nm1ToolStripMenuItem->Name = L"fileManipulationA1Nm1ToolStripMenuItem";
			this->fileManipulationA1Nm1ToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->fileManipulationA1Nm1ToolStripMenuItem->Text = L"&File Manipulation... (A^-1 -> nm^-1)";
			this->fileManipulationA1Nm1ToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::fileManipulationA1Nm1ToolStripMenuItem_Click);
			// 
			// consolidateAnalysesToolStripMenuItem
			// 
			this->consolidateAnalysesToolStripMenuItem->Name = L"consolidateAnalysesToolStripMenuItem";
			this->consolidateAnalysesToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->consolidateAnalysesToolStripMenuItem->Text = L"Consolidate Analyses...";
			this->consolidateAnalysesToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::consolidateAnalysesToolStripMenuItem_Click);
			// 
			// signalSeriesToolStripMenuItem
			// 
			this->signalSeriesToolStripMenuItem->Name = L"signalSeriesToolStripMenuItem";
			this->signalSeriesToolStripMenuItem->Size = System::Drawing::Size(269, 22);
			this->signalSeriesToolStripMenuItem->Text = L"Signal Series...";
			this->signalSeriesToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::signalSeriesToolStripMenuItem_Click);
			// 
			// plotToolStripMenuItem
			// 
			this->plotToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {this->dataAgainstBackgroundToolStripMenuItem, 
				this->formFactorAgainstModelToolStripMenuItem, this->plotElectronDensityProfileToolStripMenuItem});
			this->plotToolStripMenuItem->Name = L"plotToolStripMenuItem";
			this->plotToolStripMenuItem->Size = System::Drawing::Size(40, 20);
			this->plotToolStripMenuItem->Text = L"&Plot";
			// 
			// dataAgainstBackgroundToolStripMenuItem
			// 
			this->dataAgainstBackgroundToolStripMenuItem->Name = L"dataAgainstBackgroundToolStripMenuItem";
			this->dataAgainstBackgroundToolStripMenuItem->Size = System::Drawing::Size(297, 22);
			this->dataAgainstBackgroundToolStripMenuItem->Text = L"&Data against Background...";
			this->dataAgainstBackgroundToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::dataAgainstBackgroundToolStripMenuItem_Click);
			// 
			// formFactorAgainstModelToolStripMenuItem
			// 
			this->formFactorAgainstModelToolStripMenuItem->Name = L"formFactorAgainstModelToolStripMenuItem";
			this->formFactorAgainstModelToolStripMenuItem->Size = System::Drawing::Size(297, 22);
			this->formFactorAgainstModelToolStripMenuItem->Text = L"Signal against &Model...";
			this->formFactorAgainstModelToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::formFactorAgainstModelToolStripMenuItem_Click);
			// 
			// plotElectronDensityProfileToolStripMenuItem
			// 
			this->plotElectronDensityProfileToolStripMenuItem->Name = L"plotElectronDensityProfileToolStripMenuItem";
			this->plotElectronDensityProfileToolStripMenuItem->Size = System::Drawing::Size(297, 22);
			this->plotElectronDensityProfileToolStripMenuItem->Text = L"&Electron Density Profile... (Selected Model)";
			this->plotElectronDensityProfileToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::plotElectronDensityProfileToolStripMenuItem_Click);
			// 
			// settingsToolStripMenuItem
			// 
			this->settingsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->dragToFitToolStripMenuItem, 
				this->backendToolStripMenuItem});
			this->settingsToolStripMenuItem->Name = L"settingsToolStripMenuItem";
			this->settingsToolStripMenuItem->Size = System::Drawing::Size(61, 20);
			this->settingsToolStripMenuItem->Text = L"&Settings";
			this->settingsToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::settingsToolStripMenuItem_Click);
			// 
			// dragToFitToolStripMenuItem
			// 
			this->dragToFitToolStripMenuItem->CheckOnClick = true;
			this->dragToFitToolStripMenuItem->Name = L"dragToFitToolStripMenuItem";
			this->dragToFitToolStripMenuItem->Size = System::Drawing::Size(148, 22);
			this->dragToFitToolStripMenuItem->Text = L"&Drag to Zoom";
			this->dragToFitToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::dragToFitToolStripMenuItem_Click);
			// 
			// backendToolStripMenuItem
			// 
			this->backendToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->cPUToolStripMenuItem, 
				this->gPUToolStripMenuItem});
			this->backendToolStripMenuItem->Name = L"backendToolStripMenuItem";
			this->backendToolStripMenuItem->Size = System::Drawing::Size(148, 22);
			this->backendToolStripMenuItem->Text = L"Backend";
			// 
			// cPUToolStripMenuItem
			// 
			this->cPUToolStripMenuItem->Checked = true;
			this->cPUToolStripMenuItem->CheckState = System::Windows::Forms::CheckState::Checked;
			this->cPUToolStripMenuItem->Name = L"cPUToolStripMenuItem";
			this->cPUToolStripMenuItem->Size = System::Drawing::Size(97, 22);
			this->cPUToolStripMenuItem->Text = L"CPU";
			this->cPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::cPUToolStripMenuItem_Click);
			// 
			// gPUToolStripMenuItem
			// 
			this->gPUToolStripMenuItem->Enabled = false;
			this->gPUToolStripMenuItem->Name = L"gPUToolStripMenuItem";
			this->gPUToolStripMenuItem->Size = System::Drawing::Size(97, 22);
			this->gPUToolStripMenuItem->Text = L"GPU";
			this->gPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::gPUToolStripMenuItem_Click);
			// 
			// helpToolStripMenuItem
			// 
			this->helpToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->aboutToolStripMenuItem});
			this->helpToolStripMenuItem->Name = L"helpToolStripMenuItem";
			this->helpToolStripMenuItem->Size = System::Drawing::Size(44, 20);
			this->helpToolStripMenuItem->Text = L"&Help";
			// 
			// aboutToolStripMenuItem
			// 
			this->aboutToolStripMenuItem->Name = L"aboutToolStripMenuItem";
			this->aboutToolStripMenuItem->Size = System::Drawing::Size(116, 22);
			this->aboutToolStripMenuItem->Text = L"A&bout...";
			this->aboutToolStripMenuItem->Click += gcnew System::EventHandler(this, &OpeningWindow::aboutButton_Click);
			// 
			// timer1
			// 
			this->timer1->Enabled = true;
			this->timer1->Interval = 20;
			this->timer1->Tick += gcnew System::EventHandler(this, &OpeningWindow::timer1_Tick);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->Filter = L"Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
			// 
			// OpeningWindow
			// 
			this->AcceptButton = this->fitButton;
			this->AllowDrop = true;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(627, 333);
			this->Controls->Add(this->menuStrip1);
			this->Controls->Add(this->tableLayoutPanel1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::Fixed3D;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->KeyPreview = true;
			this->MainMenuStrip = this->menuStrip1;
			this->MaximizeBox = false;
			this->Name = L"OpeningWindow";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"X+";
			this->Load += gcnew System::EventHandler(this, &OpeningWindow::OpeningWindow_Load);
			this->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &OpeningWindow::OpeningWindow_DragDrop);
			this->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &OpeningWindow::OpeningWindow_KeyUp);
			this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &OpeningWindow::OpeningWindow_KeyDown);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel2->ResumeLayout(false);
			this->groupBox1->ResumeLayout(false);
			this->panel1->ResumeLayout(false);
			this->groupBox2->ResumeLayout(false);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

private: System::Void OpeningWindow_Load(System::Object^  sender, System::EventArgs^  e);
private: System::Void closeButton_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->Close();
		 }
private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e) {
			 if(_bRedrawGL) {
				OpenGL->Render();
				OpenGL->SwapOpenGLBuffers();
			}
		 }
private: System::Void oglPanel_Resize(System::Object^  sender, System::EventArgs^  e) {
			 if(_bRedrawGL)
				OpenGL->ReSizeGLScene(oglPanel->Width, oglPanel->Height);
		 }
private: System::Void radioButton_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void oglPanel_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 bMousing = true;
		 }
private: System::Void oglPanel_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 bMousing = false;
		 }

private: System::Void aboutButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void fitButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void extractBackgroundieAirDataToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void dataAgainstBackgroundToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void formFactorAgainstModelToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void fitExistingModelToolStripMenuItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void generateModelToolStripMenuItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void smoothDataToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void dragToFitToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void plotElectronDensityProfileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void settingsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 dragToFitToolStripMenuItem->Checked = g_bDragToZoom;
		 }
private: System::Void fileManipulationA1Nm1ToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);

private: System::Void cPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void gPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void OpeningWindow_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
private: System::Void consolidateAnalysesToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void OpeningWindow_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
private: System::Void rExternalModel_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void signalSeriesToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) ;
private: System::Void OpeningWindow_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
private: System::Void OpeningWindow_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);

		 void addDragDropEventRec(System::Windows::Forms::Control^ sender);

		 void LoadFFModelsButtons(System::Windows::Forms::FlowLayoutPanel^ fp,
								  std::vector<ModelInformation> *mds, 
								  FrontendComm *lf, const wchar_t *con);
};
}
