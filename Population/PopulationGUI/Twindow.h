#pragma once
//#include "svnrev.h" // Current SVN revision
#include "Eigen/Core" // For ArrayXd
#include "calculationExternal.h"
#include "clrfunctionality.h"

#include "FileExpressionPanel.h"
#include "calculationExternal.h"

namespace PopulationGUI {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	

	/// <summary>
	/// Summary for TWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class TWindow : public System::Windows::Forms::Form
	{
	public:
		TWindow(void);

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~TWindow()
		{
			if (components)
			{
				delete components;
			}
	
			if(coefficients)
				delete coefficients;

			if(dataArray)
				delete dataArray;

			if(xDataArray)
				delete xDataArray;

			if(a)
				delete a;
			if(pmut)
				delete pmut;
			if(pMin)
				delete pMin;
			if(pMax)
				delete pMax;
		}
	protected: FileExpressionPanel^ debugPan;
			   FileExpressionPanel^ debugPan2;
			   bool bHandled;

			   System::Collections::Generic::List<FileExpressionPanel^>^ FileExpressionList;
			   array<System::String ^> ^files;

			   Eigen::Array<double,1,26> *coefficients;
			   Eigen::ArrayXXd *dataArray, *xDataArray;
			   Eigen::ArrayXd *a;		// Parameter vector
			   Eigen::ArrayXi *pmut;	// Mutability vector
			   cons *pMin, *pMax;		// Contstraints

	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
	private: System::Windows::Forms::SplitContainer^  splitContainerFullScreen;
	private: System::Windows::Forms::SplitContainer^  splitContainerTopLeft;
	private: System::Windows::Forms::SplitContainer^  splitContainerMiddleLeft;
	private: System::Windows::Forms::SplitContainer^  splitContainerBottomLeft;
	private: System::Windows::Forms::SplitContainer^  splitContainerGraph;
	private: System::Windows::Forms::FlowLayoutPanel^  exprFlowLayoutPanel;
	private: System::Windows::Forms::ToolStripMenuItem^  exportToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^  ofd;
	private: System::Windows::Forms::ToolStripMenuItem^  addPanelToolStripMenuItem;
	private: System::Windows::Forms::CheckBox^  logIcheckBox;
	private: System::Windows::Forms::CheckBox^  logQcheckBox;
	private: System::Windows::Forms::DataGridView^  variableDataGridView;
	private: System::Windows::Forms::Timer^  timer1;
	private: System::Windows::Forms::TrackBar^  trackBar1;
	private: System::Windows::Forms::TextBox^  fitTextBox;
	private: System::Windows::Forms::GroupBox^  fitGroupBox;
	private: System::Windows::Forms::Button^  fitButton;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TextBox^  toTextBox;
	private: System::Windows::Forms::SaveFileDialog^  sfd;
	private: System::Windows::Forms::DataGridViewCheckBoxColumn^  mutColumn;
	private: System::Windows::Forms::DataGridViewTextBoxColumn^  varColumn;
	private: System::Windows::Forms::DataGridViewTextBoxColumn^  valColumn;
	private: System::Windows::Forms::DataGridViewTextBoxColumn^  minColumn;
	private: System::Windows::Forms::DataGridViewTextBoxColumn^  maxColumn;
	private: System::Windows::Forms::Button^  addButton;
	private: System::Windows::Forms::DataGridView^  seriesDataGridView;
	private: System::Windows::Forms::OpenFileDialog^  openManyFD;
	private: System::Windows::Forms::Button^  fitManyButton;

	private: System::Windows::Forms::TextBox^  fitExpressionTextBox;

	private: System::Windows::Forms::Label^  fitManyLabel;
	private: System::Windows::Forms::Button^  fitNowButton;
	private: System::Windows::Forms::Label^  rangeLabel;

	private: System::Windows::Forms::TextBox^  maxRangeTextBox;
	private: System::Windows::Forms::TextBox^  minRangeTextBox;
	private: System::Windows::Forms::Label^  rangeToLabel;
	private: System::Windows::Forms::ToolStripMenuItem^  exportSeriesTableToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  exportSeriesCurvesToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  optionsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  useLogFittingToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  fitIterationsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripTextBox^  fitIterationsToolStripTextBox;
	private: System::Windows::Forms::ToolStripMenuItem^  addMultiplePanelsToolStripMenuItem;
	private: System::Windows::Forms::Label^  LocOnGraph;
	private: GraphToolkit::Graph1D^  wgtGraph;
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
			System::Windows::Forms::DataGridViewCellStyle^  dataGridViewCellStyle4 = (gcnew System::Windows::Forms::DataGridViewCellStyle());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->addPanelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->addMultiplePanelsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exportToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exportSeriesTableToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exportSeriesCurvesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->optionsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->useLogFittingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fitIterationsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fitIterationsToolStripTextBox = (gcnew System::Windows::Forms::ToolStripTextBox());
			this->splitContainerFullScreen = (gcnew System::Windows::Forms::SplitContainer());
			this->splitContainerTopLeft = (gcnew System::Windows::Forms::SplitContainer());
			this->exprFlowLayoutPanel = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->splitContainerMiddleLeft = (gcnew System::Windows::Forms::SplitContainer());
			this->variableDataGridView = (gcnew System::Windows::Forms::DataGridView());
			this->mutColumn = (gcnew System::Windows::Forms::DataGridViewCheckBoxColumn());
			this->varColumn = (gcnew System::Windows::Forms::DataGridViewTextBoxColumn());
			this->valColumn = (gcnew System::Windows::Forms::DataGridViewTextBoxColumn());
			this->minColumn = (gcnew System::Windows::Forms::DataGridViewTextBoxColumn());
			this->maxColumn = (gcnew System::Windows::Forms::DataGridViewTextBoxColumn());
			this->splitContainerBottomLeft = (gcnew System::Windows::Forms::SplitContainer());
			this->fitNowButton = (gcnew System::Windows::Forms::Button());
			this->fitManyLabel = (gcnew System::Windows::Forms::Label());
			this->fitExpressionTextBox = (gcnew System::Windows::Forms::TextBox());
			this->fitManyButton = (gcnew System::Windows::Forms::Button());
			this->addButton = (gcnew System::Windows::Forms::Button());
			this->fitGroupBox = (gcnew System::Windows::Forms::GroupBox());
			this->fitButton = (gcnew System::Windows::Forms::Button());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->toTextBox = (gcnew System::Windows::Forms::TextBox());
			this->fitTextBox = (gcnew System::Windows::Forms::TextBox());
			this->trackBar1 = (gcnew System::Windows::Forms::TrackBar());
			this->seriesDataGridView = (gcnew System::Windows::Forms::DataGridView());
			this->splitContainerGraph = (gcnew System::Windows::Forms::SplitContainer());
			this->LocOnGraph = (gcnew System::Windows::Forms::Label());
			this->rangeToLabel = (gcnew System::Windows::Forms::Label());
			this->rangeLabel = (gcnew System::Windows::Forms::Label());
			this->maxRangeTextBox = (gcnew System::Windows::Forms::TextBox());
			this->minRangeTextBox = (gcnew System::Windows::Forms::TextBox());
			this->logIcheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->logQcheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->ofd = (gcnew System::Windows::Forms::OpenFileDialog());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->sfd = (gcnew System::Windows::Forms::SaveFileDialog());
			this->openManyFD = (gcnew System::Windows::Forms::OpenFileDialog());
			this->wgtGraph = (gcnew GraphToolkit::Graph1D());
			this->menuStrip1->SuspendLayout();
			this->splitContainerFullScreen->Panel1->SuspendLayout();
			this->splitContainerFullScreen->Panel2->SuspendLayout();
			this->splitContainerFullScreen->SuspendLayout();
			this->splitContainerTopLeft->Panel1->SuspendLayout();
			this->splitContainerTopLeft->Panel2->SuspendLayout();
			this->splitContainerTopLeft->SuspendLayout();
			this->splitContainerMiddleLeft->Panel1->SuspendLayout();
			this->splitContainerMiddleLeft->Panel2->SuspendLayout();
			this->splitContainerMiddleLeft->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->variableDataGridView))->BeginInit();
			this->splitContainerBottomLeft->Panel1->SuspendLayout();
			this->splitContainerBottomLeft->Panel2->SuspendLayout();
			this->splitContainerBottomLeft->SuspendLayout();
			this->fitGroupBox->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->trackBar1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->seriesDataGridView))->BeginInit();
			this->splitContainerGraph->Panel1->SuspendLayout();
			this->splitContainerGraph->Panel2->SuspendLayout();
			this->splitContainerGraph->SuspendLayout();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {this->fileToolStripMenuItem, 
				this->exportToolStripMenuItem, this->optionsToolStripMenuItem});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1177, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->addPanelToolStripMenuItem, 
				this->addMultiplePanelsToolStripMenuItem});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// addPanelToolStripMenuItem
			// 
			this->addPanelToolStripMenuItem->Name = L"addPanelToolStripMenuItem";
			this->addPanelToolStripMenuItem->Size = System::Drawing::Size(180, 22);
			this->addPanelToolStripMenuItem->Text = L"Add Panel";
			this->addPanelToolStripMenuItem->Click += gcnew System::EventHandler(this, &TWindow::addPanelToolStripMenuItem_Click);
			// 
			// addMultiplePanelsToolStripMenuItem
			// 
			this->addMultiplePanelsToolStripMenuItem->Name = L"addMultiplePanelsToolStripMenuItem";
			this->addMultiplePanelsToolStripMenuItem->Size = System::Drawing::Size(180, 22);
			this->addMultiplePanelsToolStripMenuItem->Text = L"Add Multiple Panels";
			this->addMultiplePanelsToolStripMenuItem->Click += gcnew System::EventHandler(this, &TWindow::addMultiplePanelsToolStripMenuItem_Click);
			// 
			// exportToolStripMenuItem
			// 
			this->exportToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->exportSeriesTableToolStripMenuItem, 
				this->exportSeriesCurvesToolStripMenuItem});
			this->exportToolStripMenuItem->Name = L"exportToolStripMenuItem";
			this->exportToolStripMenuItem->Size = System::Drawing::Size(52, 20);
			this->exportToolStripMenuItem->Text = L"Export";
			// 
			// exportSeriesTableToolStripMenuItem
			// 
			this->exportSeriesTableToolStripMenuItem->Name = L"exportSeriesTableToolStripMenuItem";
			this->exportSeriesTableToolStripMenuItem->Size = System::Drawing::Size(188, 22);
			this->exportSeriesTableToolStripMenuItem->Text = L"Export Series Table...";
			this->exportSeriesTableToolStripMenuItem->Click += gcnew System::EventHandler(this, &TWindow::exportSeriesToolStripMenuItem_Click);
			// 
			// exportSeriesCurvesToolStripMenuItem
			// 
			this->exportSeriesCurvesToolStripMenuItem->Name = L"exportSeriesCurvesToolStripMenuItem";
			this->exportSeriesCurvesToolStripMenuItem->Size = System::Drawing::Size(188, 22);
			this->exportSeriesCurvesToolStripMenuItem->Text = L"Export Series Curves...";
			this->exportSeriesCurvesToolStripMenuItem->Click += gcnew System::EventHandler(this, &TWindow::exportSeriesToolStripMenuItem_Click);
			// 
			// optionsToolStripMenuItem
			// 
			this->optionsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->useLogFittingToolStripMenuItem, 
				this->fitIterationsToolStripMenuItem});
			this->optionsToolStripMenuItem->Name = L"optionsToolStripMenuItem";
			this->optionsToolStripMenuItem->Size = System::Drawing::Size(61, 20);
			this->optionsToolStripMenuItem->Text = L"Options";
			// 
			// useLogFittingToolStripMenuItem
			// 
			this->useLogFittingToolStripMenuItem->Checked = true;
			this->useLogFittingToolStripMenuItem->CheckOnClick = true;
			this->useLogFittingToolStripMenuItem->CheckState = System::Windows::Forms::CheckState::Checked;
			this->useLogFittingToolStripMenuItem->Name = L"useLogFittingToolStripMenuItem";
			this->useLogFittingToolStripMenuItem->Size = System::Drawing::Size(150, 22);
			this->useLogFittingToolStripMenuItem->Text = L"Use LogFitting";
			this->useLogFittingToolStripMenuItem->CheckedChanged += gcnew System::EventHandler(this, &TWindow::useLogFittingToolStripMenuItem_CheckedChanged);
			// 
			// fitIterationsToolStripMenuItem
			// 
			this->fitIterationsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->fitIterationsToolStripTextBox});
			this->fitIterationsToolStripMenuItem->Name = L"fitIterationsToolStripMenuItem";
			this->fitIterationsToolStripMenuItem->Size = System::Drawing::Size(150, 22);
			this->fitIterationsToolStripMenuItem->Text = L"Fit Iterations";
			// 
			// fitIterationsToolStripTextBox
			// 
			this->fitIterationsToolStripTextBox->Name = L"fitIterationsToolStripTextBox";
			this->fitIterationsToolStripTextBox->Size = System::Drawing::Size(100, 23);
			// 
			// splitContainerFullScreen
			// 
			this->splitContainerFullScreen->BackColor = System::Drawing::SystemColors::Control;
			this->splitContainerFullScreen->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->splitContainerFullScreen->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainerFullScreen->Location = System::Drawing::Point(0, 24);
			this->splitContainerFullScreen->Name = L"splitContainerFullScreen";
			// 
			// splitContainerFullScreen.Panel1
			// 
			this->splitContainerFullScreen->Panel1->Controls->Add(this->splitContainerTopLeft);
			// 
			// splitContainerFullScreen.Panel2
			// 
			this->splitContainerFullScreen->Panel2->Controls->Add(this->splitContainerGraph);
			this->splitContainerFullScreen->Size = System::Drawing::Size(1177, 596);
			this->splitContainerFullScreen->SplitterDistance = 492;
			this->splitContainerFullScreen->TabIndex = 1;
			// 
			// splitContainerTopLeft
			// 
			this->splitContainerTopLeft->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->splitContainerTopLeft->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainerTopLeft->Location = System::Drawing::Point(0, 0);
			this->splitContainerTopLeft->Name = L"splitContainerTopLeft";
			this->splitContainerTopLeft->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainerTopLeft.Panel1
			// 
			this->splitContainerTopLeft->Panel1->Controls->Add(this->exprFlowLayoutPanel);
			// 
			// splitContainerTopLeft.Panel2
			// 
			this->splitContainerTopLeft->Panel2->Controls->Add(this->splitContainerMiddleLeft);
			this->splitContainerTopLeft->Size = System::Drawing::Size(492, 596);
			this->splitContainerTopLeft->SplitterDistance = 259;
			this->splitContainerTopLeft->TabIndex = 0;
			// 
			// exprFlowLayoutPanel
			// 
			this->exprFlowLayoutPanel->AllowDrop = true;
			this->exprFlowLayoutPanel->AutoScroll = true;
			this->exprFlowLayoutPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->exprFlowLayoutPanel->Location = System::Drawing::Point(0, 0);
			this->exprFlowLayoutPanel->Name = L"exprFlowLayoutPanel";
			this->exprFlowLayoutPanel->Size = System::Drawing::Size(488, 255);
			this->exprFlowLayoutPanel->TabIndex = 0;
			this->exprFlowLayoutPanel->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &TWindow::exprFlowLayoutPanel_DragDrop);
			this->exprFlowLayoutPanel->Resize += gcnew System::EventHandler(this, &TWindow::exprFlowLayoutPanel_Resize);
			this->exprFlowLayoutPanel->DragLeave += gcnew System::EventHandler(this, &TWindow::exprFlowLayoutPanel_DragLeave);
			this->exprFlowLayoutPanel->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &TWindow::exprFlowLayoutPanel_DragEnter);
			// 
			// splitContainerMiddleLeft
			// 
			this->splitContainerMiddleLeft->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->splitContainerMiddleLeft->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainerMiddleLeft->Location = System::Drawing::Point(0, 0);
			this->splitContainerMiddleLeft->Name = L"splitContainerMiddleLeft";
			this->splitContainerMiddleLeft->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainerMiddleLeft.Panel1
			// 
			this->splitContainerMiddleLeft->Panel1->Controls->Add(this->variableDataGridView);
			// 
			// splitContainerMiddleLeft.Panel2
			// 
			this->splitContainerMiddleLeft->Panel2->Controls->Add(this->splitContainerBottomLeft);
			this->splitContainerMiddleLeft->Size = System::Drawing::Size(492, 333);
			this->splitContainerMiddleLeft->SplitterDistance = 136;
			this->splitContainerMiddleLeft->TabIndex = 0;
			// 
			// variableDataGridView
			// 
			this->variableDataGridView->AllowUserToAddRows = false;
			this->variableDataGridView->AllowUserToDeleteRows = false;
			dataGridViewCellStyle4->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(128)), 
				static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(255)));
			this->variableDataGridView->AlternatingRowsDefaultCellStyle = dataGridViewCellStyle4;
			this->variableDataGridView->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
			this->variableDataGridView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::DataGridViewColumn^  >(5) {this->mutColumn, 
				this->varColumn, this->valColumn, this->minColumn, this->maxColumn});
			this->variableDataGridView->Dock = System::Windows::Forms::DockStyle::Fill;
			this->variableDataGridView->Location = System::Drawing::Point(0, 0);
			this->variableDataGridView->Name = L"variableDataGridView";
			this->variableDataGridView->Size = System::Drawing::Size(488, 132);
			this->variableDataGridView->TabIndex = 0;
			this->variableDataGridView->CellEndEdit += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &TWindow::variableDataGridView_CellEndEdit);
			this->variableDataGridView->SelectionChanged += gcnew System::EventHandler(this, &TWindow::variableDataGridView_SelectionChanged);
			// 
			// mutColumn
			// 
			this->mutColumn->FalseValue = L"N";
			this->mutColumn->HeaderText = L"Mut";
			this->mutColumn->Name = L"mutColumn";
			this->mutColumn->Resizable = System::Windows::Forms::DataGridViewTriState::True;
			this->mutColumn->SortMode = System::Windows::Forms::DataGridViewColumnSortMode::Automatic;
			this->mutColumn->TrueValue = L"Y";
			this->mutColumn->Width = 30;
			// 
			// varColumn
			// 
			this->varColumn->HeaderText = L"Variable";
			this->varColumn->Name = L"varColumn";
			this->varColumn->ReadOnly = true;
			// 
			// valColumn
			// 
			this->valColumn->HeaderText = L"Value";
			this->valColumn->Name = L"valColumn";
			// 
			// minColumn
			// 
			this->minColumn->HeaderText = L"Min";
			this->minColumn->Name = L"minColumn";
			// 
			// maxColumn
			// 
			this->maxColumn->HeaderText = L"Max";
			this->maxColumn->Name = L"maxColumn";
			// 
			// splitContainerBottomLeft
			// 
			this->splitContainerBottomLeft->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->splitContainerBottomLeft->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainerBottomLeft->Location = System::Drawing::Point(0, 0);
			this->splitContainerBottomLeft->Name = L"splitContainerBottomLeft";
			this->splitContainerBottomLeft->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainerBottomLeft.Panel1
			// 
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->fitNowButton);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->fitManyLabel);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->fitExpressionTextBox);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->fitManyButton);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->addButton);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->fitGroupBox);
			this->splitContainerBottomLeft->Panel1->Controls->Add(this->trackBar1);
			// 
			// splitContainerBottomLeft.Panel2
			// 
			this->splitContainerBottomLeft->Panel2->Controls->Add(this->seriesDataGridView);
			this->splitContainerBottomLeft->Panel2MinSize = 2;
			this->splitContainerBottomLeft->Size = System::Drawing::Size(492, 193);
			this->splitContainerBottomLeft->SplitterDistance = 85;
			this->splitContainerBottomLeft->TabIndex = 0;
			// 
			// fitNowButton
			// 
			this->fitNowButton->Location = System::Drawing::Point(355, 55);
			this->fitNowButton->Name = L"fitNowButton";
			this->fitNowButton->Size = System::Drawing::Size(56, 23);
			this->fitNowButton->TabIndex = 7;
			this->fitNowButton->Text = L"Fit series";
			this->fitNowButton->UseVisualStyleBackColor = true;
			this->fitNowButton->Click += gcnew System::EventHandler(this, &TWindow::fitNowButton_Click);
			// 
			// fitManyLabel
			// 
			this->fitManyLabel->AutoSize = true;
			this->fitManyLabel->Location = System::Drawing::Point(87, 60);
			this->fitManyLabel->Name = L"fitManyLabel";
			this->fitManyLabel->Size = System::Drawing::Size(75, 13);
			this->fitManyLabel->TabIndex = 6;
			this->fitManyLabel->Text = L"Fit Expression:";
			// 
			// fitExpressionTextBox
			// 
			this->fitExpressionTextBox->Location = System::Drawing::Point(162, 57);
			this->fitExpressionTextBox->Name = L"fitExpressionTextBox";
			this->fitExpressionTextBox->Size = System::Drawing::Size(187, 20);
			this->fitExpressionTextBox->TabIndex = 5;
			this->fitExpressionTextBox->Leave += gcnew System::EventHandler(this, &TWindow::fitExpressionTextBox_Leave);
			this->fitExpressionTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &TWindow::fitExpressionTextBox_KeyPress);
			// 
			// fitManyButton
			// 
			this->fitManyButton->Location = System::Drawing::Point(3, 55);
			this->fitManyButton->Name = L"fitManyButton";
			this->fitManyButton->Size = System::Drawing::Size(80, 23);
			this->fitManyButton->TabIndex = 4;
			this->fitManyButton->Text = L"Load series...";
			this->fitManyButton->UseVisualStyleBackColor = true;
			this->fitManyButton->Click += gcnew System::EventHandler(this, &TWindow::fitManyButton_Click);
			// 
			// addButton
			// 
			this->addButton->Location = System::Drawing::Point(10, 3);
			this->addButton->Name = L"addButton";
			this->addButton->Size = System::Drawing::Size(22, 23);
			this->addButton->TabIndex = 3;
			this->addButton->Text = L"+";
			this->addButton->UseVisualStyleBackColor = true;
			this->addButton->Click += gcnew System::EventHandler(this, &TWindow::addPanelToolStripMenuItem_Click);
			// 
			// fitGroupBox
			// 
			this->fitGroupBox->Controls->Add(this->fitButton);
			this->fitGroupBox->Controls->Add(this->label2);
			this->fitGroupBox->Controls->Add(this->label1);
			this->fitGroupBox->Controls->Add(this->toTextBox);
			this->fitGroupBox->Controls->Add(this->fitTextBox);
			this->fitGroupBox->Location = System::Drawing::Point(304, 5);
			this->fitGroupBox->Name = L"fitGroupBox";
			this->fitGroupBox->Size = System::Drawing::Size(181, 41);
			this->fitGroupBox->TabIndex = 2;
			this->fitGroupBox->TabStop = false;
			this->fitGroupBox->Text = L"Fit";
			// 
			// fitButton
			// 
			this->fitButton->Location = System::Drawing::Point(136, 11);
			this->fitButton->Name = L"fitButton";
			this->fitButton->Size = System::Drawing::Size(35, 23);
			this->fitButton->TabIndex = 10;
			this->fitButton->Text = L"Fit";
			this->fitButton->UseVisualStyleBackColor = true;
			this->fitButton->Click += gcnew System::EventHandler(this, &TWindow::fitButton_Click);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(72, 16);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(16, 13);
			this->label2->TabIndex = 2;
			this->label2->Text = L"to";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(6, 16);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(18, 13);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Fit";
			// 
			// toTextBox
			// 
			this->toTextBox->Location = System::Drawing::Point(94, 13);
			this->toTextBox->Name = L"toTextBox";
			this->toTextBox->Size = System::Drawing::Size(36, 20);
			this->toTextBox->TabIndex = 5;
			this->toTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &TWindow::fitTextBox_KeyPress);
			// 
			// fitTextBox
			// 
			this->fitTextBox->Location = System::Drawing::Point(30, 13);
			this->fitTextBox->Name = L"fitTextBox";
			this->fitTextBox->Size = System::Drawing::Size(36, 20);
			this->fitTextBox->TabIndex = 1;
			this->fitTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &TWindow::fitTextBox_KeyPress);
			// 
			// trackBar1
			// 
			this->trackBar1->LargeChange = 10;
			this->trackBar1->Location = System::Drawing::Point(149, 5);
			this->trackBar1->Maximum = 100;
			this->trackBar1->Name = L"trackBar1";
			this->trackBar1->Size = System::Drawing::Size(104, 45);
			this->trackBar1->TabIndex = 0;
			this->trackBar1->TickFrequency = 50;
			this->trackBar1->Value = 50;
			this->trackBar1->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &TWindow::trackBar1_MouseUp);
			// 
			// seriesDataGridView
			// 
			this->seriesDataGridView->AllowUserToAddRows = false;
			this->seriesDataGridView->AllowUserToDeleteRows = false;
			this->seriesDataGridView->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
			this->seriesDataGridView->Dock = System::Windows::Forms::DockStyle::Fill;
			this->seriesDataGridView->Location = System::Drawing::Point(0, 0);
			this->seriesDataGridView->Name = L"seriesDataGridView";
			this->seriesDataGridView->Size = System::Drawing::Size(488, 100);
			this->seriesDataGridView->TabIndex = 0;
			this->seriesDataGridView->ColumnAdded += gcnew System::Windows::Forms::DataGridViewColumnEventHandler(this, &TWindow::seriesDataGridView_ColumnAdded);
			// 
			// splitContainerGraph
			// 
			this->splitContainerGraph->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->splitContainerGraph->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainerGraph->Location = System::Drawing::Point(0, 0);
			this->splitContainerGraph->Name = L"splitContainerGraph";
			this->splitContainerGraph->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainerGraph.Panel1
			// 
			this->splitContainerGraph->Panel1->Controls->Add(this->wgtGraph);
			// 
			// splitContainerGraph.Panel2
			// 
			this->splitContainerGraph->Panel2->Controls->Add(this->LocOnGraph);
			this->splitContainerGraph->Panel2->Controls->Add(this->rangeToLabel);
			this->splitContainerGraph->Panel2->Controls->Add(this->rangeLabel);
			this->splitContainerGraph->Panel2->Controls->Add(this->maxRangeTextBox);
			this->splitContainerGraph->Panel2->Controls->Add(this->minRangeTextBox);
			this->splitContainerGraph->Panel2->Controls->Add(this->logIcheckBox);
			this->splitContainerGraph->Panel2->Controls->Add(this->logQcheckBox);
			this->splitContainerGraph->Size = System::Drawing::Size(681, 596);
			this->splitContainerGraph->SplitterDistance = 567;
			this->splitContainerGraph->TabIndex = 0;
			// 
			// LocOnGraph
			// 
			this->LocOnGraph->AutoSize = true;
			this->LocOnGraph->Location = System::Drawing::Point(297, 4);
			this->LocOnGraph->Name = L"LocOnGraph";
			this->LocOnGraph->Size = System::Drawing::Size(31, 13);
			this->LocOnGraph->TabIndex = 4;
			this->LocOnGraph->Text = L"(0, 0)";
			// 
			// rangeToLabel
			// 
			this->rangeToLabel->AutoSize = true;
			this->rangeToLabel->Location = System::Drawing::Point(153, 4);
			this->rangeToLabel->Name = L"rangeToLabel";
			this->rangeToLabel->Size = System::Drawing::Size(19, 13);
			this->rangeToLabel->TabIndex = 3;
			this->rangeToLabel->Text = L"to:";
			// 
			// rangeLabel
			// 
			this->rangeLabel->AutoSize = true;
			this->rangeLabel->Location = System::Drawing::Point(3, 3);
			this->rangeLabel->Name = L"rangeLabel";
			this->rangeLabel->Size = System::Drawing::Size(42, 13);
			this->rangeLabel->TabIndex = 3;
			this->rangeLabel->Text = L"Range:";
			// 
			// maxRangeTextBox
			// 
			this->maxRangeTextBox->Location = System::Drawing::Point(174, 1);
			this->maxRangeTextBox->Name = L"maxRangeTextBox";
			this->maxRangeTextBox->Size = System::Drawing::Size(100, 20);
			this->maxRangeTextBox->TabIndex = 3;
			this->maxRangeTextBox->Leave += gcnew System::EventHandler(this, &TWindow::rangeTB_Leave);
			this->maxRangeTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &TWindow::rangeTB_KeyPress);
			// 
			// minRangeTextBox
			// 
			this->minRangeTextBox->Location = System::Drawing::Point(47, 1);
			this->minRangeTextBox->Name = L"minRangeTextBox";
			this->minRangeTextBox->Size = System::Drawing::Size(100, 20);
			this->minRangeTextBox->TabIndex = 2;
			this->minRangeTextBox->Leave += gcnew System::EventHandler(this, &TWindow::rangeTB_Leave);
			this->minRangeTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &TWindow::rangeTB_KeyPress);
			// 
			// logIcheckBox
			// 
			this->logIcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logIcheckBox->AutoSize = true;
			this->logIcheckBox->Location = System::Drawing::Point(566, 2);
			this->logIcheckBox->Name = L"logIcheckBox";
			this->logIcheckBox->Size = System::Drawing::Size(49, 17);
			this->logIcheckBox->TabIndex = 1;
			this->logIcheckBox->Text = L"log(I)";
			this->logIcheckBox->UseVisualStyleBackColor = true;
			this->logIcheckBox->CheckedChanged += gcnew System::EventHandler(this, &TWindow::logCheckedChanged);
			// 
			// logQcheckBox
			// 
			this->logQcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logQcheckBox->AutoSize = true;
			this->logQcheckBox->Location = System::Drawing::Point(625, 3);
			this->logQcheckBox->Name = L"logQcheckBox";
			this->logQcheckBox->Size = System::Drawing::Size(54, 17);
			this->logQcheckBox->TabIndex = 0;
			this->logQcheckBox->Text = L"log(Q)";
			this->logQcheckBox->UseVisualStyleBackColor = true;
			this->logQcheckBox->CheckedChanged += gcnew System::EventHandler(this, &TWindow::logCheckedChanged);
			// 
			// timer1
			// 
			this->timer1->Tick += gcnew System::EventHandler(this, &TWindow::timer1_Tick);
			// 
			// openManyFD
			// 
			this->openManyFD->Multiselect = true;
			// 
			// wgtGraph
			// 
			this->wgtGraph->Cursor = System::Windows::Forms::Cursors::Cross;
			this->wgtGraph->Dock = System::Windows::Forms::DockStyle::Fill;
			this->wgtGraph->GraphTitle = L"";
			this->wgtGraph->Location = System::Drawing::Point(0, 0);
			this->wgtGraph->Name = L"wgtGraph";
			this->wgtGraph->Size = System::Drawing::Size(677, 563);
			this->wgtGraph->TabIndex = 0;
			this->wgtGraph->XLabel = L" [nm\u02c9\u00b9]";
			this->wgtGraph->YLabel = L"Intensity [a.u.]";
			this->wgtGraph->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &TWindow::wgt_MouseMove);
			// 
			// TWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1177, 620);
			this->Controls->Add(this->splitContainerFullScreen);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"TWindow";
			this->Text = L"TWindow";
			this->Load += gcnew System::EventHandler(this, &TWindow::TWindow_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->splitContainerFullScreen->Panel1->ResumeLayout(false);
			this->splitContainerFullScreen->Panel2->ResumeLayout(false);
			this->splitContainerFullScreen->ResumeLayout(false);
			this->splitContainerTopLeft->Panel1->ResumeLayout(false);
			this->splitContainerTopLeft->Panel2->ResumeLayout(false);
			this->splitContainerTopLeft->ResumeLayout(false);
			this->splitContainerMiddleLeft->Panel1->ResumeLayout(false);
			this->splitContainerMiddleLeft->Panel2->ResumeLayout(false);
			this->splitContainerMiddleLeft->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->variableDataGridView))->EndInit();
			this->splitContainerBottomLeft->Panel1->ResumeLayout(false);
			this->splitContainerBottomLeft->Panel1->PerformLayout();
			this->splitContainerBottomLeft->Panel2->ResumeLayout(false);
			this->splitContainerBottomLeft->ResumeLayout(false);
			this->fitGroupBox->ResumeLayout(false);
			this->fitGroupBox->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->trackBar1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->seriesDataGridView))->EndInit();
			this->splitContainerGraph->Panel1->ResumeLayout(false);
			this->splitContainerGraph->Panel2->ResumeLayout(false);
			this->splitContainerGraph->Panel2->PerformLayout();
			this->splitContainerGraph->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void exprFlowLayoutPanel_Resize(System::Object^  sender, System::EventArgs^  e);
			 System::Void TWindow_Load(System::Object^  sender, System::EventArgs^  e);
			 System::Void addPanelToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void logCheckedChanged(System::Object^  sender, System::EventArgs^  e);
			 System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e);
			 System::Void variableDataGridView_SelectionChanged(System::Object^  sender, System::EventArgs^  e);
			 System::Void trackBar1_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
			 System::Void fitTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
			 System::Void fitButton_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void fitManyButton_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void fitExpressionTextBox_Leave(System::Object^  sender, System::EventArgs^  e);
			 System::Void fitExpressionTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
			 System::Void fitNowButton_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void rangeTB_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
			 System::Void rangeTB_Leave(System::Object^  sender, System::EventArgs^  e);
			 System::Void exportSeriesToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void seriesDataGridView_ColumnAdded(System::Object^  sender, System::Windows::Forms::DataGridViewColumnEventArgs^  e);
			 System::Void useLogFittingToolStripMenuItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
			 System::Void fitIterationsToolStripTextBox_Leave(System::Object^  sender, System::EventArgs^  e);
			 System::Void addMultiplePanelsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void wgt_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e); 
			 System::Void exprFlowLayoutPanel_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
			 System::Void exprFlowLayoutPanel_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
			 System::Void exprFlowLayoutPanel_DragLeave(System::Object^  sender, System::EventArgs^  e);
		//////////////////
		// Panel Events //
		//////////////////
			 System::Void exprTextBox_Leave(System::Object^  sender, System::EventArgs^  e);
			 System::Void fileRadioButton_Click(System::Object^  sender, System::EventArgs^  e);
			 System::Void radioButton_CheckChanged(System::Object^  sender, System::EventArgs^  e);
			 System::Void visCheckedChanged(System::Object^  sender, System::EventArgs^  e);
			 System::Void ChangeColor(System::Object^  sender, System::EventArgs^  e);
			 System::Void RemoveButtonClick(System::Object^  sender, System::EventArgs^  e);
			 System::Void exportButtonClick(System::Object^  sender, System::EventArgs^  e);
			 System::Void variableDataGridView_CellEndEdit(System::Object^  sender, System::Windows::Forms::DataGridViewCellEventArgs^  e);
			 System::Void panel_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
			 System::Void panel_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
			 System::Void panel_DragLeave(System::Object^  sender, System::EventArgs^  e);
			 

			 bool AddPanel();
			 bool AddPanel(String ^ fileN, std::vector<double> x, std::vector<double> y);
			 bool checkForChanges(bool &changeVec);
			 void RemovePanel(int ind);
			 void draw();
			 bool LoadFile(System::Object ^sender);
			 bool LoadFile(System::String ^fileName);
			 bool LoadFile(System::String ^fileName, FileExpressionPanel ^par);

			 void TWindow::dealWithTrackBar(DataGridViewCell^ cell, TrackBar ^tb, double factor);
		
			 FileExpressionPanel^ GetParent(System::Object ^sender);
			 bool modifyGraph(FileExpressionPanel^ panel, bool bExpression);

			 bool loadVariables(bool bRange);
			 array<System::String ^> ^GetDroppedFilenames(ArrayList^ droppedFiles);

};
}

