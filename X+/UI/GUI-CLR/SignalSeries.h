#pragma once

#include "WGTControl.h"
#include "GUIHelperClasses.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace GUICLR {

	/// <summary>
	/// Summary for SignalSeries
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class SignalSeries : public System::Windows::Forms::Form
	{
	public:
		SignalSeries(void)
		{
			InitializeComponent();

			signalFileList = gcnew System::Collections::Generic::List<signalFile^>();
			flowLayoutPanel1->AutoScroll = true;
		}
	protected: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel1;
	protected: System::Windows::Forms::Splitter^  splitter1;
			 System::Collections::Generic::List<signalFile^>^ signalFileList;
			 array<System::String ^> ^files;
	protected: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel2;
	protected: System::Windows::Forms::CheckBox^  logICheckBox;
	protected: System::Windows::Forms::CheckBox^  logQCheckBox;
	protected: System::Windows::Forms::OpenFileDialog^  ofd;
	protected: System::Windows::Forms::FlowLayoutPanel^  flowLayoutPanel1;





	protected: System::Windows::Forms::Timer^  timer1;
	protected: System::Windows::Forms::SaveFileDialog^  sfd;
	protected: System::Windows::Forms::CheckBox^  pathCheckBox;
			 int selectedIndex, selectedIndexV;
			 double minDistance;
			 bool changingSel, changingVis;	//Flags to indicate that checkBoxes are being changed

	protected: System::Windows::Forms::MenuStrip^  menuStrip1;
	protected: System::Windows::Forms::ToolStripMenuItem^  optionsToolStripMenuItem;
	protected: System::Windows::Forms::ToolStripMenuItem^  reverseOrderToolStripMenuItem;

	protected: System::Windows::Forms::TrackBar^  minSpacingTrackBar;
	protected: System::Windows::Forms::Panel^  spacingPanel;
	protected: System::Windows::Forms::Label^  spacingLabel;
	protected: System::Windows::Forms::ToolStripMenuItem^  importTSVFileToolStripMenuItem;
	private: System::Windows::Forms::SplitContainer^  splitContainer2;
	protected: System::Windows::Forms::Label^  label3;
	private: 
	protected: System::Windows::Forms::Label^  label4;
	protected: System::Windows::Forms::Label^  label7;
	protected: System::Windows::Forms::Label^  label8;
	protected: System::Windows::Forms::Label^  label9;
	protected: System::Windows::Forms::Button^  sortButton;

	protected: System::Windows::Forms::Button^  moveDownButton;
	protected: System::Windows::Forms::Button^  moveUpButton;
	protected: System::Windows::Forms::Button^  removeButton;
	protected: System::Windows::Forms::Button^  addButton;
	private: System::Windows::Forms::Label^  LocOnGraph;

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>

		GUICLR::WGTControl^  wgtGraph;

		~SignalSeries()
		{
			if (components)
			{
				delete components;
			}
		}
	protected: System::Windows::Forms::SplitContainer^  splitContainer1;




	protected: System::Windows::Forms::Button^  exportButton;
	private: System::ComponentModel::IContainer^  components;
	protected:
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
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->splitContainer2 = (gcnew System::Windows::Forms::SplitContainer());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->sortButton = (gcnew System::Windows::Forms::Button());
			this->moveDownButton = (gcnew System::Windows::Forms::Button());
			this->moveUpButton = (gcnew System::Windows::Forms::Button());
			this->removeButton = (gcnew System::Windows::Forms::Button());
			this->addButton = (gcnew System::Windows::Forms::Button());
			this->flowLayoutPanel1 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->splitter1 = (gcnew System::Windows::Forms::Splitter());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->logICheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->logQCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->exportButton = (gcnew System::Windows::Forms::Button());
			this->pathCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->spacingPanel = (gcnew System::Windows::Forms::Panel());
			this->spacingLabel = (gcnew System::Windows::Forms::Label());
			this->minSpacingTrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->LocOnGraph = (gcnew System::Windows::Forms::Label());
			this->ofd = (gcnew System::Windows::Forms::OpenFileDialog());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->sfd = (gcnew System::Windows::Forms::SaveFileDialog());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->optionsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->reverseOrderToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->importTSVFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->splitContainer2->Panel1->SuspendLayout();
			this->splitContainer2->Panel2->SuspendLayout();
			this->splitContainer2->SuspendLayout();
			this->tableLayoutPanel1->SuspendLayout();
			this->tableLayoutPanel2->SuspendLayout();
			this->spacingPanel->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->minSpacingTrackBar))->BeginInit();
			this->menuStrip1->SuspendLayout();
			this->SuspendLayout();
			// 
			// splitContainer1
			// 
			this->splitContainer1->AllowDrop = true;
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->Location = System::Drawing::Point(0, 24);
			this->splitContainer1->Name = L"splitContainer1";
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->AllowDrop = true;
			this->splitContainer1->Panel1->Controls->Add(this->splitContainer2);
			this->splitContainer1->Panel1->Controls->Add(this->splitter1);
			this->splitContainer1->Panel1->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragDrop);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->Controls->Add(this->tableLayoutPanel1);
			this->splitContainer1->Size = System::Drawing::Size(1091, 627);
			this->splitContainer1->SplitterDistance = 457;
			this->splitContainer1->TabIndex = 0;
			this->splitContainer1->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragDrop);
			this->splitContainer1->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragEnter);
			// 
			// splitContainer2
			// 
			this->splitContainer2->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer2->IsSplitterFixed = true;
			this->splitContainer2->Location = System::Drawing::Point(3, 0);
			this->splitContainer2->Name = L"splitContainer2";
			this->splitContainer2->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainer2.Panel1
			// 
			this->splitContainer2->Panel1->Controls->Add(this->label3);
			this->splitContainer2->Panel1->Controls->Add(this->label4);
			this->splitContainer2->Panel1->Controls->Add(this->label7);
			this->splitContainer2->Panel1->Controls->Add(this->label8);
			this->splitContainer2->Panel1->Controls->Add(this->label9);
			this->splitContainer2->Panel1->Controls->Add(this->sortButton);
			this->splitContainer2->Panel1->Controls->Add(this->moveDownButton);
			this->splitContainer2->Panel1->Controls->Add(this->moveUpButton);
			this->splitContainer2->Panel1->Controls->Add(this->removeButton);
			this->splitContainer2->Panel1->Controls->Add(this->addButton);
			this->splitContainer2->Panel1MinSize = 43;
			// 
			// splitContainer2.Panel2
			// 
			this->splitContainer2->Panel2->Controls->Add(this->flowLayoutPanel1);
			this->splitContainer2->Size = System::Drawing::Size(454, 627);
			this->splitContainer2->SplitterDistance = 43;
			this->splitContainer2->SplitterWidth = 1;
			this->splitContainer2->TabIndex = 9;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(92, 30);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(23, 13);
			this->label3->TabIndex = 27;
			this->label3->Text = L"File";
			// 
			// label4
			// 
			this->label4->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(381, 30);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(65, 13);
			this->label4->TabIndex = 26;
			this->label4->Text = L"Background";
			// 
			// label7
			// 
			this->label7->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(321, 30);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(34, 13);
			this->label7->TabIndex = 28;
			this->label7->Text = L"Scale";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(32, 30);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(22, 13);
			this->label8->TabIndex = 30;
			this->label8->Text = L"Sel";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(12, 30);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(21, 13);
			this->label9->TabIndex = 29;
			this->label9->Text = L"Vis";
			// 
			// sortButton
			// 
			this->sortButton->Location = System::Drawing::Point(259, 3);
			this->sortButton->Name = L"sortButton";
			this->sortButton->Size = System::Drawing::Size(34, 23);
			this->sortButton->TabIndex = 24;
			this->sortButton->Text = L"Sort";
			this->sortButton->UseVisualStyleBackColor = true;
			this->sortButton->Click += gcnew System::EventHandler(this, &SignalSeries::sortButton_Click);
			// 
			// moveDownButton
			// 
			this->moveDownButton->AllowDrop = true;
			this->moveDownButton->Location = System::Drawing::Point(180, 3);
			this->moveDownButton->Name = L"moveDownButton";
			this->moveDownButton->Size = System::Drawing::Size(73, 23);
			this->moveDownButton->TabIndex = 23;
			this->moveDownButton->Text = L"Move Down";
			this->moveDownButton->UseVisualStyleBackColor = true;
			this->moveDownButton->Click += gcnew System::EventHandler(this, &SignalSeries::moveDownButton_Click);
			// 
			// moveUpButton
			// 
			this->moveUpButton->Location = System::Drawing::Point(115, 3);
			this->moveUpButton->Name = L"moveUpButton";
			this->moveUpButton->Size = System::Drawing::Size(59, 23);
			this->moveUpButton->TabIndex = 25;
			this->moveUpButton->Text = L"Move Up";
			this->moveUpButton->UseVisualStyleBackColor = true;
			this->moveUpButton->Click += gcnew System::EventHandler(this, &SignalSeries::moveUpButton_Click);
			// 
			// removeButton
			// 
			this->removeButton->Location = System::Drawing::Point(54, 3);
			this->removeButton->Name = L"removeButton";
			this->removeButton->Size = System::Drawing::Size(55, 23);
			this->removeButton->TabIndex = 22;
			this->removeButton->Text = L"Remove";
			this->removeButton->UseVisualStyleBackColor = true;
			this->removeButton->Click += gcnew System::EventHandler(this, &SignalSeries::removeButton_Click);
			// 
			// addButton
			// 
			this->addButton->Location = System::Drawing::Point(5, 3);
			this->addButton->Name = L"addButton";
			this->addButton->Size = System::Drawing::Size(43, 23);
			this->addButton->TabIndex = 21;
			this->addButton->Text = L"Add...";
			this->addButton->UseVisualStyleBackColor = true;
			this->addButton->Click += gcnew System::EventHandler(this, &SignalSeries::addButton_Click);
			// 
			// flowLayoutPanel1
			// 
			this->flowLayoutPanel1->AllowDrop = true;
			this->flowLayoutPanel1->AutoScroll = true;
			this->flowLayoutPanel1->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->flowLayoutPanel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->flowLayoutPanel1->Location = System::Drawing::Point(0, 0);
			this->flowLayoutPanel1->MaximumSize = System::Drawing::Size(550000, 100000);
			this->flowLayoutPanel1->Name = L"flowLayoutPanel1";
			this->flowLayoutPanel1->Size = System::Drawing::Size(454, 583);
			this->flowLayoutPanel1->TabIndex = 8;
			this->flowLayoutPanel1->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragDrop);
			this->flowLayoutPanel1->Resize += gcnew System::EventHandler(this, &SignalSeries::flowLayoutPanel1_Resize);
			this->flowLayoutPanel1->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragEnter);
			// 
			// splitter1
			// 
			this->splitter1->Location = System::Drawing::Point(0, 0);
			this->splitter1->Name = L"splitter1";
			this->splitter1->Size = System::Drawing::Size(3, 627);
			this->splitter1->TabIndex = 7;
			this->splitter1->TabStop = false;
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanel1->ColumnCount = 1;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				630)));
			this->tableLayoutPanel1->Controls->Add(this->tableLayoutPanel2, 0, 1);
			this->tableLayoutPanel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel1->Location = System::Drawing::Point(0, 0);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 2;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 34)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(630, 627);
			this->tableLayoutPanel1->TabIndex = 1;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->tableLayoutPanel2->ColumnCount = 6;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				100)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				125)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				125)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				70)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				70)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				78)));
			this->tableLayoutPanel2->Controls->Add(this->logICheckBox, 3, 0);
			this->tableLayoutPanel2->Controls->Add(this->logQCheckBox, 4, 0);
			this->tableLayoutPanel2->Controls->Add(this->exportButton, 5, 0);
			this->tableLayoutPanel2->Controls->Add(this->pathCheckBox, 0, 0);
			this->tableLayoutPanel2->Controls->Add(this->spacingPanel, 2, 0);
			this->tableLayoutPanel2->Controls->Add(this->LocOnGraph, 1, 0);
			this->tableLayoutPanel2->Location = System::Drawing::Point(3, 596);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 1;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(624, 28);
			this->tableLayoutPanel2->TabIndex = 1;
			// 
			// logICheckBox
			// 
			this->logICheckBox->AutoSize = true;
			this->logICheckBox->Dock = System::Windows::Forms::DockStyle::Fill;
			this->logICheckBox->Enabled = false;
			this->logICheckBox->Location = System::Drawing::Point(409, 3);
			this->logICheckBox->Name = L"logICheckBox";
			this->logICheckBox->Size = System::Drawing::Size(64, 22);
			this->logICheckBox->TabIndex = 1;
			this->logICheckBox->Text = L"log(I)";
			this->logICheckBox->UseVisualStyleBackColor = true;
			this->logICheckBox->CheckedChanged += gcnew System::EventHandler(this, &SignalSeries::logICheckBox_CheckedChanged);
			// 
			// logQCheckBox
			// 
			this->logQCheckBox->AutoSize = true;
			this->logQCheckBox->Dock = System::Windows::Forms::DockStyle::Fill;
			this->logQCheckBox->Enabled = false;
			this->logQCheckBox->Location = System::Drawing::Point(479, 3);
			this->logQCheckBox->Name = L"logQCheckBox";
			this->logQCheckBox->Size = System::Drawing::Size(64, 22);
			this->logQCheckBox->TabIndex = 1;
			this->logQCheckBox->Text = L"loq(q)";
			this->logQCheckBox->UseVisualStyleBackColor = true;
			this->logQCheckBox->CheckedChanged += gcnew System::EventHandler(this, &SignalSeries::logQCheckBox_CheckedChanged);
			// 
			// exportButton
			// 
			this->exportButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->exportButton->Location = System::Drawing::Point(567, 4);
			this->exportButton->Name = L"exportButton";
			this->exportButton->Size = System::Drawing::Size(54, 21);
			this->exportButton->TabIndex = 0;
			this->exportButton->Text = L"Export...";
			this->exportButton->UseVisualStyleBackColor = true;
			this->exportButton->Click += gcnew System::EventHandler(this, &SignalSeries::exportButton_Click);
			// 
			// pathCheckBox
			// 
			this->pathCheckBox->AutoSize = true;
			this->pathCheckBox->Dock = System::Windows::Forms::DockStyle::Fill;
			this->pathCheckBox->Location = System::Drawing::Point(3, 3);
			this->pathCheckBox->Name = L"pathCheckBox";
			this->pathCheckBox->Size = System::Drawing::Size(150, 22);
			this->pathCheckBox->TabIndex = 2;
			this->pathCheckBox->Text = L"Show Path";
			this->pathCheckBox->UseVisualStyleBackColor = true;
			this->pathCheckBox->CheckedChanged += gcnew System::EventHandler(this, &SignalSeries::pathCheckBox_CheckedChanged);
			// 
			// spacingPanel
			// 
			this->spacingPanel->Controls->Add(this->spacingLabel);
			this->spacingPanel->Controls->Add(this->minSpacingTrackBar);
			this->spacingPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->spacingPanel->Location = System::Drawing::Point(281, 0);
			this->spacingPanel->Margin = System::Windows::Forms::Padding(0);
			this->spacingPanel->Name = L"spacingPanel";
			this->spacingPanel->Size = System::Drawing::Size(125, 28);
			this->spacingPanel->TabIndex = 3;
			// 
			// spacingLabel
			// 
			this->spacingLabel->AutoSize = true;
			this->spacingLabel->Location = System::Drawing::Point(4, 6);
			this->spacingLabel->Name = L"spacingLabel";
			this->spacingLabel->Size = System::Drawing::Size(46, 13);
			this->spacingLabel->TabIndex = 1;
			this->spacingLabel->Text = L"Spacing";
			// 
			// minSpacingTrackBar
			// 
			this->minSpacingTrackBar->BackColor = System::Drawing::SystemColors::Control;
			this->minSpacingTrackBar->Location = System::Drawing::Point(45, 1);
			this->minSpacingTrackBar->Maximum = 100;
			this->minSpacingTrackBar->Name = L"minSpacingTrackBar";
			this->minSpacingTrackBar->Size = System::Drawing::Size(71, 45);
			this->minSpacingTrackBar->TabIndex = 0;
			this->minSpacingTrackBar->TickStyle = System::Windows::Forms::TickStyle::None;
			this->minSpacingTrackBar->Value = 50;
			this->minSpacingTrackBar->Scroll += gcnew System::EventHandler(this, &SignalSeries::minSpacingTrackBar_Scroll);
			// 
			// LocOnGraph
			// 
			this->LocOnGraph->AutoSize = true;
			this->LocOnGraph->Dock = System::Windows::Forms::DockStyle::Fill;
			this->LocOnGraph->Location = System::Drawing::Point(159, 0);
			this->LocOnGraph->Name = L"LocOnGraph";
			this->LocOnGraph->Size = System::Drawing::Size(119, 28);
			this->LocOnGraph->TabIndex = 4;
			this->LocOnGraph->Text = L"(0, 0)";
			this->LocOnGraph->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// ofd
			// 
			this->ofd->Multiselect = true;
			// 
			// timer1
			// 
			this->timer1->Tick += gcnew System::EventHandler(this, &SignalSeries::timer1_Tick);
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->optionsToolStripMenuItem});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1091, 24);
			this->menuStrip1->TabIndex = 1;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// optionsToolStripMenuItem
			// 
			this->optionsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->reverseOrderToolStripMenuItem, 
				this->importTSVFileToolStripMenuItem});
			this->optionsToolStripMenuItem->Name = L"optionsToolStripMenuItem";
			this->optionsToolStripMenuItem->Size = System::Drawing::Size(61, 20);
			this->optionsToolStripMenuItem->Text = L"Options";
			// 
			// reverseOrderToolStripMenuItem
			// 
			this->reverseOrderToolStripMenuItem->Name = L"reverseOrderToolStripMenuItem";
			this->reverseOrderToolStripMenuItem->Size = System::Drawing::Size(161, 22);
			this->reverseOrderToolStripMenuItem->Text = L"Reverse order";
			this->reverseOrderToolStripMenuItem->Click += gcnew System::EventHandler(this, &SignalSeries::reverseOrderToolStripMenuItem_Click);
			// 
			// importTSVFileToolStripMenuItem
			// 
			this->importTSVFileToolStripMenuItem->Name = L"importTSVFileToolStripMenuItem";
			this->importTSVFileToolStripMenuItem->Size = System::Drawing::Size(161, 22);
			this->importTSVFileToolStripMenuItem->Text = L"Import TSV file...";
			this->importTSVFileToolStripMenuItem->Click += gcnew System::EventHandler(this, &SignalSeries::importTSVFileToolStripMenuItem_Click);
			// 
			// SignalSeries
			// 
			this->AllowDrop = true;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1091, 651);
			this->Controls->Add(this->splitContainer1);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"SignalSeries";
			this->Text = L"SignalSeries";
			this->Load += gcnew System::EventHandler(this, &SignalSeries::SignalSeries_Load);
			this->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &SignalSeries::SignalSeries_DragDrop);
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->ResumeLayout(false);
			this->splitContainer2->Panel1->ResumeLayout(false);
			this->splitContainer2->Panel1->PerformLayout();
			this->splitContainer2->Panel2->ResumeLayout(false);
			this->splitContainer2->ResumeLayout(false);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel2->ResumeLayout(false);
			this->tableLayoutPanel2->PerformLayout();
			this->spacingPanel->ResumeLayout(false);
			this->spacingPanel->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->minSpacingTrackBar))->EndInit();
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
		/////////////////////
		//// User Events ////
		/////////////////////
#pragma endregion

	protected: 
		System::Void SignalSeries_Load(System::Object^  sender, System::EventArgs^  e);
		System::Void removeButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void addButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void moveUpButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void moveDownButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void exportButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void logICheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void logQCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void textBox_Leave(System::Object^  sender, System::EventArgs^  e);
		System::Void textBox_Changed(System::Object^  sender, System::EventArgs^  e);
		System::Void pmDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		System::Void pmUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		System::Void vis_CheckChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void color_Clicked(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		System::Void General_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
		System::Void pathCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e);
		System::Void select_CheckChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void sortButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void reverseOrderToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void minSpacingTrackBar_Scroll(System::Object^  sender, System::EventArgs^  e);
		System::Void importTSVFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void SignalSeries_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
		System::Void SignalSeries_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
		System::Void flowLayoutPanel1_Resize(System::Object^  sender, System::EventArgs^  e);
		System::Void wgt_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e); 

		void addKeyDownEventRec(System::Windows::Forms::Control^ sender);
public:
		void ArrangeList();
		void draw();

};
}