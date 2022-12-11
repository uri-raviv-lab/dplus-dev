#pragma once

#include "MainWindow.h"
#include "GraphPane3D.h"
#include "PreferencesPane.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;

namespace DPlus {

	/// <summary>
	/// Summary for SymmetryView
	/// </summary>
	
	public enum ModelsWithFiles
	{
		PDB = 999,
		EPDB = 1999,
		AMP = 1000,
		ScriptedGeometry = 1001,
		ScriptedModel = 1002,
		ScriptedSymmetry = 1003
	};

	public ref class SymmetryView : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	private: System::Windows::Forms::ContextMenuStrip^  contextMenuStrip1;
			 System::Windows::Forms::ContextMenuStrip^  contextMenuModelName;
	protected: 

	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator1;
	private: System::Windows::Forms::ToolStripMenuItem^  closeToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  modelRenameToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  modelDeleteNameToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  renameToolStripMenuItem;
			 int tabIndex, contextTab;
	private: System::Windows::Forms::CheckBox^  anomalousCheckBox;
	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	public: System::Windows::Forms::CheckBox^  avgpopsizeMut;
	public:
		SymmetryView(MainWindow ^pform)
		{
			InitializeComponent();

			parentForm = pform;
			treeViewAdv1->Model = parentForm->entityTree;
			tabIndex = 0;
			contextTab = 0;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~SymmetryView()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::GroupBox^  groupBox1;
	public: System::Windows::Forms::ComboBox^  entityCombo;

	public: System::Windows::Forms::Button^  buttonAdd;
	private: System::Windows::Forms::Button^  buttonRemove;
	private: System::Windows::Forms::Button^  buttonGroup;

	private: Aga::Controls::Tree::NodeControls::NodeTextBox^  nodeTextBox1;
	private: Aga::Controls::Tree::TreeColumn^  treeColumn1;
	private: Aga::Controls::Tree::NodeControls::NodeIcon^  nodeIcon1;
	public: Aga::Controls::Tree::TreeViewAdv^  treeViewAdv1;
	private: System::Windows::Forms::CheckBox^  centerPDBCheckBox;
	public: System::Windows::Forms::TabControl^  populationTabs;
	private: System::Windows::Forms::TabPage^  tabPage1;
	public: System::Windows::Forms::TextBox^  avgpopsizeText;

	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TabPage^  addPopulationFakeTab;
	private: System::ComponentModel::IContainer^  components;

	private: System::Windows::Forms::Label^  scaleLabel;
	public: System::Windows::Forms::TextBox^  scaleBox;
	private:
	public: System::Windows::Forms::CheckBox^  scaleMut;
	public: System::Windows::Forms::CheckBox^  constantMut;

	public: System::Windows::Forms::TextBox^  constantBox;
	private: System::Windows::Forms::Label^ constantLabel;

	public: 
	private: 

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
				System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(SymmetryView::typeid));
				this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
				this->treeViewAdv1 = (gcnew Aga::Controls::Tree::TreeViewAdv());
				this->treeColumn1 = (gcnew Aga::Controls::Tree::TreeColumn());
				this->contextMenuModelName = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
				this->modelRenameToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
				this->modelDeleteNameToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
				this->nodeIcon1 = (gcnew Aga::Controls::Tree::NodeControls::NodeIcon());
				this->nodeTextBox1 = (gcnew Aga::Controls::Tree::NodeControls::NodeTextBox());
				this->entityCombo = (gcnew System::Windows::Forms::ComboBox());
				this->buttonAdd = (gcnew System::Windows::Forms::Button());
				this->buttonRemove = (gcnew System::Windows::Forms::Button());
				this->buttonGroup = (gcnew System::Windows::Forms::Button());
				this->centerPDBCheckBox = (gcnew System::Windows::Forms::CheckBox());
				this->populationTabs = (gcnew System::Windows::Forms::TabControl());
				this->contextMenuStrip1 = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
				this->toolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
				this->renameToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
				this->closeToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
				this->tabPage1 = (gcnew System::Windows::Forms::TabPage());
				this->addPopulationFakeTab = (gcnew System::Windows::Forms::TabPage());
				this->avgpopsizeText = (gcnew System::Windows::Forms::TextBox());
				this->label1 = (gcnew System::Windows::Forms::Label());
				this->avgpopsizeMut = (gcnew System::Windows::Forms::CheckBox());
				this->anomalousCheckBox = (gcnew System::Windows::Forms::CheckBox());
				this->anomalousCheckBox->Visible = false;
				this->anomalousCheckBox->Click += gcnew System::EventHandler(this, &SymmetryView::anomalous_CheckedClick);
				this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
				this->groupBox1->SuspendLayout();
				this->contextMenuModelName->SuspendLayout();
				this->populationTabs->SuspendLayout();
				this->contextMenuStrip1->SuspendLayout();
				(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
				this->splitContainer1->Panel1->SuspendLayout();
				this->splitContainer1->Panel2->SuspendLayout();
				this->splitContainer1->SuspendLayout();
				this->SuspendLayout();

				this->scaleLabel = (gcnew System::Windows::Forms::Label());
				this->scaleBox = (gcnew System::Windows::Forms::TextBox());
				this->scaleMut = (gcnew System::Windows::Forms::CheckBox());
				this->constantMut = (gcnew System::Windows::Forms::CheckBox());
				this->constantBox = (gcnew System::Windows::Forms::TextBox());
				this->constantLabel = (gcnew System::Windows::Forms::Label());
			
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->treeViewAdv1);
			this->groupBox1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->groupBox1->Location = System::Drawing::Point(0, 0);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(306, 188);
			this->groupBox1->TabIndex = 2;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Entities:";
			// 
			// treeViewAdv1
			// 
			this->treeViewAdv1->AllowDrop = true;
			this->treeViewAdv1->BackColor = System::Drawing::SystemColors::Window;
			this->treeViewAdv1->Columns->Add(this->treeColumn1);
			this->treeViewAdv1->ContextMenuStrip = this->contextMenuModelName;
			this->treeViewAdv1->DefaultToolTipProvider = nullptr;
			this->treeViewAdv1->DisplayDraggingNodes = true;
			this->treeViewAdv1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->treeViewAdv1->DragDropMarkColor = System::Drawing::Color::Black;
			this->treeViewAdv1->FullRowSelect = true;
			this->treeViewAdv1->LineColor = System::Drawing::SystemColors::ControlDark;
			this->treeViewAdv1->Location = System::Drawing::Point(3, 16);
			this->treeViewAdv1->Model = nullptr;
			this->treeViewAdv1->Name = L"treeViewAdv1";
			this->treeViewAdv1->NodeControls->Add(this->nodeIcon1);
			this->treeViewAdv1->NodeControls->Add(this->nodeTextBox1);
			this->treeViewAdv1->SelectedNode = nullptr;
			this->treeViewAdv1->SelectionMode = Aga::Controls::Tree::TreeSelectionMode::MultiSameParent;
			this->treeViewAdv1->Size = System::Drawing::Size(300, 169);
			this->treeViewAdv1->TabIndex = 1;
			this->treeViewAdv1->Text = L"treeViewAdv1";
			this->treeViewAdv1->ItemDrag += gcnew System::Windows::Forms::ItemDragEventHandler(this, &SymmetryView::treeViewAdv1_ItemDrag);
			this->treeViewAdv1->SelectionChanged += gcnew System::EventHandler(this, &SymmetryView::treeViewAdv1_SelectionChanged);
			this->treeViewAdv1->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &SymmetryView::treeViewAdv1_DragDrop);
			this->treeViewAdv1->DragOver += gcnew System::Windows::Forms::DragEventHandler(this, &SymmetryView::treeViewAdv1_DragOver);
			this->treeViewAdv1->DoubleClick += gcnew System::EventHandler(this, &SymmetryView::treeViewAdv1_DoubleClick);
			this->treeViewAdv1->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SymmetryView::treeViewAdv1_KeyDown);
			// 
			// treeColumn1
			// 
			this->treeColumn1->Header = L"Entity";
			this->treeColumn1->SortOrder = System::Windows::Forms::SortOrder::None;
			this->treeColumn1->TooltipText = nullptr;
			// 
			// contextMenuModelName
			// 
			this->contextMenuModelName->Enabled = false;
			this->contextMenuModelName->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->modelRenameToolStripMenuItem,
					this->modelDeleteNameToolStripMenuItem
			});
			this->contextMenuModelName->Name = L"contextMenuModelName";
			this->contextMenuModelName->Size = System::Drawing::Size(180, 48);
			this->contextMenuModelName->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &SymmetryView::contextMenuStrip1_Opening);
			// 
			// modelRenameToolStripMenuItem
			// 
			this->modelRenameToolStripMenuItem->Enabled = true;
			this->modelRenameToolStripMenuItem->Name = L"modelRenameToolStripMenuItem";
			this->modelRenameToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->modelRenameToolStripMenuItem->Text = L"Rename Model";
			this->modelRenameToolStripMenuItem->Click += gcnew System::EventHandler(this, &SymmetryView::modelRenameToolStripMenuItem_Click);
			// 
			// modelDeleteNameToolStripMenuItem
			// 
			this->modelDeleteNameToolStripMenuItem->Enabled = true;
			this->modelDeleteNameToolStripMenuItem->Name = L"modelDeleteNameToolStripMenuItem";
			this->modelDeleteNameToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->modelDeleteNameToolStripMenuItem->Text = L"Delete Model Name";
			this->modelDeleteNameToolStripMenuItem->Click += gcnew System::EventHandler(this, &SymmetryView::modelDeleteNameToolStripMenuItem_Click);
			// 
			// nodeIcon1
			// 
			this->nodeIcon1->LeftMargin = 1;
			this->nodeIcon1->ParentColumn = nullptr;
			this->nodeIcon1->ScaleMode = Aga::Controls::Tree::ImageScaleMode::Clip;
			// 
			// nodeTextBox1
			// 
			this->nodeTextBox1->DataPropertyName = L"Text";
			this->nodeTextBox1->IncrementalSearchEnabled = true;
			this->nodeTextBox1->LeftMargin = 3;
			this->nodeTextBox1->ParentColumn = this->treeColumn1;
			// 
			// entityCombo
			// 
			this->entityCombo->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->entityCombo->FormattingEnabled = true;
			this->entityCombo->Location = System::Drawing::Point(5, 57);
			this->entityCombo->Name = L"entityCombo";
			this->entityCombo->Size = System::Drawing::Size(130, 21);
			this->entityCombo->TabIndex = 3;
			this->entityCombo->SelectedIndexChanged += gcnew System::EventHandler(this, &SymmetryView::entityCombo_SelectedIndexChanged);
			// 
			// buttonAdd
			// 
			this->buttonAdd->Location = System::Drawing::Point(5, 84);
			this->buttonAdd->Name = L"buttonAdd";
			this->buttonAdd->Size = System::Drawing::Size(61, 21);
			this->buttonAdd->TabIndex = 4;
			this->buttonAdd->Text = L"Add";
			this->buttonAdd->UseVisualStyleBackColor = true;
			this->buttonAdd->Click += gcnew System::EventHandler(this, &SymmetryView::buttonAdd_Click);
			// 
			// buttonRemove
			// 
			this->buttonRemove->Location = System::Drawing::Point(71, 84);
			this->buttonRemove->Name = L"buttonRemove";
			this->buttonRemove->Size = System::Drawing::Size(64, 21);
			this->buttonRemove->TabIndex = 5;
			this->buttonRemove->Text = L"Remove";
			this->buttonRemove->UseVisualStyleBackColor = true;
			this->buttonRemove->Click += gcnew System::EventHandler(this, &SymmetryView::buttonRemove_Click);
			// 
			// buttonGroup
			// 
			this->buttonGroup->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->buttonGroup->Enabled = false;
			this->buttonGroup->Location = System::Drawing::Point(231, 56);
			this->buttonGroup->Name = L"buttonGroup";
			this->buttonGroup->Size = System::Drawing::Size(68, 48);
			this->buttonGroup->TabIndex = 7;
			this->buttonGroup->Text = L"Group Selected";
			this->buttonGroup->UseVisualStyleBackColor = true;
			this->buttonGroup->Click += gcnew System::EventHandler(this, &SymmetryView::buttonGroup_Click);
			// 
			// centerPDBCheckBox
			// 
			this->centerPDBCheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->centerPDBCheckBox->AutoSize = true;
			this->centerPDBCheckBox->Checked = true;
			this->centerPDBCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->centerPDBCheckBox->Location = System::Drawing::Point(217, 80);
			this->centerPDBCheckBox->Name = L"centerPDBCheckBox";
			this->centerPDBCheckBox->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->centerPDBCheckBox->Size = System::Drawing::Size(82, 17);
			this->centerPDBCheckBox->TabIndex = 6;
			this->centerPDBCheckBox->Text = L"Center PDB";
			this->centerPDBCheckBox->UseVisualStyleBackColor = true;
			// 
			// populationTabs
			// 
			this->populationTabs->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->populationTabs->ContextMenuStrip = this->contextMenuStrip1;
			this->populationTabs->Controls->Add(this->tabPage1);
			this->populationTabs->Controls->Add(this->addPopulationFakeTab);
			this->populationTabs->Location = System::Drawing::Point(4, 67);
			this->populationTabs->Name = L"populationTabs";
			this->populationTabs->SelectedIndex = 0;
			this->populationTabs->Size = System::Drawing::Size(295, 20);
			this->populationTabs->TabIndex = 0;
			this->populationTabs->Selecting += gcnew System::Windows::Forms::TabControlCancelEventHandler(this, &SymmetryView::populationTabs_Selecting);
			this->populationTabs->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &SymmetryView::populationTabs_MouseDown);
			this->populationTabs->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &SymmetryView::populationTabs_MouseUp);
			// 
			// contextMenuStrip1
			// 
			this->contextMenuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->toolStripSeparator1,
					this->renameToolStripMenuItem, this->closeToolStripMenuItem
			});
			this->contextMenuStrip1->Name = L"contextMenuStrip1";
			this->contextMenuStrip1->Size = System::Drawing::Size(165, 54);
			this->contextMenuStrip1->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &SymmetryView::contextMenuStrip1_Opening);
			// 
			// toolStripSeparator1
			// 
			this->toolStripSeparator1->Name = L"toolStripSeparator1";
			this->toolStripSeparator1->Size = System::Drawing::Size(161, 6);
			// 
			// renameToolStripMenuItem
			// 
			this->renameToolStripMenuItem->Name = L"renameToolStripMenuItem";
			this->renameToolStripMenuItem->ShortcutKeys = System::Windows::Forms::Keys::F2;
			this->renameToolStripMenuItem->Size = System::Drawing::Size(164, 22);
			this->renameToolStripMenuItem->Text = L"Rename";
			this->renameToolStripMenuItem->Click += gcnew System::EventHandler(this, &SymmetryView::renameToolStripMenuItem_ShortcutClick);
			this->renameToolStripMenuItem->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &SymmetryView::renameToolStripMenuItem_MouseDown);
			// 
			// closeToolStripMenuItem
			// 
			this->closeToolStripMenuItem->Name = L"closeToolStripMenuItem";
			this->closeToolStripMenuItem->Size = System::Drawing::Size(164, 22);
			this->closeToolStripMenuItem->Text = L"Close Population";
			this->closeToolStripMenuItem->Click += gcnew System::EventHandler(this, &SymmetryView::closeToolStripMenuItem_Click);
			// 
			// tabPage1
			// 
			this->tabPage1->ContextMenuStrip = this->contextMenuStrip1;
			this->tabPage1->Location = System::Drawing::Point(4, 22);
			this->tabPage1->Name = L"tabPage1";
			this->tabPage1->Padding = System::Windows::Forms::Padding(3);
			this->tabPage1->Size = System::Drawing::Size(287, 0);
			this->tabPage1->TabIndex = 0;
			this->tabPage1->Text = L"Population 1";
			this->tabPage1->UseVisualStyleBackColor = true;
			// 
			// addPopulationFakeTab
			// 
			this->addPopulationFakeTab->Location = System::Drawing::Point(4, 22);
			this->addPopulationFakeTab->Name = L"addPopulationFakeTab";
			this->addPopulationFakeTab->Padding = System::Windows::Forms::Padding(3);
			this->addPopulationFakeTab->Size = System::Drawing::Size(430, 0);
			this->addPopulationFakeTab->TabIndex = 1;
			this->addPopulationFakeTab->Text = L"+";
			this->addPopulationFakeTab->ToolTipText = L"Adds a new population";
			this->addPopulationFakeTab->UseVisualStyleBackColor = true;
			// 
			// avgpopsizeText
			// 
			this->avgpopsizeText->Location = System::Drawing::Point(141, 9);
			this->avgpopsizeText->Name = L"avgpopsizeText";
			this->avgpopsizeText->Size = System::Drawing::Size(64, 20);
			this->avgpopsizeText->TabIndex = 2;
			this->avgpopsizeText->Text = L"1";
			this->avgpopsizeText->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SymmetryView::avgpopsizeText_KeyDown);
			this->avgpopsizeText->Leave += gcnew System::EventHandler(this, &SymmetryView::avgpopsizeText_Leave);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(9, 12);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(126, 13);
			this->label1->TabIndex = 3;
			this->label1->Text = L"Average Population Size:";
			// 
			// avgpopsizeMut
			// 
			this->avgpopsizeMut->AutoSize = true;
			this->avgpopsizeMut->Location = System::Drawing::Point(141, 35);
			this->avgpopsizeMut->Name = L"avgpopsizeMut";
			this->avgpopsizeMut->Size = System::Drawing::Size(64, 17);
			this->avgpopsizeMut->TabIndex = 8;
			this->avgpopsizeMut->Text = L"Mutable";
			this->avgpopsizeMut->UseVisualStyleBackColor = true;
			this->avgpopsizeMut->CheckedChanged += gcnew System::EventHandler(this, &SymmetryView::avgpopsizeMut_CheckedChanged);
			// 
			// anomalousCheckBox
			// 
			this->anomalousCheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->anomalousCheckBox->AutoSize = true;
			this->anomalousCheckBox->Location = System::Drawing::Point(217, 18);
			this->anomalousCheckBox->Name = L"anomalousCheckBox";
			this->anomalousCheckBox->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->anomalousCheckBox->Size = System::Drawing::Size(78, 17);
			this->anomalousCheckBox->TabIndex = 9;
			this->anomalousCheckBox->Text = L"Anomalous";
			this->anomalousCheckBox->UseVisualStyleBackColor = true;
			// 
			// splitContainer1
			// 
			this->splitContainer1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->splitContainer1->Location = System::Drawing::Point(0, 90);
			this->splitContainer1->Name = L"splitContainer1";
			this->splitContainer1->Orientation = System::Windows::Forms::Orientation::Horizontal;
			this->splitContainer1->AutoScroll = true;
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->AutoScroll = true;
			this->splitContainer1->Panel1->Controls->Add(this->groupBox1);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->AutoScroll = true;
			this->splitContainer1->Panel2->Controls->Add(this->label1);
			this->splitContainer1->Panel2->Controls->Add(this->entityCombo);
			this->splitContainer1->Panel2->Controls->Add(this->buttonAdd);
			this->splitContainer1->Panel2->Controls->Add(this->anomalousCheckBox);
			this->splitContainer1->Panel2->Controls->Add(this->buttonRemove);
			this->splitContainer1->Panel2->Controls->Add(this->avgpopsizeMut);
			this->splitContainer1->Panel2->Controls->Add(this->buttonGroup);
			this->splitContainer1->Panel2->Controls->Add(this->avgpopsizeText);
			this->splitContainer1->Panel2->Controls->Add(this->centerPDBCheckBox);
			this->splitContainer1->Size = System::Drawing::Size(306, 424);
			this->splitContainer1->SplitterDistance = 188;
			this->splitContainer1->TabIndex = 10;
			// 
			// SymmetryView
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(306, 450);
			this->Controls->Add(this->populationTabs);
			this->Controls->Add(this->splitContainer1);
			this->Controls->Add(this->constantMut);
			this->Controls->Add(this->constantBox);
			this->Controls->Add(this->constantLabel);
			this->Controls->Add(this->scaleMut);
			this->Controls->Add(this->scaleBox);
			this->Controls->Add(this->scaleLabel);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->AutoScroll = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"SymmetryView";
			this->ShowIcon = false;
			this->Text = L"Domain View";
			this->groupBox1->ResumeLayout(false);
			this->contextMenuModelName->ResumeLayout(false);
			this->populationTabs->ResumeLayout(false);
			this->contextMenuStrip1->ResumeLayout(false);
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->Panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			this->ResumeLayout(false);
			// 
			// scaleLabel
			// 
			this->scaleLabel->AutoSize = true;
			this->scaleLabel->Location = System::Drawing::Point(12, 9);
			this->scaleLabel->Name = L"scaleLabel";
			this->scaleLabel->Size = System::Drawing::Size(76, 13);
			this->scaleLabel->TabIndex = 16;
			this->scaleLabel->Text = L"Domain Scale:";
			// 
			// scaleBox
			// 
			this->scaleBox->Location = System::Drawing::Point(105, 7);
			this->scaleBox->Name = L"scaleBox";
			this->scaleBox->Size = System::Drawing::Size(90, 20);
			this->scaleBox->TabIndex = 18;
			this->scaleBox->Text = L"1";
			this->scaleBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SymmetryView::textBox_KeyDown);
			this->scaleBox->Leave += gcnew System::EventHandler(this, &SymmetryView::textBox_Leave);
			// 
			// scaleMut
			// 
			this->scaleMut->AutoSize = true;
			this->scaleMut->Location = System::Drawing::Point(201, 9);
			this->scaleMut->Name = L"scaleMut";
			this->scaleMut->Size = System::Drawing::Size(64, 17);
			this->scaleMut->TabIndex = 19;
			this->scaleMut->Text = L"Mutable";
			this->scaleMut->UseVisualStyleBackColor = true;
			// 
			// constantMut
			// 
			this->constantMut->AutoSize = true;
			this->constantMut->Location = System::Drawing::Point(201, 35);
			this->constantMut->Name = L"constantMut";
			this->constantMut->Size = System::Drawing::Size(64, 17);
			this->constantMut->TabIndex = 21;
			this->constantMut->Text = L"Mutable";
			this->constantMut->UseVisualStyleBackColor = true;
			// 
			// constantBox
			// 
			this->constantBox->Location = System::Drawing::Point(105, 33);
			this->constantBox->Name = L"constantBox";
			this->constantBox->Size = System::Drawing::Size(90, 20);
			this->constantBox->TabIndex = 20;
			this->constantBox->Text = L"0";
			this->constantBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SymmetryView::textBox_KeyDown);
			this->constantBox->Leave += gcnew System::EventHandler(this, &SymmetryView::textBox_Leave);
			// 
			// constantLabel
			// 
			this->constantLabel->AutoSize = true;
			this->constantLabel->Location = System::Drawing::Point(12, 36);
			this->constantLabel->Name = L"constantLabel";
			this->constantLabel->Size = System::Drawing::Size(91, 13);
			this->constantLabel->TabIndex = 23;
			this->constantLabel->Text = L"Domain Constant:";

		}
#pragma endregion
		/*
	private: System::Void dataGridView1_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
private: System::Void dataGridView1_EditingControlShowing(System::Object^  sender, System::Windows::Forms::DataGridViewEditingControlShowingEventArgs^  e);
*/
private: System::Void buttonAdd_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void buttonRemove_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void buttonGroup_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void treeViewAdv1_SelectionChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void entityCombo_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void treeViewAdv1_ItemDrag(System::Object^  sender, System::Windows::Forms::ItemDragEventArgs^  e);
private: System::Void treeViewAdv1_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
private: System::Void treeViewAdv1_DragOver(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e);
public:	 void ChangeEditorEnabled( bool en );
		 void ChangeParamEditorEnabled( bool en );
		 public: Entity^ GetSelectedEntity();
				 void tvInvalidate();
private: System::Void treeViewAdv1_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
		 public: void RemoveSelectedNodes();
				 bool CenterChecked() { return this->centerPDBCheckBox->Checked; }
				 bool AnomalousChecked() { return this->anomalousCheckBox->Checked; }
private: System::Void treeViewAdv1_DoubleClick(System::Object^  sender, System::EventArgs^  e);
private: System::Void treeViewAdv1_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs ^ e);
private: System::Void populationTabs_Selecting(System::Object^  sender, System::Windows::Forms::TabControlCancelEventArgs^  e);
private: System::Void closeToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
public:  void AddPopulation();
public:  void RemovePopulation(int index);
private: System::Void contextMenuStrip1_Opening(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e);
private: System::Void modelRenameToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void modelDeleteNameToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void renameToolStripMenuItem_ShortcutClick(System::Object^  sender, System::EventArgs^  e);
private: System::Void renameToolStripMenuItem_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
private: System::Void populationTabs_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		 System::Void populationTabs_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
		 int GetHoveredTab(System::Windows::Forms::MouseEventArgs^ e);
private: System::Void avgpopsizeText_Leave(System::Object^  sender, System::EventArgs^  e);
private: System::Void avgpopsizeText_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
private: System::Void avgpopsizeMut_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void anomalous_CheckedClick(System::Object^  sender, System::EventArgs^  e);
private: System::Void UpdateModelPtr(Entity ^ ent, String ^anomfilename);
private: System::Void AddLayerManualSymmetry(System::Object^ sender, System::EventArgs^ e, Entity^ ent);
		private: System::Void textBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
		private: System::Void textBox_Leave(System::Object^  sender, System::EventArgs^  e);
				 public: void SetDefaultParams();
};

// A class just to help with the model combobox (provides reference to model names, containers and IDs)
public ref class ModelInfo {
	String ^name;
	String ^container;
	int modelID;

public:
	ModelInfo(String ^n, String ^cont, int id) {
		name = n; container = cont; modelID = id;
	}

	int GetID() { return modelID; }
	String ^GetContainer() { return container; }	

	virtual String ^ToString() override {
		return name;
	}
};
}
