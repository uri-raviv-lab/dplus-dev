#pragma once

#include "MainWindow.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;
using namespace LuaInterface;
using namespace Aga::Controls::Tree;

namespace DPlus {

	/// <summary>
	/// Summary for FittingPrefsPane
	/// </summary>
	public ref class FittingPrefsPane : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	private: System::Windows::Forms::GroupBox^  groupBoxFittingTree;
	private: System::Windows::Forms::ComboBox^  lossFunctionComboBox;
	private: System::Windows::Forms::TextBox^  lossFunctionTextBox1;
	private: System::Windows::Forms::Label^  lossFunctionParameterLabel1;
	private: System::Windows::Forms::TextBox^  lossFunctionTextBox2;
	private: System::Windows::Forms::Label^  lossFunctionParameterLabel2;
	private: System::Windows::Forms::Label^  label1;
	public: System::Windows::Forms::TextBox^  iterationsTextBox;
	private: System::Windows::Forms::ComboBox^  residualsComboBox;
	private: System::Windows::Forms::Label^  label2;
	public: System::Windows::Forms::TextBox^  stepSizeTextBox;
	private: System::Windows::Forms::Label^  label3;
	public: System::Windows::Forms::TextBox^  convergenceTextBox;
	private: System::Windows::Forms::Label^  label4;
	public: System::Windows::Forms::TextBox^  derEpsTextBox;
	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	public:
	protected:
		Aga::Controls::Tree::Node^ checkedNode;
	public:
		FittingPrefsPane(MainWindow ^pform)
		{
			InitializeComponent();

			parentForm = pform;
			fittingTreeView->Model = parentForm->fittingPrefsTree;
		}

	private: Aga::Controls::Tree::NodeControls::NodeTextBox^  nodeTextBox1;
			 Aga::Controls::Tree::NodeControls::NodeCheckBox^ nodeCheckBox1;
	private: Aga::Controls::Tree::TreeColumn^  treeColumn1;
	private: Aga::Controls::Tree::NodeControls::NodeIcon^  nodeIcon1;
	public: Aga::Controls::Tree::TreeViewAdv^  fittingTreeView;

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~FittingPrefsPane()
		{
			if (components)
			{
				delete components;
			}
		}



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
			this->groupBoxFittingTree = (gcnew System::Windows::Forms::GroupBox());
			this->fittingTreeView = (gcnew Aga::Controls::Tree::TreeViewAdv());
			this->treeColumn1 = (gcnew Aga::Controls::Tree::TreeColumn());
			this->nodeIcon1 = (gcnew Aga::Controls::Tree::NodeControls::NodeIcon());
			this->nodeCheckBox1 = (gcnew Aga::Controls::Tree::NodeControls::NodeCheckBox());
			this->nodeTextBox1 = (gcnew Aga::Controls::Tree::NodeControls::NodeTextBox());
			this->lossFunctionComboBox = (gcnew System::Windows::Forms::ComboBox());
			this->lossFunctionTextBox1 = (gcnew System::Windows::Forms::TextBox());
			this->lossFunctionParameterLabel1 = (gcnew System::Windows::Forms::Label());
			this->lossFunctionTextBox2 = (gcnew System::Windows::Forms::TextBox());
			this->lossFunctionParameterLabel2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->iterationsTextBox = (gcnew System::Windows::Forms::TextBox());
			this->residualsComboBox = (gcnew System::Windows::Forms::ComboBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->stepSizeTextBox = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->convergenceTextBox = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->derEpsTextBox = (gcnew System::Windows::Forms::TextBox());
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			this->groupBoxFittingTree->SuspendLayout();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->SuspendLayout();
			// 
			// groupBoxFittingTree
			// 
			this->groupBoxFittingTree->Controls->Add(this->fittingTreeView);
			this->groupBoxFittingTree->Dock = System::Windows::Forms::DockStyle::Fill;
			this->groupBoxFittingTree->Location = System::Drawing::Point(0, 0);
			this->groupBoxFittingTree->Name = L"groupBoxFittingTree";
			this->groupBoxFittingTree->Size = System::Drawing::Size(306, 188);
			this->groupBoxFittingTree->TabIndex = 2;
			this->groupBoxFittingTree->TabStop = false;
			this->groupBoxFittingTree->Text = L"Minimizer type:";
			// 
			// fittingTreeView
			// 
			this->fittingTreeView->AllowDrop = true;
			this->fittingTreeView->BackColor = System::Drawing::SystemColors::Window;
			this->fittingTreeView->Columns->Add(this->treeColumn1);
			this->fittingTreeView->DefaultToolTipProvider = nullptr;
			this->fittingTreeView->DisplayDraggingNodes = true;
			this->fittingTreeView->Dock = System::Windows::Forms::DockStyle::Fill;
			this->fittingTreeView->DragDropMarkColor = System::Drawing::Color::Black;
			this->fittingTreeView->FullRowSelect = true;
			this->fittingTreeView->LineColor = System::Drawing::SystemColors::ControlDark;
			this->fittingTreeView->Location = System::Drawing::Point(3, 16);
			this->fittingTreeView->Model = nullptr;
			this->fittingTreeView->Name = L"fittingTreeView";
			this->fittingTreeView->NodeControls->Add(this->nodeIcon1);
			this->fittingTreeView->NodeControls->Add(this->nodeCheckBox1);
			this->fittingTreeView->NodeControls->Add(this->nodeTextBox1);
			this->fittingTreeView->SelectedNode = nullptr;
			this->fittingTreeView->SelectionMode = Aga::Controls::Tree::TreeSelectionMode::MultiSameParent;
			this->fittingTreeView->Size = System::Drawing::Size(261, 464);
			this->fittingTreeView->TabIndex = 0;
			this->fittingTreeView->Text = L"fittingTreeView";
			this->fittingTreeView->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &FittingPrefsPane::fittingTreeView_MouseDown);
			// 
			// treeColumn1
			// 
			this->treeColumn1->Header = L"Fitting Parameters";
			this->treeColumn1->SortOrder = System::Windows::Forms::SortOrder::None;
			this->treeColumn1->TooltipText = nullptr;
			// 
			// nodeIcon1
			// 
			this->nodeIcon1->LeftMargin = 1;
			this->nodeIcon1->ParentColumn = nullptr;
			this->nodeIcon1->ScaleMode = Aga::Controls::Tree::ImageScaleMode::Clip;
			// 
			// nodeCheckBox1
			// 
			this->nodeCheckBox1->DataPropertyName = L"Checked";
			this->nodeCheckBox1->EditEnabled = true;
			this->nodeCheckBox1->IncrementalSearchEnabled = true;
			this->nodeCheckBox1->LeftMargin = 3;
			this->nodeCheckBox1->ParentColumn = nullptr;
			this->nodeCheckBox1->IsEditEnabledValueNeeded += gcnew System::EventHandler<Aga::Controls::Tree::NodeControls::NodeControlValueEventArgs^ >(this, &FittingPrefsPane::IsNodeLeaf);
			this->nodeCheckBox1->IsVisibleValueNeeded += gcnew System::EventHandler<Aga::Controls::Tree::NodeControls::NodeControlValueEventArgs^ >(this, &FittingPrefsPane::IsNodeLeaf);
			// 
			// nodeTextBox1
			// 
			this->nodeTextBox1->DataPropertyName = L"Text";
			this->nodeTextBox1->IncrementalSearchEnabled = true;
			this->nodeTextBox1->LeftMargin = 3;
			this->nodeTextBox1->ParentColumn = nullptr;
			// 
			// lossFunctionComboBox
			// 
			this->lossFunctionComboBox->FormattingEnabled = true;
			this->lossFunctionComboBox->Location = System::Drawing::Point(4, 30);
			this->lossFunctionComboBox->Name = L"lossFunctionComboBox";
			this->lossFunctionComboBox->Size = System::Drawing::Size(121, 21);
			this->lossFunctionComboBox->TabIndex = 1;
			this->lossFunctionComboBox->SelectedIndexChanged += gcnew System::EventHandler(this, &FittingPrefsPane::lossFunctionComboBox_SelectedIndexChanged);
			// 
			// lossFunctionTextBox1
			// 
			this->lossFunctionTextBox1->Location = System::Drawing::Point(7, 85);
			this->lossFunctionTextBox1->Name = L"lossFunctionTextBox1";
			this->lossFunctionTextBox1->Size = System::Drawing::Size(100, 20);
			this->lossFunctionTextBox1->TabIndex = 2;
			this->lossFunctionTextBox1->Text = L"0.5";
			this->lossFunctionTextBox1->Leave += gcnew System::EventHandler(this, &FittingPrefsPane::textBox_Leave);
			// 
			// lossFunctionParameterLabel1
			// 
			this->lossFunctionParameterLabel1->AutoSize = true;
			this->lossFunctionParameterLabel1->Location = System::Drawing::Point(4, 69);
			this->lossFunctionParameterLabel1->Name = L"lossFunctionParameterLabel1";
			this->lossFunctionParameterLabel1->Size = System::Drawing::Size(135, 13);
			this->lossFunctionParameterLabel1->TabIndex = 3;
			this->lossFunctionParameterLabel1->Text = L"Loss Function parameter 1:";
			// 
			// lossFunctionTextBox2
			// 
			this->lossFunctionTextBox2->Location = System::Drawing::Point(7, 131);
			this->lossFunctionTextBox2->Name = L"lossFunctionTextBox2";
			this->lossFunctionTextBox2->Size = System::Drawing::Size(100, 20);
			this->lossFunctionTextBox2->TabIndex = 2;
			this->lossFunctionTextBox2->Text = L"0.5";
			this->lossFunctionTextBox2->Leave += gcnew System::EventHandler(this, &FittingPrefsPane::textBox_Leave);
			// 
			// lossFunctionParameterLabel2
			// 
			this->lossFunctionParameterLabel2->AutoSize = true;
			this->lossFunctionParameterLabel2->Location = System::Drawing::Point(4, 115);
			this->lossFunctionParameterLabel2->Name = L"lossFunctionParameterLabel2";
			this->lossFunctionParameterLabel2->Size = System::Drawing::Size(135, 13);
			this->lossFunctionParameterLabel2->TabIndex = 3;
			this->lossFunctionParameterLabel2->Text = L"Loss Function parameter 2:";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(4, 164);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(53, 13);
			this->label1->TabIndex = 3;
			this->label1->Text = L"Iterations:";
			// 
			// iterationsTextBox
			// 
			this->iterationsTextBox->Location = System::Drawing::Point(77, 161);
			this->iterationsTextBox->Name = L"iterationsTextBox";
			this->iterationsTextBox->Size = System::Drawing::Size(48, 20);
			this->iterationsTextBox->TabIndex = 4;
			this->iterationsTextBox->Text = L"20";
			// 
			// residualsComboBox
			// 
			this->residualsComboBox->FormattingEnabled = true;
			this->residualsComboBox->Location = System::Drawing::Point(3, 3);
			this->residualsComboBox->Name = L"residualsComboBox";
			this->residualsComboBox->Size = System::Drawing::Size(121, 21);
			this->residualsComboBox->TabIndex = 1;
			this->residualsComboBox->SelectedIndexChanged += gcnew System::EventHandler(this, &FittingPrefsPane::lossFunctionComboBox_SelectedIndexChanged);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(4, 184);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(55, 13);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Step Size:";
			// 
			// stepSizeTextBox
			// 
			this->stepSizeTextBox->Location = System::Drawing::Point(77, 181);
			this->stepSizeTextBox->Name = L"stepSizeTextBox";
			this->stepSizeTextBox->Size = System::Drawing::Size(48, 20);
			this->stepSizeTextBox->TabIndex = 4;
			this->stepSizeTextBox->Text = L"0.01";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(4, 204);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(74, 13);
			this->label3->TabIndex = 3;
			this->label3->Text = L"Convergence:";
			// 
			// convergenceTextBox
			// 
			this->convergenceTextBox->Location = System::Drawing::Point(77, 201);
			this->convergenceTextBox->Name = L"convergenceTextBox";
			this->convergenceTextBox->Size = System::Drawing::Size(48, 20);
			this->convergenceTextBox->TabIndex = 4;
			this->convergenceTextBox->Text = L"0.1";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(4, 224);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(47, 13);
			this->label4->TabIndex = 3;
			this->label4->Text = L"Der eps:";
			// 
			// derEpsTextBox
			// 
			this->derEpsTextBox->Location = System::Drawing::Point(77, 221);
			this->derEpsTextBox->Name = L"derEpsTextBox";
			this->derEpsTextBox->Size = System::Drawing::Size(48, 20);
			this->derEpsTextBox->TabIndex = 4;
			this->derEpsTextBox->Text = L"0.1";
			// 
			// splitContainer1
			// 
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->Location = System::Drawing::Point(0, 0);
			this->splitContainer1->Name = L"splitContainer1";
			this->splitContainer1->Orientation = System::Windows::Forms::Orientation::Vertical;
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->AutoScroll = true;
			this->splitContainer1->Panel1->Controls->Add(this->groupBoxFittingTree);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->AutoScroll = true;
			this->splitContainer1->Panel2->Controls->Add(this->residualsComboBox);
			this->splitContainer1->Panel2->Controls->Add(this->derEpsTextBox);
			this->splitContainer1->Panel2->Controls->Add(this->lossFunctionComboBox);
			this->splitContainer1->Panel2->Controls->Add(this->convergenceTextBox);
			this->splitContainer1->Panel2->Controls->Add(this->lossFunctionTextBox1);
			this->splitContainer1->Panel2->Controls->Add(this->stepSizeTextBox);
			this->splitContainer1->Panel2->Controls->Add(this->lossFunctionTextBox2);
			this->splitContainer1->Panel2->Controls->Add(this->iterationsTextBox);
			this->splitContainer1->Panel2->Controls->Add(this->lossFunctionParameterLabel1);
			this->splitContainer1->Panel2->Controls->Add(this->label4);
			this->splitContainer1->Panel2->Controls->Add(this->lossFunctionParameterLabel2);
			this->splitContainer1->Panel2->Controls->Add(this->label3);
			this->splitContainer1->Panel2->Controls->Add(this->label1);
			this->splitContainer1->Panel2->Controls->Add(this->label2);
			this->splitContainer1->Size = System::Drawing::Size(502, 464);
			this->splitContainer1->SplitterDistance = 167;
			this->splitContainer1->TabIndex = 5;
			// 
			// FittingPrefsPane
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(502, 464);
			this->Controls->Add(this->splitContainer1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->Name = L"FittingPrefsPane";
			this->ShowIcon = false;
			this->Text = L"Fitting Preferences";
			this->Load += gcnew System::EventHandler(this, &FittingPrefsPane::FittingPrefsPane_Load);
			this->groupBoxFittingTree->ResumeLayout(false);
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->Panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			this->ResumeLayout(false);

		}
#pragma endregion

	private: System::Void FittingPrefsPane_Load(System::Object^  sender, System::EventArgs^  e);
private:
	System::Void fittingTreeView_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
	
	void IsNodeLeaf(System::Object^ sender, NodeControls::NodeControlValueEventArgs^ e);
public:
	CeresProperties GetFittingMethod();
	void SetFittingMethod(CeresProperties& dp);

	System::String^ SerializePreferences();
	void DeserializePreferences(LuaTable ^contents);

	// Helper function(s)
	delegate String ^FuncNoParamsReturnString();

	void SetDefaultParams();

private: System::Void lossFunctionComboBox_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
		 System::Void textBox_Leave(System::Object^ sender, System::EventArgs^ e);
		 System::Void InitCeresItems();
}; // FittingPrefsPane

} // namespace
