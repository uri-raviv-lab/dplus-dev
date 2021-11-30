#pragma once

#include "MainWindow.h"
#include "SymmetryView.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;

namespace DPlus {

	/// <summary>
	/// Summary for ParameterEditor
	/// </summary>
	public ref class ParameterEditor : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
		SymmetryView ^controlledForm;
	public:
		Generic::List<Control^>^ allUIObjects;
		System::Windows::Forms::DataGridView^  parameterDataGridView;
		System::Windows::Forms::DataGridView^  extraParamsDataGridView;
	private: System::Windows::Forms::ContextMenuStrip^  gridViewContextMenuStrip;
	private: System::Windows::Forms::ToolStripMenuItem^  polydispersityToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  editConstraintsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  linkToolStripMenuItem;
	private: System::Windows::Forms::Button^  addLayerButton;
	private: System::Windows::Forms::Button^  removeLayerButton;
	public: 


	private: System::Windows::Forms::SplitContainer^  splitContainer1;





	public:
		ParameterEditor(MainWindow ^pform, SymmetryView ^paneSym)
		{
			InitializeComponent();

			consDummy();
			controlledForm = paneSym;
			parentForm = pform;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ParameterEditor()
		{
			if (components)
			{
				delete components;
			}
		}

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ParameterEditor::typeid));
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->addLayerButton = (gcnew System::Windows::Forms::Button());
			this->removeLayerButton = (gcnew System::Windows::Forms::Button());
			this->parameterDataGridView = (gcnew System::Windows::Forms::DataGridView());
			this->gridViewContextMenuStrip = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->polydispersityToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->editConstraintsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->linkToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->extraParamsDataGridView = (gcnew System::Windows::Forms::DataGridView());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->BeginInit();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->parameterDataGridView))->BeginInit();
			this->gridViewContextMenuStrip->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->extraParamsDataGridView))->BeginInit();
			this->SuspendLayout();
			// 
			// splitContainer1
			// 
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->Location = System::Drawing::Point(0, 0);
			this->splitContainer1->Name = L"splitContainer1";
			this->splitContainer1->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->Controls->Add(this->addLayerButton);
			this->splitContainer1->Panel1->Controls->Add(this->removeLayerButton);
			this->splitContainer1->Panel1->Controls->Add(this->parameterDataGridView);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->Controls->Add(this->extraParamsDataGridView);
			this->splitContainer1->Size = System::Drawing::Size(431, 452);
			this->splitContainer1->SplitterDistance = 253;
			this->splitContainer1->TabIndex = 0;
			// 
			// addLayerButton
			// 
			this->addLayerButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->addLayerButton->Enabled = false;
			this->addLayerButton->Location = System::Drawing::Point(244, 227);
			this->addLayerButton->Name = L"addLayerButton";
			this->addLayerButton->Size = System::Drawing::Size(89, 23);
			this->addLayerButton->TabIndex = 2;
			this->addLayerButton->Text = L"Add Layer";
			this->addLayerButton->UseVisualStyleBackColor = true;
			this->addLayerButton->Click += gcnew System::EventHandler(this, &ParameterEditor::addLayerButton_Click);
			// 
			// removeLayerButton
			// 
			this->removeLayerButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->removeLayerButton->Enabled = false;
			this->removeLayerButton->Location = System::Drawing::Point(339, 227);
			this->removeLayerButton->Name = L"removeLayerButton";
			this->removeLayerButton->Size = System::Drawing::Size(89, 23);
			this->removeLayerButton->TabIndex = 1;
			this->removeLayerButton->Text = L"Remove Layer";
			this->removeLayerButton->UseVisualStyleBackColor = true;
			this->removeLayerButton->Click += gcnew System::EventHandler(this, &ParameterEditor::removeLayerButton_Click);
			// 
			// parameterDataGridView
			// 
			this->parameterDataGridView->AllowUserToAddRows = false;
			this->parameterDataGridView->AllowUserToDeleteRows = false;
			this->parameterDataGridView->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->parameterDataGridView->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
			this->parameterDataGridView->ContextMenuStrip = this->gridViewContextMenuStrip;
			this->parameterDataGridView->Location = System::Drawing::Point(0, 0);
			this->parameterDataGridView->Name = L"parameterDataGridView";
			this->parameterDataGridView->Size = System::Drawing::Size(431, 226);
			this->parameterDataGridView->TabIndex = 0;
			this->parameterDataGridView->CellEndEdit += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::DataGridView_CellEndEdit);
			this->parameterDataGridView->CellValueChanged += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::DataGridView_OnCellValueChanged);
			this->parameterDataGridView->CellMouseUp += gcnew System::Windows::Forms::DataGridViewCellMouseEventHandler(this, &ParameterEditor::DataGridView_OnCellMouseUp);
			this->parameterDataGridView->EditingControlShowing += gcnew System::Windows::Forms::DataGridViewEditingControlShowingEventHandler(this, &ParameterEditor::dataGridView_EditingControlShowing);
			this->parameterDataGridView->SelectionChanged += gcnew System::EventHandler(this, &ParameterEditor::parameterDataGridView_SelectionChanged);
			this->parameterDataGridView->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &ParameterEditor::parameterDataGridView_MouseDown);
			this->parameterDataGridView->CellContentClick += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::dataGridView_CellContentClick);
			// 
			// gridViewContextMenuStrip
			// 
			this->gridViewContextMenuStrip->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {this->polydispersityToolStripMenuItem, 
				this->editConstraintsToolStripMenuItem, this->linkToolStripMenuItem});
			this->gridViewContextMenuStrip->Name = L"contextMenuStrip1";
			this->gridViewContextMenuStrip->Size = System::Drawing::Size(165, 70);
			this->gridViewContextMenuStrip->Opening += gcnew System::ComponentModel::CancelEventHandler(this, &ParameterEditor::gridViewContextMenuStrip_Opening);
			// 
			// polydispersityToolStripMenuItem
			// 
			this->polydispersityToolStripMenuItem->Name = L"polydispersityToolStripMenuItem";
			this->polydispersityToolStripMenuItem->Size = System::Drawing::Size(164, 22);
			this->polydispersityToolStripMenuItem->Text = L"Polydispersity...";
			this->polydispersityToolStripMenuItem->Click += gcnew System::EventHandler(this, &ParameterEditor::polydispersityToolStripMenuItem_Click);
			// 
			// editConstraintsToolStripMenuItem
			// 
			this->editConstraintsToolStripMenuItem->Name = L"editConstraintsToolStripMenuItem";
			this->editConstraintsToolStripMenuItem->Size = System::Drawing::Size(164, 22);
			this->editConstraintsToolStripMenuItem->Text = L"Edit constraints...";
			this->editConstraintsToolStripMenuItem->Click += gcnew System::EventHandler(this, &ParameterEditor::editConstraintsToolStripMenuItem_Click);
			// 
			// linkToolStripMenuItem
			// 
			this->linkToolStripMenuItem->Name = L"linkToolStripMenuItem";
			this->linkToolStripMenuItem->Size = System::Drawing::Size(164, 22);
			this->linkToolStripMenuItem->Text = L"Link";
			this->linkToolStripMenuItem->Click += gcnew System::EventHandler(this, &ParameterEditor::linkToolStripMenuItem_Click);
			// 
			// extraParamsDataGridView
			// 
			this->extraParamsDataGridView->AllowUserToAddRows = false;
			this->extraParamsDataGridView->AllowUserToDeleteRows = false;
			this->extraParamsDataGridView->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
			this->extraParamsDataGridView->ContextMenuStrip = this->gridViewContextMenuStrip;
			this->extraParamsDataGridView->Dock = System::Windows::Forms::DockStyle::Fill;
			this->extraParamsDataGridView->Location = System::Drawing::Point(0, 0);
			this->extraParamsDataGridView->Name = L"extraParamsDataGridView";
			this->extraParamsDataGridView->Size = System::Drawing::Size(431, 195);
			this->extraParamsDataGridView->TabIndex = 0;
			this->extraParamsDataGridView->CellEndEdit += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::DataGridView_CellEndEdit);
			this->extraParamsDataGridView->CellValueChanged += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::DataGridView_OnCellValueChanged);
			this->extraParamsDataGridView->CellMouseUp += gcnew System::Windows::Forms::DataGridViewCellMouseEventHandler(this, &ParameterEditor::DataGridView_OnCellMouseUp);
			this->extraParamsDataGridView->EditingControlShowing += gcnew System::Windows::Forms::DataGridViewEditingControlShowingEventHandler(this, &ParameterEditor::dataGridView_EditingControlShowing);
			this->extraParamsDataGridView->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &ParameterEditor::parameterDataGridView_MouseDown);
			this->extraParamsDataGridView->CellContentClick += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &ParameterEditor::dataGridView_CellContentClick);
			// 
			// ParameterEditor
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(431, 452);
			this->Controls->Add(this->splitContainer1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"ParameterEditor";
			this->ShowIcon = false;
			this->Text = L"Parameter Editor";
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->parameterDataGridView))->EndInit();
			this->gridViewContextMenuStrip->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->extraParamsDataGridView))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

		 void consDummy();
		 public: void FillParamGridView(Entity ^ent);
private: System::Void dataGridView_EditingControlShowing(System::Object^  sender, System::Windows::Forms::DataGridViewEditingControlShowingEventArgs^  e);
private: System::Void dataGridView_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
private: System::Void DataGridView_CellEndEdit(System::Object^  sender, System::Windows::Forms::DataGridViewCellEventArgs^  e);
private: System::Void DataGridView_OnCellValueChanged(System::Object^  sender, System::Windows::Forms::DataGridViewCellEventArgs^  e);
private: System::Void DataGridView_OnCellMouseUp(System::Object^  sender, System::Windows::Forms::DataGridViewCellMouseEventArgs^  e);
private: System::Void gridViewContextMenuStrip_Opening(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e);
private: System::Void polydispersityToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void editConstraintsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void linkToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void parameterDataGridView_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
private: System::Void removeLayerButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void addLayerButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void parameterDataGridView_SelectionChanged(System::Object^  sender, System::EventArgs^  e);
		 System::Void handleCheckChangeInDataGridView(System::Object^ sender, int col, int row);
		 System::Void handleComboBoxChangeInDataGridView(System::Object^ sender, int col, int row, ComboBox^ cb);
		 System::Void dataGridView_CellContentClick(System::Object^ sender, DataGridViewCellEventArgs^ e);
		 System::Void ComboBoxSelectedIndexChange(System::Object^ sender, System::EventArgs^ e);
};
}
