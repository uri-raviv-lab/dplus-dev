#pragma once
#include <vector>
#include <time.h>

using namespace std;
using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace PopulationGUI {

    /// <summary>
    /// Summary for FileExpressionPanel
	/// A GUI class that contains an intensity curve or an expression representing one
    /// </summary>
    public ref class FileExpressionPanel : public System::Windows::Forms::UserControl
    {
    public:
        FileExpressionPanel(void);
		FileExpressionPanel(char des);
		FileExpressionPanel(String ^fileName, std::vector<double> x, std::vector<double> y, char des);
		FileExpressionPanel(String ^expression, char des);
        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        ~FileExpressionPanel();
		int GetColorRef();

		// The index of the correlated graph
		int graphIndex;
		String^ path;
		String^ file;
		System::Windows::Forms::ToolTip^ ttip;
		// Expression and file vectors respectively
		std::vector<double> *xE, *yE, *xF, *yF, *x, *y;

	System::Windows::Forms::RadioButton^  fileRadioButton;
	System::Windows::Forms::RadioButton^  expressionRadioButton;

    System::Windows::Forms::TextBox^  filenameTextBox;
    System::Windows::Forms::CheckBox^  visCheckBox;
    System::Windows::Forms::TextBox^  expressionTextBox;
	System::Windows::Forms::TextBox^  designationTextBox;
	System::Windows::Forms::Label^  colorLabel;
	System::Windows::Forms::Button^  removeButton;
	private: System::Windows::Forms::Label^  designationLabel;
	public: System::Windows::Forms::Button^  exportButton;
	private: System::Windows::Forms::ContextMenuStrip^  fileContextMenuStrip;
	private: System::Windows::Forms::ToolStripMenuItem^  toggleFilenameToolStripMenuItem;
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
			this->fileRadioButton = (gcnew System::Windows::Forms::RadioButton());
			this->fileContextMenuStrip = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->toggleFilenameToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->expressionRadioButton = (gcnew System::Windows::Forms::RadioButton());
			this->filenameTextBox = (gcnew System::Windows::Forms::TextBox());
			this->visCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->expressionTextBox = (gcnew System::Windows::Forms::TextBox());
			this->designationTextBox = (gcnew System::Windows::Forms::TextBox());
			this->designationLabel = (gcnew System::Windows::Forms::Label());
			this->colorLabel = (gcnew System::Windows::Forms::Label());
			this->removeButton = (gcnew System::Windows::Forms::Button());
			this->exportButton = (gcnew System::Windows::Forms::Button());
			this->fileContextMenuStrip->SuspendLayout();
			this->SuspendLayout();
			// 
			// fileRadioButton
			// 
			this->fileRadioButton->AutoSize = true;
			this->fileRadioButton->ContextMenuStrip = this->fileContextMenuStrip;
			this->fileRadioButton->Location = System::Drawing::Point(100, 4);
			this->fileRadioButton->Name = L"fileRadioButton";
			this->fileRadioButton->Size = System::Drawing::Size(41, 17);
			this->fileRadioButton->TabIndex = 0;
			this->fileRadioButton->TabStop = true;
			this->fileRadioButton->Text = L"File";
			this->fileRadioButton->UseVisualStyleBackColor = true;
			this->fileRadioButton->Click += gcnew System::EventHandler(this, &FileExpressionPanel::fileRadioButton_Clicked);
			// 
			// fileContextMenuStrip
			// 
			this->fileContextMenuStrip->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->toggleFilenameToolStripMenuItem});
			this->fileContextMenuStrip->Name = L"fileContextMenuStrip";
			this->fileContextMenuStrip->Size = System::Drawing::Size(153, 48);
			// 
			// toggleFilenameToolStripMenuItem
			// 
			this->toggleFilenameToolStripMenuItem->Name = L"toggleFilenameToolStripMenuItem";
			this->toggleFilenameToolStripMenuItem->Size = System::Drawing::Size(152, 22);
			this->toggleFilenameToolStripMenuItem->Text = L"Toggle Filename";
			this->toggleFilenameToolStripMenuItem->Click += gcnew System::EventHandler(this, &FileExpressionPanel::toggleFilenameToolStripMenuItem_Click);
			// 
			// expressionRadioButton
			// 
			this->expressionRadioButton->AutoSize = true;
			this->expressionRadioButton->Location = System::Drawing::Point(100, 27);
			this->expressionRadioButton->Name = L"expressionRadioButton";
			this->expressionRadioButton->Size = System::Drawing::Size(76, 17);
			this->expressionRadioButton->TabIndex = 1;
			this->expressionRadioButton->TabStop = true;
			this->expressionRadioButton->Text = L"Expression";
			this->expressionRadioButton->UseVisualStyleBackColor = true;
			// 
			// filenameTextBox
			// 
			this->filenameTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->filenameTextBox->ForeColor = System::Drawing::SystemColors::WindowText;
			this->filenameTextBox->Location = System::Drawing::Point(174, 3);
			this->filenameTextBox->Name = L"filenameTextBox";
			this->filenameTextBox->ReadOnly = true;
			this->filenameTextBox->Size = System::Drawing::Size(189, 20);
			this->filenameTextBox->TabIndex = 2;
			this->filenameTextBox->Text = L"Filename";
			this->filenameTextBox->MouseEnter += gcnew System::EventHandler(this, &FileExpressionPanel::ResetTT);
			// 
			// visCheckBox
			// 
			this->visCheckBox->AutoSize = true;
			this->visCheckBox->CheckAlign = System::Drawing::ContentAlignment::TopCenter;
			this->visCheckBox->Location = System::Drawing::Point(-1, 17);
			this->visCheckBox->Name = L"visCheckBox";
			this->visCheckBox->Size = System::Drawing::Size(41, 31);
			this->visCheckBox->TabIndex = 3;
			this->visCheckBox->Text = L"Visible";
			this->visCheckBox->UseVisualStyleBackColor = true;
			// 
			// expressionTextBox
			// 
			this->expressionTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->expressionTextBox->Location = System::Drawing::Point(174, 25);
			this->expressionTextBox->Name = L"expressionTextBox";
			this->expressionTextBox->Size = System::Drawing::Size(189, 20);
			this->expressionTextBox->TabIndex = 4;
			this->expressionTextBox->Leave += gcnew System::EventHandler(this, &FileExpressionPanel::expressionTextBox_Leave);
			this->expressionTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &FileExpressionPanel::expressionTextBox_KeyPress);
			// 
			// designationTextBox
			// 
			this->designationTextBox->Location = System::Drawing::Point(56, 23);
			this->designationTextBox->Name = L"designationTextBox";
			this->designationTextBox->ReadOnly = true;
			this->designationTextBox->Size = System::Drawing::Size(21, 20);
			this->designationTextBox->TabIndex = 5;
			this->designationTextBox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// designationLabel
			// 
			this->designationLabel->AutoSize = true;
			this->designationLabel->Location = System::Drawing::Point(35, 6);
			this->designationLabel->Name = L"designationLabel";
			this->designationLabel->Size = System::Drawing::Size(63, 13);
			this->designationLabel->TabIndex = 6;
			this->designationLabel->Text = L"Designation";
			// 
			// colorLabel
			// 
			this->colorLabel->AutoSize = true;
			this->colorLabel->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->colorLabel->Location = System::Drawing::Point(12, 3);
			this->colorLabel->Name = L"colorLabel";
			this->colorLabel->Size = System::Drawing::Size(13, 13);
			this->colorLabel->TabIndex = 7;
			this->colorLabel->Text = L"  ";
			this->colorLabel->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &FileExpressionPanel::colorLabel_MouseClick);
			// 
			// removeButton
			// 
			this->removeButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->removeButton->Location = System::Drawing::Point(367, 25);
			this->removeButton->Name = L"removeButton";
			this->removeButton->Size = System::Drawing::Size(55, 20);
			this->removeButton->TabIndex = 8;
			this->removeButton->Text = L"Remove";
			this->removeButton->UseVisualStyleBackColor = true;
			// 
			// exportButton
			// 
			this->exportButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->exportButton->Location = System::Drawing::Point(367, 4);
			this->exportButton->Name = L"exportButton";
			this->exportButton->Size = System::Drawing::Size(55, 20);
			this->exportButton->TabIndex = 8;
			this->exportButton->Text = L"Export...";
			this->exportButton->UseVisualStyleBackColor = true;
			// 
			// FileExpressionPanel
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->Controls->Add(this->exportButton);
			this->Controls->Add(this->removeButton);
			this->Controls->Add(this->colorLabel);
			this->Controls->Add(this->designationLabel);
			this->Controls->Add(this->designationTextBox);
			this->Controls->Add(this->expressionTextBox);
			this->Controls->Add(this->visCheckBox);
			this->Controls->Add(this->filenameTextBox);
			this->Controls->Add(this->expressionRadioButton);
			this->Controls->Add(this->fileRadioButton);
			this->Name = L"FileExpressionPanel";
			this->Size = System::Drawing::Size(427, 47);
			this->fileContextMenuStrip->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
    private: 
		void init();
		System::Void expressionTextBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
		System::Void expressionTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void ResetTT(System::Object^  sender, System::EventArgs^  e);
		System::Void fileRadioButton_Clicked(System::Object^  sender, System::EventArgs^  e);
		System::Void expressionTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
		System::Void expressionTextBox_Leave(System::Object^  sender, System::EventArgs^  e);
		System::Void colorLabel_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		System::Void toggleFilenameToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
};
}

