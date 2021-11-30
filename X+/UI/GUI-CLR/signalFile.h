#pragma once
#include <vector>

//#define PUSH 75

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace GUICLR {

	/// <summary>
	/// Summary for signalFile
	/// </summary>
	public ref class signalFile : public System::Windows::Forms::UserControl
	{
		/* Contains:
		 * 			2 textboxes ( * scale + BG)
		 * 			2 check boxes (visible, random color)
		 * 			label for filename + ToolTip for the path
		 * 			+/- buttons for each textbox
		 */
	public:
		signalFile(void)
		{
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~signalFile()
		{
			if (components)
			{
				delete components;
			}
			if(x)
				delete x;
			if(y)
				delete y;
		}
		public:
		System::Windows::Forms::ComboBox^  comboBox1;
		System::Windows::Forms::CheckBox^  visCheckBox;
		System::Windows::Forms::TextBox^  bgTextBox;
		System::Windows::Forms::TextBox^  scaleTextBox;
		System::Windows::Forms::Label^  filenameLabel;
		System::Windows::Forms::CheckBox^  selectedCheckBox;
		System::Windows::Forms::Label^  colorLabel;
		System::Windows::Forms::Label^  scaleMinusLabel;
		System::Windows::Forms::Label^  bgMinusLabel;
		System::Windows::Forms::Label^  bgPlusLabel;
		System::Windows::Forms::Label^  scalePlusLabel;

	public: 
		int index;
		System::String^ path;
		System::String^ file;
		System::Windows::Forms::ToolTip^ ttip;
		std::vector<double> *x, *y;
		bool bScalePlus, bScaleMinus, bBGPlus, bBGMinus;
	private: System::ComponentModel::IContainer^  components;
	public: 
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
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->visCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->selectedCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->filenameLabel = (gcnew System::Windows::Forms::Label());
			this->scaleTextBox = (gcnew System::Windows::Forms::TextBox());
			this->bgTextBox = (gcnew System::Windows::Forms::TextBox());
			this->scalePlusLabel = (gcnew System::Windows::Forms::Label());
			this->bgPlusLabel = (gcnew System::Windows::Forms::Label());
			this->bgMinusLabel = (gcnew System::Windows::Forms::Label());
			this->scaleMinusLabel = (gcnew System::Windows::Forms::Label());
			this->colorLabel = (gcnew System::Windows::Forms::Label());
			this->ttip = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->SuspendLayout();
			// 
			// comboBox1
			// 
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Location = System::Drawing::Point(61, 70);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(121, 21);
			this->comboBox1->TabIndex = 0;
			// 
			// visCheckBox
			// 
			this->visCheckBox->AutoSize = true;
			this->visCheckBox->Checked = true;
			this->visCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->visCheckBox->Location = System::Drawing::Point(4, 7);
			this->visCheckBox->Name = L"visCheckBox";
			this->visCheckBox->Size = System::Drawing::Size(15, 14);
			this->visCheckBox->TabIndex = 0;
			this->visCheckBox->UseVisualStyleBackColor = true;
			// 
			// selectedCheckBox
			// 
			this->selectedCheckBox->AutoSize = true;
			this->selectedCheckBox->Location = System::Drawing::Point(25, 7);
			this->selectedCheckBox->Name = L"selectedCheckBox";
			this->selectedCheckBox->Size = System::Drawing::Size(15, 14);
			this->selectedCheckBox->TabIndex = 0;
			this->selectedCheckBox->UseVisualStyleBackColor = true;
			// 
			// filenameLabel
			// 
			this->filenameLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->filenameLabel->AutoSize = true;
			this->filenameLabel->Location = System::Drawing::Point(63, 7);
			this->filenameLabel->Name = L"filenameLabel";
			this->filenameLabel->Size = System::Drawing::Size(153, 13);
			this->filenameLabel->TabIndex = 1;
			this->filenameLabel->Text = L"This is a filename or entire path";
			this->filenameLabel->MouseEnter += gcnew System::EventHandler(this, &signalFile::ResetTT);
			// 
			// scaleTextBox
			// 
			this->scaleTextBox->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->scaleTextBox->Location = System::Drawing::Point(207, 4);
			this->scaleTextBox->Name = L"scaleTextBox";
			this->scaleTextBox->Size = System::Drawing::Size(50, 20);
			this->scaleTextBox->TabIndex = 2;
			this->scaleTextBox->Text = L"1.0000";
			// 
			// bgTextBox
			// 
			this->bgTextBox->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->bgTextBox->Location = System::Drawing::Point(280, 4);
			this->bgTextBox->Name = L"bgTextBox";
			this->bgTextBox->Size = System::Drawing::Size(47, 20);
			this->bgTextBox->TabIndex = 2;
			this->bgTextBox->Text = L"0.0000";
			// 
			// scalePlusLabel
			// 
			this->scalePlusLabel->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->scalePlusLabel->AutoSize = true;
			this->scalePlusLabel->Location = System::Drawing::Point(258, 1);
			this->scalePlusLabel->Name = L"scalePlusLabel";
			this->scalePlusLabel->Size = System::Drawing::Size(13, 13);
			this->scalePlusLabel->TabIndex = 3;
			this->scalePlusLabel->Text = L"+";
			this->scalePlusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// bgPlusLabel
			// 
			this->bgPlusLabel->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->bgPlusLabel->AutoSize = true;
			this->bgPlusLabel->Location = System::Drawing::Point(328, 1);
			this->bgPlusLabel->Name = L"bgPlusLabel";
			this->bgPlusLabel->Size = System::Drawing::Size(13, 13);
			this->bgPlusLabel->TabIndex = 3;
			this->bgPlusLabel->Text = L"+";
			this->bgPlusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// bgMinusLabel
			// 
			this->bgMinusLabel->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->bgMinusLabel->AutoSize = true;
			this->bgMinusLabel->Location = System::Drawing::Point(329, 15);
			this->bgMinusLabel->Name = L"bgMinusLabel";
			this->bgMinusLabel->Size = System::Drawing::Size(10, 13);
			this->bgMinusLabel->TabIndex = 3;
			this->bgMinusLabel->Text = L"-";
			this->bgMinusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// scaleMinusLabel
			// 
			this->scaleMinusLabel->Anchor = System::Windows::Forms::AnchorStyles::Right;
			this->scaleMinusLabel->AutoSize = true;
			this->scaleMinusLabel->Location = System::Drawing::Point(260, 15);
			this->scaleMinusLabel->Name = L"scaleMinusLabel";
			this->scaleMinusLabel->Size = System::Drawing::Size(10, 13);
			this->scaleMinusLabel->TabIndex = 3;
			this->scaleMinusLabel->Text = L"-";
			this->scaleMinusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// colorLabel
			// 
			this->colorLabel->AutoSize = true;
			this->colorLabel->Location = System::Drawing::Point(45, 7);
			this->colorLabel->Name = L"colorLabel";
			this->colorLabel->Size = System::Drawing::Size(13, 13);
			this->colorLabel->TabIndex = 1;
			this->colorLabel->Text = L"  ";
			// 
			// signalFile
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->Controls->Add(this->scaleMinusLabel);
			this->Controls->Add(this->bgMinusLabel);
			this->Controls->Add(this->bgPlusLabel);
			this->Controls->Add(this->scalePlusLabel);
			this->Controls->Add(this->bgTextBox);
			this->Controls->Add(this->scaleTextBox);
			this->Controls->Add(this->filenameLabel);
			this->Controls->Add(this->selectedCheckBox);
			this->Controls->Add(this->visCheckBox);
			this->Controls->Add(this->colorLabel);
			this->Controls->Add(this->comboBox1);
			this->Name = L"signalFile";
			this->Size = System::Drawing::Size(340, 32);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
public:
		// Display filename or full path
		void fileText(bool fullPath) {
			this->filenameLabel->Text = (fullPath) ? path : file;
			this->ttip->SetToolTip(this->filenameLabel, (fullPath) ? file : path);
			this->filenameLabel->Visible = true;
		}

		// Required in order to show the tool tip after it pops
		void ResetTT(System::Object^ sender, System::EventArgs^ e) {
			this->ttip->Active = false;
			this->ttip->Active = true;
		}

		// Change the color of the legend
		void setColor(int r, int g, int b) {
			this->colorLabel->BackColor = System::Drawing::Color::FromArgb(r, g, b);
		}
		
		void setColor(System::Drawing::Color col) {
			this->colorLabel->BackColor = col;			
		}

		// Constructor. The filename and index need to be defined
		signalFile(System::String^ filepath, bool fullPath, int pos, std::vector<double> nx, std::vector<double> ny) {
			signalFile();
			InitializeComponent();
			this->path	= (gcnew System::String(L""));
			this->file	= (gcnew System::String(L""));
			this->x		= (new std::vector<double>());
			this->y		= (new std::vector<double>());
			index = pos;
			path = String::Copy(filepath);
			file = path->Substring(path->LastIndexOf(L"\\") + 1);
			*x = nx;
			*y = ny;
 
			bScalePlus	= false;
			bScaleMinus	= false;
			bBGPlus		= false;
			bBGMinus	= false;
			fileText(fullPath);
			this->Size = System::Drawing::Size(340, 32);
		}

	};
}
