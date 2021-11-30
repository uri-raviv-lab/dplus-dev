#pragma once

#include "UnicodeChars.h"
#include "signalFile.h"
#include <time.h>

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::Threading;

#define PUSH 75

namespace GUICLR {
	//////////////////////////////////////////////////////////////////////////
	// Container for OpeningWindow
	// Contains a radioButton for loading ModelInformation names and indices.
	//////////////////////////////////////////////////////////////////////////
	public ref class modelInfoRadioButton : public System::Windows::Forms::RadioButton {
	public:
		int catIndex, modelIndex, indexInVec;
		String^ contName;

		modelInfoRadioButton(int catI, int modI, int vecI, const char* name, const wchar_t* container) : RadioButton() {
			catIndex = catI;
			modelIndex = modI;
			indexInVec = vecI;

			contName = gcnew String(stringToClr(std::wstring(container)));

			this->Name = gcnew String(stringToClr(std::string(name)));
			this->Text = this->Name;
		}

		virtual String^ ToString() override {
			return gcnew String(this->Name);
		}

		~modelInfoRadioButton() {
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// Container for comboBoxes
	// Contains an Object for loading ModelInformation names and indices.
	//////////////////////////////////////////////////////////////////////////
	public ref class modelInfoObject : public System::Object {
	public:
		int catIndex, modelIndex, indexInVec;
		String ^contName, ^Name;

		modelInfoObject(int catI, int modI, int vecI, const char* name, const wchar_t* container) : Object() {
			catIndex = catI;
			modelIndex = modI;
			indexInVec = vecI;

			contName = gcnew String(stringToClr(std::wstring(container)));

			Name = gcnew String(stringToClr(std::string(name)));
		}
		modelInfoObject(int catI, int modI, int vecI, const char* name, String ^container) : Object() {
			catIndex = catI;
			modelIndex = modI;
			indexInVec = vecI;

			//contName = container->ToString();
			contName = String::Copy(container);

			Name = gcnew String(stringToClr(std::string(name)));
		}
		virtual String^ ToString() override {
			return gcnew String(Name);
		}

		~modelInfoObject() {
		}
	};

	//////////////////////////////////////////////////////////
	// This class is a groupBox with a textBox, trackBar	//
	// and checkBox used in the GUI to display and change	//
	// the values of the parameters in the listViews		//
	//////////////////////////////////////////////////////////
	public ref class ParamGroupBox: public System::Windows::Forms::GroupBox
	{
	public: 
		System::Windows::Forms::TextBox^  text;
		System::Windows::Forms::CheckBox^ check;
		System::Windows::Forms::TrackBar^ track;
		System::Windows::Forms::RadioButton^ rValue;
		System::Windows::Forms::RadioButton^ rStddev;

		int col;
		bool bPD;

		void Init() {
			this->text = (gcnew System::Windows::Forms::TextBox());
			this->check = (gcnew System::Windows::Forms::CheckBox());
			this->track = (gcnew System::Windows::Forms::TrackBar());
			this->rValue = (gcnew System::Windows::Forms::RadioButton());
			this->rStddev = (gcnew System::Windows::Forms::RadioButton());

			/////
			// GroupBox
			/////
			this->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left));
			this->Controls->Add(this->text);
			this->Controls->Add(this->check);
			this->Controls->Add(this->track);
			this->Controls->Add(this->rValue);
			this->Controls->Add(this->rStddev);
			this->Location = System::Drawing::Point(441, 15);
			this->Name = L"groupBoxI";
			this->Size = System::Drawing::Size(107, 89);
			this->TabIndex = 732;
			this->TabStop = false;
			this->Text = L"Mistake";
		
			/////
			// TextBox
			/////
			this->text->Name = L"textBox";
			this->text->Size = System::Drawing::Size(152, 21);
			this->text->Text = L"0.000000";
			this->text->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->text->Location = System::Drawing::Point(6, 44);
			this->text->Size = System::Drawing::Size(74, 20);


			/////
			// CheckBox
			/////
			this->check->AutoSize = true;
			this->check->Location = System::Drawing::Point(86, 47);
			this->check->Name = L"checkBox1";
			this->check->Size = System::Drawing::Size(15, 14);
			this->check->TabIndex = 0;
			this->check->UseVisualStyleBackColor = true;

			/////
			// rValue
			/////
			this->rValue->AutoSize = true;
			this->rValue->Location = System::Drawing::Point(6, 67);
			this->rValue->Name = L"rValue";
			this->rValue->Checked = true;
			this->rValue->Size = System::Drawing::Size(15, 14);
			this->rValue->Text = L"Value";
			this->rValue->TabIndex = 2;
			this->rValue->UseVisualStyleBackColor = true;
			this->rValue->Visible = bPD;
			
			/////
			// rStddev
			/////
			this->rStddev->AutoSize = true;
			this->rStddev->Location = System::Drawing::Point(70, 67);
			this->rStddev->Name = L"rStddev";
			this->rStddev->Checked = false;
			this->rStddev->Size = System::Drawing::Size(15, 14);
			this->rStddev->Text = UnicodeChars::sigmaUnicode + UnicodeChars::sqr;
			this->rStddev->TabIndex = 2;
			this->rStddev->UseVisualStyleBackColor = true;
			this->rStddev->Visible = bPD;

			/////
			// TrackBar
			/////
			this->track->Location = System::Drawing::Point(6, 19);
			this->track->Maximum = 100;
			this->track->Name = L"";
			this->track->Size = System::Drawing::Size(96, 45);
			this->track->TabIndex = 1;
			this->track->TickStyle = System::Windows::Forms::TickStyle::None;
		}

		ParamGroupBox(System::String^ str, int columnInd, bool bCanBePD) {
			bPD = bCanBePD;
			col = columnInd;
			Init();
			this->Text = str;
			this->Enabled = false;
		    track->Value = int((track->Maximum - track->Minimum) / 2.0);
		}
	};

	////////////////////////////////////////////////////////
	////	A groupBox that contains all the needed 	////
	////	GUI elements for the constraints window		////
	////////////////////////////////////////////////////////
	public ref class ConstraintGroupBox: public System::Windows::Forms::GroupBox
	{
	public:
		System::Windows::Forms::Label^		absMinLabel;
		System::Windows::Forms::Label^		absMaxLabel;
		System::Windows::Forms::Label^		largerLabel;
		System::Windows::Forms::Label^		smallerLabel;
		System::Windows::Forms::Label^		linkedLabel;
		System::Windows::Forms::TextBox^	minText;
		System::Windows::Forms::TextBox^	maxText;
		System::Windows::Forms::CheckBox^	consCheckBox;
		System::Windows::Forms::ComboBox^	largerCombo;
		System::Windows::Forms::ComboBox^	smallerCombo;
		System::Windows::Forms::ComboBox^	linkedCombo;

		System::Collections::Generic::List<ComboBox^>^ comboBoxes;
		System::Collections::Generic::List<TextBox^>^ textBoxes;

		int paramNum;

		void Init() {
			this->comboBoxes = gcnew System::Collections::Generic::List<ComboBox^>();
			this->textBoxes = gcnew System::Collections::Generic::List<TextBox^>();

			this->maxText		= (gcnew System::Windows::Forms::TextBox());
			this->minText		= (gcnew System::Windows::Forms::TextBox());
			this->consCheckBox	= (gcnew System::Windows::Forms::CheckBox());
			this->linkedLabel	= (gcnew System::Windows::Forms::Label());
			this->smallerLabel	= (gcnew System::Windows::Forms::Label());
			this->largerLabel	= (gcnew System::Windows::Forms::Label());
			this->absMaxLabel	= (gcnew System::Windows::Forms::Label());
			this->absMinLabel	= (gcnew System::Windows::Forms::Label());

			this->linkedCombo	= (gcnew System::Windows::Forms::ComboBox());
			this->smallerCombo	= (gcnew System::Windows::Forms::ComboBox());
			this->largerCombo	= (gcnew System::Windows::Forms::ComboBox());

			this->Controls->Add(this->absMinLabel);
			this->Controls->Add(this->absMaxLabel);
			this->Controls->Add(this->largerLabel);
			this->Controls->Add(this->smallerLabel);
			this->Controls->Add(this->linkedLabel);
			this->Controls->Add(this->minText);
			this->Controls->Add(this->maxText);
			this->Controls->Add(this->consCheckBox);
			this->Controls->Add(this->largerCombo);
			this->Controls->Add(this->smallerCombo);
			this->Controls->Add(this->linkedCombo);

			////////////////////////////////////////////////
			//////////////////// Labels ////////////////////
			////////////////////////////////////////////////
			this->absMinLabel->AutoSize = true;
			this->absMinLabel->Location = System::Drawing::Point(10, 16);
			this->absMinLabel->Name = L"absMinLabel";
			this->absMinLabel->Size = System::Drawing::Size(95, 13);
			this->absMinLabel->TabIndex = 13;
			this->absMinLabel->Text = L"Absolute Minimum:";
			
			this->absMaxLabel->AutoSize = true;
			this->absMaxLabel->Location = System::Drawing::Point(299, 16);
			this->absMaxLabel->Name = L"absMaxLabel";
			this->absMaxLabel->Size = System::Drawing::Size(98, 13);
			this->absMaxLabel->TabIndex = 15;
			this->absMaxLabel->Text = L"Absolute Maximum:";

			this->largerLabel->AutoSize = true;
			this->largerLabel->Location = System::Drawing::Point(10, 48);
			this->largerLabel->Name = L"largerLabel";
			this->largerLabel->Size = System::Drawing::Size(68, 13);
			this->largerLabel->TabIndex = 14;
			this->largerLabel->Text = L"Larger Than:";

			this->smallerLabel->AutoSize = true;
			this->smallerLabel->Location = System::Drawing::Point(299, 48);
			this->smallerLabel->Name = L"smallerLabel";
			this->smallerLabel->Size = System::Drawing::Size(72, 13);
			this->smallerLabel->TabIndex = 16;
			this->smallerLabel->Text = L"Smaller Than:";

			this->linkedLabel->AutoSize = true;
			this->linkedLabel->Location = System::Drawing::Point(190, 80);
			this->linkedLabel->Name = L"linkedLabel";
			this->linkedLabel->Size = System::Drawing::Size(54, 13);
			this->linkedLabel->TabIndex = 21;
			this->linkedLabel->Text = L"Linked to:";

			////////////////////////////////////////////////
			////////////////// Text Boxes //////////////////
			////////////////////////////////////////////////
			this->minText->Location = System::Drawing::Point(111, 15);
			this->minText->Name = L"minText";
			this->minText->Size = System::Drawing::Size(100, 20);
			this->minText->TabIndex = 7;
			this->minText->Text = L"0.000000";

			this->maxText->Location = System::Drawing::Point(403, 15);
			this->maxText->Name = L"maxText";
			this->maxText->Size = System::Drawing::Size(100, 20);
			this->maxText->TabIndex = 9;
			this->maxText->Text = L"0.000000";

			////////////////////////////////////////////////
			////////////////// Check Box ///////////////////
			////////////////////////////////////////////////
			this->consCheckBox->AutoSize = true;
			this->consCheckBox->Location = System::Drawing::Point(225, 12);
			this->consCheckBox->Name = L"consCheckBox";
			this->consCheckBox->Size = System::Drawing::Size(59, 17);
			this->consCheckBox->TabIndex = 71;
			this->consCheckBox->Text = L"Use this\nconstraint";
			this->consCheckBox->UseVisualStyleBackColor = true;



			////////////////////////////////////////////////
			////////////////// ComboBoxes //////////////////
			////////////////////////////////////////////////
			this->largerCombo->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->largerCombo->FormattingEnabled = true;
			this->largerCombo->Items->AddRange(gcnew cli::array< System::Object^  >(1) {L"None"});
			this->largerCombo->Location = System::Drawing::Point(111, 45);
			this->largerCombo->Name = L"largerCombo";
			this->largerCombo->Size = System::Drawing::Size(121, 21);
			this->largerCombo->TabIndex = 4;

			this->smallerCombo->DisplayMember = L"None";
			this->smallerCombo->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->smallerCombo->FormattingEnabled = true;
			this->smallerCombo->Items->AddRange(gcnew cli::array< System::Object^  >(1) {L"None"});
			this->smallerCombo->Location = System::Drawing::Point(382, 45);
			this->smallerCombo->Name = L"smallerCombo";
			this->smallerCombo->Size = System::Drawing::Size(121, 21);
			this->smallerCombo->TabIndex = 5;
			this->smallerCombo->ValueMember = L"None";
			
			this->linkedCombo->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->linkedCombo->FormattingEnabled = true;
			this->linkedCombo->Items->AddRange(gcnew cli::array< System::Object^  >(1) {L"None"});
			this->linkedCombo->Location = System::Drawing::Point(245, 77);
			this->linkedCombo->Name = L"linkedCombo";
			this->linkedCombo->Size = System::Drawing::Size(121, 21);
			this->linkedCombo->TabIndex = 2;

			////////////////////////////////////////////////
			////////////////// Init Lists //////////////////
			////////////////////////////////////////////////
			this->textBoxes->Add(this->minText);
			this->textBoxes->Add(this->maxText);
			this->comboBoxes->Add(this->largerCombo);
			this->comboBoxes->Add(this->smallerCombo);
			this->comboBoxes->Add(this->linkedCombo);
		}
		// Constructor. The parameter name and index need to be defined
		ConstraintGroupBox(System::String^ str, int ind) {
			Init();
			this->Text	= str;
			paramNum	= ind;

			this->Size = System::Drawing::Size(510, 110);
		}
	};

	//////////////////////////////////////////////////////////
	//// Fill this in later...
	////
	//////////////////////////////////////////////////////////
	//public ref class signalFile : public System::Windows::Forms::Panel
	//{
	//	/* Contains:
	//	 * 			2 textboxes ( * scale + BG)
	//	 * 			2 check boxes (visible, random color)
	//	 * 			label for filename + ToolTip for the path
	//	 * 			+/- buttons for each textbox
	//	 */
	//public:
	//	int index;
	//	String^ path;
	//	String^ file;
	//	System::Windows::Forms::ToolTip^ ttip;
	//	std::vector<double> *x, *y;
	//	System::Windows::Forms::CheckBox^  visCheckBox;
	//	System::Windows::Forms::TextBox^  bgTextBox;
	//	System::Windows::Forms::TextBox^  scaleTextBox;
	//	System::Windows::Forms::Label^  filenameLabel;
	//	System::Windows::Forms::CheckBox^  selectedCheckBox;
	//	System::Windows::Forms::Label^  colorLabel;
	//	System::Windows::Forms::Label^  scaleMinusLabel;
	//	System::Windows::Forms::Label^  bgMinusLabel;
	//	System::Windows::Forms::Label^  bgPlusLabel;
	//	System::Windows::Forms::Label^  scalePlusLabel;

	//	bool bScalePlus, bScaleMinus, bBGPlus, bBGMinus;

	//	void Init() {
	//		this->path				= (gcnew String(L""));
	//		this->file				= (gcnew String(L""));
	//		this->x					= (new std::vector<double>());
	//		this->y					= (new std::vector<double>());
	//		this->visCheckBox		= (gcnew System::Windows::Forms::CheckBox());
	//		this->selectedCheckBox	= (gcnew System::Windows::Forms::CheckBox());
	//		this->filenameLabel		= (gcnew System::Windows::Forms::Label());
	//		this->scaleTextBox		= (gcnew System::Windows::Forms::TextBox());
	//		this->bgTextBox			= (gcnew System::Windows::Forms::TextBox());
	//		this->scalePlusLabel	= (gcnew System::Windows::Forms::Label());
	//		this->bgPlusLabel		= (gcnew System::Windows::Forms::Label());
	//		this->bgMinusLabel		= (gcnew System::Windows::Forms::Label());
	//		this->scaleMinusLabel	= (gcnew System::Windows::Forms::Label());
	//		this->colorLabel		= (gcnew System::Windows::Forms::Label);
	//		this->ttip				= (gcnew ToolTip(gcnew System::ComponentModel::Container()));
	//		// 
	//		// visCheckBox
	//		// 
	//		this->visCheckBox->AutoSize = true;
	//		this->visCheckBox->Location = System::Drawing::Point(4, 7);
	//		this->visCheckBox->Name = L"visCheckBox";
	//		this->visCheckBox->Size = System::Drawing::Size(15, 14);
	//		this->visCheckBox->TabIndex = 0;
	//		this->visCheckBox->Checked = true;
	//		this->visCheckBox->UseVisualStyleBackColor = true;
	//		// 
	//		// selectedCheckBox
	//		// 
	//		this->selectedCheckBox->AutoSize = true;
	//		this->selectedCheckBox->Location = System::Drawing::Point(25, 7);
	//		this->selectedCheckBox->Name = L"selectedCheckBox";
	//		this->selectedCheckBox->Size = System::Drawing::Size(15, 14);
	//		this->selectedCheckBox->TabIndex = 0;
	//		this->selectedCheckBox->UseVisualStyleBackColor = true;
	//		// 
	//		// filenameLabel
	//		// 
	//		this->filenameLabel->AutoSize = true;
	//		this->filenameLabel->Location = System::Drawing::Point(63, 7);
	//		this->filenameLabel->Name = L"filenameLabel";
	//		this->filenameLabel->Size = System::Drawing::Size(153, 13);
	//		this->filenameLabel->TabIndex = 1;
	//		this->filenameLabel->Text = L"This is a filename or entire path";
	//		this->filenameLabel->MouseEnter += gcnew System::EventHandler(this, &signalFile::ResetTT);
	//		this->filenameLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// colorLabel
	//		// 
	//		this->colorLabel->AutoSize = true;
	//		this->colorLabel->Location = System::Drawing::Point(45, 7);
	//		this->colorLabel->Name = L"colorLabel";
	//		this->colorLabel->Size = System::Drawing::Size(15, 15);
	//		this->colorLabel->TabIndex = 1;
	//		this->colorLabel->Text = L"  ";
	//		this->colorLabel->BackColor =  System::Drawing::Color::FromArgb(
	//			(rand() % 256),
	//			(rand() % 256), 
	//			(rand() % 256));
	//		// 
	//		// scaleTextBox
	//		// 
	//		this->scaleTextBox->Location = System::Drawing::Point(207 + PUSH, 4);
	//		this->scaleTextBox->Name = L"scaleTextBox";
	//		this->scaleTextBox->Size = System::Drawing::Size(50, 20);
	//		this->scaleTextBox->TabIndex = 2;
	//		this->scaleTextBox->Text = L"1.0000";
	//		this->scaleTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// bgTextBox
	//		// 
	//		this->bgTextBox->Location = System::Drawing::Point(280 + PUSH, 4);
	//		this->bgTextBox->Name = L"bgTextBox";
	//		this->bgTextBox->Size = System::Drawing::Size(47, 20);
	//		this->bgTextBox->TabIndex = 2;
	//		this->bgTextBox->Text = L"0.0000";
	//		this->bgTextBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// scalePlusLabel
	//		// 
	//		this->scalePlusLabel->AutoSize = true;
	//		this->scalePlusLabel->Location = System::Drawing::Point(258 + PUSH, 1);
	//		this->scalePlusLabel->Name = L"scalePlusLabel";
	//		this->scalePlusLabel->Size = System::Drawing::Size(13, 13);
	//		this->scalePlusLabel->TabIndex = 3;
	//		this->scalePlusLabel->Text = L"+";
	//		this->scalePlusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
	//		this->scalePlusLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// bgPlusLabel
	//		// 
	//		this->bgPlusLabel->AutoSize = true;
	//		this->bgPlusLabel->Location = System::Drawing::Point(328 + PUSH, 1);
	//		this->bgPlusLabel->Name = L"bgPlusLabel";
	//		this->bgPlusLabel->Size = System::Drawing::Size(13, 13);
	//		this->bgPlusLabel->TabIndex = 3;
	//		this->bgPlusLabel->Text = L"+";
	//		this->bgPlusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
	//		this->bgPlusLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// bgMinusLabel
	//		// 
	//		this->bgMinusLabel->AutoSize = true;
	//		this->bgMinusLabel->Location = System::Drawing::Point(329 + PUSH, 15);
	//		this->bgMinusLabel->Name = L"bgMinusLabel";
	//		this->bgMinusLabel->Size = System::Drawing::Size(10, 13);
	//		this->bgMinusLabel->TabIndex = 3;
	//		this->bgMinusLabel->Text = L"-";
	//		this->bgMinusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
	//		this->bgMinusLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//			| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// scaleMinusLabel
	//		// 
	//		this->scaleMinusLabel->AutoSize = true;
	//		this->scaleMinusLabel->Location = System::Drawing::Point(260 + PUSH, 15);
	//		this->scaleMinusLabel->Name = L"scaleMinusLabel";
	//		this->scaleMinusLabel->Size = System::Drawing::Size(10, 13);
	//		this->scaleMinusLabel->TabIndex = 3;
	//		this->scaleMinusLabel->Text = L"-";
	//		this->scaleMinusLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
	//		//this->scaleMinusLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top 
	//		//	| System::Windows::Forms::AnchorStyles::Right));
	//		// 
	//		// panel
	//		// 
	//		this->Controls->Add(this->scaleMinusLabel);
	//		this->Controls->Add(this->bgMinusLabel);
	//		this->Controls->Add(this->bgPlusLabel);
	//		this->Controls->Add(this->scalePlusLabel);
	//		this->Controls->Add(this->bgTextBox);
	//		this->Controls->Add(this->scaleTextBox);
	//		this->Controls->Add(this->filenameLabel);
	//		this->Controls->Add(this->selectedCheckBox);
	//		this->Controls->Add(this->visCheckBox);
	//		this->Controls->Add(this->colorLabel);
	//		this->Location = System::Drawing::Point(0, 0);
	//		this->Size = System::Drawing::Size(458, 32);
	//		this->TabIndex = 0;
	//		this->Dock = System::Windows::Forms::DockStyle::Top;

	//	}

	//	// Display filename or full path
	//	void fileText(bool fullPath) {
	//		this->filenameLabel->Text = (fullPath) ? path : file;
	//		this->ttip->SetToolTip(this->filenameLabel, (fullPath) ? file : path);
	//		this->filenameLabel->Visible = true;
	//	}

	//	// Required in order to show the tool tip after it pops
	//	void ResetTT(System::Object^ sender, System::EventArgs^ e) {
	//		this->ttip->Active = false;
	//		this->ttip->Active = true;
	//	}

	//	// Change the color of the legend
	//	void setColor(int r, int g, int b) {
	//		this->colorLabel->BackColor = System::Drawing::Color::FromArgb(r, g, b);
	//	}
	//	
	//	void setColor(System::Drawing::Color col) {
	//		this->colorLabel->BackColor = col;			
	//	}

	//	// Constructor. The filename and index need to be defined
	//	signalFile(System::String^ filepath, bool fullPath, int pos, std::vector<double> nx, std::vector<double> ny) {
	//		Init();
	//		index = pos;
	//		path = String::Copy(filepath);
	//		file = path->Substring(path->LastIndexOf(L"\\") + 1);
	//		*x = nx;
	//		*y = ny;
 //
	//		bScalePlus	= false;
	//		bScaleMinus	= false;
	//		bBGPlus		= false;
	//		bBGMinus	= false;
	//		fileText(fullPath);
	//		this->Size = System::Drawing::Size(458, 32);
	//	}

	//	~signalFile() {
	//		if(x)
	//			delete x;
	//		if(y)
	//			delete y;
	//	}

	//};
}