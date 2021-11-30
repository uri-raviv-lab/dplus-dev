#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace GUICLR {

	/// <summary>
	/// Summary for DataManip
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class DataManip : public System::Windows::Forms::Form
	{
	protected:
	public:
		DataManip(void)
		{
			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~DataManip()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
	private: System::Windows::Forms::Button^  chooseFilesButton;

	private: System::Windows::Forms::Button^  doneButton;

	private: System::Windows::Forms::CheckBox^  angstromCheckbox;
	private: System::Windows::Forms::CheckBox^  factorCheckbox;
	private: System::Windows::Forms::CheckBox^  rmvHeadersCheckbox;
	private: System::Windows::Forms::TextBox^  factorTextBox;


	protected: 

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
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->chooseFilesButton = (gcnew System::Windows::Forms::Button());
			this->doneButton = (gcnew System::Windows::Forms::Button());
			this->angstromCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->factorCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->rmvHeadersCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->factorTextBox = (gcnew System::Windows::Forms::TextBox());
			this->SuspendLayout();
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->Multiselect = true;
			// 
			// chooseFilesButton
			// 
			this->chooseFilesButton->Location = System::Drawing::Point(34, 118);
			this->chooseFilesButton->Name = L"chooseFilesButton";
			this->chooseFilesButton->Size = System::Drawing::Size(92, 23);
			this->chooseFilesButton->TabIndex = 0;
			this->chooseFilesButton->Text = L"&Choose Files...";
			this->chooseFilesButton->UseVisualStyleBackColor = true;
			this->chooseFilesButton->Click += gcnew System::EventHandler(this, &DataManip::chooseFilesButton_Click);
			// 
			// doneButton
			// 
			this->doneButton->Location = System::Drawing::Point(43, 158);
			this->doneButton->Name = L"doneButton";
			this->doneButton->Size = System::Drawing::Size(75, 23);
			this->doneButton->TabIndex = 1;
			this->doneButton->Text = L"&Done";
			this->doneButton->UseVisualStyleBackColor = true;
			this->doneButton->Click += gcnew System::EventHandler(this, &DataManip::doneButton_Click);
			// 
			// angstromCheckbox
			// 
			this->angstromCheckbox->AutoSize = true;
			this->angstromCheckbox->Checked = true;
			this->angstromCheckbox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->angstromCheckbox->Location = System::Drawing::Point(39, 16);
			this->angstromCheckbox->Name = L"angstromCheckbox";
			this->angstromCheckbox->Size = System::Drawing::Size(83, 17);
			this->angstromCheckbox->TabIndex = 2;
			this->angstromCheckbox->Text = L"&A-1 --> nm-1";
			this->angstromCheckbox->UseVisualStyleBackColor = true;
			this->angstromCheckbox->CheckedChanged += gcnew System::EventHandler(this, &DataManip::Checkbox_CheckedChanged);
			// 
			// factorCheckbox
			// 
			this->factorCheckbox->AutoSize = true;
			this->factorCheckbox->Checked = true;
			this->factorCheckbox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->factorCheckbox->Location = System::Drawing::Point(54, 50);
			this->factorCheckbox->Name = L"factorCheckbox";
			this->factorCheckbox->Size = System::Drawing::Size(46, 17);
			this->factorCheckbox->TabIndex = 3;
			this->factorCheckbox->Text = L"&x 1e";
			this->factorCheckbox->UseVisualStyleBackColor = true;
			this->factorCheckbox->CheckedChanged += gcnew System::EventHandler(this, &DataManip::Checkbox_CheckedChanged);
			// 
			// rmvHeadersCheckbox
			// 
			this->rmvHeadersCheckbox->AutoSize = true;
			this->rmvHeadersCheckbox->Location = System::Drawing::Point(12, 84);
			this->rmvHeadersCheckbox->Name = L"rmvHeadersCheckbox";
			this->rmvHeadersCheckbox->Size = System::Drawing::Size(142, 17);
			this->rmvHeadersCheckbox->TabIndex = 3;
			this->rmvHeadersCheckbox->Text = L"&Remove Headers && Text";
			this->rmvHeadersCheckbox->UseVisualStyleBackColor = true;
			this->rmvHeadersCheckbox->CheckedChanged += gcnew System::EventHandler(this, &DataManip::Checkbox_CheckedChanged);
			// 
			// factorTextBox
			// 
			this->factorTextBox->Location = System::Drawing::Point(95, 49);
			this->factorTextBox->Name = L"factorTextBox";
			this->factorTextBox->Size = System::Drawing::Size(21, 20);
			this->factorTextBox->TabIndex = 4;
			this->factorTextBox->Text = L"5";
			this->factorTextBox->Leave += gcnew System::EventHandler(this, &DataManip::factorTextBox_Leave);
			// 
			// DataManip
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(161, 197);
			this->Controls->Add(this->factorTextBox);
			this->Controls->Add(this->rmvHeadersCheckbox);
			this->Controls->Add(this->factorCheckbox);
			this->Controls->Add(this->angstromCheckbox);
			this->Controls->Add(this->doneButton);
			this->Controls->Add(this->chooseFilesButton);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Name = L"DataManip";
			this->Text = L"File Manipulation";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
			
/**
 * Opens an open file dialog and accepts multiple files, converts from inverse angstrom
 * to inverse nm and multplies the intensity/sigma (if any) by a factor of 1e9 based 
 * on checkboxes
**/
private: System::Void chooseFilesButton_Click(System::Object^  sender, System::EventArgs^  e);

/**
 * Closes the popup window
**/
private: System::Void doneButton_Click(System::Object^  sender, System::EventArgs^  e);

/**
 * Enables/Disables the select files button
**/
private: System::Void Checkbox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);

private: System::Void factorTextBox_Leave(System::Object^  sender, System::EventArgs^  e);
};
}
