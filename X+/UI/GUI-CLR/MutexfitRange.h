#pragma once

#include "clrfunctionality.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace xraytools {

	/// <summary>
	/// Summary for MutexfitRange
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class MutexfitRange : public System::Windows::Forms::Form
	{
	public:
		ListView::ListViewItemCollection ^_lc; 
		ListViewItem ^_lvi;
		int _prevItem;
		MutexfitRange(ListView::ListViewItemCollection ^s)
		{
			InitializeComponent();
			_prevItem = -1;
			_lc = s;
			_lvi = s[0];
			exmin->Text = Double::Parse(_lvi->SubItems[2]->Text).ToString("0.000000");
			exmax->Text = Double::Parse(_lvi->SubItems[3]->Text).ToString("0.000000");
			
			for (int i=0; i<_lc->Count ;i++) {
				if ( s[i]->SubItems[1]->Text->Equals("0") ) continue;
				params->Items->Add(_lc[i]->Text);   
			}
			params->SelectedIndex=0;
			_prevItem=0;
		}
/*		MutexfitRange(void)
		{
			InitializeComponent();
			exmin->Text = "0.000000";
			exmax->Text = "0.000000";
		}
*/
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MutexfitRange()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^  ok;

	private: System::Windows::Forms::Label^  minim;
	private: System::Windows::Forms::Label^  maxim;
	private: System::Windows::Forms::TextBox^  exmax;



	private: System::Windows::Forms::TextBox^  exmin;
	private: System::Windows::Forms::ComboBox^  params;
	private: System::Windows::Forms::Label^  Parameter;

	protected: 

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
			this->ok = (gcnew System::Windows::Forms::Button());
			this->minim = (gcnew System::Windows::Forms::Label());
			this->maxim = (gcnew System::Windows::Forms::Label());
			this->exmax = (gcnew System::Windows::Forms::TextBox());
			this->exmin = (gcnew System::Windows::Forms::TextBox());
			this->params = (gcnew System::Windows::Forms::ComboBox());
			this->Parameter = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// ok
			// 
			this->ok->Location = System::Drawing::Point(60, 117);
			this->ok->Name = L"ok";
			this->ok->Size = System::Drawing::Size(78, 23);
			this->ok->TabIndex = 0;
			this->ok->Text = L"OK";
			this->ok->UseVisualStyleBackColor = true;
			this->ok->Click += gcnew System::EventHandler(this, &MutexfitRange::ok_Click);
			// 
			// minim
			// 
			this->minim->AutoSize = true;
			this->minim->Location = System::Drawing::Point(17, 46);
			this->minim->Name = L"minim";
			this->minim->Size = System::Drawing::Size(51, 13);
			this->minim->TabIndex = 5;
			this->minim->Text = L"Minimum:";
			// 
			// maxim
			// 
			this->maxim->AutoSize = true;
			this->maxim->Location = System::Drawing::Point(17, 85);
			this->maxim->Name = L"maxim";
			this->maxim->Size = System::Drawing::Size(54, 13);
			this->maxim->TabIndex = 6;
			this->maxim->Text = L"Maximum:";
			// 
			// exmax
			// 
			this->exmax->Location = System::Drawing::Point(76, 85);
			this->exmax->Name = L"exmax";
			this->exmax->Size = System::Drawing::Size(106, 20);
			this->exmax->TabIndex = 3;
			this->exmax->Text = L"0.000000";
			this->exmax->Leave += gcnew System::EventHandler(this, &MutexfitRange::double_TextChanged);
			// 
			// exmin
			// 
			this->exmin->Location = System::Drawing::Point(76, 46);
			this->exmin->Name = L"exmin";
			this->exmin->Size = System::Drawing::Size(106, 20);
			this->exmin->TabIndex = 4;
			this->exmin->Text = L"0.000000";
			this->exmin->Leave += gcnew System::EventHandler(this, &MutexfitRange::double_TextChanged);
			// 
			// params
			// 
			this->params->AllowDrop = true;
			this->params->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->params->FormattingEnabled = true;
			this->params->Location = System::Drawing::Point(76, 7);
			this->params->Name = L"params";
			this->params->Size = System::Drawing::Size(106, 21);
			this->params->TabIndex = 7;
			this->params->SelectedIndexChanged += gcnew System::EventHandler(this, &MutexfitRange::params_SelectedIndexChanged);
			// 
			// Parameter
			// 
			this->Parameter->AutoSize = true;
			this->Parameter->Location = System::Drawing::Point(17, 7);
			this->Parameter->Name = L"Parameter";
			this->Parameter->Size = System::Drawing::Size(58, 13);
			this->Parameter->TabIndex = 5;
			this->Parameter->Text = L"Parameter:";
			// 
			// MutexfitRange
			// 
			this->AcceptButton = this->ok;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(199, 152);
			this->Controls->Add(this->params);
			this->Controls->Add(this->Parameter);
			this->Controls->Add(this->minim);
			this->Controls->Add(this->maxim);
			this->Controls->Add(this->exmax);
			this->Controls->Add(this->exmin);
			this->Controls->Add(this->ok);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"MutexfitRange";
			this->Text = L"Choose Constraints";
			this->Load += gcnew System::EventHandler(this, &MutexfitRange::MutexfitRange_Load_1);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void MutexfitRange_Load_1(System::Object^  sender, System::EventArgs^  e) {

			 }
private: System::Void ok_Click(System::Object^  sender, System::EventArgs^  e) {
			 _lc[trueIndex(params->SelectedIndex)]->SubItems[2]->Text= Double::Parse(exmin->Text).ToString("0.000000");
			 _lc[trueIndex(params->SelectedIndex)]->SubItems[3]->Text= Double::Parse(exmax->Text).ToString("0.000000");
			 this->Close();
		 }

		 int trueIndex(int num) {
			int j = -1;
			for (int i=0; i<_lc->Count ;i++) {
				if ( !_lc[i]->SubItems[1]->Text->Equals("0") ) j++;
				if(num == j) return i;
			}
			 return 0;
		 }



		 void double_TextChanged(System::Object^  sender, System::EventArgs^  e) {
				double res;
				std::string str;
				char f[128] = {0};
		
				clrToString(((TextBox ^)(sender))->Text, str);

				res = (strtod(str.c_str(), NULL));
		
				sprintf(f, "%.6f", res);		
				((TextBox ^)(sender))->Text = gcnew String(f);
			 }

private: System::Void params_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(_prevItem == -1)
				 return;

			_lvi = _lc[trueIndex(_prevItem)];
			_lvi->SubItems[2]->Text = Double::Parse(exmin->Text).ToString("0.000000");
			_lvi->SubItems[3]->Text = Double::Parse(exmax->Text).ToString("0.000000");

			_lvi = _lc[trueIndex(params->SelectedIndex)];
			exmin->Text = Double::Parse(_lvi->SubItems[2]->Text).ToString("0.000000");
			exmax->Text = Double::Parse(_lvi->SubItems[3]->Text).ToString("0.000000");
			_prevItem = params->SelectedIndex;
		 }
};
}
