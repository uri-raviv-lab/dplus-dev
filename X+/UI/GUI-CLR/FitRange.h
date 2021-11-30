#pragma once

#include "ModelUI.h"
#include "clrfunctionality.h"
#include "GUIHelperClasses.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace GUICLR {

#ifndef max
	#define max(a,b)	(((a) > (b)) ? (a) : (b))
#endif

#ifndef min
	#define min(a,b)	(((a) < (b)) ? (a) : (b))
#endif

#ifndef LV_POSITIONS
#define LV_POSITIONS

#define LV_PAR_SUBITEMS       (7)
#define LV_NAME               (0)
#define LV_VALUE(n)           (2 * n + 1)
#define LV_MUTABLE(n)         (2 * n + 2)
#define LV_CONSMIN(n, nlp)    (2 * nlp + LV_PAR_SUBITEMS * n + 1)
#define LV_CONSMAX(n, nlp)    (2 * nlp + LV_PAR_SUBITEMS * n + 2)
#define LV_CONSIMIN(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 3)
#define LV_CONSIMAX(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 4)
#define LV_CONSLINK(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 5)
#define LV_CONS(n, nlp)       (2 * nlp + LV_PAR_SUBITEMS * n + 6)
#define LV_SIGMA(n, nlp)      (2 * nlp + LV_PAR_SUBITEMS * n + 7)
#endif

	/// <summary>
	/// Summary for FitRange
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class FitRange : public System::Windows::Forms::Form
	{
	public:
		ListView::ListViewItemCollection ^_lc; 

			 System::Collections::Generic::List<ConstraintGroupBox^>^ GroupBoxList;

	public:


			 ListViewItem ^_lvi;
			 //FFModel *_mod;
			 ModelUI *_mod;
			 int _layer, _nLayerParams;
	private: System::Windows::Forms::FlowLayoutPanel^  GroupBoxPanel;
	public: 
		FitRange(ListView::ListViewItemCollection ^s, int layer, int nlp, ModelUI *_model) {
			InitializeComponent();

			_layer = layer;
			_nLayerParams = nlp;
			_mod = _model;

			GroupBoxList = gcnew System::Collections::Generic::List<ConstraintGroupBox^>();
			for(int i = 0; i < _mod->GetNumLayerParams(); i++) {
				String ^lpName = stringToClr(_mod->GetLayerParamName(i));

				// Add GroupBoxes
				GroupBoxList->Add(gcnew ConstraintGroupBox(lpName, layer/*2 * i + 1*/));
				
				// GroupBox Events
				// Add event handlers for the textboxes
				for(int j = 0; j < GroupBoxList[i]->textBoxes->Count; j++) 
					GroupBoxList[i]->textBoxes[j]->Leave += gcnew System::EventHandler(this, &FitRange::Parameter_TextChanged);


				GroupBoxPanel->Controls->Add(GroupBoxList[i]);
			}
			GroupBoxPanel->Height = min(_mod->GetNumLayerParams() * 118, 650);
			this->Height = GroupBoxPanel->Height + 70;
			this->OK->Location = System::Drawing::Point(366, this->Height - this->OK->Height - 28);
			this->Cancel->Location = System::Drawing::Point(447, this->Height - this->Cancel->Height - 28);

			_lc = s;
			_lvi = s[layer];

			this->Text += " - " + _lvi->SubItems[0]->Text;

			for(int i = 0; i < nlp; i++) {
				// Disable areas that cannot have values
				GroupBoxList[i]->Enabled = _mod->IsParamApplicable(layer, i);
				// Fill in the min and max fields
				GroupBoxList[i]->minText->Text = _lvi->SubItems[LV_CONSMIN(i, nlp)]->Text;
				GroupBoxList[i]->maxText->Text = _lvi->SubItems[LV_CONSMAX(i, nlp)]->Text;

				// Parse the values correctly
				Parameter_TextChanged(GroupBoxList[i]->minText, gcnew EventArgs());
				Parameter_TextChanged(GroupBoxList[i]->maxText, gcnew EventArgs());

				// Set the use contraint checkbox
				GroupBoxList[i]->consCheckBox->Checked = 
					_lvi->SubItems[LV_CONS(i, nlp)]->Text->Equals("Y");

				// Fill in the drop down menus
				for(int j = 0; j < _lc->Count; j++) {
					if(j == layer) continue;
					for(int k = 0; k < 3; k++)
						GroupBoxList[i]->comboBoxes[k]->Items->Add(_lc[j]->SubItems[0]->Text);
				}
				// Initialize the comboBoxes selected indices
				for(int j = 0; j < 3; j++) {
					int ind = Int32::Parse(_lvi->SubItems[LV_CONSIMIN(i, nlp) + j]->Text);
					if(ind < layer)
						ind++;
					GroupBoxList[i]->comboBoxes[j]->SelectedIndex = 
						(ind >= GroupBoxList[i]->comboBoxes[j]->Items->Count) ? 0 : ind;
				}
			}	// end for(i)
		}	// end constructor

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~FitRange()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^  OK;
	private: System::Windows::Forms::Button^  Cancel;

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
			this->OK = (gcnew System::Windows::Forms::Button());
			this->Cancel = (gcnew System::Windows::Forms::Button());
			this->GroupBoxPanel = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->SuspendLayout();
			// 
			// OK
			// 
			this->OK->Location = System::Drawing::Point(382, 139);
			this->OK->Name = L"OK";
			this->OK->Size = System::Drawing::Size(75, 23);
			this->OK->TabIndex = 0;
			this->OK->Text = L"OK";
			this->OK->UseVisualStyleBackColor = true;
			this->OK->Click += gcnew System::EventHandler(this, &FitRange::OKButton_Click);
			// 
			// Cancel
			// 
			this->Cancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->Cancel->Location = System::Drawing::Point(463, 139);
			this->Cancel->Name = L"Cancel";
			this->Cancel->Size = System::Drawing::Size(75, 23);
			this->Cancel->TabIndex = 1;
			this->Cancel->Text = L"Cancel";
			this->Cancel->UseVisualStyleBackColor = true;
			// 
			// GroupBoxPanel
			// 
			this->GroupBoxPanel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->GroupBoxPanel->AutoScroll = true;
			this->GroupBoxPanel->Location = System::Drawing::Point(13, 13);
			this->GroupBoxPanel->Name = L"GroupBoxPanel";
			this->GroupBoxPanel->Size = System::Drawing::Size(535, 120);
			this->GroupBoxPanel->TabIndex = 12;
			// 
			// FitRange
			// 
			this->AcceptButton = this->OK;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->Cancel;
			this->ClientSize = System::Drawing::Size(559, 168);
			this->Controls->Add(this->GroupBoxPanel);
			this->Controls->Add(this->Cancel);
			this->Controls->Add(this->OK);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"FitRange";
			this->Text = L"Choose Constraints";
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void OKButton_Click(System::Object^  sender, System::EventArgs^  e) {
				int index;
				int nlp = _mod->GetNumLayerParams();

				// Set values in listViewFF
				for(int i = 0; i < nlp; i++) {
					for(int j = 0;  j < GroupBoxList[i]->comboBoxes->Count; j++) {
						index = GroupBoxList[i]->comboBoxes[j]->SelectedIndex - 1;
						if(index + 1 > _layer)
							index++;
						// Don't allow linking to items that have no value
						if(index > -1 && !(_mod->IsParamApplicable(index, i)))
							index = -1;
						_lvi->SubItems[LV_CONSIMIN(i, nlp) + j]->Text = 
							index.ToString();
					}
					for(int j = 0;  j < GroupBoxList[i]->textBoxes->Count; j++) {
						_lvi->SubItems[LV_CONSMIN(i, nlp) + j]->Text = 
							GroupBoxList[i]->textBoxes[j]->Text;
					}
					_lvi->SubItems[LV_CONS(i, nlp)]->Text = 
						GroupBoxList[i]->consCheckBox->Checked ? "Y" : "N";
				}

				this->Close();
			 }	// end OKButton_Click

			 void Parameter_TextChanged(System::Object^  sender, System::EventArgs^  e) {
				double res;
				std::string str;
				char f[128] = {0};
				System::String ^mstr = ((TextBox ^)(sender))->Text;
		
				clrToString(mstr, str);

				mstr = mstr->ToLower();
				
				// Infinity
				if(mstr->Equals("inf") || mstr->Equals("infinity") ||
				   mstr->StartsWith("1.#INF")) {
					((TextBox ^)(sender))->Text = "inf";	
					return;
				} else if(mstr->Equals("-inf") || mstr->Equals("-infinity") ||
						  mstr->StartsWith("-1.#INF")) {
					// Negative infinity
					((TextBox ^)(sender))->Text = "-inf";	
					return;
				}

				res = fabs(strtod(str.c_str(), NULL));
		
				sprintf(f, "%.6f", res);		
				((TextBox ^)(sender))->Text = gcnew String(f);
			 }

};
}
