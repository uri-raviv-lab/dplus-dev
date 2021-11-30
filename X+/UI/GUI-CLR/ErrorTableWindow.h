#pragma once
//#include "calculation_external.h"
#include "FrontendExported.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace GUICLR {

	/// <summary>
	/// Summary for ErrorTableWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ErrorTableWindow : public System::Windows::Forms::Form
	{
	public:
		ErrorTableWindow(System::Collections::Generic::List<ListView^>^ LV, std::vector <std::vector <double>>& errorsVector, System::String ^dataFile)
		{
			InitializeComponent();
			_dataFile = dataFile;;
			LVs = LV;
			errorsVectors = &errorsVector;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ErrorTableWindow()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::ListView^  FFLV;
	private: System::Windows::Forms::ColumnHeader^  FFName;
	private: System::Windows::Forms::ColumnHeader^  Value;
	private: System::Windows::Forms::ColumnHeader^  Error;
	private: System::Windows::Forms::ListView^  SFLV;
	private: System::Windows::Forms::ColumnHeader^  columnHeader1;
	private: System::Windows::Forms::ColumnHeader^  columnHeader2;
	private: System::Windows::Forms::ColumnHeader^  columnHeader3;
	private: System::Windows::Forms::ListView^  PhLV;
	private: System::Windows::Forms::ColumnHeader^  columnHeader4;
	private: System::Windows::Forms::ColumnHeader^  columnHeader5;
	private: System::Windows::Forms::ColumnHeader^  columnHeader6;
	private: System::Windows::Forms::ListView^  BGLV;
	private: System::Windows::Forms::ColumnHeader^  columnHeader7;
	private: System::Windows::Forms::ColumnHeader^  columnHeader8;
	private: System::Windows::Forms::ColumnHeader^  columnHeader9;
	private: System::Windows::Forms::Button^  ExportButton;
	private: System::Windows::Forms::Button^  CloseButton;
	private: System::Windows::Forms::SaveFileDialog^  sfd;
	protected: 
	System::Collections::Generic::List<ListView^>^ LVs;
	std::vector <std::vector <double>>*  errorsVectors;
	System::String ^_dataFile;




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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ErrorTableWindow::typeid));
			this->FFLV = (gcnew System::Windows::Forms::ListView());
			this->FFName = (gcnew System::Windows::Forms::ColumnHeader());
			this->Value = (gcnew System::Windows::Forms::ColumnHeader());
			this->Error = (gcnew System::Windows::Forms::ColumnHeader());
			this->SFLV = (gcnew System::Windows::Forms::ListView());
			this->columnHeader1 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader2 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader3 = (gcnew System::Windows::Forms::ColumnHeader());
			this->PhLV = (gcnew System::Windows::Forms::ListView());
			this->columnHeader4 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader5 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader6 = (gcnew System::Windows::Forms::ColumnHeader());
			this->BGLV = (gcnew System::Windows::Forms::ListView());
			this->columnHeader7 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader8 = (gcnew System::Windows::Forms::ColumnHeader());
			this->columnHeader9 = (gcnew System::Windows::Forms::ColumnHeader());
			this->ExportButton = (gcnew System::Windows::Forms::Button());
			this->CloseButton = (gcnew System::Windows::Forms::Button());
			this->sfd = (gcnew System::Windows::Forms::SaveFileDialog());
			this->SuspendLayout();
			// 
			// FFLV
			// 
			this->FFLV->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(3) {this->FFName, this->Value, this->Error});
			this->FFLV->FullRowSelect = true;
			this->FFLV->GridLines = true;
			this->FFLV->Location = System::Drawing::Point(12, 12);
			this->FFLV->Name = L"FFLV";
			this->FFLV->Size = System::Drawing::Size(292, 136);
			this->FFLV->TabIndex = 0;
			this->FFLV->UseCompatibleStateImageBehavior = false;
			this->FFLV->View = System::Windows::Forms::View::Details;
			// 
			// FFName
			// 
			this->FFName->Text = L"FF: Name";
			// 
			// Value
			// 
			this->Value->Text = L"Value";
			// 
			// Error
			// 
			this->Error->Text = L"Error";
			// 
			// SFLV
			// 
			this->SFLV->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(3) {this->columnHeader1, this->columnHeader2, 
				this->columnHeader3});
			this->SFLV->FullRowSelect = true;
			this->SFLV->GridLines = true;
			this->SFLV->Location = System::Drawing::Point(12, 169);
			this->SFLV->Name = L"SFLV";
			this->SFLV->Size = System::Drawing::Size(292, 136);
			this->SFLV->TabIndex = 1;
			this->SFLV->UseCompatibleStateImageBehavior = false;
			this->SFLV->View = System::Windows::Forms::View::Details;
			// 
			// columnHeader1
			// 
			this->columnHeader1->Text = L"SF: Name";
			// 
			// columnHeader2
			// 
			this->columnHeader2->Text = L"Value";
			// 
			// columnHeader3
			// 
			this->columnHeader3->Text = L"Error";
			// 
			// PhLV
			// 
			this->PhLV->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(3) {this->columnHeader4, this->columnHeader5, 
				this->columnHeader6});
			this->PhLV->FullRowSelect = true;
			this->PhLV->GridLines = true;
			this->PhLV->Location = System::Drawing::Point(337, 169);
			this->PhLV->Name = L"PhLV";
			this->PhLV->Size = System::Drawing::Size(292, 136);
			this->PhLV->TabIndex = 2;
			this->PhLV->UseCompatibleStateImageBehavior = false;
			this->PhLV->View = System::Windows::Forms::View::Details;
			// 
			// columnHeader4
			// 
			this->columnHeader4->Text = L"Phases: Name";
			// 
			// columnHeader5
			// 
			this->columnHeader5->Text = L"Value";
			// 
			// columnHeader6
			// 
			this->columnHeader6->Text = L"Error";
			// 
			// BGLV
			// 
			this->BGLV->Columns->AddRange(gcnew cli::array< System::Windows::Forms::ColumnHeader^  >(3) {this->columnHeader7, this->columnHeader8, 
				this->columnHeader9});
			this->BGLV->FullRowSelect = true;
			this->BGLV->GridLines = true;
			this->BGLV->Location = System::Drawing::Point(12, 331);
			this->BGLV->Name = L"BGLV";
			this->BGLV->Size = System::Drawing::Size(292, 136);
			this->BGLV->TabIndex = 3;
			this->BGLV->UseCompatibleStateImageBehavior = false;
			this->BGLV->View = System::Windows::Forms::View::Details;
			// 
			// columnHeader7
			// 
			this->columnHeader7->Text = L"BG: Name";
			// 
			// columnHeader8
			// 
			this->columnHeader8->Text = L"Value";
			// 
			// columnHeader9
			// 
			this->columnHeader9->Text = L"Error";
			// 
			// ExportButton
			// 
			this->ExportButton->Location = System::Drawing::Point(331, 496);
			this->ExportButton->Name = L"ExportButton";
			this->ExportButton->Size = System::Drawing::Size(70, 23);
			this->ExportButton->TabIndex = 4;
			this->ExportButton->Text = L"Export...";
			this->ExportButton->UseVisualStyleBackColor = true;
			this->ExportButton->Click += gcnew System::EventHandler(this, &ErrorTableWindow::ExportButton_Click);
			// 
			// CloseButton
			// 
			this->CloseButton->Location = System::Drawing::Point(487, 496);
			this->CloseButton->Name = L"CloseButton";
			this->CloseButton->Size = System::Drawing::Size(55, 23);
			this->CloseButton->TabIndex = 4;
			this->CloseButton->Text = L"Close";
			this->CloseButton->UseVisualStyleBackColor = true;
			this->CloseButton->Click += gcnew System::EventHandler(this, &ErrorTableWindow::CloseButton_Click);
			// 
			// sfd
			// 
			this->sfd->Filter = L"Tab Separated Values (*.tsv)|*.tsv";
			// 
			// ErrorTableWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(691, 595);
			this->Controls->Add(this->CloseButton);
			this->Controls->Add(this->ExportButton);
			this->Controls->Add(this->BGLV);
			this->Controls->Add(this->PhLV);
			this->Controls->Add(this->SFLV);
			this->Controls->Add(this->FFLV);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"ErrorTableWindow";
			this->Text = L"Parameter Report";
			this->Load += gcnew System::EventHandler(this, &ErrorTableWindow::ErrorTableWindow_Load);
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void ErrorTableWindow_Load(System::Object^  sender, System::EventArgs^  e);
private: System::Void CloseButton_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->Close();
		 }
private: System::Void ExportButton_Click(System::Object^  sender, System::EventArgs^  e);

};
}
