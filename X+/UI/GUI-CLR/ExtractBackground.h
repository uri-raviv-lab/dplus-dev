#ifndef __EXTRACTBACKGROUND_H
#define __EXTRACTBACKGROUND_H

#pragma once
#include "DUMMY_HEADER_FILE.h"
#include "WGTControl.h"
#include "ResultsWindow.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace GUICLR {

	/// <summary>
	/// Summary for ExtractBackground
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ExtractBackground : public System::Windows::Forms::Form
	{
	public:
	private: System::Windows::Forms::CheckBox^  logScale;
			 System::String ^_dataFile;
	private: System::Windows::Forms::FolderBrowserDialog^  folderBrowserDialog1;
	private: System::Windows::Forms::SaveFileDialog^  saveFileDialog1;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
			 System::String ^_bgFile;

	public: 
		ExtractBackground(const wchar_t *data, const wchar_t *bg)
		{
			InitializeComponent();
			_dataFile = gcnew System::String(data);
			_bgFile = gcnew System::String(bg);
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ExtractBackground()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel1;
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel2;
	private: System::Windows::Forms::Button^  extractOneFile;
	private: System::Windows::Forms::Button^  batchButton;


	private: System::Windows::Forms::Button^  dataButton;
	private: System::Windows::Forms::Button^  backgroundButton;



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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ExtractBackground::typeid));
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->extractOneFile = (gcnew System::Windows::Forms::Button());
			this->batchButton = (gcnew System::Windows::Forms::Button());
			this->dataButton = (gcnew System::Windows::Forms::Button());
			this->backgroundButton = (gcnew System::Windows::Forms::Button());
			this->logScale = (gcnew System::Windows::Forms::CheckBox());
			this->folderBrowserDialog1 = (gcnew System::Windows::Forms::FolderBrowserDialog());
			this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->tableLayoutPanel1->SuspendLayout();
			this->tableLayoutPanel2->SuspendLayout();
			this->SuspendLayout();
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 1;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				50)));
			this->tableLayoutPanel1->Controls->Add(this->tableLayoutPanel2, 0, 1);
			this->tableLayoutPanel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel1->Location = System::Drawing::Point(0, 0);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 2;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 87.77429F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 12.22571F)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(595, 348);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->ColumnCount = 5;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				20.20374F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				21.22241F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				24.31395F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				18.89007F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				15.14946F)));
			this->tableLayoutPanel2->Controls->Add(this->extractOneFile, 4, 0);
			this->tableLayoutPanel2->Controls->Add(this->batchButton, 3, 0);
			this->tableLayoutPanel2->Controls->Add(this->dataButton, 1, 0);
			this->tableLayoutPanel2->Controls->Add(this->backgroundButton, 2, 0);
			this->tableLayoutPanel2->Controls->Add(this->logScale, 0, 0);
			this->tableLayoutPanel2->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel2->Location = System::Drawing::Point(3, 308);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 1;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(589, 37);
			this->tableLayoutPanel2->TabIndex = 0;
			// 
			// extractOneFile
			// 
			this->extractOneFile->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->extractOneFile->Location = System::Drawing::Point(511, 11);
			this->extractOneFile->Name = L"extractOneFile";
			this->extractOneFile->Size = System::Drawing::Size(75, 23);
			this->extractOneFile->TabIndex = 1;
			this->extractOneFile->Text = L"Extract File";
			this->extractOneFile->UseVisualStyleBackColor = true;
			this->extractOneFile->Click += gcnew System::EventHandler(this, &ExtractBackground::extractOneFile_Click);
			// 
			// batchButton
			// 
			this->batchButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->batchButton->Location = System::Drawing::Point(401, 11);
			this->batchButton->Name = L"batchButton";
			this->batchButton->Size = System::Drawing::Size(94, 23);
			this->batchButton->TabIndex = 2;
			this->batchButton->Text = L"Batch Extract...";
			this->batchButton->UseVisualStyleBackColor = true;
			this->batchButton->Click += gcnew System::EventHandler(this, &ExtractBackground::batchButton_Click);
			// 
			// dataButton
			// 
			this->dataButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->dataButton->Location = System::Drawing::Point(151, 11);
			this->dataButton->Name = L"dataButton";
			this->dataButton->Size = System::Drawing::Size(90, 23);
			this->dataButton->TabIndex = 3;
			this->dataButton->Text = L"Choose Data...";
			this->dataButton->UseVisualStyleBackColor = true;
			this->dataButton->Click += gcnew System::EventHandler(this, &ExtractBackground::dataButton_Click);
			// 
			// backgroundButton
			// 
			this->backgroundButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->backgroundButton->Location = System::Drawing::Point(261, 11);
			this->backgroundButton->Name = L"backgroundButton";
			this->backgroundButton->Size = System::Drawing::Size(123, 23);
			this->backgroundButton->TabIndex = 4;
			this->backgroundButton->Text = L"Choose Background...";
			this->backgroundButton->UseVisualStyleBackColor = true;
			this->backgroundButton->Click += gcnew System::EventHandler(this, &ExtractBackground::backgroundButton_Click);
			// 
			// logScale
			// 
			this->logScale->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->logScale->AutoSize = true;
			this->logScale->Location = System::Drawing::Point(3, 17);
			this->logScale->Name = L"logScale";
			this->logScale->Size = System::Drawing::Size(77, 17);
			this->logScale->TabIndex = 5;
			this->logScale->Text = L"Log. Scale";
			this->logScale->UseVisualStyleBackColor = true;
			this->logScale->CheckedChanged += gcnew System::EventHandler(this, &ExtractBackground::logScale_CheckedChanged);
			// 
			// saveFileDialog1
			// 
			this->saveFileDialog1->Filter = L"Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->Filter = L"Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
			// 
			// ExtractBackground
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(595, 348);
			this->Controls->Add(this->tableLayoutPanel1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->MinimumSize = System::Drawing::Size(511, 305);
			this->Name = L"ExtractBackground";
			this->Text = L"Extract Background from File";
			this->Load += gcnew System::EventHandler(this, &ExtractBackground::ExtractBackground_Load);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel2->ResumeLayout(false);
			this->tableLayoutPanel2->PerformLayout();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: GUICLR::WGTControl^  wgtGraph;
			 void OpenInitialGraph();
			 void ExtractBackground::chooseFile(int num);
		
	private: System::Void ExtractBackground_Load(System::Object^  sender, System::EventArgs^  e) {
				this->wgtGraph = (gcnew GUICLR::WGTControl());
				this->tableLayoutPanel1->Controls->Add(this->wgtGraph);
				// 
				// wgtGraph
				// 
				this->wgtGraph->Cursor = System::Windows::Forms::Cursors::Cross;
				this->wgtGraph->Dock = System::Windows::Forms::DockStyle::Fill;
				this->wgtGraph->Location = System::Drawing::Point(0, 0);
				this->wgtGraph->Name = L"wgtGraph";
				this->wgtGraph->Size = System::Drawing::Size(343, 349);
				this->wgtGraph->TabIndex = 0;

				// Opening an initial graph
				OpenInitialGraph();
			}
private: System::Void logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void extractOneFile_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void batchButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void backgroundButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void dataButton_Click(System::Object^  sender, System::EventArgs^  e);
};
}
#endif