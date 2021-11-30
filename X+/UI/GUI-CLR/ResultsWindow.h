#ifndef __RESULTSWINDOW_H
#define __RESULTSWINDOW_H
#pragma once
#include <vector>
#include "WGTControl.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace GUICLR {

	/// <summary>
	/// Summary for ResultsWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ResultsWindow : public System::Windows::Forms::Form
	{
	private: struct graphLine *_graphs;
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel2;
	private: System::Windows::Forms::Button^  exportGraph;
	private: System::Windows::Forms::CheckBox^  logScale;
	private: System::Windows::Forms::Label^  subtitle;
			 int _cnt;
	public:
		ResultsWindow(struct graphLine *graphs, int number)
		{
			_graphs = graphs;
			_cnt = number;
			InitializeComponent();
		}

		ResultsWindow(struct graphLine *graphs, int number, String ^subt)
		{
			_graphs = graphs;
			_cnt = number;
			InitializeComponent();
			subtitle->Text = subt;
			subtitle->Visible = true;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ResultsWindow()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel1;
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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ResultsWindow::typeid));
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->logScale = (gcnew System::Windows::Forms::CheckBox());
			this->exportGraph = (gcnew System::Windows::Forms::Button());
			this->subtitle = (gcnew System::Windows::Forms::Label());
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
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 89.27335F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 10.72664F)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(563, 351);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->ColumnCount = 3;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				33.58974F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				66.41026F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute, 
				170)));
			this->tableLayoutPanel2->Controls->Add(this->logScale, 0, 0);
			this->tableLayoutPanel2->Controls->Add(this->exportGraph, 2, 0);
			this->tableLayoutPanel2->Controls->Add(this->subtitle, 1, 0);
			this->tableLayoutPanel2->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel2->Location = System::Drawing::Point(3, 316);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 1;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(557, 32);
			this->tableLayoutPanel2->TabIndex = 0;
			// 
			// logScale
			// 
			this->logScale->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->logScale->AutoSize = true;
			this->logScale->Location = System::Drawing::Point(3, 12);
			this->logScale->Name = L"logScale";
			this->logScale->Size = System::Drawing::Size(77, 17);
			this->logScale->TabIndex = 2;
			this->logScale->Text = L"Log. Scale";
			this->logScale->UseVisualStyleBackColor = true;
			this->logScale->CheckedChanged += gcnew System::EventHandler(this, &ResultsWindow::logScale_CheckedChanged);
			// 
			// exportGraph
			// 
			this->exportGraph->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->exportGraph->Location = System::Drawing::Point(466, 6);
			this->exportGraph->Name = L"exportGraph";
			this->exportGraph->Size = System::Drawing::Size(88, 23);
			this->exportGraph->TabIndex = 1;
			this->exportGraph->Text = L"Export Graph...";
			this->exportGraph->UseVisualStyleBackColor = true;
			this->exportGraph->Click += gcnew System::EventHandler(this, &ResultsWindow::exportGraph_Click);
			// 
			// subtitle
			// 
			this->subtitle->Anchor = System::Windows::Forms::AnchorStyles::Top;
			this->subtitle->AutoSize = true;
			this->subtitle->Location = System::Drawing::Point(236, 0);
			this->subtitle->Name = L"subtitle";
			this->subtitle->Size = System::Drawing::Size(42, 13);
			this->subtitle->TabIndex = 3;
			this->subtitle->Text = L"Subtitle";
			this->subtitle->TextAlign = System::Drawing::ContentAlignment::TopCenter;
			this->subtitle->Visible = false;
			// 
			// ResultsWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(563, 351);
			this->Controls->Add(this->tableLayoutPanel1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"ResultsWindow";
			this->Text = L"Results Window";
			this->Load += gcnew System::EventHandler(this, &ResultsWindow::ResultsWindow_Load);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel2->ResumeLayout(false);
			this->tableLayoutPanel2->PerformLayout();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: GUICLR::WGTControl^  wgtGraph;
			 void OpenInitialGraph();
		
	private: System::Void ResultsWindow_Load(System::Object^  sender, System::EventArgs^  e) {
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
	private: System::Void exportGraph_Click(System::Object^  sender, System::EventArgs^  e);
};
}
#endif