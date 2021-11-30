#pragma once

#include "WGTControl.h"
#include "clrfunctionality.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace GUICLR {

	/// <summary>
	/// Summary for ExportGraph
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ExportGraph : public System::Windows::Forms::Form
	{
	private: struct graphLine *_graphs;
	private: System::Windows::Forms::GroupBox^  groupBox3;
	private: System::Windows::Forms::TextBox^  textBox3;
	private: System::Windows::Forms::GroupBox^  groupBox2;
	private: System::Windows::Forms::TextBox^  textBox2;
			 int _cnt;
	public:
		ExportGraph(struct graphLine *graphs, int number)
		{
			_graphs = graphs;
			_cnt = number;

			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ExportGraph()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	private: System::Windows::Forms::NumericUpDown^  numericUpDown1;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::CheckBox^  xTicks;
	private: System::Windows::Forms::CheckBox^  legend;


	private: System::Windows::Forms::CheckBox^  grid;

	private: System::Windows::Forms::CheckBox^  yTicks;
	private: System::Windows::Forms::CheckBox^  logScale;
	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Button^  button2;


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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ExportGraph::typeid));
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->logScale = (gcnew System::Windows::Forms::CheckBox());
			this->legend = (gcnew System::Windows::Forms::CheckBox());
			this->grid = (gcnew System::Windows::Forms::CheckBox());
			this->yTicks = (gcnew System::Windows::Forms::CheckBox());
			this->xTicks = (gcnew System::Windows::Forms::CheckBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->groupBox3->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->numericUpDown1))->BeginInit();
			this->SuspendLayout();
			// 
			// splitContainer1
			// 
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->Location = System::Drawing::Point(0, 0);
			this->splitContainer1->Name = L"splitContainer1";
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->Controls->Add(this->groupBox3);
			this->splitContainer1->Panel1->Controls->Add(this->groupBox2);
			this->splitContainer1->Panel1->Controls->Add(this->button2);
			this->splitContainer1->Panel1->Controls->Add(this->button1);
			this->splitContainer1->Panel1->Controls->Add(this->logScale);
			this->splitContainer1->Panel1->Controls->Add(this->legend);
			this->splitContainer1->Panel1->Controls->Add(this->grid);
			this->splitContainer1->Panel1->Controls->Add(this->yTicks);
			this->splitContainer1->Panel1->Controls->Add(this->xTicks);
			this->splitContainer1->Panel1->Controls->Add(this->label1);
			this->splitContainer1->Panel1->Controls->Add(this->groupBox1);
			this->splitContainer1->Panel1->Controls->Add(this->numericUpDown1);
			this->splitContainer1->Size = System::Drawing::Size(524, 431);
			this->splitContainer1->SplitterDistance = 151;
			this->splitContainer1->TabIndex = 0;
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->textBox3);
			this->groupBox3->Location = System::Drawing::Point(9, 298);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(136, 50);
			this->groupBox3->TabIndex = 12;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"Y Axis Label:";
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(6, 19);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(124, 20);
			this->textBox3->TabIndex = 1;
			this->textBox3->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &ExportGraph::textBox3_KeyUp);
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->textBox2);
			this->groupBox2->Location = System::Drawing::Point(9, 230);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(136, 50);
			this->groupBox2->TabIndex = 11;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"X Axis Label:";
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(6, 19);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(124, 20);
			this->textBox2->TabIndex = 1;
			this->textBox2->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &ExportGraph::textBox2_KeyUp);
			// 
			// button2
			// 
			this->button2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->button2->Location = System::Drawing::Point(4, 396);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(75, 23);
			this->button2->TabIndex = 10;
			this->button2->Text = L"Export...";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &ExportGraph::button2_Click);
			// 
			// button1
			// 
			this->button1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->button1->Location = System::Drawing::Point(85, 396);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(60, 23);
			this->button1->TabIndex = 9;
			this->button1->Text = L"Cancel";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &ExportGraph::button1_Click);
			// 
			// logScale
			// 
			this->logScale->AutoSize = true;
			this->logScale->Location = System::Drawing::Point(9, 198);
			this->logScale->Name = L"logScale";
			this->logScale->Size = System::Drawing::Size(77, 17);
			this->logScale->TabIndex = 8;
			this->logScale->Text = L"Log. Scale";
			this->logScale->UseVisualStyleBackColor = true;
			this->logScale->CheckedChanged += gcnew System::EventHandler(this, &ExportGraph::logScale_CheckedChanged);
			// 
			// legend
			// 
			this->legend->AutoSize = true;
			this->legend->Checked = true;
			this->legend->CheckState = System::Windows::Forms::CheckState::Checked;
			this->legend->Location = System::Drawing::Point(9, 174);
			this->legend->Name = L"legend";
			this->legend->Size = System::Drawing::Size(62, 17);
			this->legend->TabIndex = 7;
			this->legend->Text = L"Legend";
			this->legend->UseVisualStyleBackColor = true;
			this->legend->CheckedChanged += gcnew System::EventHandler(this, &ExportGraph::legend_CheckedChanged);
			// 
			// grid
			// 
			this->grid->AutoSize = true;
			this->grid->Checked = true;
			this->grid->CheckState = System::Windows::Forms::CheckState::Checked;
			this->grid->Location = System::Drawing::Point(9, 150);
			this->grid->Name = L"grid";
			this->grid->Size = System::Drawing::Size(45, 17);
			this->grid->TabIndex = 6;
			this->grid->Text = L"Grid";
			this->grid->UseVisualStyleBackColor = true;
			this->grid->CheckedChanged += gcnew System::EventHandler(this, &ExportGraph::grid_CheckedChanged);
			// 
			// yTicks
			// 
			this->yTicks->AutoSize = true;
			this->yTicks->Checked = true;
			this->yTicks->CheckState = System::Windows::Forms::CheckState::Checked;
			this->yTicks->Location = System::Drawing::Point(9, 126);
			this->yTicks->Name = L"yTicks";
			this->yTicks->Size = System::Drawing::Size(62, 17);
			this->yTicks->TabIndex = 5;
			this->yTicks->Text = L"Y Ticks";
			this->yTicks->UseVisualStyleBackColor = true;
			this->yTicks->CheckedChanged += gcnew System::EventHandler(this, &ExportGraph::yTicks_CheckedChanged);
			// 
			// xTicks
			// 
			this->xTicks->AutoSize = true;
			this->xTicks->Checked = true;
			this->xTicks->CheckState = System::Windows::Forms::CheckState::Checked;
			this->xTicks->Location = System::Drawing::Point(9, 103);
			this->xTicks->Name = L"xTicks";
			this->xTicks->Size = System::Drawing::Size(62, 17);
			this->xTicks->TabIndex = 4;
			this->xTicks->Text = L"X Ticks";
			this->xTicks->UseVisualStyleBackColor = true;
			this->xTicks->CheckedChanged += gcnew System::EventHandler(this, &ExportGraph::xTicks_CheckedChanged);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(9, 65);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(82, 13);
			this->label1->TabIndex = 3;
			this->label1->Text = L"Grid Resolution:";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->textBox1);
			this->groupBox1->Location = System::Drawing::Point(9, 12);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(136, 50);
			this->groupBox1->TabIndex = 2;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Graph Title:";
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(6, 19);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(124, 20);
			this->textBox1->TabIndex = 1;
			this->textBox1->KeyUp += gcnew System::Windows::Forms::KeyEventHandler(this, &ExportGraph::textBox1_KeyUp);
			// 
			// numericUpDown1
			// 
			this->numericUpDown1->Location = System::Drawing::Point(97, 63);
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(48, 20);
			this->numericUpDown1->TabIndex = 0;
			this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) {5, 0, 0, 0});
			this->numericUpDown1->ValueChanged += gcnew System::EventHandler(this, &ExportGraph::numericUpDown1_ValueChanged);
			// 
			// ExportGraph
			// 
			this->AcceptButton = this->button2;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(524, 431);
			this->Controls->Add(this->splitContainer1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"ExportGraph";
			this->Text = L"Export Graph";
			this->Load += gcnew System::EventHandler(this, &ExportGraph::ExportGraph_Load);
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel1->PerformLayout();
			this->splitContainer1->ResumeLayout(false);
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->numericUpDown1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: GUICLR::WGTControl^  wgtGraph;
			 void OpenInitialGraph();
		
	private: System::Void ExportGraph_Load(System::Object^  sender, System::EventArgs^  e) {
				this->wgtGraph = (gcnew GUICLR::WGTControl());
				this->splitContainer1->Panel2->Controls->Add(this->wgtGraph);
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

private: System::Void textBox1_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			 std::string text;
			 if(!wgtGraph->graph)
				 return;
			
			 clrToString(textBox1->Text, text);

			 wgtGraph->graph->SetTitle(text.c_str());
			 wgtGraph->Invalidate();
		 }
private: System::Void numericUpDown1_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;

			 wgtGraph->graph->SetTickRes(
				 System::Decimal::ToInt32(numericUpDown1->Value));
			 wgtGraph->Invalidate();
		 }
private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
			 this->Close();
		 }
private: System::Void xTicks_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
			
			 wgtGraph->graph->ToggleXTicks();
			 wgtGraph->graph->FitToAllGraphs();
			 wgtGraph->Invalidate();
		 }
private: System::Void yTicks_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
			 
			 wgtGraph->graph->ToggleYTicks();
			 wgtGraph->graph->FitToAllGraphs();
			 wgtGraph->Invalidate();
		 }
private: System::Void grid_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if(!wgtGraph->graph)
				 return;

			wgtGraph->graph->ToggleGrid();
			wgtGraph->Invalidate();
		 }
private: System::Void legend_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if(!wgtGraph->graph)
				 return;

			wgtGraph->graph->bLegend = legend->Checked;
			wgtGraph->Invalidate();
		 }
private: System::Void logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
				
			 wgtGraph->graph->SetScale(0, logScale->Checked ? 
					SCALE_LOG : SCALE_LIN);
			 wgtGraph->Invalidate();
		 }
private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
			 wgtGraph->Invalidate();
			 wgtGraph->graph->ExportImage(wgtGraph->Handle);
		 }
private: System::Void textBox2_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
				
			 wgtGraph->graph->SetXLabel(clrToString(textBox2->Text));
			 wgtGraph->Invalidate();
		 }
private: System::Void textBox3_KeyUp(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			 if(!wgtGraph->graph)
				 return;
				
			 wgtGraph->graph->SetYLabel(clrToString(textBox3->Text));
			 wgtGraph->Invalidate();
		 }
};
}
