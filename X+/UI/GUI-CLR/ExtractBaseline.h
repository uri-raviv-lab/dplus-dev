#pragma once

#include <map>

#include "WGTControl.h"

//#include "clrfunctionality.h"
//#include "calculation_external.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

struct graphMap {
	std::map<double, double> m;
};

namespace GUICLR {

	/// <summary>
	/// Summary for ExtractBaseline
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ExtractBaseline : public System::Windows::Forms::Form
	{
	private: System::String ^_dataFile, ^_targetFile, ^_workspace;
			 struct graphMap *_map;
			 bool _bCrop;
			 bool _bAuto;
			 bool _shiftpressed;
			 bool _angstrom;
			 int _firstCropX;
			 int _oldleft,_oldright;

	public:
			 int _curleft, _curright;
			 bool _bHasBaseline;
			 int _Xdown;

			 			 
	private: System::Windows::Forms::Button^  Crop;
	private: System::Windows::Forms::Panel^  panel1;
	private: System::Windows::Forms::Button^  cancelcrop;

	public:
		static bool bUsingOld = false;
		static void DontAsk() { bUsingOld = true; }

		ExtractBaseline(const wchar_t *data, const wchar_t *target, bool angstrom)
		{
			_dataFile = gcnew System::String(data);
			_targetFile = gcnew System::String(target);
			_workspace = CLRDirectory(gcnew System::String(data));
			_bCrop=false;
			_bAuto = false;
			_bHasBaseline = false; 
			_angstrom = angstrom;
			_firstCropX=-1;

			if(CheckSizeOfFile(target) > 0) {
				if(bUsingOld || MessageBox::Show("Previous baseline information has been found, "
								 "would you really like to use it?", "Baseline", 
								 Windows::Forms::MessageBoxButtons::YesNo, 
								 Windows::Forms::MessageBoxIcon::Question)
								 == Windows::Forms::DialogResult::Yes)
					bUsingOld = true;
			}

			_map = new struct graphMap;

			InitializeComponent();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ExtractBaseline()
		{
			if(_map)
				delete _map;
			if (components)
			{
				delete components;
			}
		}

private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel1;
	protected: 
	private: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanel2;
	private: System::Windows::Forms::Button^  resetZoom;

	private: System::Windows::Forms::Button^  saveAs;


	private: System::Windows::Forms::CheckBox^  logScale;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Button^  removeAll;
	private: System::Windows::Forms::Button^  AutoBaseline;



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
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanel2 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->AutoBaseline = (gcnew System::Windows::Forms::Button());
			this->resetZoom = (gcnew System::Windows::Forms::Button());
			this->saveAs = (gcnew System::Windows::Forms::Button());
			this->logScale = (gcnew System::Windows::Forms::CheckBox());
			this->removeAll = (gcnew System::Windows::Forms::Button());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->cancelcrop = (gcnew System::Windows::Forms::Button());
			this->Crop = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->tableLayoutPanel1->SuspendLayout();
			this->tableLayoutPanel2->SuspendLayout();
			this->panel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 1;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				50)));
			this->tableLayoutPanel1->Controls->Add(this->tableLayoutPanel2, 0, 2);
			this->tableLayoutPanel1->Controls->Add(this->label1, 0, 1);
			this->tableLayoutPanel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel1->Location = System::Drawing::Point(0, 0);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 3;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 96.59367F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 3.406326F)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 32)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 20)));
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Absolute, 20)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(625, 444);
			this->tableLayoutPanel1->TabIndex = 0;
			// 
			// tableLayoutPanel2
			// 
			this->tableLayoutPanel2->ColumnCount = 6;
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				25.12156F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				20.25932F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				13.77634F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				14.58671F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				11.66936F)));
			this->tableLayoutPanel2->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent, 
				14.58671F)));
			this->tableLayoutPanel2->Controls->Add(this->AutoBaseline, 0, 0);
			this->tableLayoutPanel2->Controls->Add(this->resetZoom, 5, 0);
			this->tableLayoutPanel2->Controls->Add(this->saveAs, 4, 0);
			this->tableLayoutPanel2->Controls->Add(this->logScale, 3, 0);
			this->tableLayoutPanel2->Controls->Add(this->removeAll, 2, 0);
			this->tableLayoutPanel2->Controls->Add(this->panel1, 1, 0);
			this->tableLayoutPanel2->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanel2->Location = System::Drawing::Point(3, 414);
			this->tableLayoutPanel2->Name = L"tableLayoutPanel2";
			this->tableLayoutPanel2->RowCount = 1;
			this->tableLayoutPanel2->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 100)));
			this->tableLayoutPanel2->Size = System::Drawing::Size(619, 27);
			this->tableLayoutPanel2->TabIndex = 0;
			// 
			// AutoBaseline
			// 
			this->AutoBaseline->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->AutoBaseline->Location = System::Drawing::Point(3, 3);
			this->AutoBaseline->Name = L"AutoBaseline";
			this->AutoBaseline->Size = System::Drawing::Size(125, 21);
			this->AutoBaseline->TabIndex = 0;
			this->AutoBaseline->Text = L"Automatic Generation";
			this->AutoBaseline->UseVisualStyleBackColor = true;
			this->AutoBaseline->Click += gcnew System::EventHandler(this, &ExtractBaseline::AutoBaseline_Click);
			// 
			// resetZoom
			// 
			this->resetZoom->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->resetZoom->Location = System::Drawing::Point(541, 3);
			this->resetZoom->Name = L"resetZoom";
			this->resetZoom->Size = System::Drawing::Size(75, 21);
			this->resetZoom->TabIndex = 4;
			this->resetZoom->Text = L"Reset Zoom";
			this->resetZoom->UseVisualStyleBackColor = true;
			this->resetZoom->Click += gcnew System::EventHandler(this, &ExtractBaseline::button1_Click);
			// 
			// saveAs
			// 
			this->saveAs->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->saveAs->Location = System::Drawing::Point(458, 3);
			this->saveAs->Name = L"saveAs";
			this->saveAs->Size = System::Drawing::Size(66, 21);
			this->saveAs->TabIndex = 3;
			this->saveAs->Text = L"OK";
			this->saveAs->UseVisualStyleBackColor = true;
			this->saveAs->Click += gcnew System::EventHandler(this, &ExtractBaseline::saveAs_Click);
			// 
			// logScale
			// 
			this->logScale->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->logScale->AutoSize = true;
			this->logScale->Location = System::Drawing::Point(375, 7);
			this->logScale->Name = L"logScale";
			this->logScale->Size = System::Drawing::Size(77, 17);
			this->logScale->TabIndex = 2;
			this->logScale->Text = L"Log. Scale";
			this->logScale->UseVisualStyleBackColor = true;
			this->logScale->CheckedChanged += gcnew System::EventHandler(this, &ExtractBaseline::logScale_CheckedChanged);
			// 
			// removeAll
			// 
			this->removeAll->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->removeAll->Location = System::Drawing::Point(287, 3);
			this->removeAll->Name = L"removeAll";
			this->removeAll->Size = System::Drawing::Size(75, 21);
			this->removeAll->TabIndex = 1;
			this->removeAll->Text = L"Remove All";
			this->removeAll->UseVisualStyleBackColor = true;
			this->removeAll->Click += gcnew System::EventHandler(this, &ExtractBaseline::removeAll_Click);
			// 
			// panel1
			// 
			this->panel1->Controls->Add(this->cancelcrop);
			this->panel1->Controls->Add(this->Crop);
			this->panel1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->panel1->Location = System::Drawing::Point(158, 3);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(119, 21);
			this->panel1->TabIndex = 5;
			// 
			// cancelcrop
			// 
			this->cancelcrop->Enabled = false;
			this->cancelcrop->Location = System::Drawing::Point(70, 0);
			this->cancelcrop->Name = L"cancelcrop";
			this->cancelcrop->Size = System::Drawing::Size(49, 21);
			this->cancelcrop->TabIndex = 6;
			this->cancelcrop->Text = L"Cancel";
			this->cancelcrop->UseVisualStyleBackColor = true;
			this->cancelcrop->Click += gcnew System::EventHandler(this, &ExtractBaseline::cancelcrop_Click);
			// 
			// Crop
			// 
			this->Crop->Location = System::Drawing::Point(3, 0);
			this->Crop->Name = L"Crop";
			this->Crop->Size = System::Drawing::Size(61, 21);
			this->Crop->TabIndex = 5;
			this->Crop->Text = L"Crop";
			this->Crop->UseVisualStyleBackColor = true;
			this->Crop->Click += gcnew System::EventHandler(this, &ExtractBaseline::Crop_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->label1->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(192)), static_cast<System::Int32>(static_cast<System::Byte>(0)), 
				static_cast<System::Int32>(static_cast<System::Byte>(0)));
			this->label1->Location = System::Drawing::Point(3, 397);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(619, 14);
			this->label1->TabIndex = 1;
			this->label1->Text = L"An intersection was found in the background.";
			this->label1->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			this->label1->Visible = false;
			// 
			// ExtractBaseline
			// 
			this->AcceptButton = this->saveAs;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(625, 444);
			this->Controls->Add(this->tableLayoutPanel1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->MinimumSize = System::Drawing::Size(512, 26);
			this->Name = L"ExtractBaseline";
			this->Text = L"Extract Baseline";
			this->Load += gcnew System::EventHandler(this, &ExtractBaseline::ExtractBaseline_Load);
			this->tableLayoutPanel1->ResumeLayout(false);
			this->tableLayoutPanel1->PerformLayout();
			this->tableLayoutPanel2->ResumeLayout(false);
			this->tableLayoutPanel2->PerformLayout();
			this->panel1->ResumeLayout(false);
			this->ResumeLayout(false);

		}
#pragma endregion
		private: GUICLR::WGTControl^  wgtGraph;
				void OpenInitialGraph();
		
		private: System::Void ExtractBaseline_Load(System::Object^  sender, System::EventArgs^  e) {
				this->wgtGraph = (gcnew GUICLR::WGTControl());
				this->tableLayoutPanel1->Controls->Add(this->wgtGraph);
				// 
				// wgtGraph
				// 
				this->wgtGraph->Cursor = System::Windows::Forms::Cursors::Cross;
				this->wgtGraph->Dock = System::Windows::Forms::DockStyle::Fill;
				this->wgtGraph->Location = System::Drawing::Point(0, 0);
				this->wgtGraph->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &ExtractBaseline::wgtGraph_MouseClick);
				this->wgtGraph->MouseUp    += gcnew System::Windows::Forms::MouseEventHandler(this, &ExtractBaseline::wgtGraph_MouseUp);
				this->wgtGraph->MouseDown  += gcnew System::Windows::Forms::MouseEventHandler(this, &ExtractBaseline::wgtGraph_MouseDown);
				this->wgtGraph->MouseMove  += gcnew System::Windows::Forms::MouseEventHandler(this, &ExtractBaseline::wgtGraph_MouseMove);
				this->wgtGraph->Name = L"wgtGraph";
				this->wgtGraph->Size = System::Drawing::Size(343, 349);
				this->wgtGraph->TabIndex = 0;

				// Opening an initial graph
				OpenInitialGraph();
				
			}
		private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
			if(wgtGraph->graph)
				wgtGraph->graph->ResetZoom();
			wgtGraph->Invalidate();
		}
		private: System::Void saveAs_Click(System::Object^  sender, System::EventArgs^  e);

		private: System::Void removeAll_Click(System::Object^  sender, System::EventArgs^  e);

		private: System::Void logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if(wgtGraph->graph)
				wgtGraph->graph->SetScale(0, (logScale->Checked) ? 
						SCALE_LOG : SCALE_LIN);
			wgtGraph->Invalidate();
		}

		private: System::Void wgtGraph_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		private: System::Void wgtGraph_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		private: System::Void wgtGraph_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		private: System::Void wgtGraph_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
		void ExtractBaseline::__findIntersections();
		private: System::Void Crop_Click(System::Object^  sender, System::EventArgs^  e); 
		public: static int PosToDataIndex(double x, std::vector <double> datax);
private: System::Void cancelcrop_Click(System::Object^  sender, System::EventArgs^  e) {
			 _bCrop = false;
			 AutoBaseline->Enabled = true;
			 _firstCropX = -1;
			 _oldleft = -1; _oldright = -1;
			 wgtGraph->graph->RemoveMask();
			 wgtGraph->Invalidate();
			 cancelcrop->Enabled = false;
			 removeAll->Enabled = true;
			 Crop->Enabled = true;
			 saveAs->Enabled = true;

		 }

private: System::Void AutoBaseline_Click(System::Object^  sender, System::EventArgs^  e);

public: bool bAutoBaseline() { return _bAuto; }

public: void getAutoBaseline(std::vector<double> &bl);
public: void getCroppedSignal(std::vector<double> &x, std::vector<double> &y);

public: void mapToVectors( std::map<double, double>& m, 
										std::vector<double>& sbx, 
										std::vector<double>& sby);

public: void VectorsToLogscale(const std::vector<double>& datax,
						   const std::vector<double>& datay,
						   std::vector<double>& sx,
						   std::vector<double>& sy);

public:	static double InterpolatePoint(double x0, const std::vector<double>& x, const std::vector<double>& y);
};

}
