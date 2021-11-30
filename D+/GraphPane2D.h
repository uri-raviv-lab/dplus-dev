#pragma once

#include "MainWindow.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;

namespace DPlus {

	/// <summary>
	/// Summary for GraphPane2D
	/// </summary>
	public ref class GraphPane2D : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	public:
		GraphPane2D(MainWindow ^pform);

		bool bSignalSet, bModelSet;
		array<double> ^sigy, ^mody;

		void SetSignalGraph(array<double> ^x, array<double> ^y);
		void ClearSignalGraph();

		void GetModelGraph(array<double> ^%x, array<double> ^%y);
		void SetModelGraph(array<double> ^x, array<double> ^y);
		void ClearModelGraph();

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~GraphPane2D()
		{
			if (components)
			{
				delete components;
			}
		}
	public: GraphToolkit::Graph1D^  graph1D1;
	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	private: System::Windows::Forms::Label^  locationLabel;

	private: System::Windows::Forms::CheckBox^  logQcheckBox;
	private: System::Windows::Forms::CheckBox^  logIcheckBox;
	private: System::Windows::Forms::Label^  rSqrLabel;
	private: System::Windows::Forms::Label^  chiSqrLabel;

	protected: 

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(GraphPane2D::typeid));
			this->graph1D1 = (gcnew GraphToolkit::Graph1D());
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->rSqrLabel = (gcnew System::Windows::Forms::Label());
			this->chiSqrLabel = (gcnew System::Windows::Forms::Label());
			this->locationLabel = (gcnew System::Windows::Forms::Label());
			this->logQcheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->logIcheckBox = (gcnew System::Windows::Forms::CheckBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->BeginInit();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->SuspendLayout();
			// 
			// graph1D1
			// 
			this->graph1D1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->graph1D1->GraphTitle = L"";
			this->graph1D1->Location = System::Drawing::Point(0, 0);
			this->graph1D1->Name = L"graph1D1";
			this->graph1D1->Size = System::Drawing::Size(431, 402);
			this->graph1D1->TabIndex = 0;
			this->graph1D1->XLabel = L"Reciprocal Space [nm^-1]";
			this->graph1D1->YLabel = L"Intensity [a.u.]";
			this->graph1D1->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &GraphPane2D::graph1D1_MouseMove);
			// 
			// splitContainer1
			// 
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->Location = System::Drawing::Point(0, 0);
			this->splitContainer1->Name = L"splitContainer1";
			this->splitContainer1->Orientation = System::Windows::Forms::Orientation::Horizontal;
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->Controls->Add(this->graph1D1);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->Controls->Add(this->rSqrLabel);
			this->splitContainer1->Panel2->Controls->Add(this->chiSqrLabel);
			this->splitContainer1->Panel2->Controls->Add(this->locationLabel);
			this->splitContainer1->Panel2->Controls->Add(this->logQcheckBox);
			this->splitContainer1->Panel2->Controls->Add(this->logIcheckBox);
			this->splitContainer1->Size = System::Drawing::Size(431, 452);
			this->splitContainer1->SplitterDistance = 402;
			this->splitContainer1->TabIndex = 1;
			// 
			// rSqrLabel
			// 
			this->rSqrLabel->AutoSize = true;
			this->rSqrLabel->Location = System::Drawing::Point(3, 24);
			this->rSqrLabel->Name = L"rSqrLabel";
			this->rSqrLabel->Size = System::Drawing::Size(59, 13);
			this->rSqrLabel->TabIndex = 2;
			this->rSqrLabel->Text = L"R^2 = N/A";
			// 
			// chiSqrLabel
			// 
			this->chiSqrLabel->AutoSize = true;
			this->chiSqrLabel->Location = System::Drawing::Point(3, 7);
			this->chiSqrLabel->Name = L"chiSqrLabel";
			this->chiSqrLabel->Size = System::Drawing::Size(65, 13);
			this->chiSqrLabel->TabIndex = 2;
			this->chiSqrLabel->Text = L"chi^2 = N/A";
			// 
			// locationLabel
			// 
			this->locationLabel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->locationLabel->Location = System::Drawing::Point(188, 7);
			this->locationLabel->Name = L"locationLabel";
			this->locationLabel->Size = System::Drawing::Size(170, 16);
			this->locationLabel->TabIndex = 1;
			this->locationLabel->Text = L"(0,0)";
			this->locationLabel->TextAlign = System::Drawing::ContentAlignment::TopRight;
			// 
			// logQcheckBox
			// 
			this->logQcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logQcheckBox->AutoSize = true;
			this->logQcheckBox->Location = System::Drawing::Point(364, 23);
			this->logQcheckBox->Name = L"logQcheckBox";
			this->logQcheckBox->Size = System::Drawing::Size(52, 17);
			this->logQcheckBox->TabIndex = 0;
			this->logQcheckBox->Text = L"log(q)";
			this->logQcheckBox->UseVisualStyleBackColor = true;
			this->logQcheckBox->CheckedChanged += gcnew System::EventHandler(this, &GraphPane2D::logQcheckBox_CheckedChanged);
			// 
			// logIcheckBox
			// 
			this->logIcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logIcheckBox->AutoSize = true;
			this->logIcheckBox->Location = System::Drawing::Point(364, 6);
			this->logIcheckBox->Name = L"logIcheckBox";
			this->logIcheckBox->Size = System::Drawing::Size(49, 17);
			this->logIcheckBox->TabIndex = 0;
			this->logIcheckBox->Text = L"log(I)";
			this->logIcheckBox->UseVisualStyleBackColor = true;
			this->logIcheckBox->CheckedChanged += gcnew System::EventHandler(this, &GraphPane2D::logIcheckBox_CheckedChanged);
			// 
			// GraphPane2D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(431, 452);
			this->Controls->Add(this->splitContainer1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"GraphPane2D";
			this->ShowIcon = false;
			this->Text = L"2D Graph";
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->Panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			this->ResumeLayout(false);

		}
#pragma endregion
		// Methods
		void SetYAxisText(System::String^ yAxis);
		void SetXAxisText(System::String^ xAxis);
		void SetGraphTitleText(System::String^ title);



		// Events
	private: System::Void logQcheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
	private: System::Void logIcheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void graph1D1_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
};
}
