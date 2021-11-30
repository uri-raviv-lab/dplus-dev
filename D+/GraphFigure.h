#pragma once

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for GraphFigure
	/// </summary>
	public ref class GraphFigure : public System::Windows::Forms::Form
	{
	public:
		int figure;


		GraphFigure(int fignum)
		{
			InitializeComponent();
			
			figure = fignum;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~GraphFigure()
		{
			if (components)
			{
				delete components;
			}
		}
	public: GraphToolkit::Graph1D^  graph;
	protected: 
	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	public: 
	private: System::Windows::Forms::Label^  locationLabel;
	private: System::Windows::Forms::CheckBox^  logQcheckBox;
	private: System::Windows::Forms::CheckBox^  logIcheckBox;

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(GraphFigure::typeid));
			this->graph = (gcnew GraphToolkit::Graph1D());
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->locationLabel = (gcnew System::Windows::Forms::Label());
			this->logQcheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->logIcheckBox = (gcnew System::Windows::Forms::CheckBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->BeginInit();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->SuspendLayout();
			// 
			// graph
			// 
			this->graph->Dock = System::Windows::Forms::DockStyle::Fill;
			this->graph->GraphTitle = L"Figure";
			this->graph->Location = System::Drawing::Point(0, 0);
			this->graph->Name = L"graph";
			this->graph->Size = System::Drawing::Size(416, 322);
			this->graph->TabIndex = 0;
			this->graph->XLabel = L"X";
			this->graph->YLabel = L"Y";
			this->graph->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &GraphFigure::graph_MouseMove);
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
			this->splitContainer1->Panel1->Controls->Add(this->graph);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->Controls->Add(this->locationLabel);
			this->splitContainer1->Panel2->Controls->Add(this->logQcheckBox);
			this->splitContainer1->Panel2->Controls->Add(this->logIcheckBox);
			this->splitContainer1->Size = System::Drawing::Size(416, 351);
			this->splitContainer1->SplitterDistance = 322;
			this->splitContainer1->TabIndex = 2;
			// 
			// locationLabel
			// 
			this->locationLabel->AutoSize = true;
			this->locationLabel->Location = System::Drawing::Point(12, 6);
			this->locationLabel->Name = L"locationLabel";
			this->locationLabel->Size = System::Drawing::Size(28, 13);
			this->locationLabel->TabIndex = 1;
			this->locationLabel->Text = L"(0,0)";
			// 
			// logQcheckBox
			// 
			this->logQcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logQcheckBox->AutoSize = true;
			this->logQcheckBox->Location = System::Drawing::Point(296, 5);
			this->logQcheckBox->Name = L"logQcheckBox";
			this->logQcheckBox->Size = System::Drawing::Size(51, 17);
			this->logQcheckBox->TabIndex = 0;
			this->logQcheckBox->Text = L"log(x)";
			this->logQcheckBox->UseVisualStyleBackColor = true;
			this->logQcheckBox->CheckedChanged += gcnew System::EventHandler(this, &GraphFigure::logQcheckBox_CheckedChanged);
			// 
			// logIcheckBox
			// 
			this->logIcheckBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->logIcheckBox->AutoSize = true;
			this->logIcheckBox->Location = System::Drawing::Point(353, 5);
			this->logIcheckBox->Name = L"logIcheckBox";
			this->logIcheckBox->Size = System::Drawing::Size(51, 17);
			this->logIcheckBox->TabIndex = 0;
			this->logIcheckBox->Text = L"log(y)";
			this->logIcheckBox->UseVisualStyleBackColor = true;
			this->logIcheckBox->CheckedChanged += gcnew System::EventHandler(this, &GraphFigure::logIcheckBox_CheckedChanged);
			// 
			// GraphFigure
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(416, 351);
			this->Controls->Add(this->splitContainer1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"GraphFigure";
			this->Text = L"Figure";
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->Panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void logIcheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
				 graph->LogScaleY = logIcheckBox->Checked;
			 }
private: System::Void logQcheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
				 graph->LogScaleX = logQcheckBox->Checked;
			 }
private: System::Void graph_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 GraphToolkit::DoublePair dp = graph->PointToData(e->X, e->Y);
			 locationLabel->Text = "(" + Double(dp.first).ToString() + ", " + Double(dp.second).ToString() + ")";
		 }
};
}
