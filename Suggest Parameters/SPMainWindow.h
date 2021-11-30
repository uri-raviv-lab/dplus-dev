#pragma once

namespace SuggestParameters {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MainWindow
	/// </summary>
	public ref class SPMainWindow : public System::Windows::Forms::Form
	{
	public:
		SPMainWindow(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			should_be_adaptive = false;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~SPMainWindow()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TextBox^  textBoxX;
	protected:
	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  labelX;
	private: System::Windows::Forms::TextBox^  textBoxY;

	private: System::Windows::Forms::Label^  labelY;
	private: System::Windows::Forms::TextBox^  textBoxZ;

	private: System::Windows::Forms::Label^  labelZ;
	private: System::Windows::Forms::TextBox^  textBoxQ;

	private: System::Windows::Forms::Label^  labelQMax;
	private: System::Windows::Forms::CheckBox^  checkBoxGPU;
	private: System::Windows::Forms::CheckBox^  checkBoxRemote;
	private: System::Windows::Forms::Label^  labelNote;
	private: System::Windows::Forms::TextBox^  textBoxGridSize;
	private: System::Windows::Forms::Label^  labelGridSize;
	private: System::Windows::Forms::TextBox^  textBoxMemReq;

	private: System::Windows::Forms::Label^  labelMemReq;
	private: System::Windows::Forms::Label^  labelIntegrationIters;
	private: System::Windows::Forms::Label^  labelConvergence;
	private: System::Windows::Forms::Label^  labelGenPoints;
	private: System::Windows::Forms::Label^  labelUpdate;

	private: System::Windows::Forms::SplitContainer^  splitContainer1;
	private: System::Windows::Forms::Label^  labelWarning;
	private: System::Windows::Forms::Label^  labelIntegrationMethod;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::Label^  label9;




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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(SPMainWindow::typeid));
			this->textBoxX = (gcnew System::Windows::Forms::TextBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->labelX = (gcnew System::Windows::Forms::Label());
			this->textBoxY = (gcnew System::Windows::Forms::TextBox());
			this->labelY = (gcnew System::Windows::Forms::Label());
			this->textBoxZ = (gcnew System::Windows::Forms::TextBox());
			this->labelZ = (gcnew System::Windows::Forms::Label());
			this->textBoxQ = (gcnew System::Windows::Forms::TextBox());
			this->labelQMax = (gcnew System::Windows::Forms::Label());
			this->checkBoxGPU = (gcnew System::Windows::Forms::CheckBox());
			this->checkBoxRemote = (gcnew System::Windows::Forms::CheckBox());
			this->labelNote = (gcnew System::Windows::Forms::Label());
			this->textBoxGridSize = (gcnew System::Windows::Forms::TextBox());
			this->labelGridSize = (gcnew System::Windows::Forms::Label());
			this->textBoxMemReq = (gcnew System::Windows::Forms::TextBox());
			this->labelMemReq = (gcnew System::Windows::Forms::Label());
			this->labelIntegrationIters = (gcnew System::Windows::Forms::Label());
			this->labelConvergence = (gcnew System::Windows::Forms::Label());
			this->labelGenPoints = (gcnew System::Windows::Forms::Label());
			this->labelUpdate = (gcnew System::Windows::Forms::Label());
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->labelWarning = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->labelIntegrationMethod = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			this->SuspendLayout();
			// 
			// textBoxX
			// 
			this->textBoxX->Location = System::Drawing::Point(51, 43);
			this->textBoxX->Name = L"textBoxX";
			this->textBoxX->Size = System::Drawing::Size(100, 20);
			this->textBoxX->TabIndex = 0;
			this->textBoxX->TextChanged += gcnew System::EventHandler(this, &SPMainWindow::textBox_TextChanged);
			this->textBoxX->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &SPMainWindow::textBox_Validating);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->InitialImage = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.InitialImage")));
			this->pictureBox1->Location = System::Drawing::Point(0, 0);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(284, 335);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox1->TabIndex = 1;
			this->pictureBox1->TabStop = false;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(58, 27);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(89, 13);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Max. Length (nm)";
			// 
			// labelX
			// 
			this->labelX->AutoSize = true;
			this->labelX->Location = System::Drawing::Point(32, 46);
			this->labelX->Name = L"labelX";
			this->labelX->Size = System::Drawing::Size(12, 13);
			this->labelX->TabIndex = 3;
			this->labelX->Text = L"x";
			// 
			// textBoxY
			// 
			this->textBoxY->Location = System::Drawing::Point(51, 69);
			this->textBoxY->Name = L"textBoxY";
			this->textBoxY->Size = System::Drawing::Size(100, 20);
			this->textBoxY->TabIndex = 1;
			this->textBoxY->TextChanged += gcnew System::EventHandler(this, &SPMainWindow::textBox_TextChanged);
			this->textBoxY->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &SPMainWindow::textBox_Validating);
			// 
			// labelY
			// 
			this->labelY->AutoSize = true;
			this->labelY->Location = System::Drawing::Point(32, 72);
			this->labelY->Name = L"labelY";
			this->labelY->Size = System::Drawing::Size(12, 13);
			this->labelY->TabIndex = 3;
			this->labelY->Text = L"y";
			// 
			// textBoxZ
			// 
			this->textBoxZ->Location = System::Drawing::Point(51, 95);
			this->textBoxZ->Name = L"textBoxZ";
			this->textBoxZ->Size = System::Drawing::Size(100, 20);
			this->textBoxZ->TabIndex = 2;
			this->textBoxZ->TextChanged += gcnew System::EventHandler(this, &SPMainWindow::textBox_TextChanged);
			this->textBoxZ->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &SPMainWindow::textBox_Validating);
			// 
			// labelZ
			// 
			this->labelZ->AutoSize = true;
			this->labelZ->Location = System::Drawing::Point(32, 98);
			this->labelZ->Name = L"labelZ";
			this->labelZ->Size = System::Drawing::Size(12, 13);
			this->labelZ->TabIndex = 3;
			this->labelZ->Text = L"z";
			// 
			// textBoxQ
			// 
			this->textBoxQ->Location = System::Drawing::Point(51, 121);
			this->textBoxQ->Name = L"textBoxQ";
			this->textBoxQ->Size = System::Drawing::Size(100, 20);
			this->textBoxQ->TabIndex = 3;
			this->textBoxQ->TextChanged += gcnew System::EventHandler(this, &SPMainWindow::textBox_TextChanged);
			this->textBoxQ->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &SPMainWindow::textBox_Validating);
			// 
			// labelQMax
			// 
			this->labelQMax->AutoSize = true;
			this->labelQMax->Location = System::Drawing::Point(6, 124);
			this->labelQMax->Name = L"labelQMax";
			this->labelQMax->Size = System::Drawing::Size(39, 13);
			this->labelQMax->TabIndex = 3;
			this->labelQMax->Text = L"q Max.";
			// 
			// checkBoxGPU
			// 
			this->checkBoxGPU->AutoSize = true;
			this->checkBoxGPU->Location = System::Drawing::Point(51, 148);
			this->checkBoxGPU->Name = L"checkBoxGPU";
			this->checkBoxGPU->Size = System::Drawing::Size(110, 17);
			this->checkBoxGPU->TabIndex = 4;
			this->checkBoxGPU->Text = L"Use GPU (CUDA)";
			this->checkBoxGPU->UseVisualStyleBackColor = true;
			this->checkBoxGPU->CheckedChanged += gcnew System::EventHandler(this, &SPMainWindow::checkBoxGPU_CheckedChanged);
			// 
			// checkBoxRemote
			// 
			this->checkBoxRemote->AutoSize = true;
			this->checkBoxRemote->Location = System::Drawing::Point(51, 171);
			this->checkBoxRemote->Name = L"checkBoxRemote";
			this->checkBoxRemote->Size = System::Drawing::Size(117, 17);
			this->checkBoxRemote->TabIndex = 5;
			this->checkBoxRemote->Text = L"Calculate Remotely";
			this->checkBoxRemote->UseVisualStyleBackColor = true;
			this->checkBoxRemote->CheckedChanged += gcnew System::EventHandler(this, &SPMainWindow::checkBoxRemote_CheckedChanged);
			// 
			// labelNote
			// 
			this->labelNote->AutoSize = true;
			this->labelNote->Location = System::Drawing::Point(32, 227);
			this->labelNote->MaximumSize = System::Drawing::Size(264, 0);
			this->labelNote->Name = L"labelNote";
			this->labelNote->Size = System::Drawing::Size(261, 26);
			this->labelNote->TabIndex = 5;
			this->labelNote->Text = L"NOTE: Make sure that the Level of Detail is on the minimum when loading a large o"
				L"bject with many atoms";
			this->labelNote->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// textBoxGridSize
			// 
			this->textBoxGridSize->Location = System::Drawing::Point(276, 43);
			this->textBoxGridSize->Name = L"textBoxGridSize";
			this->textBoxGridSize->ReadOnly = true;
			this->textBoxGridSize->Size = System::Drawing::Size(100, 20);
			this->textBoxGridSize->TabIndex = 10;
			// 
			// labelGridSize
			// 
			this->labelGridSize->AutoSize = true;
			this->labelGridSize->Location = System::Drawing::Point(221, 46);
			this->labelGridSize->Name = L"labelGridSize";
			this->labelGridSize->Size = System::Drawing::Size(49, 13);
			this->labelGridSize->TabIndex = 3;
			this->labelGridSize->Text = L"Grid Size";
			// 
			// textBoxMemReq
			// 
			this->textBoxMemReq->Location = System::Drawing::Point(276, 69);
			this->textBoxMemReq->Name = L"textBoxMemReq";
			this->textBoxMemReq->ReadOnly = true;
			this->textBoxMemReq->Size = System::Drawing::Size(100, 20);
			this->textBoxMemReq->TabIndex = 11;
			// 
			// labelMemReq
			// 
			this->labelMemReq->AutoSize = true;
			this->labelMemReq->Location = System::Drawing::Point(155, 72);
			this->labelMemReq->Name = L"labelMemReq";
			this->labelMemReq->Size = System::Drawing::Size(115, 13);
			this->labelMemReq->TabIndex = 3;
			this->labelMemReq->Text = L"Memory Required (MB)";
			// 
			// labelIntegrationIters
			// 
			this->labelIntegrationIters->AutoSize = true;
			this->labelIntegrationIters->Location = System::Drawing::Point(184, 148);
			this->labelIntegrationIters->Name = L"labelIntegrationIters";
			this->labelIntegrationIters->Size = System::Drawing::Size(128, 13);
			this->labelIntegrationIters->TabIndex = 6;
			this->labelIntegrationIters->Text = L"Integration Iterations: 1E6";
			// 
			// labelConvergence
			// 
			this->labelConvergence->AutoSize = true;
			this->labelConvergence->Location = System::Drawing::Point(216, 167);
			this->labelConvergence->Name = L"labelConvergence";
			this->labelConvergence->Size = System::Drawing::Size(104, 13);
			this->labelConvergence->TabIndex = 6;
			this->labelConvergence->Text = L"Convergence: 0.001";
			// 
			// labelGenPoints
			// 
			this->labelGenPoints->AutoSize = true;
			this->labelGenPoints->Location = System::Drawing::Point(198, 187);
			this->labelGenPoints->Name = L"labelGenPoints";
			this->labelGenPoints->Size = System::Drawing::Size(95, 13);
			this->labelGenPoints->TabIndex = 6;
			this->labelGenPoints->Text = L"Generated Points: ";
			// 
			// labelUpdate
			// 
			this->labelUpdate->AutoSize = true;
			this->labelUpdate->Location = System::Drawing::Point(207, 205);
			this->labelUpdate->Name = L"labelUpdate";
			this->labelUpdate->Size = System::Drawing::Size(117, 13);
			this->labelUpdate->TabIndex = 6;
			this->labelUpdate->Text = L"Update Interval: 500ms";
			// 
			// splitContainer1
			// 
			this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->splitContainer1->FixedPanel = System::Windows::Forms::FixedPanel::Panel1;
			this->splitContainer1->IsSplitterFixed = true;
			this->splitContainer1->Location = System::Drawing::Point(0, 0);
			this->splitContainer1->Name = L"splitContainer1";
			// 
			// splitContainer1.Panel1
			// 
			this->splitContainer1->Panel1->AutoScroll = true;
			this->splitContainer1->Panel1->Controls->Add(this->labelWarning);
			this->splitContainer1->Panel1->Controls->Add(this->labelNote);
			this->splitContainer1->Panel1->Controls->Add(this->labelUpdate);
			this->splitContainer1->Panel1->Controls->Add(this->labelX);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxX);
			this->splitContainer1->Panel1->Controls->Add(this->labelMemReq);
			this->splitContainer1->Panel1->Controls->Add(this->labelGenPoints);
			this->splitContainer1->Panel1->Controls->Add(this->labelGridSize);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxGridSize);
			this->splitContainer1->Panel1->Controls->Add(this->labelY);
			this->splitContainer1->Panel1->Controls->Add(this->labelConvergence);
			this->splitContainer1->Panel1->Controls->Add(this->labelZ);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxMemReq);
			this->splitContainer1->Panel1->Controls->Add(this->label10);
			this->splitContainer1->Panel1->Controls->Add(this->label9);
			this->splitContainer1->Panel1->Controls->Add(this->label1);
			this->splitContainer1->Panel1->Controls->Add(this->label6);
			this->splitContainer1->Panel1->Controls->Add(this->label5);
			this->splitContainer1->Panel1->Controls->Add(this->label8);
			this->splitContainer1->Panel1->Controls->Add(this->label7);
			this->splitContainer1->Panel1->Controls->Add(this->label4);
			this->splitContainer1->Panel1->Controls->Add(this->label3);
			this->splitContainer1->Panel1->Controls->Add(this->label2);
			this->splitContainer1->Panel1->Controls->Add(this->labelIntegrationMethod);
			this->splitContainer1->Panel1->Controls->Add(this->labelIntegrationIters);
			this->splitContainer1->Panel1->Controls->Add(this->labelQMax);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxY);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxQ);
			this->splitContainer1->Panel1->Controls->Add(this->checkBoxRemote);
			this->splitContainer1->Panel1->Controls->Add(this->checkBoxGPU);
			this->splitContainer1->Panel1->Controls->Add(this->textBoxZ);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->Controls->Add(this->pictureBox1);
			this->splitContainer1->Size = System::Drawing::Size(671, 335);
			this->splitContainer1->SplitterDistance = 383;
			this->splitContainer1->TabIndex = 7;
			// 
			// labelWarning
			// 
			this->labelWarning->AutoSize = true;
			this->labelWarning->Location = System::Drawing::Point(184, 95);
			this->labelWarning->MaximumSize = System::Drawing::Size(195, 0);
			this->labelWarning->Name = L"labelWarning";
			this->labelWarning->Size = System::Drawing::Size(0, 13);
			this->labelWarning->TabIndex = 12;
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label10->Location = System::Drawing::Point(122, 271);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(123, 17);
			this->label10->TabIndex = 2;
			this->label10->Text = L"Fitting Parameters";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label9->Location = System::Drawing::Point(116, 5);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(156, 17);
			this->label9->TabIndex = 2;
			this->label9->Text = L"Generation Parameters";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(304, 312);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(65, 13);
			this->label6->TabIndex = 6;
			this->label6->Text = L"Der eps: 0.1";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(270, 297);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(104, 13);
			this->label5->TabIndex = 6;
			this->label5->Text = L"Converegence: 0.01";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(99, 312);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(60, 13);
			this->label8->TabIndex = 6;
			this->label8->Text = L"Trivial Loss";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(12, 312);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(81, 13);
			this->label7->TabIndex = 6;
			this->label7->Text = L"Ratio Residuals";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(199, 312);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(73, 13);
			this->label4->TabIndex = 6;
			this->label4->Text = L"Step Size: 0.1";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(201, 297);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(62, 13);
			this->label3->TabIndex = 6;
			this->label3->Text = L"Iterations: 6";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(12, 297);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(182, 13);
			this->label2->TabIndex = 6;
			this->label2->Text = L"Fitting Method: Levenberg-Marquardt";
			// 
			// labelIntegrationMethod
			// 
			this->labelIntegrationMethod->AutoSize = true;
			this->labelIntegrationMethod->Location = System::Drawing::Point(191, 128);
			this->labelIntegrationMethod->Name = L"labelIntegrationMethod";
			this->labelIntegrationMethod->Size = System::Drawing::Size(159, 13);
			this->labelIntegrationMethod->TabIndex = 6;
			this->labelIntegrationMethod->Text = L"Integration Method: Monte Carlo";
			// 
			// SPMainWindow
			// 
			this->AutoScroll = true;
			this->ClientSize = System::Drawing::Size(671, 335);
			this->Controls->Add(this->splitContainer1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"SPMainWindow";
			this->Text = L"Suggest Parameters";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel1->PerformLayout();
			this->splitContainer1->Panel2->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			this->ResumeLayout(false);

		}
#pragma endregion

private: System::Void textBox_Validating(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e);
private: System::Void checkBoxRemote_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void checkBoxGPU_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		 bool should_be_adaptive;
private: System::Void textBox_TextChanged(System::Object^  sender, System::EventArgs^  e);
};
}
