#pragma once
#include "dllmain.h"
#include "PDBReader.h"

namespace DockingTest {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for DockingWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class DockingWindow : public System::Windows::Forms::Form
	{
	public:
		void ProgressReport(void *args, double progress);
		void NotifyCompletion(void *args, int error);
		delegate void CLRProgressFunc(void *args, double progress);
		delegate void CLRNotifyCompletionFunc(void *args, int error);

	protected:
		PDBModel *pdb;
		std::vector<FACC> *Q, *res;
		bool bLoadedGrid;
		// Delegate handles for handler functions
		CLRProgressFunc ^progressrep;
		CLRNotifyCompletionFunc ^notifycomp;

	private: System::Windows::Forms::TrackBar^  iterationsTrackBar;
	private: System::Windows::Forms::Label^  iterationsLabel;
	private: System::Windows::Forms::Label^  qMaxlabel;
	private: System::Windows::Forms::TextBox^  qMaxTextBox;
	private: System::Windows::Forms::ProgressBar^  gridProgressBar;
	private: System::Windows::Forms::ProgressBar^  iterationsProgressBar;
	private: System::Windows::Forms::Button^  loadButton;
	public:
		DockingWindow(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			DummyCons();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~DockingWindow()
		{
			if (components)
			{
				delete components;
			}

			delete Q;
			Q = NULL;
			delete res;
			res = NULL;
 			delete pdb;
 			pdb = NULL;
		}
	private: System::Windows::Forms::Button^  loadPDB_Button;
	private: System::Windows::Forms::OpenFileDialog^  PDB_openFileDialog;
	private: System::Windows::Forms::TrackBar^  L_trackBar;
	private: System::Windows::Forms::Label^  L_label;
	private: System::Windows::Forms::Button^  calculateButton;
	private: System::Windows::Forms::Button^  save_Button;
	private: System::Windows::Forms::Label^  matrixLabel;
	private: System::Windows::Forms::Label^  orientationLabel;
	private: System::Windows::Forms::SaveFileDialog^  sfd;
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
			this->loadPDB_Button = (gcnew System::Windows::Forms::Button());
			this->PDB_openFileDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->L_trackBar = (gcnew System::Windows::Forms::TrackBar());
			this->L_label = (gcnew System::Windows::Forms::Label());
			this->calculateButton = (gcnew System::Windows::Forms::Button());
			this->save_Button = (gcnew System::Windows::Forms::Button());
			this->matrixLabel = (gcnew System::Windows::Forms::Label());
			this->orientationLabel = (gcnew System::Windows::Forms::Label());
			this->sfd = (gcnew System::Windows::Forms::SaveFileDialog());
			this->loadButton = (gcnew System::Windows::Forms::Button());
			this->iterationsTrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->iterationsLabel = (gcnew System::Windows::Forms::Label());
			this->qMaxlabel = (gcnew System::Windows::Forms::Label());
			this->qMaxTextBox = (gcnew System::Windows::Forms::TextBox());
			this->gridProgressBar = (gcnew System::Windows::Forms::ProgressBar());
			this->iterationsProgressBar = (gcnew System::Windows::Forms::ProgressBar());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->L_trackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->iterationsTrackBar))->BeginInit();
			this->SuspendLayout();
			// 
			// loadPDB_Button
			// 
			this->loadPDB_Button->Location = System::Drawing::Point(18, 14);
			this->loadPDB_Button->Name = L"loadPDB_Button";
			this->loadPDB_Button->Size = System::Drawing::Size(75, 23);
			this->loadPDB_Button->TabIndex = 0;
			this->loadPDB_Button->Text = L"Load PDB...";
			this->loadPDB_Button->UseVisualStyleBackColor = true;
			this->loadPDB_Button->Click += gcnew System::EventHandler(this, &DockingWindow::loadPDB_Button_Click);
			// 
			// L_trackBar
			// 
			this->L_trackBar->LargeChange = 50;
			this->L_trackBar->Location = System::Drawing::Point(24, 90);
			this->L_trackBar->Maximum = 1000;
			this->L_trackBar->Minimum = 50;
			this->L_trackBar->Name = L"L_trackBar";
			this->L_trackBar->Size = System::Drawing::Size(169, 45);
			this->L_trackBar->SmallChange = 10;
			this->L_trackBar->TabIndex = 1;
			this->L_trackBar->TickFrequency = 50;
			this->L_trackBar->Value = 50;
			this->L_trackBar->Scroll += gcnew System::EventHandler(this, &DockingWindow::L_trackBar_Scroll);
			// 
			// L_label
			// 
			this->L_label->AutoSize = true;
			this->L_label->Location = System::Drawing::Point(21, 74);
			this->L_label->Name = L"L_label";
			this->L_label->Size = System::Drawing::Size(83, 13);
			this->L_label->TabIndex = 2;
			this->L_label->Text = L"Subsections: 50";
			// 
			// calculateButton
			// 
			this->calculateButton->Location = System::Drawing::Point(18, 182);
			this->calculateButton->Name = L"calculateButton";
			this->calculateButton->Size = System::Drawing::Size(75, 23);
			this->calculateButton->TabIndex = 3;
			this->calculateButton->Text = L"Calculate...";
			this->calculateButton->UseVisualStyleBackColor = true;
			this->calculateButton->Click += gcnew System::EventHandler(this, &DockingWindow::calculateButton_Click);
			// 
			// save_Button
			// 
			this->save_Button->Location = System::Drawing::Point(99, 182);
			this->save_Button->Name = L"save_Button";
			this->save_Button->Size = System::Drawing::Size(88, 23);
			this->save_Button->TabIndex = 4;
			this->save_Button->Text = L"...Save Output";
			this->save_Button->UseVisualStyleBackColor = true;
			this->save_Button->Click += gcnew System::EventHandler(this, &DockingWindow::save_Button_Click);
			// 
			// matrixLabel
			// 
			this->matrixLabel->AutoSize = true;
			this->matrixLabel->Location = System::Drawing::Point(18, 212);
			this->matrixLabel->Name = L"matrixLabel";
			this->matrixLabel->Size = System::Drawing::Size(67, 13);
			this->matrixLabel->TabIndex = 5;
			this->matrixLabel->Text = L"Matrix Time: ";
			// 
			// orientationLabel
			// 
			this->orientationLabel->AutoSize = true;
			this->orientationLabel->Location = System::Drawing::Point(18, 235);
			this->orientationLabel->Name = L"orientationLabel";
			this->orientationLabel->Size = System::Drawing::Size(90, 13);
			this->orientationLabel->TabIndex = 5;
			this->orientationLabel->Text = L"Orientation Time: ";
			// 
			// loadButton
			// 
			this->loadButton->Location = System::Drawing::Point(99, 14);
			this->loadButton->Name = L"loadButton";
			this->loadButton->Size = System::Drawing::Size(75, 23);
			this->loadButton->TabIndex = 0;
			this->loadButton->Text = L"Load grid...";
			this->loadButton->UseVisualStyleBackColor = true;
			this->loadButton->Click += gcnew System::EventHandler(this, &DockingWindow::loadPDB_Button_Click);
			// 
			// iterationsTrackBar
			// 
			this->iterationsTrackBar->Location = System::Drawing::Point(24, 141);
			this->iterationsTrackBar->Name = L"iterationsTrackBar";
			this->iterationsTrackBar->Size = System::Drawing::Size(169, 45);
			this->iterationsTrackBar->TabIndex = 1;
			this->iterationsTrackBar->TickFrequency = 2;
			this->iterationsTrackBar->Value = 3;
			this->iterationsTrackBar->Scroll += gcnew System::EventHandler(this, &DockingWindow::L_trackBar_Scroll);
			// 
			// iterationsLabel
			// 
			this->iterationsLabel->AutoSize = true;
			this->iterationsLabel->Location = System::Drawing::Point(21, 125);
			this->iterationsLabel->Name = L"iterationsLabel";
			this->iterationsLabel->Size = System::Drawing::Size(80, 13);
			this->iterationsLabel->TabIndex = 2;
			this->iterationsLabel->Text = L"Iterations: 10^3";
			// 
			// qMaxlabel
			// 
			this->qMaxlabel->AutoSize = true;
			this->qMaxlabel->Location = System::Drawing::Point(21, 46);
			this->qMaxlabel->Name = L"qMaxlabel";
			this->qMaxlabel->Size = System::Drawing::Size(38, 13);
			this->qMaxlabel->TabIndex = 6;
			this->qMaxlabel->Text = L"q max:";
			// 
			// qMaxTextBox
			// 
			this->qMaxTextBox->Location = System::Drawing::Point(62, 43);
			this->qMaxTextBox->Name = L"qMaxTextBox";
			this->qMaxTextBox->Size = System::Drawing::Size(100, 20);
			this->qMaxTextBox->TabIndex = 7;
			this->qMaxTextBox->Text = L"5.00";
			this->qMaxTextBox->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &DockingWindow::qMaxTextBox_KeyPress);
			// 
			// gridProgressBar
			// 
			this->gridProgressBar->Location = System::Drawing::Point(110, 74);
			this->gridProgressBar->Name = L"gridProgressBar";
			this->gridProgressBar->Size = System::Drawing::Size(100, 14);
			this->gridProgressBar->TabIndex = 8;
			this->gridProgressBar->Visible = false;
			// 
			// iterationsProgressBar
			// 
			this->iterationsProgressBar->Location = System::Drawing::Point(110, 125);
			this->iterationsProgressBar->Name = L"iterationsProgressBar";
			this->iterationsProgressBar->Size = System::Drawing::Size(100, 14);
			this->iterationsProgressBar->TabIndex = 8;
			this->iterationsProgressBar->Visible = false;
			// 
			// DockingWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(246, 259);
			this->Controls->Add(this->iterationsProgressBar);
			this->Controls->Add(this->gridProgressBar);
			this->Controls->Add(this->qMaxTextBox);
			this->Controls->Add(this->qMaxlabel);
			this->Controls->Add(this->orientationLabel);
			this->Controls->Add(this->matrixLabel);
			this->Controls->Add(this->save_Button);
			this->Controls->Add(this->calculateButton);
			this->Controls->Add(this->iterationsLabel);
			this->Controls->Add(this->L_label);
			this->Controls->Add(this->iterationsTrackBar);
			this->Controls->Add(this->L_trackBar);
			this->Controls->Add(this->loadButton);
			this->Controls->Add(this->loadPDB_Button);
			this->Name = L"DockingWindow";
			this->Text = L"DockingWindow";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->L_trackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->iterationsTrackBar))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
#pragma region UI Events
	private: System::Void L_trackBar_Scroll(System::Object^  sender, System::EventArgs^  e);
	private: System::Void loadPDB_Button_Click(System::Object^  sender, System::EventArgs^  e);
			 void DummyCons();
#pragma endregion
	private: System::Void calculateButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void save_Button_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void qMaxTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e);
		 void ChangeEnabled(bool en);
};
}

