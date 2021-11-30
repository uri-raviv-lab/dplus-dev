#pragma once

#include "MainWindow.h"
using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;
using namespace LuaInterface;

namespace DPlus {

	public ref class ControlButtonsPane : WeifenLuo::WinFormsUI::Docking::DockContent
	{

	protected:
		MainWindow ^ parentForm;
	private: System::ComponentModel::IContainer^  components;
	public: System::Windows::Forms::Button^  generateButton;
	public: System::Windows::Forms::Button^  fitButton;
	public: System::Windows::Forms::Button^  stopButton;

	public: System::Windows::Forms::CheckBox^ applyResolutionCheckBox;
	public: System::Windows::Forms::TextBox^ resolutionSigmaTextBox;
	private: System::Windows::Forms::Label^ resolutionSigmaLabel;

	public:
		ControlButtonsPane(MainWindow ^pform)
		{
			InitializeComponent();
			parentForm = pform;
			validate_qs = true;
			Double::TryParse(resolutionSigmaTextBox->Text, prev_sigma);
		}
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ControlButtonsPane()
		{
			if (components)
			{
				delete components;
			}
		}
#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ControlButtonsPane::typeid));
			this->components = (gcnew System::ComponentModel::Container());
			this->generateButton = (gcnew System::Windows::Forms::Button());
			this->fitButton = (gcnew System::Windows::Forms::Button());
			this->stopButton = (gcnew System::Windows::Forms::Button());

			this->applyResolutionCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->resolutionSigmaLabel = (gcnew System::Windows::Forms::Label());
			this->resolutionSigmaTextBox = (gcnew System::Windows::Forms::TextBox());

			// 
			// generateButton
			// 
			this->generateButton->Location = System::Drawing::Point(12, 12);
			this->generateButton->Name = L"generateButton";
			this->generateButton->Size = System::Drawing::Size(73, 23);
			this->generateButton->TabIndex = 30;
			this->generateButton->Text = L"Generate";
			this->generateButton->UseVisualStyleBackColor = true;
			this->generateButton->Click += gcnew System::EventHandler(this, &ControlButtonsPane::generateButton_Click);
			// 
			// fitButton
			// 
			this->fitButton->Location = System::Drawing::Point(12, 37);
			this->fitButton->Name = L"fitButton";
			this->fitButton->Size = System::Drawing::Size(73, 23);
			this->fitButton->TabIndex = 31;
			this->fitButton->Text = L"Fit";
			this->fitButton->UseVisualStyleBackColor = true;
			this->fitButton->Click += gcnew System::EventHandler(this, &ControlButtonsPane::fitButton_Click);
			// 
			// stopButton
			// 
			this->stopButton->Enabled = false;
			this->stopButton->Location = System::Drawing::Point(91, 37);
			this->stopButton->Name = L"stopButton";
			this->stopButton->Size = System::Drawing::Size(73, 23);
			this->stopButton->TabIndex = 32;
			this->stopButton->Text = L"Stop";
			this->stopButton->UseVisualStyleBackColor = true;
			this->stopButton->Click += gcnew System::EventHandler(this, &ControlButtonsPane::stopButton_Click);
			// 
			// applyResolutionCheckBox
			// 
			this->applyResolutionCheckBox->AutoSize = true;
			this->applyResolutionCheckBox->Checked = true;
			this->applyResolutionCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->applyResolutionCheckBox->Location = System::Drawing::Point(180, 14);
			this->applyResolutionCheckBox->Name = L"applyResolutionCheckBox";
			this->applyResolutionCheckBox->Size = System::Drawing::Size(105, 17);
			this->applyResolutionCheckBox->TabIndex = 3;
			this->applyResolutionCheckBox->Text = L"Apply Resolution";
			this->applyResolutionCheckBox->UseVisualStyleBackColor = false;
			this->applyResolutionCheckBox->CheckedChanged += gcnew System::EventHandler(this, &ControlButtonsPane::applyResolutionCheckBox_CheckedChanged);
			// 
			// resolutionSigmaLabel
			// 
			this->resolutionSigmaLabel->AutoSize = true;
			this->resolutionSigmaLabel->Location = System::Drawing::Point(91, 14);
			this->resolutionSigmaLabel->Name = L"resolutionSigmaLabel";
			this->resolutionSigmaLabel->Size = System::Drawing::Size(36, 13);
			this->resolutionSigmaLabel->TabIndex = 1;
			this->resolutionSigmaLabel->Text = L"Sigma";
			this->resolutionSigmaLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// resolutionSigmaTextBox
			// 
			this->resolutionSigmaTextBox->Location = System::Drawing::Point(131, 12);
			this->resolutionSigmaTextBox->Name = L"resolutionSigmaTextBox";
			this->resolutionSigmaTextBox->Size = System::Drawing::Size(45, 20);
			this->resolutionSigmaTextBox->TabIndex = 4;
			this->resolutionSigmaTextBox->Text = L"0.02";
			this->resolutionSigmaTextBox->TextChanged += gcnew System::EventHandler(this, &ControlButtonsPane::resolutionSigmaTextBox_TextChanged);
			this->resolutionSigmaTextBox->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &ControlButtonsPane::resolutionSigmaTextBox_Validating);

			//
			//ControlButtonsPane
			//
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->AutoScroll = true;
			this->ClientSize = System::Drawing::Size(300, 293);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->HideOnClose = true;
			//this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));//1q2w3e4r5t6y7u123456789987654321
			this->Name = L"ControlButtonsPane";
			this->ShowIcon = false;
			this->Text = L"Controls";
			this->ResumeLayout(false);
			this->Controls->Add(this->generateButton);
			this->Controls->Add(this->stopButton);
			this->Controls->Add(this->fitButton);

			this->Controls->Add(this->applyResolutionCheckBox);
			this->Controls->Add(this->resolutionSigmaLabel);
			this->Controls->Add(this->resolutionSigmaTextBox);
			
		}
#pragma endregion

	public:
		bool validate_qs;
		double prev_sigma;

	private: System::Void generateButton_Click(System::Object^  sender, System::EventArgs^  e);
	private: System::Void fitButton_Click(System::Object^  sender, System::EventArgs^  e);
	private: System::Void stopButton_Click(System::Object^  sender, System::EventArgs^  e);

	private: System::Void resolutionSigmaTextBox_TextChanged(System::Object^  sender, System::EventArgs ^ e);
	private: System::Void resolutionSigmaTextBox_Validating(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e);
	private: System::Void applyResolutionCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e);

	public:	 System::Void SetDefaultParams();


	};
}
