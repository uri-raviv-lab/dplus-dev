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

	/// <summary>
	/// Summary for PreferencesPane
	/// </summary>
	public ref class PreferencesPane : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	public:
		PreferencesPane(MainWindow ^pform)
		{
			InitializeComponent();
			this->integrationMethodComboBox->Items->Clear();
			for (int i = 0; i < OAMethod_SIZE; i++) {
				this->integrationMethodComboBox->Items->Add(gcnew System::String(OAMethodToCString(OAMethod_Enum(i))));
			}

			integrationMethodComboBox->SelectedIndex = 0;
			validate_qs = true;
			Double::TryParse(qMaxTextBox->Text, prev_qmax);
			Double::TryParse(qMinTextBox->Text, prev_qmin);
			parentForm = pform;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~PreferencesPane()
		{
			if (components)
			{
				delete components;
			}
		}
	public: System::Windows::Forms::TextBox^  integIterTextBox;
	private: System::Windows::Forms::Label^  integIterLabel;
	private: System::Windows::Forms::TextBox^  gridSizeTextBox;
	private: System::Windows::Forms::Label^  gridSizeLabel;
	private: System::Windows::Forms::CheckBox^  useGridCheckBox;


	public: System::Windows::Forms::Label^  qMaxLabel;
	public: System::Windows::Forms::TextBox^  qMaxTextBox;
	public: System::Windows::Forms::Label^  genResLabel;
	public: System::Windows::Forms::TextBox^  genResTextBox;
	private: System::Windows::Forms::Label^  convLabel;
	public: System::Windows::Forms::TextBox^  convTextBox;
	public: System::Windows::Forms::TextBox^  updateIntervalMSTextBox;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  label2;
	public: System::Windows::Forms::TrackBar^  lodTrackbar;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::Label^  label4;
	public: System::Windows::Forms::TrackBar^  drawDistTrackbar;
	private: System::Windows::Forms::Label^  integrationMethodTextBox;
	public: System::Windows::Forms::ComboBox^  integrationMethodComboBox;
	public: System::Windows::Forms::TextBox^  qMinTextBox;
	public: System::Windows::Forms::Label^  qMinLabel;

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(PreferencesPane::typeid));
			this->integIterTextBox = (gcnew System::Windows::Forms::TextBox());
			this->integIterLabel = (gcnew System::Windows::Forms::Label());
			this->gridSizeTextBox = (gcnew System::Windows::Forms::TextBox());
			this->gridSizeLabel = (gcnew System::Windows::Forms::Label());
			this->useGridCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->qMaxLabel = (gcnew System::Windows::Forms::Label());
			this->qMaxTextBox = (gcnew System::Windows::Forms::TextBox());
			this->genResLabel = (gcnew System::Windows::Forms::Label());
			this->genResTextBox = (gcnew System::Windows::Forms::TextBox());
			this->convLabel = (gcnew System::Windows::Forms::Label());
			this->convTextBox = (gcnew System::Windows::Forms::TextBox());
			this->updateIntervalMSTextBox = (gcnew System::Windows::Forms::TextBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->lodTrackbar = (gcnew System::Windows::Forms::TrackBar());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->drawDistTrackbar = (gcnew System::Windows::Forms::TrackBar());
			this->integrationMethodTextBox = (gcnew System::Windows::Forms::Label());
			this->integrationMethodComboBox = (gcnew System::Windows::Forms::ComboBox());
			this->qMinTextBox = (gcnew System::Windows::Forms::TextBox());
			this->qMinLabel = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->lodTrackbar))->BeginInit();
			this->groupBox1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->drawDistTrackbar))->BeginInit();
			this->SuspendLayout();
			// 
			// integIterTextBox
			// 
			this->integIterTextBox->Location = System::Drawing::Point(121, 36);
			this->integIterTextBox->Name = L"integIterTextBox";
			this->integIterTextBox->Size = System::Drawing::Size(100, 20);
			this->integIterTextBox->TabIndex = 2;
			this->integIterTextBox->Text = L"1000000";
			// 
			// integIterLabel
			// 
			this->integIterLabel->AutoSize = true;
			this->integIterLabel->Location = System::Drawing::Point(12, 39);
			this->integIterLabel->Name = L"integIterLabel";
			this->integIterLabel->Size = System::Drawing::Size(103, 13);
			this->integIterLabel->TabIndex = 1;
			this->integIterLabel->Text = L"Integration Iterations";
			// 
			// gridSizeTextBox
			// 
			this->gridSizeTextBox->Location = System::Drawing::Point(103, 72);
			this->gridSizeTextBox->Name = L"gridSizeTextBox";
			this->gridSizeTextBox->Size = System::Drawing::Size(100, 20);
			this->gridSizeTextBox->TabIndex = 4;
			this->gridSizeTextBox->Text = L"80";
			this->gridSizeTextBox->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &PreferencesPane::gridSizeTextBox_Validating);
			// 
			// gridSizeLabel
			// 
			this->gridSizeLabel->AutoSize = true;
			this->gridSizeLabel->Location = System::Drawing::Point(15, 75);
			this->gridSizeLabel->Name = L"gridSizeLabel";
			this->gridSizeLabel->Size = System::Drawing::Size(49, 13);
			this->gridSizeLabel->TabIndex = 1;
			this->gridSizeLabel->Text = L"Grid Size";
			// 
			// useGridCheckBox
			// 
			this->useGridCheckBox->AutoSize = true;
			this->useGridCheckBox->Checked = true;
			this->useGridCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->useGridCheckBox->Location = System::Drawing::Point(15, 55);
			this->useGridCheckBox->Name = L"useGridCheckBox";
			this->useGridCheckBox->Size = System::Drawing::Size(67, 17);
			this->useGridCheckBox->TabIndex = 3;
			this->useGridCheckBox->Text = L"Use Grid";
			this->useGridCheckBox->UseVisualStyleBackColor = true;
			this->useGridCheckBox->CheckedChanged += gcnew System::EventHandler(this, &PreferencesPane::usGridCheckBox_CheckedChanged);
			// 
			// qMaxLabel
			// 
			this->qMaxLabel->AutoSize = true;
			this->qMaxLabel->Location = System::Drawing::Point(15, 127);
			this->qMaxLabel->Name = L"qMaxLabel";
			this->qMaxLabel->Size = System::Drawing::Size(36, 13);
			this->qMaxLabel->TabIndex = 1;
			this->qMaxLabel->Text = L"q Max";
			this->qMaxLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// qMaxTextBox
			// 
			this->qMaxTextBox->Location = System::Drawing::Point(103, 124);
			this->qMaxTextBox->Name = L"qMaxTextBox";
			this->qMaxTextBox->Size = System::Drawing::Size(100, 20);
			this->qMaxTextBox->TabIndex = 5;
			this->qMaxTextBox->Text = L"7.5";
			this->qMaxTextBox->TextChanged += gcnew System::EventHandler(this, &PreferencesPane::qMaxTextBox_TextChanged);
			// 
			// genResLabel
			// 
			this->genResLabel->AutoSize = true;
			this->genResLabel->Location = System::Drawing::Point(15, 153);
			this->genResLabel->Name = L"genResLabel";
			this->genResLabel->Size = System::Drawing::Size(88, 13);
			this->genResLabel->TabIndex = 1;
			this->genResLabel->Text = L"Generated points";
			// 
			// genResTextBox
			// 
			this->genResTextBox->Location = System::Drawing::Point(103, 151);
			this->genResTextBox->Name = L"genResTextBox";
			this->genResTextBox->Size = System::Drawing::Size(100, 20);
			this->genResTextBox->TabIndex = 6;
			this->genResTextBox->Text = L"800";
			// 
			// convLabel
			// 
			this->convLabel->AutoSize = true;
			this->convLabel->Location = System::Drawing::Point(15, 176);
			this->convLabel->Name = L"convLabel";
			this->convLabel->Size = System::Drawing::Size(71, 13);
			this->convLabel->TabIndex = 1;
			this->convLabel->Text = L"Convergence";
			// 
			// convTextBox
			// 
			this->convTextBox->Location = System::Drawing::Point(103, 176);
			this->convTextBox->Name = L"convTextBox";
			this->convTextBox->Size = System::Drawing::Size(100, 20);
			this->convTextBox->TabIndex = 7;
			this->convTextBox->Text = L"0.001";
			// 
			// updateIntervalMSTextBox
			// 
			this->updateIntervalMSTextBox->Location = System::Drawing::Point(104, 205);
			this->updateIntervalMSTextBox->Name = L"updateIntervalMSTextBox";
			this->updateIntervalMSTextBox->Size = System::Drawing::Size(74, 20);
			this->updateIntervalMSTextBox->TabIndex = 8;
			this->updateIntervalMSTextBox->Text = L"100";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(15, 205);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(80, 13);
			this->label1->TabIndex = 6;
			this->label1->Text = L"Update Interval";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(184, 205);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(20, 13);
			this->label2->TabIndex = 8;
			this->label2->Text = L"ms";
			// 
			// lodTrackbar
			// 
			this->lodTrackbar->Location = System::Drawing::Point(86, 23);
			this->lodTrackbar->Maximum = 5;
			this->lodTrackbar->Minimum = 1;
			this->lodTrackbar->Name = L"lodTrackbar";
			this->lodTrackbar->Size = System::Drawing::Size(104, 45);
			this->lodTrackbar->TabIndex = 9;
			this->lodTrackbar->Value = 1;
			this->lodTrackbar->Scroll += gcnew System::EventHandler(this, &PreferencesPane::lodTrackbar_Scroll);
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(5, 26);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(75, 13);
			this->label3->TabIndex = 10;
			this->label3->Text = L"Level of Detail";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->label4);
			this->groupBox1->Controls->Add(this->drawDistTrackbar);
			this->groupBox1->Controls->Add(this->label3);
			this->groupBox1->Controls->Add(this->lodTrackbar);
			this->groupBox1->Location = System::Drawing::Point(15, 228);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(200, 100);
			this->groupBox1->TabIndex = 11;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Graphics:";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(6, 67);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(77, 13);
			this->label4->TabIndex = 12;
			this->label4->Text = L"Draw Distance";
			// 
			// drawDistTrackbar
			// 
			this->drawDistTrackbar->LargeChange = 50;
			this->drawDistTrackbar->Location = System::Drawing::Point(87, 58);
			this->drawDistTrackbar->Maximum = 200;
			this->drawDistTrackbar->Minimum = 1;
			this->drawDistTrackbar->Name = L"drawDistTrackbar";
			this->drawDistTrackbar->Size = System::Drawing::Size(104, 45);
			this->drawDistTrackbar->TabIndex = 11;
			this->drawDistTrackbar->TickStyle = System::Windows::Forms::TickStyle::None;
			this->drawDistTrackbar->Value = 200;
			this->drawDistTrackbar->Scroll += gcnew System::EventHandler(this, &PreferencesPane::drawDistTrack_Scroll);
			// 
			// integrationMethodTextBox
			// 
			this->integrationMethodTextBox->AutoSize = true;
			this->integrationMethodTextBox->Location = System::Drawing::Point(12, 14);
			this->integrationMethodTextBox->Name = L"integrationMethodTextBox";
			this->integrationMethodTextBox->Size = System::Drawing::Size(96, 13);
			this->integrationMethodTextBox->TabIndex = 1;
			this->integrationMethodTextBox->Text = L"Integration Method";
			// 
			// integrationMethodComboBox
			// 
			this->integrationMethodComboBox->DropDownWidth = 175;
			this->integrationMethodComboBox->FormattingEnabled = true;
			this->integrationMethodComboBox->Items->AddRange(gcnew cli::array< System::Object^  >(4) {
				L"Monte Carlo", L"Adaptive Gauss Kronrod",
					L"Cubic Spline (Unused)", L"Direct Computation - MC"
			});
			this->integrationMethodComboBox->Location = System::Drawing::Point(114, 11);
			this->integrationMethodComboBox->Name = L"integrationMethodComboBox";
			this->integrationMethodComboBox->Size = System::Drawing::Size(175, 21);
			this->integrationMethodComboBox->TabIndex = 1;
			// 
			// qMinTextBox
			// 
			this->qMinTextBox->Location = System::Drawing::Point(103, 98);
			this->qMinTextBox->Name = L"qMinTextBox";
			this->qMinTextBox->Size = System::Drawing::Size(100, 20);
			this->qMinTextBox->TabIndex = 12;
			this->qMinTextBox->Text = L"0";
			this->qMinTextBox->TextChanged += gcnew System::EventHandler(this, &PreferencesPane::qMinTextBox_TextChanged);
			// 
			// qMinLabel
			// 
			this->qMinLabel->AutoSize = true;
			this->qMinLabel->Location = System::Drawing::Point(15, 101);
			this->qMinLabel->Name = L"qMinLabel";
			this->qMinLabel->Size = System::Drawing::Size(33, 13);
			this->qMinLabel->TabIndex = 13;
			this->qMinLabel->Text = L"q Min";
			this->qMinLabel->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// PreferencesPane
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->AutoScroll = true;
			this->ClientSize = System::Drawing::Size(431, 452);
			this->Controls->Add(this->qMinLabel);
			this->Controls->Add(this->qMinTextBox);
			this->Controls->Add(this->integrationMethodComboBox);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->updateIntervalMSTextBox);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->useGridCheckBox);
			this->Controls->Add(this->convTextBox);
			this->Controls->Add(this->convLabel);
			this->Controls->Add(this->genResTextBox);
			this->Controls->Add(this->genResLabel);
			this->Controls->Add(this->qMaxTextBox);
			this->Controls->Add(this->qMaxLabel);
			this->Controls->Add(this->gridSizeTextBox);
			this->Controls->Add(this->gridSizeLabel);
			this->Controls->Add(this->integrationMethodTextBox);
			this->Controls->Add(this->integIterLabel);
			this->Controls->Add(this->integIterTextBox);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::SizableToolWindow;
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"PreferencesPane";
			this->ShowIcon = false;
			this->Text = L"Preferences";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->lodTrackbar))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->drawDistTrackbar))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	private: System::Void usGridCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
	public:
		paramStruct GetDomainPreferences();
		String ^SerializePreferences();
		void DeserializePreferences(LuaTable ^contents);
		void SetDefaultParams();
		// Helper function(s)
		delegate String ^FuncNoParamsReturnString();
		bool validate_qs;
		double prev_qmin;
		double prev_qmax;
	private: System::Void drawDistTrack_Scroll(System::Object^  sender, System::EventArgs^  e);
	private: System::Void lodTrackbar_Scroll(System::Object^  sender, System::EventArgs^  e);
	private: System::Void gridSizeTextBox_Validating(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e);
	private: System::Void qMinTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e);
    private: System::Void qMaxTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e);
};

}