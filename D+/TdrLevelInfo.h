#pragma once

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for TdrLevelInfo
	/// </summary>
	public ref class TdrLevelInfo : public System::Windows::Forms::Form
	{
	public:
		TdrLevelInfo(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			useCPU = false;
			retry = false;
			this->textBoxMessage->Text = "Your GPU is not configured properly for running D+.";
			this->linkLabel->Text = "Explanation of how to configure your GPU";
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~TdrLevelInfo()
		{
			if (components)
			{
				delete components;
			}
		}
	protected: System::Windows::Forms::Button^  buttonRetry;
	protected: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanelMain;
	protected: System::Windows::Forms::TableLayoutPanel^  tableLayoutPanelButtons;

	protected: System::Windows::Forms::Button^  buttonUseCPU;

	protected: System::Windows::Forms::TextBox^  textBoxMessage;
	public: bool useCPU, retry;
	private: System::Windows::Forms::LinkLabel^  linkLabel;
	public:
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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(TdrLevelInfo::typeid));
			this->buttonRetry = (gcnew System::Windows::Forms::Button());
			this->tableLayoutPanelMain = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->tableLayoutPanelButtons = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->buttonUseCPU = (gcnew System::Windows::Forms::Button());
			this->textBoxMessage = (gcnew System::Windows::Forms::TextBox());
			this->linkLabel = (gcnew System::Windows::Forms::LinkLabel());
			this->tableLayoutPanelMain->SuspendLayout();
			this->tableLayoutPanelButtons->SuspendLayout();
			this->SuspendLayout();
			// 
			// buttonRetry
			// 
			this->buttonRetry->AutoSize = true;
			this->buttonRetry->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->buttonRetry->Dock = System::Windows::Forms::DockStyle::Fill;
			this->buttonRetry->Location = System::Drawing::Point(139, 3);
			this->buttonRetry->Name = L"buttonRetry";
			this->buttonRetry->Size = System::Drawing::Size(130, 28);
			this->buttonRetry->TabIndex = 2;
			this->buttonRetry->Text = L"Retry";
			this->buttonRetry->UseVisualStyleBackColor = true;
			this->buttonRetry->Click += gcnew System::EventHandler(this, &TdrLevelInfo::buttonRetry_Click);
			// 
			// tableLayoutPanelMain
			// 
			this->tableLayoutPanelMain->AutoSize = true;
			this->tableLayoutPanelMain->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanelMain->ColumnCount = 1;
			this->tableLayoutPanelMain->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				100)));
			this->tableLayoutPanelMain->Controls->Add(this->tableLayoutPanelButtons, 0, 2);
			this->tableLayoutPanelMain->Controls->Add(this->textBoxMessage, 0, 0);
			this->tableLayoutPanelMain->Controls->Add(this->linkLabel, 0, 1);
			this->tableLayoutPanelMain->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanelMain->Location = System::Drawing::Point(0, 0);
			this->tableLayoutPanelMain->Name = L"tableLayoutPanelMain";
			this->tableLayoutPanelMain->RowCount = 4;
			this->tableLayoutPanelMain->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
				40)));
			this->tableLayoutPanelMain->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
				30)));
			this->tableLayoutPanelMain->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
				25)));
			this->tableLayoutPanelMain->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
				5)));
			this->tableLayoutPanelMain->Size = System::Drawing::Size(272, 139);
			this->tableLayoutPanelMain->TabIndex = 3;
			// 
			// tableLayoutPanelButtons
			// 
			this->tableLayoutPanelButtons->AutoSize = true;
			this->tableLayoutPanelButtons->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->tableLayoutPanelButtons->ColumnCount = 2;
			this->tableLayoutPanelButtons->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanelButtons->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanelButtons->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Absolute,
				20)));
			this->tableLayoutPanelButtons->Controls->Add(this->buttonUseCPU, 0, 0);
			this->tableLayoutPanelButtons->Controls->Add(this->buttonRetry, 1, 0);
			this->tableLayoutPanelButtons->Dock = System::Windows::Forms::DockStyle::Fill;
			this->tableLayoutPanelButtons->Location = System::Drawing::Point(0, 96);
			this->tableLayoutPanelButtons->Margin = System::Windows::Forms::Padding(0);
			this->tableLayoutPanelButtons->Name = L"tableLayoutPanelButtons";
			this->tableLayoutPanelButtons->RowCount = 1;
			this->tableLayoutPanelButtons->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent,
				100)));
			this->tableLayoutPanelButtons->Size = System::Drawing::Size(272, 34);
			this->tableLayoutPanelButtons->TabIndex = 0;
			// 
			// buttonUseCPU
			// 
			this->buttonUseCPU->AutoSize = true;
			this->buttonUseCPU->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->buttonUseCPU->Dock = System::Windows::Forms::DockStyle::Fill;
			this->buttonUseCPU->Location = System::Drawing::Point(3, 3);
			this->buttonUseCPU->Name = L"buttonUseCPU";
			this->buttonUseCPU->Size = System::Drawing::Size(130, 28);
			this->buttonUseCPU->TabIndex = 3;
			this->buttonUseCPU->Text = L"Use CPU";
			this->buttonUseCPU->UseVisualStyleBackColor = true;
			this->buttonUseCPU->Click += gcnew System::EventHandler(this, &TdrLevelInfo::buttonUseCPU_Click);
			// 
			// textBoxMessage
			// 
			this->textBoxMessage->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->textBoxMessage->Dock = System::Windows::Forms::DockStyle::Fill;
			this->textBoxMessage->Location = System::Drawing::Point(3, 30);
			this->textBoxMessage->Margin = System::Windows::Forms::Padding(3, 30, 3, 3);
			this->textBoxMessage->Multiline = true;
			this->textBoxMessage->Name = L"textBoxMessage";
			this->textBoxMessage->ReadOnly = true;
			this->textBoxMessage->Size = System::Drawing::Size(266, 22);
			this->textBoxMessage->TabIndex = 1;
			// 
			// linkLabel
			// 
			this->linkLabel->AutoSize = true;
			this->linkLabel->Location = System::Drawing::Point(3, 58);
			this->linkLabel->Margin = System::Windows::Forms::Padding(3);
			this->linkLabel->Name = L"linkLabel";
			this->linkLabel->Size = System::Drawing::Size(49, 13);
			this->linkLabel->TabIndex = 2;
			this->linkLabel->TabStop = true;
			this->linkLabel->Text = L"linkLabel";
			this->linkLabel->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &TdrLevelInfo::linkLabel_LinkClicked);
			// 
			// TdrLevelInfo
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(272, 139);
			this->Controls->Add(this->tableLayoutPanelMain);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"TdrLevelInfo";
			this->Text = L"Tdr Level Info";
			this->VisibleChanged += gcnew System::EventHandler(this, &TdrLevelInfo::TdrLevelInfo_VisibleChanged);
			this->tableLayoutPanelMain->ResumeLayout(false);
			this->tableLayoutPanelMain->PerformLayout();
			this->tableLayoutPanelButtons->ResumeLayout(false);
			this->tableLayoutPanelButtons->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private:
		System::Void buttonUseCPU_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void buttonRetry_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void TdrLevelInfo_VisibleChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void linkLabel_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e);
		void VisitLink();
};
}
