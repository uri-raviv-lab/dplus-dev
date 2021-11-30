#pragma once

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for ServerConfigForm
	/// </summary>
	public ref class ServerConfigForm : public System::Windows::Forms::Form
	{
	public:

		ServerConfigForm(String ^ server, String ^ code)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			if (server == "" || code == "")
			{
				EnableEdit();
			}
			else
			{
				this->serverAddressTextbox->Text = server;
				this->codeTextbox->Text = code;
				this->serverAddress = server;
				this->validationCode = code;
			}

		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ServerConfigForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  label2;

	private: System::Windows::Forms::TextBox^  serverAddressTextbox;
	private: System::Windows::Forms::TextBox^  codeTextbox;


	private: System::Windows::Forms::Button^  ok_button;

	private: System::Windows::Forms::Label^  errorMessage;
	private: System::Windows::Forms::CheckBox^  EnableEditCheck;








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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ServerConfigForm::typeid));
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->serverAddressTextbox = (gcnew System::Windows::Forms::TextBox());
			this->ok_button = (gcnew System::Windows::Forms::Button());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->codeTextbox = (gcnew System::Windows::Forms::TextBox());
			this->errorMessage = (gcnew System::Windows::Forms::Label());
			this->EnableEditCheck = (gcnew System::Windows::Forms::CheckBox());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(31, 49);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(82, 13);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Server Address:";
			// 
			// serverAddressTextbox
			// 
			this->serverAddressTextbox->Enabled = false;
			this->serverAddressTextbox->Location = System::Drawing::Point(33, 65);
			this->serverAddressTextbox->Name = L"serverAddressTextbox";
			this->serverAddressTextbox->Size = System::Drawing::Size(212, 20);
			this->serverAddressTextbox->TabIndex = 1;
			this->serverAddressTextbox->TextChanged += gcnew System::EventHandler(this, &ServerConfigForm::serverAddressTextbox_TextChanged);
			// 
			// ok_button
			// 
			this->ok_button->Location = System::Drawing::Point(231, 143);
			this->ok_button->Name = L"ok_button";
			this->ok_button->Size = System::Drawing::Size(41, 27);
			this->ok_button->TabIndex = 2;
			this->ok_button->Text = L"OK";
			this->ok_button->UseVisualStyleBackColor = true;
			this->ok_button->Click += gcnew System::EventHandler(this, &ServerConfigForm::ok_button_Click);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(34, 88);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(82, 13);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Activation Code";
			// 
			// codeTextbox
			// 
			this->codeTextbox->Enabled = false;
			this->codeTextbox->Location = System::Drawing::Point(34, 104);
			this->codeTextbox->Name = L"codeTextbox";
			this->codeTextbox->Size = System::Drawing::Size(211, 20);
			this->codeTextbox->TabIndex = 5;
			// 
			// errorMessage
			// 
			this->errorMessage->AutoSize = true;
			this->errorMessage->ForeColor = System::Drawing::Color::Red;
			this->errorMessage->Location = System::Drawing::Point(18, 150);
			this->errorMessage->Name = L"errorMessage";
			this->errorMessage->Size = System::Drawing::Size(35, 13);
			this->errorMessage->TabIndex = 8;
			this->errorMessage->Text = L"label5";
			this->errorMessage->Visible = false;
			// 
			// EnableEditCheck
			// 
			this->EnableEditCheck->AutoSize = true;
			this->EnableEditCheck->Location = System::Drawing::Point(61, 12);
			this->EnableEditCheck->Name = L"EnableEditCheck";
			this->EnableEditCheck->Size = System::Drawing::Size(142, 17);
			this->EnableEditCheck->TabIndex = 9;
			this->EnableEditCheck->Text = L"Enable values for editing";
			this->EnableEditCheck->UseVisualStyleBackColor = true;
			this->EnableEditCheck->CheckedChanged += gcnew System::EventHandler(this, &ServerConfigForm::EnableEditCheck_CheckedChanged);
			// 
			// ServerConfigForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(284, 210);
			this->Controls->Add(this->EnableEditCheck);
			this->Controls->Add(this->errorMessage);
			this->Controls->Add(this->ok_button);
			this->Controls->Add(this->codeTextbox);
			this->Controls->Add(this->serverAddressTextbox);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"ServerConfigForm";
			this->Text = L"Connection Settings";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	public: String ^ serverAddress;
	public: String ^ validationCode;
	private: System::Void ok_button_Click(System::Object^  sender, System::EventArgs^  e);
	private: System::Void setErrorMessage(String ^ msg);
	private: bool checkServerAddress(String ^ address);
			 bool checkCode(String ^ code);

private: System::Void EnableEditCheck_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		 void EnableEdit();
private: System::Void serverAddressTextbox_TextChanged(System::Object^  sender, System::EventArgs^  e);

};
}
