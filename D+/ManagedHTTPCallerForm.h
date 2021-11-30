#pragma once
class BackendCall;

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Net;
	using namespace System::Text;
	using namespace System::IO;
	using namespace System::Threading;
	using namespace::System::Runtime::InteropServices;
	using namespace Newtonsoft::Json;
	using namespace Newtonsoft::Json::Linq;
	using namespace System::Security::Cryptography;
	using namespace System::Collections::Generic;


	/// <summary>
	/// Summary for ManagedHTTPCallerForm
	/// </summary>
	public ref class ManagedHTTPCallerForm : public System::Windows::Forms::Form
	{
	public:
		delegate void CallBackendDelegate(BackendCall &call, bool runInBackground);


		ManagedHTTPCallerForm(String ^ serveraddress, String ^ code)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			_callBackendDelegate = gcnew CallBackendDelegate(this, &DPlus::ManagedHTTPCallerForm::PerformCall);
			setBaseUrl(serveraddress); //gcnew String("http://localhost:8000/"); /**/ /*http://10.0.0.4:8000/");/**/
			token = code;//"a42ebef6c125dac4f7b081888d2c84b60ddfed13";
			Guid g;
			sessionID = g.NewGuid().ToString();
		}

	CallBackendDelegate ^GetDelegate(){ return _callBackendDelegate; }

	private: System::Windows::Forms::Button^  cancelButton;
	private: System::Windows::Forms::Button^  restartButton;

	private: System::Windows::Forms::LinkLabel^  serverLabel;
	public: 
		delegate void cancelClickEventHandler(System::Object^  sender, System::EventArgs^  e);
		event cancelClickEventHandler ^ cancelButtonClicked;
		delegate void restartClickEventHandler(System::Object^  sender, System::EventArgs^  e);
		event restartClickEventHandler ^ restartButtonClicked;
		delegate void severlabelHandler(System::Object^  sender, System::EventArgs^  e);
		event severlabelHandler ^ serverLabelClicked;

	public: void setBaseUrl(String ^ url){ base_url = url + "api/v1/"; }
	public: void setToken(String ^ code){ token = code; }
	private: System::Windows::Forms::Button^  retryButton;
	
	private: String ^ username;
	private: String ^ password;
	private: String ^ token;
	private: String ^ sessionID;
	public: String ^ getSessionID(){ return sessionID;}
	public: void setSessionID(String ^ id){ sessionID = id; }
	private: bool tokenfailed;

	protected:
		BackendCall * class_call;

		void PerformCall(BackendCall &call, bool runInBackground);
	private: System::Windows::Forms::Label^  currentlydoing;
	protected:

	private: System::Windows::Forms::Label^  errorLabel;

			 //private: System::Timers::Timer^ _statusPollingTimer;
	private: System::Windows::Forms::ProgressBar^  progressBar1;

	protected:
		CallBackendDelegate ^_callBackendDelegate;
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ManagedHTTPCallerForm()
		{
			if (components)
			{
				delete components;
			}
		}

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ManagedHTTPCallerForm::typeid));
			this->currentlydoing = (gcnew System::Windows::Forms::Label());
			this->errorLabel = (gcnew System::Windows::Forms::Label());
			this->progressBar1 = (gcnew System::Windows::Forms::ProgressBar());
			this->cancelButton = (gcnew System::Windows::Forms::Button());
			this->retryButton = (gcnew System::Windows::Forms::Button());
			this->serverLabel = (gcnew System::Windows::Forms::LinkLabel());
			this->restartButton = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// currentlydoing
			// 
			this->currentlydoing->AutoSize = true;
			this->currentlydoing->Location = System::Drawing::Point(95, 38);
			this->currentlydoing->Name = L"currentlydoing";
			this->currentlydoing->Size = System::Drawing::Size(92, 13);
			this->currentlydoing->TabIndex = 0;
			this->currentlydoing->Text = L"Contacting Server";
			// 
			// errorLabel
			// 
			this->errorLabel->AutoSize = true;
			this->errorLabel->Location = System::Drawing::Point(12, 9);
			this->errorLabel->MaximumSize = System::Drawing::Size(250, 0);
			this->errorLabel->MinimumSize = System::Drawing::Size(250, 0);
			this->errorLabel->Name = L"errorLabel";
			this->errorLabel->Size = System::Drawing::Size(250, 13);
			this->errorLabel->TabIndex = 1;
			this->errorLabel->Text = L"errormsg";
			this->errorLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			this->errorLabel->Visible = false;
			// 
			// progressBar1
			// 
			this->progressBar1->Location = System::Drawing::Point(59, 71);
			this->progressBar1->Name = L"progressBar1";
			this->progressBar1->Size = System::Drawing::Size(167, 23);
			this->progressBar1->Style = System::Windows::Forms::ProgressBarStyle::Marquee;
			this->progressBar1->TabIndex = 2;
			// 
			// cancelButton
			// 
			this->cancelButton->Location = System::Drawing::Point(15, 71);
			this->cancelButton->Name = L"cancelButton";
			this->cancelButton->Size = System::Drawing::Size(75, 23);
			this->cancelButton->TabIndex = 3;
			this->cancelButton->Text = L"Close DPlus";
			this->cancelButton->UseVisualStyleBackColor = true;
			this->cancelButton->Visible = false;
			this->cancelButton->Click += gcnew System::EventHandler(this, &ManagedHTTPCallerForm::cancelButton_Click);
			// 
			// retryButton
			// 
			this->retryButton->Cursor = System::Windows::Forms::Cursors::Default;
			this->retryButton->Location = System::Drawing::Point(187, 71);
			this->retryButton->Name = L"retryButton";
			this->retryButton->Size = System::Drawing::Size(75, 23);
			this->retryButton->TabIndex = 5;
			this->retryButton->Text = L"Retry";
			this->retryButton->UseVisualStyleBackColor = true;
			this->retryButton->Visible = false;
			this->retryButton->Click += gcnew System::EventHandler(this, &ManagedHTTPCallerForm::retryButton_Click);
			// 
			// serverLabel
			// 
			this->serverLabel->AutoSize = true;
			this->serverLabel->Location = System::Drawing::Point(78, 101);
			this->serverLabel->Name = L"serverLabel";
			this->serverLabel->Size = System::Drawing::Size(102, 13);
			this->serverLabel->TabIndex = 5;
			this->serverLabel->TabStop = true;
			this->serverLabel->Text = L"Connection Settings";
			this->serverLabel->Visible = false;
			this->serverLabel->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &ManagedHTTPCallerForm::serverLabel_LinkClicked);
			// 
			// restartButton
			// 
			this->restartButton->Location = System::Drawing::Point(98, 71);
			this->restartButton->Name = L"restartButton";
			this->restartButton->Size = System::Drawing::Size(82, 23);
			this->restartButton->TabIndex = 4;
			this->restartButton->Text = L"Restart DPlus";
			this->restartButton->UseVisualStyleBackColor = true;
			this->restartButton->Visible = false;
			this->restartButton->Click += gcnew System::EventHandler(this, &ManagedHTTPCallerForm::restartButton_Click);
			// 
			// ManagedHTTPCallerForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(285, 149);
			this->Controls->Add(this->restartButton);
			this->Controls->Add(this->serverLabel);
			this->Controls->Add(this->retryButton);
			this->Controls->Add(this->cancelButton);
			this->Controls->Add(this->progressBar1);
			this->Controls->Add(this->errorLabel);
			this->Controls->Add(this->currentlydoing);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"ManagedHTTPCallerForm";
			this->FormClosed += gcnew System::Windows::Forms::FormClosedEventHandler(this, &ManagedHTTPCallerForm::ManagedHTTPCallerForm_FormClosed);
			this->Shown += gcnew System::EventHandler(this, &ManagedHTTPCallerForm::On_Shown);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

		void OnDoWork(System::Object ^sender, System::ComponentModel::DoWorkEventArgs ^e);


	private:
		System::Void On_Shown(System::Object^  sender, System::EventArgs^  e);
		void OnRunWorkerCompleted(System::Object ^sender, System::ComponentModel::RunWorkerCompletedEventArgs ^e);
		System::Void cancelButton_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void retryButton_Click(System::Object^  sender, System::EventArgs^  e);
		String ^ httpCall(String ^ _URL, String^ postData, String^ method);
		void DownloadFile(String ^url, BackendCall * call);
		void executeHttp(BackendCall * call);
		void BGWCall();
		Stream ^_WebStream;
		void handleErrors(String ^ e);
		void my_setWindowDisplay(bool error, String ^ job);
		void syncFilesWithServer(BackendCall * call);
		bool checkFiles(BackendCall * call);
		bool uploadFiles();
		Dictionary<String^, String^> missingfiles;
		String ^ base_url;
		String ^ buildModelPtrUrl(String ^ postData, String ^ url);

private: System::Void serverLabel_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e); 
private: System::Void ManagedHTTPCallerForm_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e);

		 bool _ForcedClose = true;
private: System::Void restartButton_Click(System::Object^  sender, System::EventArgs^  e);

};

}