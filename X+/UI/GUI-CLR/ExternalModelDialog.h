#pragma once

#include "clrfunctionality.h"	//includes "FrontendExported.h"
//#include "calculation_external.h" // For FFModel
//#include "FrontendExported.h"
#include "ModelUI.h"
#include "LocalComm.h"

// TODO::ChangeModel
// Need to either replace or add access to Model and FFModel
// 

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

using namespace System::Reflection;

namespace GUICLR {

	typedef int (__cdecl *numModelsFunc)(void);
	typedef std::string (__cdecl *modelNameFunc)(int index);
	//typedef Model* (__cdecl *getModelFunc)(int index, ProfileShape shape);

	/// <summary>
	/// Summary for ExternalModelDialog
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class ExternalModelDialog : public System::Windows::Forms::Form
	{
	protected:
		//FFModel *_selectedModel;
		//ModelUI *_selectedModel;
		ModelInformation *_selectedModel;
		System::String ^filename;

		int cats;
		std::vector<ModelInformation> *modelInfos;
		FrontendComm *lf;
		wchar_t *container;

		// DLL handle
		HMODULE hModule;
		
		// Function handles
		// TODO::handles numModelsFunc NumModels;
		// TODO::handles modelNameFunc ModelName;
		// TODO::handles getModelFunc  GetModelF;

		// Adds all models to the combobox
		bool GetAllModels();
	public:
		
		ExternalModelDialog()
		{
			_selectedModel = NULL;
			lf = new LocalFrontend();
			modelInfos = new std::vector<ModelInformation>;
			container = new wchar_t[MAX_PATH];

			// Initialize UI
			InitializeComponent();	
		}

		void ClearModelSelection() {
			_selectedModel = NULL;
		}

		void GetContainer(wchar_t *res) {
			wcsncpy(res, container, MAX_PATH);
		}

		ModelInformation GetSelectedModel() {
			return *_selectedModel;
		}

		System::String ^GetSelectedModelName() {
			if(!_selectedModel)
				return "N/A";

			return stringToClr(string(_selectedModel->name));
		}
	
		void LoadDefaultModels();

		bool ChooseModelContainer() {
			return ChooseModelContainer(L"");
		}

		bool ChooseModelContainer(System::String^ path) {
			if(path->Length > 0) {
				//filename = path;
				clrToWchar_tp(path, container);
				//container = clrToWstring(path).c_str();
				return GetAllModels();
			}

			OpenFileDialog ^ofd = gcnew OpenFileDialog();
			ofd->Title = "Choose a model container...";
			ofd->Filter = "Model Containers (*.dll)|*.dll|All Files|*.*";

			if(ofd->ShowDialog() != System::Windows::Forms::DialogResult::Cancel) {
				filename = ofd->FileName;
				clrToWchar_tp(path, container);
// 				container = clrToWstring(filename).c_str();
				return GetAllModels();
			} else
				return false;
		}

		bool LoadModelContainer();
		void FreeModelContainer();

private: 
		System::Void changeFile_Click(System::Object^  sender, System::EventArgs^  e) {
			ChooseModelContainer();
		}
		
		System::Void ok_Click(System::Object^  sender, System::EventArgs^  e) {			
			this->DialogResult = ::DialogResult::OK;
			Close();
		}
		
		System::Void cancel_Click(System::Object^  sender, System::EventArgs^  e) {
			if(_selectedModel)
				delete _selectedModel;
			_selectedModel = NULL;

			Close();
		}
		
		System::Void models_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ExternalModelDialog()
		{
			if(modelInfos)
				delete modelInfos;

			if(lf)
				delete lf;

			if(container) {
				delete container;
			}

			if (components)
			{
				delete components;
			}

			FreeModelContainer();
		}
	private: System::Windows::Forms::ComboBox^  models;
	protected: 

	private: System::Windows::Forms::Button^  cancel;
	private: System::Windows::Forms::Button^  ok;
	private: System::Windows::Forms::Button^  changeFile;
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
			this->models = (gcnew System::Windows::Forms::ComboBox());
			this->cancel = (gcnew System::Windows::Forms::Button());
			this->ok = (gcnew System::Windows::Forms::Button());
			this->changeFile = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// models
			// 
			this->models->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->models->Enabled = false;
			this->models->FormattingEnabled = true;
			this->models->Location = System::Drawing::Point(12, 12);
			this->models->Name = L"models";
			this->models->Size = System::Drawing::Size(268, 21);
			this->models->TabIndex = 0;
			this->models->SelectedIndexChanged += gcnew System::EventHandler(this, &ExternalModelDialog::models_SelectedIndexChanged);
			// 
			// cancel
			// 
			this->cancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->cancel->Location = System::Drawing::Point(205, 54);
			this->cancel->Name = L"cancel";
			this->cancel->Size = System::Drawing::Size(75, 23);
			this->cancel->TabIndex = 1;
			this->cancel->Text = L"Cancel";
			this->cancel->UseVisualStyleBackColor = true;
			this->cancel->Click += gcnew System::EventHandler(this, &ExternalModelDialog::cancel_Click);
			// 
			// ok
			// 
			this->ok->Enabled = false;
			this->ok->Location = System::Drawing::Point(124, 54);
			this->ok->Name = L"ok";
			this->ok->Size = System::Drawing::Size(75, 23);
			this->ok->TabIndex = 2;
			this->ok->Text = L"OK";
			this->ok->UseVisualStyleBackColor = true;
			this->ok->Click += gcnew System::EventHandler(this, &ExternalModelDialog::ok_Click);
			// 
			// changeFile
			// 
			this->changeFile->Location = System::Drawing::Point(12, 54);
			this->changeFile->Name = L"changeFile";
			this->changeFile->Size = System::Drawing::Size(106, 23);
			this->changeFile->TabIndex = 3;
			this->changeFile->Text = L"Change File...";
			this->changeFile->UseVisualStyleBackColor = true;
			this->changeFile->Click += gcnew System::EventHandler(this, &ExternalModelDialog::changeFile_Click);
			// 
			// ExternalModelDialog
			// 
			this->AcceptButton = this->ok;
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->cancel;
			this->ClientSize = System::Drawing::Size(292, 89);
			this->Controls->Add(this->changeFile);
			this->Controls->Add(this->ok);
			this->Controls->Add(this->cancel);
			this->Controls->Add(this->models);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Name = L"ExternalModelDialog";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"Choose External Model";
			this->ResumeLayout(false);

		}
#pragma endregion
};
}
