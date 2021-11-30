#pragma once
#include "..\frontend_version.h"

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for AboutDPlus
	/// </summary>
	public ref class AboutDPlus : public System::Windows::Forms::Form
	{
	public:
		AboutDPlus(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			versionLabel->Text += FRONTEND_VERSION_DECIMAL;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~AboutDPlus()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  versionLabel;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label2;
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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(AboutDPlus::typeid));
			this->versionLabel = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// label1
			// 
			this->versionLabel->AutoSize = true;
			this->versionLabel->Location = System::Drawing::Point(21, 206);
			this->versionLabel->Name = L"label1";
			this->versionLabel->Size = System::Drawing::Size(83, 13);
			this->versionLabel->TabIndex = 0;
			this->versionLabel->Text = L"D+ Version ";
			// 
			// label3
			// 
			this->label3->Location = System::Drawing::Point(21, 21);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(196, 46);
			this->label3->TabIndex = 2;
			this->label3->Text = L"(C) Copyrights for this program belong to Uri Raviv, Avi Ginsburg and Tal Ben-Nun"
				L". All rights reserved.";
			// 
			// label2
			// 
			this->label2->Location = System::Drawing::Point(21, 91);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(251, 105);
			this->label2->TabIndex = 2;
			this->label2->Text = resources->GetString(L"label2.Text");
			// 
			// AboutDPlus
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(284, 262);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->versionLabel);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"AboutDPlus";
			this->Text = L"AboutDPlus";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	};
}
