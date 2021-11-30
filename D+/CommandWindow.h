#pragma once

#include "MainWindow.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::Collections::Generic;
using namespace WeifenLuo::WinFormsUI::Docking;
using namespace LuaInterface;


namespace DPlus {	

	/// <summary>
	/// Summary for CommandWindow
	/// </summary>
	public ref class CommandWindow : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	private: ScintillaNET::Scintilla^  scriptBox;	
	private: System::ComponentModel::BackgroundWorker^  scriptRunner;
	protected: 
		Lua ^lua;
		bool bStopRequired, bScriptRunning;
		
		static const int HISTORY_SIZE = 100;
		List<String ^> ^commandHistory;
		int curHistIndex;

	private: System::Windows::Forms::Label^  labelPrompt;
	private: ScintillaNET::Scintilla^  commandBox;
		
	public:
		CommandWindow(MainWindow ^pform, Lua ^ctx);

		void LuaPrint(String ^message);

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~CommandWindow()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::ComponentModel::IContainer^  components;
	protected: 

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(CommandWindow::typeid));
			this->scriptBox = (gcnew ScintillaNET::Scintilla());
			this->scriptRunner = (gcnew System::ComponentModel::BackgroundWorker());
			this->labelPrompt = (gcnew System::Windows::Forms::Label());
			this->commandBox = (gcnew ScintillaNET::Scintilla());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->scriptBox))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->commandBox))->BeginInit();
			this->SuspendLayout();
			// 
			// scriptBox
			// 
			this->scriptBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->scriptBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->scriptBox->ConfigurationManager->Language = L"lua";
			this->scriptBox->Indentation->TabWidth = 4;
			this->scriptBox->IsReadOnly = true;
			this->scriptBox->Lexing->Lexer = ScintillaNET::Lexer::Lua;
			this->scriptBox->Lexing->LexerName = L"lua";
			this->scriptBox->Lexing->LineCommentPrefix = L"";
			this->scriptBox->Lexing->StreamCommentPrefix = L"";
			this->scriptBox->Lexing->StreamCommentSufix = L"";
			this->scriptBox->Location = System::Drawing::Point(0, 0);
			this->scriptBox->Margins->Margin1->AutoToggleMarkerNumber = 0;
			this->scriptBox->Margins->Margin1->Width = 0;
			this->scriptBox->Name = L"scriptBox";
			this->scriptBox->Size = System::Drawing::Size(600, 122);
			this->scriptBox->TabIndex = 1;
			this->scriptBox->UndoRedo->IsUndoEnabled = false;
			// 
			// scriptRunner
			// 
			this->scriptRunner->WorkerSupportsCancellation = true;
			this->scriptRunner->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &CommandWindow::scriptRunner_DoWork);
			this->scriptRunner->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &CommandWindow::scriptRunner_RunWorkerCompleted);
			// 
			// labelPrompt
			// 
			this->labelPrompt->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->labelPrompt->AutoSize = true;
			this->labelPrompt->Location = System::Drawing::Point(-1, 130);
			this->labelPrompt->Name = L"labelPrompt";
			this->labelPrompt->Size = System::Drawing::Size(16, 13);
			this->labelPrompt->TabIndex = 2;
			this->labelPrompt->Text = L"> ";
			// 
			// commandBox
			// 
			this->commandBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->commandBox->BorderStyle = System::Windows::Forms::BorderStyle::None;
			this->commandBox->ConfigurationManager->Language = L"lua";
			this->commandBox->Indentation->TabWidth = 4;
			this->commandBox->Lexing->Lexer = ScintillaNET::Lexer::Lua;
			this->commandBox->Lexing->LexerName = L"lua";
			this->commandBox->Lexing->LineCommentPrefix = L"";
			this->commandBox->Lexing->StreamCommentPrefix = L"";
			this->commandBox->Lexing->StreamCommentSufix = L"";
			this->commandBox->Location = System::Drawing::Point(19, 129);
			this->commandBox->Margins->Margin1->AutoToggleMarkerNumber = 0;
			this->commandBox->Margins->Margin1->Width = 0;
			this->commandBox->Name = L"commandBox";
			this->commandBox->Scrolling->HorizontalScrollTracking = false;
			this->commandBox->Scrolling->ScrollBars = System::Windows::Forms::ScrollBars::None;
			this->commandBox->Size = System::Drawing::Size(579, 24);
			this->commandBox->TabIndex = 0;
			this->commandBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &CommandWindow::commandBox_KeyDown);
			// 
			// CommandWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::White;
			this->ClientSize = System::Drawing::Size(600, 146);
			this->Controls->Add(this->commandBox);
			this->Controls->Add(this->labelPrompt);
			this->Controls->Add(this->scriptBox);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"CommandWindow";
			this->ShowIcon = false;
			this->Text = L"Command Window";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->scriptBox))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->commandBox))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		 public:
			 void LuaLineHook(System::Object^  sender, LuaInterface::DebugHookEventArgs ^args);
			 void HookExceptionForwarder(System::Object^  sender, LuaInterface::HookExceptionEventArgs ^args);
private: System::Void scriptRunner_RunWorkerCompleted(System::Object^  sender, System::ComponentModel::RunWorkerCompletedEventArgs^  e);
private: System::Void scriptRunner_DoWork(System::Object^  sender, System::ComponentModel::DoWorkEventArgs^  e);
		 void PerformAction();
private: System::Void commandBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
};
}
