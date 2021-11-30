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
	/// Summary for ScriptPane
	/// </summary>
	public ref class ScriptPane : public WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	private: ScintillaNET::Scintilla^  scriptBox;
	private: System::Windows::Forms::ToolStrip^  toolStrip;
	private: System::Windows::Forms::ToolStrip^  toolStrip1;
	private: System::Windows::Forms::ToolStripButton^  newToolStripButton;
	private: System::Windows::Forms::ToolStripButton^  openToolStripButton;
	private: System::Windows::Forms::ToolStripButton^  saveToolStripButton;

	private: System::Windows::Forms::ToolStripSeparator^  cutToolStripButtonSeparator;
	private: System::Windows::Forms::ToolStripButton^  cutToolStripButton;
	private: System::Windows::Forms::ToolStripButton^  copyToolStripButton;
	private: System::Windows::Forms::ToolStripButton^  pasteToolStripButton;
	private: System::Windows::Forms::ToolStripSeparator^  undoToolStripButtonSeparator;
	private: System::Windows::Forms::ToolStripButton^  undoToolStripButton;
	private: System::Windows::Forms::ToolStripButton^  redoToolStripButton;
	private: ScintillaNET::ToolStripIncrementalSearcher^  toolIncremental;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator1;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
	private: System::Windows::Forms::ToolStripButton^  commentLineButton;
	private: System::Windows::Forms::ToolStripButton^  commentRegionButton;
	private: System::Windows::Forms::ToolStripButton^  uncommentButton;
	private: System::Windows::Forms::ToolStripButton^  breakpointButton;
	private: System::Windows::Forms::Button^  pauseScript;
	private: System::ComponentModel::BackgroundWorker^  scriptRunner;
	protected: 
		Lua ^lua;
		bool bPauseRequired, bStopRequired, bScriptRunning;
		List<int> ^breakpoints;
		int nextLine;
		// The next index variable denotes where to look for the next index, or -1 if there are
		// no more breakpoints.
		int nextIndex;
		int bCount;

	public:
		ScriptPane(MainWindow ^pform, Lua ^ctx);
		
		void OpenFile(String ^filename);
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~ScriptPane()
		{
			if (components)
			{
				delete components;
			}
		}

	protected: 
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Button^  runScript;


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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(ScriptPane::typeid));
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->runScript = (gcnew System::Windows::Forms::Button());
			this->scriptBox = (gcnew ScintillaNET::Scintilla());
			this->toolStrip = (gcnew System::Windows::Forms::ToolStrip());
			this->toolStrip1 = (gcnew System::Windows::Forms::ToolStrip());
			this->newToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->openToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->saveToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->cutToolStripButtonSeparator = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->cutToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->copyToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->pasteToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->breakpointButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->commentLineButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->uncommentButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->commentRegionButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->undoToolStripButtonSeparator = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->undoToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->redoToolStripButton = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->toolIncremental = (gcnew ScintillaNET::ToolStripIncrementalSearcher());
			this->pauseScript = (gcnew System::Windows::Forms::Button());
			this->scriptRunner = (gcnew System::ComponentModel::BackgroundWorker());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->scriptBox))->BeginInit();
			this->toolStrip1->SuspendLayout();
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(9, 27);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(37, 13);
			this->label1->TabIndex = 1;
			this->label1->Text = L"Script:";
			// 
			// runScript
			// 
			this->runScript->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->runScript->Location = System::Drawing::Point(497, 302);
			this->runScript->Name = L"runScript";
			this->runScript->Size = System::Drawing::Size(75, 23);
			this->runScript->TabIndex = 2;
			this->runScript->Text = L"Run";
			this->runScript->UseVisualStyleBackColor = true;
			this->runScript->Click += gcnew System::EventHandler(this, &ScriptPane::runScript_Click);
			// 
			// scriptBox
			// 
			this->scriptBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->scriptBox->ConfigurationManager->Language = L"lua";
			this->scriptBox->Indentation->TabWidth = 4;
			this->scriptBox->Lexing->Lexer = ScintillaNET::Lexer::Lua;
			this->scriptBox->Lexing->LexerName = L"lua";
			this->scriptBox->Lexing->LineCommentPrefix = L"";
			this->scriptBox->Lexing->StreamCommentPrefix = L"";
			this->scriptBox->Lexing->StreamCommentSufix = L"";
			this->scriptBox->Location = System::Drawing::Point(12, 43);
			this->scriptBox->Margins->Margin0->Width = 30;
			this->scriptBox->Margins->Margin1->AutoToggleMarkerNumber = 0;
			this->scriptBox->Margins->Margin1->IsClickable = true;
			this->scriptBox->Name = L"scriptBox";
			this->scriptBox->Size = System::Drawing::Size(560, 253);
			this->scriptBox->TabIndex = 3;
			this->scriptBox->Text = resources->GetString(L"scriptBox.Text");
			this->scriptBox->ModifiedChanged += gcnew System::EventHandler(this, &ScriptPane::scriptBox_ModifiedChanged);
			// 
			// toolStrip
			// 
			this->toolStrip->GripStyle = System::Windows::Forms::ToolStripGripStyle::Hidden;
			this->toolStrip->Location = System::Drawing::Point(0, 0);
			this->toolStrip->Name = L"toolStrip";
			this->toolStrip->Size = System::Drawing::Size(100, 25);
			this->toolStrip->TabIndex = 0;
			// 
			// toolStrip1
			// 
			this->toolStrip1->GripStyle = System::Windows::Forms::ToolStripGripStyle::Hidden;
			this->toolStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(17) {this->newToolStripButton, 
				this->openToolStripButton, this->saveToolStripButton, this->cutToolStripButtonSeparator, this->cutToolStripButton, this->copyToolStripButton, 
				this->pasteToolStripButton, this->toolStripSeparator2, this->breakpointButton, this->commentLineButton, this->uncommentButton, 
				this->commentRegionButton, this->undoToolStripButtonSeparator, this->undoToolStripButton, this->redoToolStripButton, this->toolStripSeparator1, 
				this->toolIncremental});
			this->toolStrip1->Location = System::Drawing::Point(0, 0);
			this->toolStrip1->Name = L"toolStrip1";
			this->toolStrip1->Padding = System::Windows::Forms::Padding(10, 0, 1, 0);
			this->toolStrip1->Size = System::Drawing::Size(584, 27);
			this->toolStrip1->TabIndex = 10;
			// 
			// newToolStripButton
			// 
			this->newToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->newToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"newToolStripButton.Image")));
			this->newToolStripButton->Name = L"newToolStripButton";
			this->newToolStripButton->Size = System::Drawing::Size(23, 24);
			this->newToolStripButton->Text = L"New File (Ctrl+N)";
			this->newToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::newToolStripButton_Click);
			// 
			// openToolStripButton
			// 
			this->openToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->openToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"openToolStripButton.Image")));
			this->openToolStripButton->Name = L"openToolStripButton";
			this->openToolStripButton->Size = System::Drawing::Size(23, 24);
			this->openToolStripButton->Text = L"Open File (Ctrl+O)";
			this->openToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::openToolStripButton_Click);
			// 
			// saveToolStripButton
			// 
			this->saveToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->saveToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"saveToolStripButton.Image")));
			this->saveToolStripButton->Name = L"saveToolStripButton";
			this->saveToolStripButton->Size = System::Drawing::Size(23, 24);
			this->saveToolStripButton->Text = L"Save File (Ctrl+S)";
			this->saveToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::saveToolStripButton_Click);
			// 
			// cutToolStripButtonSeparator
			// 
			this->cutToolStripButtonSeparator->Name = L"cutToolStripButtonSeparator";
			this->cutToolStripButtonSeparator->Size = System::Drawing::Size(6, 27);
			// 
			// cutToolStripButton
			// 
			this->cutToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->cutToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"cutToolStripButton.Image")));
			this->cutToolStripButton->Name = L"cutToolStripButton";
			this->cutToolStripButton->Size = System::Drawing::Size(23, 24);
			this->cutToolStripButton->Text = L"Cut (Ctrl+X)";
			this->cutToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::cutToolStripButton_Click);
			// 
			// copyToolStripButton
			// 
			this->copyToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->copyToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"copyToolStripButton.Image")));
			this->copyToolStripButton->Name = L"copyToolStripButton";
			this->copyToolStripButton->Size = System::Drawing::Size(23, 24);
			this->copyToolStripButton->Text = L"Copy (Ctrl+C)";
			this->copyToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::copyToolStripButton_Click);
			// 
			// pasteToolStripButton
			// 
			this->pasteToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->pasteToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"pasteToolStripButton.Image")));
			this->pasteToolStripButton->Name = L"pasteToolStripButton";
			this->pasteToolStripButton->Size = System::Drawing::Size(23, 24);
			this->pasteToolStripButton->Text = L"Paste (Ctrl+V)";
			this->pasteToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::pasteToolStripButton_Click);
			// 
			// toolStripSeparator2
			// 
			this->toolStripSeparator2->Name = L"toolStripSeparator2";
			this->toolStripSeparator2->Size = System::Drawing::Size(6, 27);
			// 
			// breakpointButton
			// 
			this->breakpointButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->breakpointButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"breakpointButton.Image")));
			this->breakpointButton->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->breakpointButton->Name = L"breakpointButton";
			this->breakpointButton->Size = System::Drawing::Size(23, 24);
			this->breakpointButton->Text = L"Add/Remove Breakpoint (Ctrl+B)";
			this->breakpointButton->Click += gcnew System::EventHandler(this, &ScriptPane::breakpointButton_Click);
			// 
			// commentLineButton
			// 
			this->commentLineButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->commentLineButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"commentLineButton.Image")));
			this->commentLineButton->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->commentLineButton->Name = L"commentLineButton";
			this->commentLineButton->Size = System::Drawing::Size(23, 24);
			this->commentLineButton->Text = L"Comment Line (Ctrl+L)";
			this->commentLineButton->Click += gcnew System::EventHandler(this, &ScriptPane::commentLineButton_Click);
			// 
			// uncommentButton
			// 
			this->uncommentButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->uncommentButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"uncommentButton.Image")));
			this->uncommentButton->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->uncommentButton->Name = L"uncommentButton";
			this->uncommentButton->Size = System::Drawing::Size(23, 24);
			this->uncommentButton->Text = L"Uncomment Line (Ctrl+U)";
			this->uncommentButton->Click += gcnew System::EventHandler(this, &ScriptPane::uncommentButton_Click);
			// 
			// commentRegionButton
			// 
			this->commentRegionButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->commentRegionButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"commentRegionButton.Image")));
			this->commentRegionButton->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->commentRegionButton->Name = L"commentRegionButton";
			this->commentRegionButton->Size = System::Drawing::Size(23, 24);
			this->commentRegionButton->Text = L"Comment Region (Ctrl+M)";
			this->commentRegionButton->Click += gcnew System::EventHandler(this, &ScriptPane::commentRegionButton_Click);
			// 
			// undoToolStripButtonSeparator
			// 
			this->undoToolStripButtonSeparator->Name = L"undoToolStripButtonSeparator";
			this->undoToolStripButtonSeparator->Size = System::Drawing::Size(6, 27);
			// 
			// undoToolStripButton
			// 
			this->undoToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->undoToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"undoToolStripButton.Image")));
			this->undoToolStripButton->Name = L"undoToolStripButton";
			this->undoToolStripButton->Size = System::Drawing::Size(23, 24);
			this->undoToolStripButton->Text = L"Undo (Ctrl+Z)";
			this->undoToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::undoToolStripButton_Click);
			// 
			// redoToolStripButton
			// 
			this->redoToolStripButton->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->redoToolStripButton->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"redoToolStripButton.Image")));
			this->redoToolStripButton->Name = L"redoToolStripButton";
			this->redoToolStripButton->Size = System::Drawing::Size(23, 24);
			this->redoToolStripButton->Text = L"Redo (Ctrl+Y)";
			this->redoToolStripButton->Click += gcnew System::EventHandler(this, &ScriptPane::redoToolStripButton_Click);
			// 
			// toolStripSeparator1
			// 
			this->toolStripSeparator1->Name = L"toolStripSeparator1";
			this->toolStripSeparator1->Size = System::Drawing::Size(6, 27);
			// 
			// toolIncremental
			// 
			this->toolIncremental->BackColor = System::Drawing::Color::Transparent;
			this->toolIncremental->Name = L"toolIncremental";
			this->toolIncremental->Scintilla = this->scriptBox;
			this->toolIncremental->Size = System::Drawing::Size(262, 24);
			// 
			// pauseScript
			// 
			this->pauseScript->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->pauseScript->Enabled = false;
			this->pauseScript->Location = System::Drawing::Point(416, 302);
			this->pauseScript->Name = L"pauseScript";
			this->pauseScript->Size = System::Drawing::Size(75, 23);
			this->pauseScript->TabIndex = 11;
			this->pauseScript->Text = L"Pause";
			this->pauseScript->UseVisualStyleBackColor = true;
			this->pauseScript->Click += gcnew System::EventHandler(this, &ScriptPane::pauseScript_Click);
			// 
			// scriptRunner
			// 
			this->scriptRunner->WorkerSupportsCancellation = true;
			this->scriptRunner->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &ScriptPane::scriptRunner_DoWork);
			this->scriptRunner->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &ScriptPane::scriptRunner_RunWorkerCompleted);
			// 
			// ScriptPane
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(584, 337);
			this->Controls->Add(this->pauseScript);
			this->Controls->Add(this->toolStrip1);
			this->Controls->Add(this->scriptBox);
			this->Controls->Add(this->runScript);
			this->Controls->Add(this->label1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^  >(resources->GetObject(L"$this.Icon")));
			this->Name = L"ScriptPane";
			this->ShowIcon = false;
			this->Text = L"Script Editor";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->scriptBox))->EndInit();
			this->toolStrip1->ResumeLayout(false);
			this->toolStrip1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void runScript_Click(System::Object^  sender, System::EventArgs^  e);
	private: System::Void pauseScript_Click(System::Object^  sender, System::EventArgs^  e);
	private: System::Void newToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void openToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void saveToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void cutToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void copyToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void pasteToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void undoToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void redoToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void commentLineButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void commentRegionButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void uncommentButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void breakpointButton_Click(System::Object^  sender, System::EventArgs^  e);
		 public:
			 void LuaLineHook(System::Object^  sender, LuaInterface::DebugHookEventArgs ^args);
			 void HookExceptionForwarder(System::Object^  sender, LuaInterface::HookExceptionEventArgs ^args);
			 void HighlightPausedLine(int line);
private: System::Void scriptRunner_RunWorkerCompleted(System::Object^  sender, System::ComponentModel::RunWorkerCompletedEventArgs^  e);
private: System::Void scriptRunner_DoWork(System::Object^  sender, System::ComponentModel::DoWorkEventArgs^  e);
private: System::Void scriptBox_ModifiedChanged(System::Object^  sender, System::EventArgs^  e);
};
}
