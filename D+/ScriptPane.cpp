#include "ScriptPane.h"

using namespace ScintillaNET;
using namespace LuaInterface;
using namespace System::Text::RegularExpressions;

namespace DPlus {

ScriptPane::ScriptPane( MainWindow ^pform, Lua ^ctx ) {
	InitializeComponent();

	parentForm = pform;
	lua = ctx;

	breakpoints = gcnew List<int>();

	bPauseRequired = false;
	bStopRequired = false;	
	bScriptRunning = false;

	lua->DebugHook += gcnew System::EventHandler<DebugHookEventArgs ^>(this, &ScriptPane::LuaLineHook);	
	lua->HookException += gcnew System::EventHandler<HookExceptionEventArgs ^>(this, &ScriptPane::HookExceptionForwarder);

	lua->SetDebugHook(EventMasks::LUA_MASKLINE, 0);	
}

System::Void ScriptPane::runScript_Click(System::Object^  sender, System::EventArgs^  e) {	
	if(bScriptRunning) {
		bStopRequired = true;
		runScript->Enabled = false;
		return;
	}		
	if(lua->IsExecuting) {
		MessageBox::Show("Cannot run script while command is running in the command window.", 
						 "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}

	// Remove previous highlights
	scriptBox->NativeInterface->IndicatorClearRange(0, scriptBox->Text->Length);

	// Create the breakpoint list
	breakpoints->Clear();
	int i = 0;
	for each (Line ^l in scriptBox->Lines) {
		if(l->GetMarkers()->Count > 0)
			breakpoints->Add(i);
		i++;
	}
	breakpoints->Sort();

	// UI stuff
	bPauseRequired = false;
	bStopRequired = false;	
	bScriptRunning = true;
	pauseScript->Enabled = true;
	toolStrip1->Enabled = false;
	scriptBox->Enabled = false;
	runScript->Text = "Stop";

	// Initialize next line with breakpoint
	bCount = breakpoints->Count;
	if(bCount > 0) {
		nextLine = breakpoints[0];		
		nextIndex = (bCount > 1) ? 1 : -1; 
	} else {
		nextLine = -1;
		nextIndex = -1;
	}

	scriptRunner->RunWorkerAsync(scriptBox->Text);
}

System::Void ScriptPane::scriptRunner_DoWork( System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e ) {
	String ^msg = "";
	try {
		lua->DoString((String ^)e->Argument);
	} catch(LuaInterface::LuaCompileException ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		MessageBox::Show(ex->Message, "Error Compiling Script", MessageBoxButtons::OK, MessageBoxIcon::Error);
		msg = ex->Message;
	} catch(LuaInterface::LuaScriptException ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		MessageBox::Show(ex->Message, "Error Running Script", MessageBoxButtons::OK, MessageBoxIcon::Error);
		msg = ex->Message;
	} catch(LuaInterface::LuaException ^ex) {
		if(bStopRequired) // Execution was stopped
			return;
		
		MessageBox::Show(ex->Message, "Error Executing Script", MessageBoxButtons::OK, MessageBoxIcon::Error);
		msg = ex->Message;
	} catch(Exception ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		MessageBox::Show(ex->Message, "General Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		msg = ex->Message;
	} finally {
		e->Result = msg;
	}
}

System::Void ScriptPane::scriptRunner_RunWorkerCompleted( System::Object^ sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e ) {
	String ^msg = (String ^)e->Result;

	if(bStopRequired)
		scriptBox->NativeInterface->IndicatorClearRange(0, scriptBox->Text->Length);

	// If result contains a line number (the one that caused the exception), 
	// highlight it in red.
	Match ^m = Regex::Match(msg, "\\]:(\\d+):");
	if(m->Success) {				
		scriptBox->Indicators[0]->Style = IndicatorStyle::RoundBox;
		scriptBox->Indicators[0]->Color = System::Drawing::Color::Red;

		UInt32 lnum = UInt32::Parse(m->Result("$1")) - 1;
		scriptBox->Lines[lnum]->Range->SetIndicator(0);
	}

	bScriptRunning = false;
	runScript->Text = "Run";
	runScript->Enabled = true;
	pauseScript->Enabled = false;
	toolStrip1->Enabled = true;
	scriptBox->Enabled = true;
	pauseScript->Text = "Pause";
}

System::Void ScriptPane::pauseScript_Click( System::Object^ sender, System::EventArgs^ e ) {
	if(!bScriptRunning)
		return;

	if(!bPauseRequired) {
		bPauseRequired = true;
		pauseScript->Text = "Resume";
	} else {
		scriptBox->NativeInterface->IndicatorClearRange(0, scriptBox->Text->Length);
		pauseScript->Text = "Pause";
		bPauseRequired = false;		
	}
}


System::Void ScriptPane::newToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Text = "";
}

System::Void ScriptPane::openToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	OpenFileDialog ^ofd = gcnew OpenFileDialog();
	ofd->Filter = "Lua Scripts (*.lua)|*.lua|All Files (*.*)|*.*";
	ofd->FileName = "";
	ofd->Title = "Choose a script to load...";
	ofd->Multiselect = false;
	if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
		return;

	OpenFile(ofd->FileName);	
}

System::Void ScriptPane::saveToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	SaveFileDialog ^sfd = gcnew SaveFileDialog();
	sfd->Filter = "Lua Scripts (*.lua)|*.lua|All Files (*.*)|*.*";
	sfd->FileName = "";
	sfd->Title = "Save As...";	
	sfd->OverwritePrompt = true;
	if(sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
		return;

	System::IO::File::WriteAllText(sfd->FileName, scriptBox->Text);
}

System::Void ScriptPane::cutToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Clipboard->Cut();
}

System::Void ScriptPane::copyToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Clipboard->Copy();
}

System::Void ScriptPane::pasteToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Clipboard->Paste();
}

System::Void ScriptPane::undoToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->UndoRedo->Undo();
}

System::Void ScriptPane::redoToolStripButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->UndoRedo->Redo();
}

System::Void ScriptPane::commentLineButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Commands->Execute(ScintillaNET::BindableCommand::LineComment);
}

System::Void ScriptPane::commentRegionButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Commands->Execute(ScintillaNET::BindableCommand::StreamComment);
}

System::Void ScriptPane::uncommentButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	scriptBox->Commands->Execute(ScintillaNET::BindableCommand::LineUncomment);
}

System::Void ScriptPane::breakpointButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	ScintillaNET::Line ^currentLine = scriptBox->Lines->Current;
	if (scriptBox->Markers->GetMarkerMask(currentLine) == 0)
	{
		currentLine->AddMarker(0);
	}
	else
	{
		currentLine->DeleteMarker(0);
	}
}

void ScriptPane::LuaLineHook(System::Object^  sender, LuaInterface::DebugHookEventArgs ^args) {	
	if(!bScriptRunning)
		return;

	if(bStopRequired)
		throw gcnew LuaException("STOP");

	int line = args->LuaDebug.currentline - 1;
	if(!bPauseRequired) {
		if(nextLine == -1) // No more breakpoints
			return;

		if(nextLine > line)
			return;
		else if(nextLine <= line) { // If we passed a marked line or it is the line to stop at
			if(nextIndex >= 0) {
				nextLine = breakpoints[nextIndex];
				nextIndex = (bCount > (nextIndex + 1)) ? (nextIndex + 1) : -1;
			} else // if(nextIndex < 0)
				nextLine = -1;
		}
		
		bPauseRequired = true;
	}

	// Highlight paused line
	HighlightPausedLine(line);

	// Pause execution
	while(bPauseRequired && !bStopRequired)
		System::Threading::Thread::Sleep(1000);

	if(bStopRequired)
		throw gcnew LuaException("STOP");
}

delegate void HighlightHandler(int line); 
void ScriptPane::HighlightPausedLine(int line) {
	if(this->InvokeRequired) {
		Invoke(gcnew HighlightHandler(this, &ScriptPane::HighlightPausedLine), gcnew array<Object ^>{line});
		return;
	}

	pauseScript->Text = "Resume";
	scriptBox->NativeInterface->GotoLine(line); // Scroll to line

	scriptBox->Indicators[1]->Style = IndicatorStyle::RoundBox;
	scriptBox->Indicators[1]->Color = System::Drawing::Color::Blue;
	scriptBox->Lines[line]->Range->SetIndicator(1);
}

void ScriptPane::HookExceptionForwarder( System::Object^ sender, LuaInterface::HookExceptionEventArgs ^args ) {
	if(!bScriptRunning)
		return;

	throw args->Exception;
}

void ScriptPane::OpenFile(String ^filename) {
	this->Text = "Script Editor - " + System::IO::Path::GetFileName(filename);

	scriptBox->Text = System::IO::File::ReadAllText(filename);
	scriptBox->Modified = false;

	this->Show();
}

System::Void ScriptPane::scriptBox_ModifiedChanged(System::Object^  sender, System::EventArgs^  e) {
	if(scriptBox->Modified == true) {
		if(!this->Text->EndsWith("*"))
			this->Text += " *";	
	} else {
		if(this->Text->EndsWith("*"))
			Text = Text->Substring(0, Text->Length - 2);	
	}
}

};
