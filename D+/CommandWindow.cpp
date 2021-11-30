#include "CommandWindow.h"

using namespace ScintillaNET;
using namespace LuaInterface;

namespace DPlus {

CommandWindow::CommandWindow( MainWindow ^pform, Lua ^ctx ) {
	InitializeComponent();

	parentForm = pform;
	lua = ctx;

	commandHistory = gcnew List<String ^>();
	curHistIndex = 0;

	bScriptRunning = false;
	bStopRequired = false;

	lua->DebugHook += gcnew System::EventHandler<DebugHookEventArgs ^>(this, &CommandWindow::LuaLineHook);	
	lua->HookException += gcnew System::EventHandler<HookExceptionEventArgs ^>(this, &CommandWindow::HookExceptionForwarder);
	lua->SetDebugHook(EventMasks::LUA_MASKLINE, 0);	

	// Register the print command
	lua->RegisterFunction("print", this, CommandWindow::typeid->GetMethod("LuaPrint"));
}

void CommandWindow::PerformAction() {
	if(bScriptRunning) {
		MessageBox::Show("Command still running!", 
						 "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	if(lua->IsExecuting) {
		MessageBox::Show("Cannot run command while a script is running in the script editor.", 
			"ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}

	String ^command = commandBox->Text->Trim();	

	// Add to history, only if a command was typed
	if(command->Length > 0)	{	
		commandHistory->Add(command);
		if(commandHistory->Count > HISTORY_SIZE)
			commandHistory->RemoveAt(0);
		curHistIndex = commandHistory->Count;
	}

	LuaPrint("> " + command);
	commandBox->Text = "";

	if(command->EndsWith(";"))
		command = command->Substring(0, command->Length - 1);

	// UI stuff
	bStopRequired = false;	
	bScriptRunning = true;
	labelPrompt->Text = "% ";

	scriptRunner->RunWorkerAsync(command);
}

System::Void CommandWindow::scriptRunner_DoWork( System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e ) {
	String ^msg = "";
	String ^cmd = ((String ^)e->Argument);

	// Check if something was written
	if(cmd->Length == 0) {
		e->Result = msg;
		return;
	}

 	// Find the last expression in the input string
	String ^evalExpr = cmd->Substring(cmd->LastIndexOf(";") + 1)->Trim();

	// All the expressions, not including the last one
	String ^runExpr = cmd->Substring(0, cmd->LastIndexOf(";") + 1);

	// Run the expressions (except the last one)
	try {
		lua->DoString(runExpr);
	} catch(LuaInterface::LuaCompileException ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		e->Result = "Compile error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1);
		return;
	} catch(LuaInterface::LuaScriptException ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		e->Result = "Runtime error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1);
		return;
	} catch(Exception ^ex) {
		if(bStopRequired) // Execution was stopped
			return;

		e->Result = "General error: " + ex->Message;
		return;
	}

	//////////////////////////////////////////////////////////////////////////

	// Try to evaluate the final expression	

	// As a global
	try {
		Object ^obj = lua[evalExpr];
		if(obj != nullptr) { // If evaluation
			bool isString = (dynamic_cast<String ^>(obj) != nullptr);

			e->Result = evalExpr + " = " + (isString ? "[[" : "") + obj->ToString() + (isString ? "]]" : "");
			return;
		}
	} catch(Exception ^) {}

	// As an expression
	try {
		lua->DoString("ans = (" + evalExpr + ")");
	} catch(Exception ^) {
		// As a command
		try {
			lua->DoString(evalExpr);
		} catch(LuaInterface::LuaCompileException ^ex) {
			if(bStopRequired) // Execution was stopped
				return;

			// If evaluation of a nonexistent global
			if(ex->Message->EndsWith("'=' expected near '<eof>'"))
				e->Result = evalExpr + " = nil";
			else
				e->Result = "Compile error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1);
		} catch(LuaInterface::LuaScriptException ^ex) {
			if(bStopRequired) // Execution was stopped
				return;

			e->Result = "Runtime error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1);
		} catch(LuaInterface::LuaException ^ex) {
			if(bStopRequired) // Execution was stopped
				return;

			e->Result = "Execution error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1);
		} catch(Exception ^ex) {
			if(bStopRequired) // Execution was stopped
				return;

			e->Result = "General error: " + ex->Message;
		}

		return;
	}

	// As an expression (continued)
	try {
		Object ^obj = lua["ans"];
		if(obj != nullptr) { // If evaluation
			bool isString = (dynamic_cast<String ^>(obj) != nullptr);

			e->Result = "ans = " + (isString ? "[[" : "") + obj->ToString() + (isString ? "]]" : "");
			return;
		} else {
			e->Result = "Invalid expression or nonexistent variable";
			return;
		}
	} catch(Exception ^) {}	
}

System::Void CommandWindow::scriptRunner_RunWorkerCompleted( System::Object^ sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e ) {
	String ^msg = (String ^)e->Result;
	
	if(bStopRequired)
		LuaPrint("Command stopped.");
	else if(msg != nullptr && msg->Length > 0)
		LuaPrint(msg);

	bScriptRunning = false;
	bStopRequired = false;
	labelPrompt->Text = "> ";
	
	commandBox->Focus();

	fflush(stdout);
}

void CommandWindow::LuaLineHook(System::Object^  sender, LuaInterface::DebugHookEventArgs ^args) {	
	if(!bScriptRunning)
		return;

	if(bStopRequired)
		throw gcnew LuaException("STOP");
}


void CommandWindow::HookExceptionForwarder( System::Object^ sender, LuaInterface::HookExceptionEventArgs ^args ) {
	if(!bScriptRunning)
		return;

	throw args->Exception;
}

System::Void CommandWindow::commandBox_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	// History up
	if(e->KeyCode == Keys::Up) {
		if(commandBox->AutoComplete->IsActive)
			return;

		// History up
		if(curHistIndex > -1)
			curHistIndex--;
		if(curHistIndex < 0) // Beginning of history
			commandBox->Text = "";
		else {
			commandBox->Text = commandHistory[curHistIndex];		
			commandBox->Selection->Start = commandBox->Text->Length;
			commandBox->Selection->End = commandBox->Text->Length;
			commandBox->Caret->Position = commandBox->Text->Length;
		}

		e->Handled = true;
		return;
	}

	// History down
	if(e->KeyCode == Keys::Down) {
		if(commandBox->AutoComplete->IsActive)
			return;

		// History down
		if(curHistIndex < commandHistory->Count)
			curHistIndex++;
		if(curHistIndex >= commandHistory->Count) // End of history
			commandBox->Text = "";
		else {
			commandBox->Text = commandHistory[curHistIndex];
			commandBox->Selection->Start = commandBox->Text->Length;
			commandBox->Selection->End = commandBox->Text->Length;
			commandBox->Caret->Position = commandBox->Text->Length;
		}

		e->Handled = true;
		return;
	}

	// Action
	if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return) {
		if(commandBox->AutoComplete->IsActive)
			return;


		PerformAction();
		e->Handled = true;
		return;
	}

	// Stop current action
	if(e->Control && e->KeyCode == Keys::C) {
		if(!bStopRequired) {
			bStopRequired = true;
			LuaPrint("^C");
		}
		e->Handled = true;
		return;
	}
}

delegate void PrintMethod(String ^message);
void CommandWindow::LuaPrint(String ^message) {
	if(this->InvokeRequired) {
		this->Invoke(gcnew PrintMethod(this, &CommandWindow::LuaPrint), gcnew array<Object ^> {message});
		return;
	}


	scriptBox->IsReadOnly = false;
	scriptBox->AppendText(message + "\n");
	scriptBox->IsReadOnly = true;

	// Scroll to bottom line
	int textlen = scriptBox->Text->Length;	
	scriptBox->NativeInterface->GotoLine(scriptBox->Lines->Count - 1); 
	scriptBox->Selection->Start = textlen;
	scriptBox->Selection->End = textlen;
	scriptBox->Caret->Position = textlen;
}

};
