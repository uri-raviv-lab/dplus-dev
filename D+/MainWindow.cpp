#include <windows.h> // For COLORREF

#include "MainWindow.h"
#include "AboutDPlus.h"
#include "clrfunctionality.h"

#include "FrontendExported.h" // For ReadDataFile/WriteDataFile
//#include "../../Frontend/Frontend/BackendAccessors.h"

#include "../../Frontend/Frontend/BackendCallers.h"
#include "LuaBinding.h"
#include "../Conversions/LUAConversions.h"
// #include "DockSampleControl.h"
// #include "AnotherDockControl.h"
#include "GraphPane2D.h"
#include "GraphPane3D.h"
#include "SymmetryEditor.h"
#include "SymmetryView.h"
#include "ControlButtonsPane.h"
#include "PreferencesPane.h"
#include "Controls3D.h"
#include "ParameterEditor.h"
#include "ScriptPane.h"
#include "CommandWindow.h"
#include "FittingPrefsPane.h"

#include "GraphFigure.h"

#include "CommProtocol.h"
#include "LocalComm.h"
#include <iostream>
#include <fstream>
#include <msclr\marshal_cppstd.h>
#include "TdrLevelInfo.h"
#include <time.h>
using namespace System::Runtime::InteropServices;
using Microsoft::WindowsAPICodePack::Taskbar::TaskbarManager;
using Microsoft::WindowsAPICodePack::Taskbar::TaskbarProgressBarState;


namespace DPlus {

	/// <summary>
	/// Creates a relative path from one file or folder to another.
	/// </summary>
	/// <param name="fromPath">Contains the directory that defines the start of the relative path.</param>
	/// <param name="toPath">Contains the path that defines the endpoint of the relative path.</param>
	/// <param name="dontEscape">Boolean indicating whether to add uri safe escapes to the relative path</param>
	/// <returns>The relative path from the start directory to the end path.</returns>
	/// <exception cref="ArgumentNullException"></exception>
	/*public static String ^MakeRelativePath(String fromPath, String toPath)
	{
	if (String.IsNullOrEmpty(fromPath)) throw new ArgumentNullException("fromPath");
	if (String.IsNullOrEmpty(toPath))   throw new ArgumentNullException("toPath");

	Uri fromUri = new Uri(fromPath);
	Uri toUri = new Uri(toPath);

	Uri relativeUri = fromUri.MakeRelativeUri(toUri);
	String relativePath = Uri.UnescapeDataString(relativeUri.ToString());

	return relativePath.Replace('/', Path.DirectorySeparatorChar);
	}*/

	MainWindow::~MainWindow() {
		if (components)
		{
			delete components;
		}
		if (_statusPollingTimer)
			_statusPollingTimer->Stop();

		if (frontend) {
			if (compositeModel)
				frontend->DestroyModel(job, compositeModel, false);
			for (int i = 0; i < domainModels->Count; i++)
				frontend->DestroyModel(job, domainModels[i], false);

			frontend->DestroyJob(job);

			delete frontend;
		}
		frontend = NULL;
		if (backendCaller)
			delete backendCaller;
	}

	void MainWindow::MainWindow_Load(System::Object^  sender, System::EventArgs^  e) {

		//DPlus::Scripting::OpenConsole();
		// Initialize UI scripting core
		luaState = nullptr;
		try {
			luaState = gcnew Lua();
			BindLuaFunctions();
		}
		catch (Exception ^ex) {
			MessageBox::Show("Lua backend failed to load: " + ex->Message, "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			this->Close();
			return;
		}

#ifndef _DEBUG	// Hide debug items
		dEBUGItemsToolStripMenuItem->Enabled = false;
		dEBUGItemsToolStripMenuItem->Visible = false;
#endif

		// Set the default directory where files will be read and written (in the
		// working directory if there are permissions, somewhere in appdata otherwise).
		do {
			char* appdata = getenv("APPDATA");

			rootWriteDir = Path::Combine(Environment::GetFolderPath(Environment::SpecialFolder::ApplicationData), "D+");

			bool haveWritePermissionsInWD = true;
			bool haveWritePermissionsInAppData = true;

			try
			{
				{ System::IO::File::WriteAllText("permissionTest.txt", "This is a test;"); }
				{ System::IO::File::Delete("permissionTest.txt"); }
			}
			catch (const System::IO::IOException ^)
			{
				haveWritePermissionsInWD = false;
			}
			catch (const System::UnauthorizedAccessException ^)
			{
				haveWritePermissionsInWD = false;
			}

			// Just write where ever the working directory is
			if (haveWritePermissionsInWD)
			{
				break;
			}

			TCHAR wd[MAX_PATH];
			// This is the directory in which we did not have permissions.
			DWORD dwRet = GetCurrentDirectory(MAX_PATH, wd);
			if (dwRet == 0)
			{
				// GetCurrentDirectory failed. What do we do now?
			}
			else if (dwRet > MAX_PATH)
			{
				// The path is longer than the MAX_PATH, also failed. What do we do now?
			}
			else
			{
				// All's good
			}

			String^ problem = "";
			try
			{
				
				if (!System::IO::Directory::Exists(rootWriteDir) && !System::IO::Directory::CreateDirectory(rootWriteDir))
				{
					haveWritePermissionsInAppData = false;
					problem = System::String::Format("did not manage to create {0} dir", rootWriteDir);
				}
				else if (SetCurrentDirectory(clrToWstring(rootWriteDir).c_str()))
				{
					{ System::IO::File::WriteAllText("permissionTest2.txt", "This is a test;"); }
					{ System::IO::File::Delete("permissionTest2.txt"); }
				}
				else
				{
					haveWritePermissionsInAppData = false;
					problem = System::String::Format("did not manage to set current dir to {0} ", rootWriteDir);
				}
			}
			catch (System::IO::IOException ^ errori)
			{		
				haveWritePermissionsInAppData = false;
				problem += e->GetType()->Name;
			}
			catch (System::UnauthorizedAccessException ^ errorU)
			{
				haveWritePermissionsInAppData = false;
				problem += "UnauthorizedAccessException: " +  errorU->Message;
			}
			// Notify the user that weird things may happen. WE LIKE OUR FILES!!
			if (!haveWritePermissionsInAppData)
			{
				String ^ msg;
				std::wstring wstring(&wd[0]); //convert to wstring
				string stringWd(wstring.begin(), wstring.end()); //and convert to string.
				String^ cSharpString = gcnew String(stringWd.c_str());
				msg = String::Concat(gcnew String("D+ seems not to have write permissions in both \""), cSharpString, "\" and \"");
				msg = String::Concat(msg, rootWriteDir, "\". Please run with write permissions in either of these places or weird things will happen. D+ will now terminate to prevent those weird things.");
				msg = String::Concat(msg, "\"Problem: ", problem);
				MessageBox::Show(this, msg, "Permissions Error", MessageBoxButtons::OK);
				rootWriteDir = "";
				Environment::Exit(int(1531));
			}

		} while (false);

		GraphPane2D ^grph2D = gcnew GraphPane2D(this);
		GraphPane3D ^grph3D = gcnew GraphPane3D(this);
		SymmetryView ^sv = gcnew SymmetryView(this);
		ControlButtonsPane ^ gf = gcnew ControlButtonsPane(this);
		SymmetryEditor ^se = gcnew SymmetryEditor(this, sv);
		PreferencesPane ^pr = gcnew PreferencesPane(this);
		Controls3D ^cn3 = gcnew Controls3D(this, grph3D);
		ParameterEditor ^pe = gcnew ParameterEditor(this, sv);
		FittingPrefsPane^fp = gcnew FittingPrefsPane(this);
		ScriptPane ^sp = nullptr;
		CommandWindow ^cw = nullptr;

		PaneList->Add(grph2D);
		PaneList->Add(grph3D);
		PaneList->Add(se);
		PaneList->Add(sv);
		PaneList->Add(pr);
		PaneList->Add(cn3);
		PaneList->Add(pe);
		PaneList->Add(fp);
		PaneList->Add(gf);

		MenuPaneList->Add(Graph2DToolStripMenuItem);
		MenuPaneList->Add(Graph3DToolStripMenuItem);
		MenuPaneList->Add(symmetryEditorToolStripMenuItem);
		MenuPaneList->Add(symmetryViewToolStripMenuItem);
		MenuPaneList->Add(preferencesToolStripMenuItem);
		MenuPaneList->Add(controls3DToolStripMenuItem);
		MenuPaneList->Add(parameterEditorToolStripMenuItem);
		MenuPaneList->Add(fittingPreferencesToolStripMenuItem);
		MenuPaneList->Add(GenerateFitToolStripMenuItem);

		if (luaState != nullptr) {
			sp = gcnew ScriptPane(this, luaState);
			cw = gcnew CommandWindow(this, luaState);

			PaneList->Add(sp);
			PaneList->Add(cw);
			MenuPaneList->Add(scriptWindowToolStripMenuItem);
			MenuPaneList->Add(commandWindowToolStripMenuItem);
		}

		// Link events
		for (int i = 0; i < PaneList->Count; i++) {
			PaneList[i]->VisibleChanged += gcnew System::EventHandler(this, &MainWindow::PaneVisibilityToggled);
			PaneList[i]->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &MainWindow::MainWindow_KeyDown);
		}
		grph3D->glCanvas3D1->RenderDone += gcnew GLView::GLCanvas3D::RenderDoneHandler(this, &MainWindow::graph3DChanged);


		mainDockPanel->SuspendLayout(true);

		if (System::IO::File::Exists("Latest.dlayout")) {
			if (!LoadLayout("Latest.dlayout"))
				LoadDefaultLayout();
		}
		else
			LoadDefaultLayout();

		mainDockPanel->ResumeLayout(true, true);
		for (int i = 0; i < MenuPaneList->Count; i++)
			MenuPaneList[i]->Checked = PaneList[i]->Visible;



		//clear the cache, if no other D+ processes are running
		System::Diagnostics::Process ^ current = System::Diagnostics::Process::GetCurrentProcess();
		array <System::Diagnostics::Process^, 1> ^ processes = System::Diagnostics::Process::GetProcessesByName(current->ProcessName);
		if (processes->Length == 1)
		{
#undef GetTempPath
			String ^ temp = Path::GetTempPath();
			String ^ dplus = "dplus";
			String ^ temp_dplus = Path::Combine(temp, dplus);
			DirectoryInfo ^ di = gcnew DirectoryInfo(temp_dplus);
			if (di->Exists){
				for each (DirectoryInfo ^ dir in di->GetDirectories())
				{
					dir->Refresh();
					System::DateTime now = System::DateTime::Now;
					TimeSpan timePassed = now.Subtract(dir->LastWriteTime);
					TimeSpan day = TimeSpan(1, 0, 0, 0);
					if (TimeSpan::Compare(timePassed, day) == 1)
					{
						dir->Delete(true);
					}
				}
			}
		}



		// Initialize and load frontend
		
		bool useLocal = false;
		bool userChose = false;
		array<String^>^ args = Environment::GetCommandLineArgs();

		for each (String^ arg in args)
		{
			if (!userChose && arg == "--local")
			{
				useLocal = true;
				DPlus::Scripting::OpenConsole();
				userChose = true;
			}

			if (!userChose && arg == "--remote")
			{
				useLocal = false;
				userChose = true;
			}

			if (!userChose && arg== "--devel")
				while (!userChose)
				{
					String^ message = "Please specify if you wish to use locally or on a remote server. Would you like to run calculations on the local computer?";
					String^ caption = "No backend specified";
					MessageBoxButtons buttons = MessageBoxButtons::YesNo;
					System::Windows::Forms::DialogResult result;

					// Displays the MessageBox.
					result = MessageBox::Show(this, message, caption, buttons);
					if (result == ::DialogResult::Yes)
					{
						useLocal = true;
						DPlus::Scripting::OpenConsole();
						userChose = true;
					}

					if (result == ::DialogResult::No)
					{
						useLocal = false;
						userChose = true;
					}

				}
		}

		if (!userChose)
		{
			userChose = true;
			useLocal = true;
			DPlus::Scripting::OpenConsole();
		}


		if (useLocal)
		{
			//backendCaller = new LocalBackendCaller();
			String^ current_path = Path::GetDirectoryName(Application::ExecutablePath);
			msclr::interop::marshal_context context;
			std::string cur_dir_str = context.marshal_as<std::string>(current_path);

			pythonCall = gcnew ManagedPythonPreCaller(cur_dir_str);
			ManagedPythonPreCaller::CallBackendDelegate ^ pyDel = pythonCall->GetDelegate();
			ManagedBackendCaller::callFunc funcPtr = (ManagedBackendCaller::callFunc)Marshal::GetFunctionPointerForDelegate(pyDel).ToPointer();
			backendCaller = new ManagedBackendCaller(funcPtr, true);
		}

		else
		{
			//get server address
			if (System::IO::File::Exists("Server.dserv"))
			{
				this->serverAddress = System::IO::File::ReadAllText("Server.dserv");
			}
			if (System::IO::File::Exists("Code.dserv"))
			{
				this->validationCode = System::IO::File::ReadAllText("Code.dserv");
			}

			if (!this->serverAddress || !this->validationCode)
				this->openServerWindow();

			httpCallForm = gcnew ManagedHTTPCallerForm(this->serverAddress, this->validationCode);
			httpCallForm->serverLabelClicked += gcnew DPlus::ManagedHTTPCallerForm::severlabelHandler(this, &DPlus::MainWindow::OnserverLabelClicked);
			httpCallForm->cancelButtonClicked += gcnew DPlus::ManagedHTTPCallerForm::cancelClickEventHandler(this, &DPlus::MainWindow::OncancelButtonClicked);
			httpCallForm->restartButtonClicked += gcnew DPlus::ManagedHTTPCallerForm::restartClickEventHandler(this, &DPlus::MainWindow::OnRestartButtonClicked);

			ManagedHTTPCallerForm::CallBackendDelegate ^del = httpCallForm->GetDelegate();
			ManagedBackendCaller::callFunc funcPtr = (ManagedBackendCaller::callFunc)Marshal::GetFunctionPointerForDelegate(del).ToPointer();
			backendCaller = new ManagedBackendCaller(funcPtr);
			this->useGPUToolStripMenuItem->Visible = false;
		}

		frontend = new LocalFrontend(backendCaller);

		if (!frontend->IsValid()) {
			MessageBox::Show("Cannot use frontend.", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);

			delete frontend;
			frontend = NULL;
			return;
		}

		checkCapabilitiesMainWind();

		// Initialize a job for this instance
		String ^str = String::Format("D+ ({0})", System::Diagnostics::Process::GetCurrentProcess()->Id);

		pcf = gcnew ProgressCallbackFunc(this, &MainWindow::ProgressCallback);
		ccf = gcnew CompletionCallbackFunc(this, &MainWindow::CompletionCallback);
		job = frontend->CreateJob(clrToWstring(str).c_str(),
			(progressFunc)Marshal::GetFunctionPointerForDelegate(pcf).ToPointer(),
			(notifyCompletionFunc)Marshal::GetFunctionPointerForDelegate(ccf).ToPointer());

		// Initialize the status polling timer
		_statusPollingTimer = gcnew System::Timers::Timer(1000);  // Poll every second
		_statusPollingTimer->Elapsed += gcnew System::Timers::ElapsedEventHandler(this, &MainWindow::PollStatus);
		_statusPollingTimer->Enabled = true;

		// Create the two main models
		compositeModel = frontend->CreateCompositeModel(job);
		domainModels = gcnew Generic::List<ModelPtr>();
		domainModels->Add(frontend->CreateDomainModel(job));
		if (!compositeModel || !domainModels[0]) {
			MessageBox::Show("Critical error creating the base models", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);

			delete frontend;
			frontend = NULL;
			return;
		}

		// Load latest preferences
		if (System::IO::File::Exists("Latest.dprefs")) {
			try {
				Lua ^prefLoadState = gcnew Lua();
				prefLoadState->DoString("os=nil;", "LoadPreferences");
				prefLoadState->DoFile("Latest.dprefs");

				if (prefLoadState->GetTable("DomainPreferences") != nullptr)
					pr->DeserializePreferences((LuaTable ^)prefLoadState->GetTable("DomainPreferences"));

				if (prefLoadState->GetTable("FittingPreferences") != nullptr)
					((FittingPrefsPane^)PaneList[FITTINGPREFS])->DeserializePreferences(
					(LuaTable ^)prefLoadState->GetTable("FittingPreferences"));

			}
			catch (Exception ^ex) {
				MessageBox::Show("Failed to load preferences: " + ex->Message, "ERROR", MessageBoxButtons::OK,
					MessageBoxIcon::Error);
			}
		}

		this->WindowState = FormWindowState::Maximized;

		// Populate entities
		RepopulateEntities(sv->entityCombo, true);
		sv->ChangeEditorEnabled(false);
	}

	void MainWindow::SaveLayout() {
		// Saves the current configuration to XML
		mainDockPanel->SaveAsXml("Latest.dlayout");
	}

	void MainWindow::savelayoutonclose()
	{
		if (luaState == nullptr)
			return;

		SaveLayout();

		//set loaded signal to null, so it doesn't get saved in the layout
		this->loadedSignal = nullptr;
		// Save the latest preferences
		try
		{
			System::IO::File::WriteAllText("Latest.dprefs",
				((PreferencesPane ^)PaneList[PREFERENCES])->SerializePreferences() + "\n" +
				((FittingPrefsPane^)PaneList[FITTINGPREFS])->SerializePreferences(),
				System::Text::Encoding::ASCII);
		}
		catch (Exception ^ex)
		{
			MessageBox::Show(ex->Message + "\n dplus will be closed without saving last preference", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}

	}

	System::Void MainWindow::MainWindow_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {

		savelayoutonclose();
	}

	void MainWindow::LoadDefaultLayout() {
		// Reset the docking panel (clear it from all of its contents)
		for (int index = mainDockPanel->Contents->Count - 1; index >= 0; index--)
		{
			if (dynamic_cast<IDockContent ^>(mainDockPanel->Contents[index]))
			{
				IDockContent ^content = (IDockContent ^)mainDockPanel->Contents[index];
				content->DockHandler->DockPanel = nullptr;
				content->DockHandler->FloatPane = nullptr;
				content->DockHandler->Pane = nullptr;
			}
		}

		GraphPane2D ^grph2D = (GraphPane2D ^)PaneList[GRAPH2D];
		GraphPane3D ^grph3D = (GraphPane3D ^)PaneList[GRAPH3D];
		SymmetryView ^sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
		SymmetryEditor ^se = (SymmetryEditor ^)PaneList[SYMMETRY_EDITOR];
		Controls3D ^cn3 = (Controls3D ^)PaneList[CONTROLS];
		ParameterEditor ^pe = (ParameterEditor ^)PaneList[PARAMETER_EDITOR];
		PreferencesPane ^pr = (PreferencesPane ^)PaneList[PREFERENCES];
		FittingPrefsPane ^fp = (FittingPrefsPane ^)PaneList[FITTINGPREFS];
		ControlButtonsPane ^ gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		ScriptPane ^sp = nullptr;
		CommandWindow ^cw = nullptr;


		if (luaState != nullptr) {
			sp = (ScriptPane ^)PaneList[SCRIPT_EDITOR];
			cw = (CommandWindow ^)PaneList[COMMAND_WINDOW];
		}


		// The actual form presentation inside the docking panel

		if (cw) cw->Show(mainDockPanel, DockState::DockBottom);
		pe->Show(mainDockPanel, DockState::DockBottom);
		fp->Show(pe->Pane, DockAlignment::Right, 0.5);

		if (sp) sp->Show(mainDockPanel, DockState::Document);
		grph2D->Show(mainDockPanel, DockState::Document);
		grph3D->Show(mainDockPanel, DockState::Document);


		sv->Show(mainDockPanel, DockState::DockLeft);
		se->Show(sv->Pane, DockAlignment::Bottom, 0.4);

		
		cn3->Show(mainDockPanel, DockState::DockRight);
		gf->Show(cn3->Pane, DockAlignment::Top, 0.15);
		pr->Show(cn3->Pane, DockAlignment::Bottom, 0.5);

		mainDockPanel->UpdateDockWindowZOrder(DockStyle::Bottom, true);

	}
	void MainWindow::SetDefaultParams() {

		Controls3D ^cn3 = (Controls3D ^)PaneList[CONTROLS];
		PreferencesPane ^pr = (PreferencesPane ^)PaneList[PREFERENCES];
		FittingPrefsPane ^fp = (FittingPrefsPane ^)PaneList[FITTINGPREFS];
		SymmetryView ^ sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
		ControlButtonsPane ^ gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];

		pr->SetDefaultParams();
		fp->SetDefaultParams();
		cn3->SetDefaultParams();
		sv->SetDefaultParams();
		gf->SetDefaultParams();
	}
	bool MainWindow::LoadLayout(System::String^ fileName) {
		// Confirm existence of file
		if (!System::IO::File::Exists(fileName)) {
			MessageBox::Show("File " + fileName + " does not seem to exist.");
			return false;
		}

		try {
			// Reset the docking panel (clear it from all of its contents)
			for (int index = mainDockPanel->Contents->Count - 1; index >= 0; index--)
			{
				if (dynamic_cast<IDockContent ^>(mainDockPanel->Contents[index]))
				{
					IDockContent ^content = (IDockContent ^)mainDockPanel->Contents[index];
					content->DockHandler->DockPanel = nullptr;
					content->DockHandler->FloatPane = nullptr;
					content->DockHandler->Pane = nullptr;
				}
			}

			// Actually load the layout
			mainDockPanel->LoadFromXml(fileName, gcnew DeserializeDockContent(this, &MainWindow::GetContentFromPersistString));
		}
		catch (Exception ^) {
			LoadDefaultLayout();
			return false;
		}

		return true;
	}

	// MINI-DOCUMENTATION on how this works (I think): The layout of the docking panel is
	// saved inside the XML. However, the contents of the windows (and their names)
	// are stored in a way that requires the developer to convert them from string to
	// IDockContent. This function is called for each such sub-window to re-create it.
	// 
	// Since we use our own data files as well, we initialize our windows on our own.
	// Therefore, we only need to determine the type of the window and create it here
	IDockContent ^MainWindow::GetContentFromPersistString(String ^persistString)
	{
		if (persistString->Equals(GraphPane2D::typeid->FullName))
			return PaneList[GRAPH2D];
		if (persistString->Equals(GraphPane3D::typeid->FullName))
			return PaneList[GRAPH3D];
		if (persistString->Equals(SymmetryEditor::typeid->FullName))
			return PaneList[SYMMETRY_EDITOR];
		if (persistString->Equals(SymmetryView::typeid->FullName))
			return PaneList[SYMMETRY_VIEWER];
		if (persistString->Equals(PreferencesPane::typeid->FullName))
			return PaneList[PREFERENCES];
		if (persistString->Equals(Controls3D::typeid->FullName))
			return PaneList[CONTROLS];
		if (persistString->Equals(ParameterEditor::typeid->FullName))
			return PaneList[PARAMETER_EDITOR];
		if (persistString->Equals(ScriptPane::typeid->FullName))
			return PaneList[SCRIPT_EDITOR];
		if (persistString->Equals(CommandWindow::typeid->FullName))
			return PaneList[COMMAND_WINDOW];
		if (persistString->Equals(FittingPrefsPane::typeid->FullName))
			return PaneList[FITTINGPREFS];
		if (persistString->Equals(ControlButtonsPane::typeid->FullName))
			return PaneList[CALC_BUTTONS];

		MessageBox::Show("Unrecognized type: " + persistString);

		return nullptr;
	}

	System::Void MainWindow::defaultLayoutToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		this->LoadDefaultLayout();
	}

	System::Void MainWindow::defaultParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		this->SetDefaultParams();
	}

	System::Void MainWindow::configureServerToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		openServerWindow();
	}
	System::Void MainWindow::useGPUToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		bool newVal = !this->useGPUToolStripMenuItem->Checked == true;
		changeUseGPUDisplayValue(newVal);
		checkCapabilitiesMainWind();
	}
	void MainWindow::changeUseGPUDisplayValue(bool newVal){
		if (newVal == true)
		{
			UseGPU = true;
			this->useGPUToolStripMenuItem->Checked = true;
		}
		else
		{
			UseGPU = false;
			this->useGPUToolStripMenuItem->Checked = false;
		}
	}
	System::Void MainWindow::OnserverLabelClicked(System::Object^  sender, System::EventArgs^  e)
	{
		openServerWindow();
	}
	System::Void MainWindow::OncancelButtonClicked(System::Object^  sender, System::EventArgs^  e)
	{
		savelayoutonclose();
		System::Environment::Exit(-1);
	}
	System::Void MainWindow::OnRestartButtonClicked(System::Object^  sender, System::EventArgs^  e)
	{
		httpCallForm->Close();
		
		savelayoutonclose();
		System::String^ current_path = System::IO::Path::GetDirectoryName(Application::ExecutablePath);
		System::String^ combined_path = Path::Combine(current_path, "Resources//restartdplus.bat");
		System::Diagnostics::Process::Start(combined_path);
		this->Close();
		System::Environment::Exit(-1);
		
	}
	System::Void MainWindow::openServerWindow()
	{
		ServerConfigForm ^ scw = gcnew ServerConfigForm(serverAddress, validationCode);
		scw->ShowDialog();
		this->serverAddress = scw->serverAddress;
		this->validationCode = scw->validationCode;
		if (this->httpCallForm)
		{
			this->httpCallForm->setBaseUrl(serverAddress);
			this->httpCallForm->setToken(validationCode);
		}//update server address 
		//save server address
		System::IO::File::WriteAllText("Server.dserv", serverAddress);
		System::IO::File::WriteAllText("Code.dserv", validationCode);
	}

	System::Void MainWindow::saveLayoutToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		System::Windows::Forms::DialogResult ans = layoutSFD->ShowDialog();
		if (ans != System::Windows::Forms::DialogResult::OK)
			return;
		mainDockPanel->SaveAsXml(layoutSFD->FileName);

	}

	System::Void MainWindow::loadLayoutToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		System::Windows::Forms::DialogResult ans = layoutOFD->ShowDialog();
		if (ans != System::Windows::Forms::DialogResult::OK)
			return;
		this->LoadLayout(layoutOFD->FileName);
	}

	System::Void MainWindow::MenuPaneStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (!dynamic_cast<ToolStripMenuItem^>(sender))
			return;

		int ind = MenuPaneList->IndexOf((ToolStripMenuItem^)(sender));
		if (PaneList->Count <= ind || ind < 0 || MenuPaneList->Count <= ind)
			return;


		bool prevState = MenuPaneList[ind]->Checked;
		if (!prevState)
			(PaneList[ind])->Show();	// Causes the checked state to be toggled
		else
			(PaneList[ind])->Hide();
	}

	void MainWindow::PaneVisibilityToggled(System::Object^ sender, System::EventArgs^ e) {
		if (!dynamic_cast<DockContent^>(sender))
			return;
		int ind = PaneList->IndexOf((DockContent^)(sender));
		bool vis = PaneList[ind]->Visible;
		MenuPaneList[ind]->Checked = vis;
	}

	System::Void MainWindow::quitToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		this->Close();
	}

	System::Void MainWindow::graph3DChanged(System::Object^ sender) {
		Controls3D^ pan = nullptr;
		GraphPane3D^ g3 = nullptr;
		pan = dynamic_cast<Controls3D^>(PaneList[CONTROLS]);
		g3 = dynamic_cast<GraphPane3D^>(PaneList[GRAPH3D]);
		if (!pan || !g3)
			return;
		pan->pitchTextBox->Text = gcnew System::String(System::Double(g3->glCanvas3D1->Pitch).ToString("0.0000"));
		pan->yawTextBox->Text = gcnew System::String(System::Double(g3->glCanvas3D1->Yaw).ToString("0.0000"));
		pan->rollTextBox->Text = gcnew System::String(System::Double(g3->glCanvas3D1->Roll).ToString("0.0000"));
		pan->zoomTextBox->Text = gcnew System::String(System::Double(g3->glCanvas3D1->Distance).ToString("0.0000"));

		g3->glCanvas3D1->ShowAxis = pan->showAxesCheckBox->Checked;
		g3->glCanvas3D1->ShowConstantSizedAxis = pan->fixedSize_checkBox->Checked;
	}

	System::Void MainWindow::luaTestToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		luaState->DoString("msgbox('THIS IS A TEST');");
		luaState->DoString("aa = 'Hello in console'");
		luaState->DoString("dplus.openconsole(); print(aa); msgbox('Press OK to close console'); dplus.closeconsole();");
	}

	void MainWindow::takeFocus(System::Object^ sender, System::EventArgs^ e) {
		mainDockPanel->Focus();
	}

	static void InsertEntitiesToParamTree(ParameterTree *pt, Entity ^parent) {
		int ind = 0;
		for each(Entity ^ent in parent->Nodes) {
			// Add the sub-model and set its parameters
			pt->AddSubModel(ent->BackendModel, ent->GetParameters());

			// Add its children, if applicable
			if (ent->Nodes->Count > 0)
				InsertEntitiesToParamTree(pt->GetSubModel(ind), ent);

			ind++;
		}
	}

	ParameterTree MainWindow::PrepareForWork(FittingProperties& fp) {
		// TODO: Take these from somewhere
		fp.accurateDerivative = false; fp.accurateFitting = true; fp.fitIterations = 20;
		fp.logScaleFitting = false; fp.method = FIT_LBFGS;
		fp.minSignal = 0.0; fp.wssrFitting = false;
		//	fp.lossFuncType = LossFunction_Enum::TRIVIAL_LOSS;	// TODO::LOSS Add to GUI and read from there
		fp.ceresProps = ((FittingPrefsPane^)PaneList[FITTINGPREFS])->GetFittingMethod();
		if (!Int32::TryParse(((FittingPrefsPane^)(PaneList[FITTINGPREFS]))->iterationsTextBox->Text, fp.fitIterations))
		{
			throw gcnew UserInputException("The text in the \"Iterations\" field in the Fitting Preferences Pane must be a valid number.");
		}

		fp.bProgressReport = true;
		fp.liveFitting = updateFitGraphToolStripMenuItem->Checked;
		//fp.liveGenerate = liveGenerationToolStripMenuItem->Checked;
		if (!Int32::TryParse(((PreferencesPane^)(PaneList[PREFERENCES]))->updateIntervalMSTextBox->Text, fp.msUpdateInterval))
		{
			throw gcnew UserInputException("The text in the \"Update Interval\" field in the Preferences Pane must be a valid number.");
		}
		double qmin;
		if (!Double::TryParse(((PreferencesPane^)(PaneList[PREFERENCES]))->qMinTextBox->Text, qmin) && loadedSignal == nullptr)
		{
			throw gcnew UserInputException("The text in the \"q Min\" field in the Preferences Pane must be a valid number.");
		}

		double qmax;
		if (!Double::TryParse(((PreferencesPane^)(PaneList[PREFERENCES]))->qMaxTextBox->Text, qmax) && loadedSignal == nullptr)
		{
			throw gcnew UserInputException("The text in the \"q Max\" field in the Preferences Pane must be a valid number.");
		}

		if (qmin < 0)
			throw gcnew UserInputException("q Min must be positive");
		if (qmin > qmax)
			throw gcnew UserInputException("q Min bigger than q Max");

		int resSteps;
		if (!Int32::TryParse(((PreferencesPane^)(PaneList[PREFERENCES]))->genResTextBox->Text, resSteps) && loadedSignal == nullptr)
		{
			throw gcnew UserInputException("The text in the \"Generated points\" field in the Preferences Pane must be a valid number.");
		}
		resSteps++;

		if (loadedSignal == nullptr) {
			std::vector<double> qvecv(resSteps);
			for (int i = 0; i < resSteps; i++){
				qvecv[i] = qmin + ((qmax - qmin) * (double)(i) / (resSteps - 1)); // [qmin, qmax]
			}
			qvec = vectortoarray(qvecv);
		} // Else, the qvec is already loaded
		else
		{
			qmax = qvec[qvec->Length - 1]; // This is effectively ignored, as the backend takes the qmax from the domain preferences. See PreferencesPane.cpp for the actual fix.
		}

		if (entityTree->Nodes->Count == 0)
			return ParameterTree();

		// Initialize an empty graph
		/*if (liveGenerationToolStripMenuItem->Checked) {
			std::vector<double> graph(resSteps, POSINF);

			((GraphPane2D ^)PaneList[GRAPH2D])->SetModelGraph(qvec, vectortoarray(graph));
		}*/


		ParameterTree pt;
		pt.SetNodeModel(compositeModel);

		paramStruct ps = GetCompositeDomainParameters();
		pt.SetNodeParameters(ps);

		// Create the parameter tree from the entity tree

		// Get and set the parameters for the domain
		paramStruct domps = ((PreferencesPane ^)(this->PaneList[PREFERENCES]))->GetDomainPreferences();

		// Add all populations
		for (int i = 0; i < populationTrees->Count; i++)
		{
			ParameterTree *domain = pt.AddSubModel(domainModels[i], domps);

			int ind = 0;
			for each(Entity ^ent in populationTrees[i]->Nodes) {
				// Add the sub-model and set its parameters
				domain->AddSubModel(ent->BackendModel, ent->GetParameters());

				// Add its children, if applicable
				if (ent->Nodes->Count > 0)
					InsertEntitiesToParamTree(domain->GetSubModel(ind), ent);

				ind++;
			}
		}
		
		//pt.PrintTree();
		return pt;
	}

	void MainWindow::HandleErr(String ^ title, ErrorCode err)
	{
		String ^detailStr = gcnew String("N/A");
		wchar_t details[1024];
		if (frontend->GetLastErrorMessage(job, details, 1024) && wcslen(details) > 0)
			detailStr = gcnew String(details);

		String ^ errorString = gcnew String(g_errorStrings[err]);
		String ^ messageboxstr = title +": "+ errorString;


		if (detailStr != errorString)
			if (detailStr != "N/A")
				messageboxstr += ".\nDetails: " + detailStr;

		MessageBox::Show(messageboxstr, "Error", MessageBoxButtons::OK,
			MessageBoxIcon::Warning);
	}

	void MainWindow::Generate() {
		stopInProcess = false;
		if (this->InvokeRequired) {
			Invoke(gcnew FuncNoParams(this, &MainWindow::Generate));
			return;
		}

		FittingProperties fp;
		ParameterTree pt;
		try
		{
			pt = PrepareForWork(fp);
		}
		catch (UserInputException^ e)
		{
			MessageBox::Show("Error in input fields:\n" + e->Message + "\nCorrect field and try again.", "Error", MessageBoxButtons::OK,
				MessageBoxIcon::Error);
			return;
		}

		if (pt.GetNumSubModels() == 0)
			return;

		ErrorCode err = SaveStateAndGenerate();
		//ErrorCode err = frontend->Generate(job, pt, arraytovector(qvec), fp);
		if (err) {
			HandleErr("ERROR initializing generation", err);
			return;
		}

		statusLabel->Text = "Generating Domain: 0%";

		EnableStopButton();
	}

	void MainWindow::Fit() {
		stopInProcess = false;
		if (this->InvokeRequired) {
			Invoke(gcnew FuncNoParams(this, &MainWindow::Fit));
			return;
		}

		try
		{
			if (loadedSignal == nullptr) {
				MessageBox::Show("Load the signal to fit", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return;
			}

			FittingProperties fp;
			ParameterTree pt = PrepareForWork(fp);
			if (pt.GetNumSubModels() == 0)
				return;

			// Check to make sure that there are mutable parameters and they're within the constraints
			{
				auto len = pt.ToMutabilityVector();
				std::vector<int> mut;
				mut.resize(len);
				pt.ToMutabilityVector(mut.data());

				bool anyMutables = false;
				for (const auto& m : mut)
					anyMutables |= m > 0;

				if (!anyMutables)
				{
					MessageBox::Show("There must be at least one mutable parameter in order to fit. Mark at least one parameter mutable and try again.",
						"No mutable parameters selected", MessageBoxButtons::OK);
					return;
				}

				// Make sure the mutable parameters are within the constraints
				bool within_constraints = true;
				std::vector<double> mins, maxs, vals;
				mins.resize(len);
				maxs.resize(len);
				vals.resize(len);
				pt.ToConstraintVector(mins.data(), ParameterTree::ConstraintType::CT_MINVAL);
				pt.ToConstraintVector(maxs.data(), ParameterTree::ConstraintType::CT_MAXVAL);
				pt.ToParamVector(vals.data());
				std::string bad_parameters = "Value\t\tMin\t\tMax\n";
				for (auto i = len * 0; i < len; i++)
				{
					if (mut[i] > 0)
					{
						if (vals[i] < mins[i] || vals[i] > maxs[i] || mins[i] >= maxs[i])
						{
							std::string bad_line =
								std::to_string(vals[i]) + "\t\t" +
								std::to_string(mins[i]) + "\t\t" +
								std::to_string(maxs[i]) + "\n";
							bad_parameters.append(bad_line);
							within_constraints = false;

						}
					}
				} // for

				if (!within_constraints)
				{
					MessageBox::Show(stringToClr("There are infeasible constraints:\n" + bad_parameters + "\nFix them and try again."),
						"Parameter(s) outside of constraints", MessageBoxButtons::OK);
					return;
				}

			}
			// TODO::OneStepAhead: Mask
			std::vector<int> nomask(loadedSignal->Length, 0);

			std::string message = "N/A";
			ErrorCode err = SaveStateAndFit(nomask, message);
			//ErrorCode err = frontend->Fit(job, pt, arraytovector(qvec), arraytovector(loadedSignal), nomask, fp);
			if (err) {
				HandleErr("ERROR initializing fitting", err);
				return;
			}
		}
		catch (UserInputException^ e)
		{
			MessageBox::Show(e->Message, "Error", MessageBoxButtons::OK,
				MessageBoxIcon::Warning);
			return;
		}

		statusLabel->Text = "Fitting Domain: 0%";

		EnableStopButton();
	}

	void MainWindow::Stop() {
		ControlButtonsPane ^gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		gf->stopButton->Enabled = false;
		EnableGenDFitButton(true);
		stopInProcess = true;
		frontend->Stop(job);
	}

	void MainWindow::ProgressCallback(IntPtr args, double progress) {
		if (this->InvokeRequired) {
			this->Invoke(gcnew ProgressCallbackFunc(this, &MainWindow::ProgressCallback), gcnew array<Object ^> { args, progress });
			return;
		}

		fflush(stdout);

		TaskbarManager ^tbm = TaskbarManager::Instance;

		//if(!progressBar->Visible)
		//	progressBar->Visible = true;		

		ControlButtonsPane ^ gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		if (!gf->stopButton->Enabled && !stopInProcess)
		{
			gf->stopButton->Enabled = true;
			EnableGenDFitButton(!gf->stopButton->Enabled);
		}

		if (progress < 0)
			progress = 0;
		progressBar->Value = (int)Math::Round(progress * 100.0);

		if (progressBar->Value > 0)
		{
			if (tbm) tbm->SetProgressState(TaskbarProgressBarState::Normal);
			if (tbm) tbm->SetProgressValue(progressBar->Value, 100);
		}
		else
		{
			if (tbm) tbm->SetProgressState(TaskbarProgressBarState::Indeterminate);
		}

		statusLabel->Text = statusLabel->Text->Substring(0, statusLabel->Text->LastIndexOf(" ")) + String::Format(" {0}%", Math::Round(progress * 100.0));

		if (/*liveGenerationToolStripMenuItem->Checked ||*/
			(updateFitGraphToolStripMenuItem->Checked && frontend->GetJobType(job) == JT_FIT)) {
			// Get back the intermediate result
			int gSize = frontend->GetGraphSize(job);
			if (gSize == 0)
				return;

			std::vector<double> graph(gSize, POSINF);

			if (graph.size() < 1 || !frontend->GetGraph(job, &graph[0], gSize)) {
				statusLabel->Text += "...Error";
				return;
			}


			((GraphPane2D ^)PaneList[GRAPH2D])->SetModelGraph(qvec_may_be_cropped, vectortoarray(graph));
		}

		if (updateFitDomainToolStripMenuItem->Checked && frontend->GetJobType(job) == JT_FIT) {
			// Update parameters
			ParameterTree pt;
			frontend->GetResults(job, pt);

			ParameterTreeCLI ^ptref = gcnew ParameterTreeCLI();
			ptref->pt = &pt;

			UpdateParameters(ptref);

			SymmetryView ^sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
			ParameterEditor ^pe = (ParameterEditor ^)PaneList[PARAMETER_EDITOR];

			Entity ^ent = sv->GetSelectedEntity();
			// If an entity is selected, reload parameters in UI
			if (ent) {
				ent->selected = false;
				sv->treeViewAdv1->FindNodeByTag(ent)->IsSelected = false;
				ent->selected = true;
				sv->treeViewAdv1->FindNodeByTag(ent)->IsSelected = true;
			}
		}
	}

	void MainWindow::CompletionCallback(IntPtr args, int error) {
		if (this->InvokeRequired) {
			this->Invoke(gcnew CompletionCallbackFunc(this, &MainWindow::CompletionCallback), gcnew array<Object ^> { args, error });
			return;
		}

		fflush(stdout);

		TaskbarManager ^tbm = TaskbarManager::Instance;

		if (tbm) tbm->SetProgressState(TaskbarProgressBarState::NoProgress);
		progressBar->Value = 0;

		ControlButtonsPane ^ gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		gf->stopButton->Enabled = false;
		EnableGenDFitButton(!gf->stopButton->Enabled);
		progressBar->Visible = false;

		if (error) {
			HandleErr("ERROR while computing", ErrorCode(error));
			statusLabel->Text = statusLabel->Text->Substring(0, statusLabel->Text->LastIndexOf(" ")) + " Error (" + gcnew String(g_errorStrings[error]) + ").";
			return;
		}

		statusLabel->Text = statusLabel->Text->Substring(0, statusLabel->Text->LastIndexOf(" ")) + " Done.";

		if (bIsScriptComputing) {
			bIsScriptComputing = false;
			return;
		}

		// Get back the result
		int gSize = frontend->GetGraphSize(job);
		if (gSize == 0) {
			MessageBox::Show("No graph returned");
			return;
		}

		std::vector<double> graph;
		graph.reserve(qvec->Length);
		graph.resize(gSize);

		if (graph.size() < 1 || !frontend->GetGraph(job, &graph[0], gSize)) {
			MessageBox::Show("ERROR getting graph");
			return;
		}

		if (gSize != qvec->Length)
		{
			int ctr = 0;
			for (int i = 0; i < qvec->Length; i++)
			{
				if (loadedSignal[i] < 0.)
				{
					graph.insert(graph.begin() + i, std::numeric_limits<double>::quiet_NaN());
					ctr++;
				}
			}
		}
		((GraphPane2D ^)PaneList[GRAPH2D])->SetModelGraph(qvec, vectortoarray(graph));

		// If we just finished a fitting job
		if (frontend->GetJobType(job) == JT_FIT) {
			// Update parameters
			ParameterTree pt;
			frontend->GetResults(job, pt);

			ParameterTreeCLI ^ptref = gcnew ParameterTreeCLI();
			ptref->pt = &pt;

			UpdateParameters(ptref);

			SymmetryView ^sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
			ParameterEditor ^pe = (ParameterEditor ^)PaneList[PARAMETER_EDITOR];
			Entity ^ent = sv->GetSelectedEntity();
			// If an entity is selected, reload parameters in UI
			if (ent) {
				ent->selected = false;
				sv->treeViewAdv1->FindNodeByTag(ent)->IsSelected = false;
				ent->selected = true;
				sv->treeViewAdv1->FindNodeByTag(ent)->IsSelected = true;
			}
		}
	}
	
	void MainWindow::RepopulateEntities(System::Windows::Forms::ComboBox^ entityCombo, bool ffOnly) {
		entityCombo->Items->Clear();

		// Load available primitives (Default models)
		int catCount = frontend->QueryCategoryCount(NULL);
		for (int i = 0; i < catCount; i++) {
			ModelCategory cat = frontend->QueryCategory(NULL, i);
			if (cat.type != MT_FORMFACTOR)
				continue;

			for (int j = 0; j < 16; j++) {
				if (cat.models[j] == -1)
					break;

				ModelInformation mi = frontend->QueryModel(NULL, cat.models[j]);

				if (!ffOnly || mi.ffImplemented)
					entityCombo->Items->Add(gcnew ModelInfo(gcnew String(mi.name), nullptr, cat.models[j]));
			}
		}

		if (ffOnly)
			entityCombo->Items->Add(gcnew ModelInfo("Scripted Geometry...", nullptr, 1001));
		else
			entityCombo->Items->Add(gcnew ModelInfo("Scripted Model...", nullptr, 1002));

		if (ffOnly) {
			for (int i = 0; i < catCount; i++) {
				ModelCategory cat = frontend->QueryCategory(NULL, i);
				if (cat.type != MT_SYMMETRY)
					continue;

				entityCombo->Items->Add(gcnew ModelInfo("-----------", nullptr, -1));

				for (int j = 0; j < 16; j++) {
					if (cat.models[j] == -1)
						break;

					ModelInformation mi = frontend->QueryModel(NULL, cat.models[j]);

					entityCombo->Items->Add(gcnew ModelInfo(gcnew String(mi.name), nullptr, cat.models[j]));
				}
			}

			entityCombo->Items->Add(gcnew ModelInfo("Scripted Symmetry...", nullptr, 1003));
		}

		// Populate entities from other loaded containers
		for each (String ^cont in loadedContainers) {
			std::wstring cppcont = clrToWstring(cont);
			const wchar_t *cstr = cppcont.c_str();

			// Separator
			entityCombo->Items->Add(gcnew ModelInfo("-----------", nullptr, -1));

			// Load available primitives from the container
			int catCount = frontend->QueryCategoryCount(cstr);
			for (int i = 0; i < catCount; i++) {
				ModelCategory cat = frontend->QueryCategory(cstr, i);
				if (cat.type != MT_FORMFACTOR)
					continue;

				for (int j = 0; j < 16; j++) {
					if (cat.models[j] == -1)
						break;

					ModelInformation mi = frontend->QueryModel(cstr, cat.models[j]);

					if (!ffOnly || mi.ffImplemented)
						entityCombo->Items->Add(gcnew ModelInfo(gcnew String(mi.name), cont, cat.models[j]));
				}
			}

			// Load available symmetries from the container
			if (ffOnly) {
				for (int i = 0; i < catCount; i++) {
					ModelCategory cat = frontend->QueryCategory(cstr, i);
					if (cat.type != MT_SYMMETRY)
						continue;

					entityCombo->Items->Add(gcnew ModelInfo("-----------", nullptr, -1));

					for (int j = 0; j < 16; j++) {
						if (cat.models[j] == -1)
							break;

						ModelInformation mi = frontend->QueryModel(cstr, cat.models[j]);

						entityCombo->Items->Add(gcnew ModelInfo(gcnew String(mi.name), cont, cat.models[j]));
					}
				}
			}
		}

		if (ffOnly) {
			// Separator
			entityCombo->Items->Add(gcnew ModelInfo("-----------", nullptr, -1));

			// Manually add the Electron PDB
			entityCombo->Items->Add(gcnew ModelInfo("Electron PDB File...", nullptr, 999));
			// Manually add the PDB
			entityCombo->Items->Add(gcnew ModelInfo("PDB File...", nullptr, 1999));

			// Manually add the Amplitude Grid
			entityCombo->Items->Add(gcnew ModelInfo("Amplitude Grid (AMP)...", nullptr, 1000));
		}

		entityCombo->SelectedIndex = 0;
	}

	static bool IsTable(Object ^obj) {
		if (obj == nullptr)
			return false;
		return (dynamic_cast<LuaTable ^>(obj) != nullptr);
	}

	static paramStruct ParamStructFromLuaTable(LuaTable ^subtbl) {
		paramStruct ps;

		if (!IsTable(subtbl["Parameters"])) {
			MessageBox::Show("ERROR: Parameters not specified for model");
			throw gcnew LuaException("ERROR: Parameters not specified for model");
		}

//		LuaTable ^nlp_tbl = (LuaTable ^);
		ps.nlp = Int32(LuaItemToDouble(subtbl["nlp"]) + 0.5);

		LuaTable ^ptbl = (LuaTable ^)subtbl["Parameters"];
		ps.layers = ptbl->Keys->Count;
		if (ps.layers > 0) {
			if (!IsTable(ptbl[1])) {
				MessageBox::Show("ERROR: Invalid type for parameters (Should be a table within a table)");
				throw gcnew LuaException("ERROR: Invalid type for parameters (Should be a table within a table)");
			}

			const auto number_of_parameters = ((LuaTable ^)ptbl[1])->Keys->Count;
			if (ps.nlp != number_of_parameters)
			{
				MessageBox::Show("ERROR: The number of parameters in the Parameters table does not match the number denoted in the model (nlp).");
				throw gcnew LuaException("ERROR: The number of parameters in the Parameters table does not match the number denoted in the model (nlp).");
			}
		}

		if (!IsTable(subtbl["ExtraParameters"])) {
			MessageBox::Show("ERROR: Extra parameters not specified for model");
			throw gcnew LuaException("ERROR: Extra parameters not specified for model");
		}

		ps.nExtraParams = ((LuaTable ^)subtbl["ExtraParameters"])->Keys->Count;

		if (subtbl["Use_Grid"] != nullptr) {
			ps.bSpecificUseGrid = LuaItemToBoolean(subtbl["Use_Grid"]);
		}
		else {
			ps.bSpecificUseGrid = true;
		}


		// Create the vectors
		ps.params.resize(ps.nlp);
		ps.extraParams.resize(ps.nExtraParams);
		for (int i = 0; i < ps.nlp; i++)
			ps.params[i].resize(ps.layers);

		// Layer parameters
		for (int l = 0; l < ps.layers; l++) {
			if (!IsTable(((LuaTable ^)subtbl["Parameters"])[l + 1]) ||
				(IsTable(subtbl["Mutables"]) && !IsTable(((LuaTable ^)subtbl["Mutables"])[l + 1])) ||
				(IsTable(subtbl["Constraints"]) && !IsTable(((LuaTable ^)subtbl["Constraints"])[l + 1])) ||
				(IsTable(subtbl["Sigma"]) && !IsTable(((LuaTable ^)subtbl["Sigma"])[l + 1]))) {
				MessageBox::Show("ERROR: Parameters and/or constraints are of an invalid format in layer " + Int32(l + 1).ToString());
				throw gcnew LuaException("ERROR: Parameters and/or constraints are of an invalid format in layer " + Int32(l + 1).ToString());
			}
			LuaTable ^layerv = (LuaTable ^)((LuaTable ^)subtbl["Parameters"])[l + 1];
			LuaTable ^layerm = IsTable(subtbl["Mutables"]) ? (LuaTable ^)((LuaTable ^)subtbl["Mutables"])[l + 1] : nullptr;
			LuaTable ^layerc = IsTable(subtbl["Constraints"]) ? (LuaTable ^)((LuaTable ^)subtbl["Constraints"])[l + 1] : nullptr;
			LuaTable ^layersig = IsTable(subtbl["Sigma"]) ? (LuaTable ^)((LuaTable ^)subtbl["Sigma"])[l + 1] : nullptr;

			// Setting Parameters
			for (int i = 0; i < ps.nlp; i++) {
				double val = 0.0;
				bool bMutable = false;
				bool bCons = false;
				double consMin = NEGINF, consMax = POSINF;
				int consMinInd = -1, consMaxInd = -1, linkInd = -1;
				double sigma = 0.0;

				if (layerv != nullptr && layerv[i + 1] != nullptr)
					val = LuaItemToDouble(layerv[i + 1]);
				if (layerm != nullptr && layerm[i + 1] != nullptr)
					bMutable = LuaItemToBoolean(layerm[i + 1]);
				if (layersig != nullptr && layersig[i + 1] != nullptr)
					sigma = (Double)layersig[i + 1];
				if (layerc != nullptr && layerc[i + 1] != nullptr) { // Constraints
					LuaTable ^ctbl = (LuaTable ^)layerc[i + 1];
					if (ctbl["MinValue"] != nullptr) {
						consMin = LuaItemToDouble(ctbl["MinValue"]);
						if (consMin != NEGINF)
							bCons = true;
					}
					if (ctbl["MaxValue"] != nullptr) {
						consMax = LuaItemToDouble(ctbl["MaxValue"]);
						if (consMax != POSINF)
							bCons = true;
					}
					if (ctbl["MinIndex"] != nullptr) {
						consMinInd = Int32((Double)ctbl["MinIndex"]);
						if (consMinInd != -1)
							bCons = true;
					}
					if (ctbl["MaxIndex"] != nullptr) {
						consMaxInd = Int32((Double)ctbl["MaxIndex"]);
						if (consMaxInd != -1)
							bCons = true;
					}
					if (ctbl["Link"] != nullptr) {
						linkInd = Int32((Double)ctbl["Link"]);
						if (linkInd != -1)
							bCons = true;
					}
				}

				ps.params[i][l] = Parameter(val, bMutable, bCons, consMin, consMax,
					consMinInd, consMaxInd, linkInd, sigma);
			}
		}

		// Extra parameters
		if (!IsTable(subtbl["ExtraParameters"]) ||
			(subtbl["ExtraMutables"] && !IsTable(subtbl["ExtraMutables"])) ||
			(subtbl["ExtraConstraints"] && !IsTable(subtbl["ExtraConstraints"])) ||
			(subtbl["ExtraSigma"] && !IsTable(subtbl["ExtraSigma"]))) {
			MessageBox::Show("ERROR: Parameters and/or constraints are of an invalid format in extra parameters");
			throw gcnew LuaException("ERROR: Parameters and/or constraints are of an invalid format in extra parameters");
		}
		LuaTable ^ev = (LuaTable ^)subtbl["ExtraParameters"];
		LuaTable ^em = (LuaTable ^)subtbl["ExtraMutables"];
		LuaTable ^ec = (LuaTable ^)subtbl["ExtraConstraints"];
		LuaTable ^esig = (LuaTable ^)subtbl["ExtraSigma"];
		for (int i = 0; i < ps.nExtraParams; i++) {
			double val = 0.0;
			bool bMutable = false;
			bool bCons = false;
			double consMin = NEGINF, consMax = POSINF;
			int consMinInd = -1, consMaxInd = -1, linkInd = -1;
			double sigma = 0.0;

			if (ev != nullptr && ev[i + 1] != nullptr)
				val = LuaItemToDouble(ev[i + 1]);
			if (em != nullptr && em[i + 1] != nullptr)
				bMutable = LuaItemToBoolean(em[i + 1]);
			if (esig != nullptr && esig[i + 1] != nullptr)
				sigma = (Double)esig[i + 1];
			if (ec != nullptr && ec[i + 1] != nullptr) { // Constraints
				LuaTable ^ctbl = (LuaTable ^)ec[i + 1];
				if (ctbl["MinValue"] != nullptr) {
					consMin = LuaItemToDouble(ctbl["MinValue"]);
					if (consMin != NEGINF)
						bCons = true;
				}
				if (ctbl["MaxValue"] != nullptr) {
					consMax = LuaItemToDouble(ctbl["MaxValue"]);
					if (consMax != POSINF)
						bCons = true;
				}
				if (ctbl["MinIndex"] != nullptr) {
					consMinInd = Int32((Double)ctbl["MinIndex"]);
					if (consMinInd != -1)
						bCons = true;
				}
				if (ctbl["MaxIndex"] != nullptr) {
					consMaxInd = Int32((Double)ctbl["MaxIndex"]);
					if (consMaxInd != -1)
						bCons = true;
				}
				if (ctbl["Link"] != nullptr) {
					linkInd = Int32((Double)ctbl["Link"]);
					if (linkInd != -1)
						bCons = true;
				}
			}

			ps.extraParams[i] = Parameter(val, bMutable, bCons, consMin, consMax,
				consMinInd, consMaxInd, linkInd, sigma);
		}

		// Location/rotation parameters
		if (!IsTable(subtbl["Location"]) ||
			(subtbl["LocationMutables"] && !IsTable(subtbl["LocationMutables"])) ||
			(subtbl["LocationConstraints"] && !IsTable(subtbl["LocationConstraints"])) ||
			(subtbl["LocationSigma"] && !IsTable(subtbl["LocationSigma"]))) {
			MessageBox::Show("ERROR: Location/Rotation parameters and/or constraints are of an invalid format");
			throw gcnew LuaException("ERROR: Location/Rotation parameters and/or constraints are of an invalid format");
		}
		LuaTable ^lrv = (LuaTable ^)subtbl["Location"];
		LuaTable ^lrm = (LuaTable ^)subtbl["LocationMutables"];
		LuaTable ^lrc = (LuaTable ^)subtbl["LocationConstraints"];
		LuaTable ^lrsig = (LuaTable ^)subtbl["LocationSigma"];

#define LOAD_LOCATION_PARAM(LOCP) do {											\
		double val = 0.0;															\
		bool bMutable = false;														\
		bool bCons = false;															\
		double consMin = NEGINF, consMax = POSINF;									\
		int consMinInd = -1, consMaxInd = -1, linkInd = -1;							\
		double sigma = 0.0;															\
																					\
		if(lrv != nullptr && lrv[#LOCP] != nullptr)									\
			val = LuaItemToDouble(lrv[#LOCP]);										\
		if(lrm != nullptr && lrm[#LOCP] != nullptr)									\
			bMutable = LuaItemToBoolean(lrm[#LOCP]);								\
		if(lrsig != nullptr && lrsig[#LOCP] != nullptr)								\
			sigma = (Double)lrsig[#LOCP];											\
		if(lrc != nullptr && lrc[#LOCP] != nullptr) {								\
			LuaTable ^ctbl = (LuaTable ^)lrc[#LOCP];								\
			if(ctbl["MinValue"] != nullptr) {										\
				consMin = LuaItemToDouble(ctbl["MinValue"]);						\
				if(consMin != NEGINF)												\
					bCons = true;													\
						}																		\
			if(ctbl["MaxValue"] != nullptr) {										\
				consMax = LuaItemToDouble(ctbl["MaxValue"]);						\
				if(consMax != POSINF)												\
					bCons = true;													\
						}																		\
			if(ctbl["MinIndex"] != nullptr) {										\
				consMinInd = Int32((Double)ctbl["MinIndex"]);						\
				if(consMinInd != -1)												\
					bCons = true;													\
						}																		\
			if(ctbl["MaxIndex"] != nullptr) {										\
				consMaxInd = Int32((Double)ctbl["MaxIndex"]);						\
				if(consMaxInd != -1)												\
					bCons = true;													\
						}																		\
			if(ctbl["Link"] != nullptr) {											\
				linkInd = Int32((Double)ctbl["Link"]);								\
				if(linkInd != -1)													\
					bCons = true;													\
						}																		\
				}																			\
																					\
		ps.LOCP = Parameter(val, bMutable, bCons, consMin, consMax, 				\
			consMinInd, consMaxInd, linkInd, sigma);								\
			} while(false);

		LOAD_LOCATION_PARAM(x);
		LOAD_LOCATION_PARAM(y);
		LOAD_LOCATION_PARAM(z);
		LOAD_LOCATION_PARAM(alpha);
		LOAD_LOCATION_PARAM(beta);
		LOAD_LOCATION_PARAM(gamma);

#undef LOAD_LOCATION_PARAM

		return ps;
	}

	static int modelCounter = 0;
	// Creates either a model or a symmetry, depending on the input
	Entity ^MainWindow::CreateEntityFromID(const wchar_t *container, int id, String ^name) {
		Entity ^ent = gcnew Entity();
		ent->frontend = frontend;
		ent->job = job;

		ModelInformation mi = frontend->QueryModel(container, id);
		ModelCategory cat = frontend->QueryCategory(container, mi.category);
		ModelUI *mui = new ModelUI();
		mui->setModel(frontend, container, mi, 10);
		ent->modelUI = mui;
		ent->modelName = (name != nullptr ? name : gcnew String(mi.name));
		ent->displayName = "";

		ModelRenderer mren(container, id);

		if (cat.type == MT_SYMMETRY) {
			ent->type = EntityType::TYPE_SYMMETRY;
			ent->BackendModel = frontend->CreateSymmetry(job, container, mi.modelIndex);
			ent->symmrender = mren.GetSymmetryRenderer();
		}
		else {
			ent->type = EntityType::TYPE_PRIMITIVE;
			ent->BackendModel = frontend->CreateModel(job, container, mi.modelIndex, EDProfile());
			ent->render = mren.GetRenderer();

			ent->BackendModel = frontend->CreateGeometricAmplitude(job, ent->BackendModel);
		}



		// Set default model parameters
		//////////////////////////////////////////////////////////////////////////
		paramStruct ps(mi);
		ps.params.resize(mi.nlp);

		// If there are layers to add
		if (mi.minLayers > 0) {
			ps.layers = mi.minLayers;

			// Set default parameters
			for (int k = 0; k < mi.nlp; k++) {
				ps.params[k].resize(mi.minLayers);
				for (int i = 0; i < mi.minLayers; i++)
					ps.params[k][i] = Parameter(mui->GetDefaultParamValue(i, k));
			}
		}

		// Set extra parameters' default values
		ps.extraParams.resize(mi.nExtraParams);
		for (int i = 0; i < mi.nExtraParams; i++) {
			ExtraParam ep = mui->GetExtraParameter(i);
			ps.extraParams[i] = Parameter(ep.defaultVal, false, false, ep.rangeMin, ep.rangeMax);
		}

		// Finally, update the parameters in the entity
		ent->SetParameters(ps, GetLevelOfDetail());

		return ent;
	}

	Entity ^MainWindow::RegisterManualSymmetry(String ^fname) {
		if (fname != nullptr) {
			// this todo currently irrelevant as this code is never called TODO::DOL: Load from file
		}

		// this todo currently irrelevant as this code is never called  TODO::DOL: The whole thing goes here

		return nullptr;
	}

	Entity ^MainWindow::InnerSetParameterTree(ParameterTree *pt, LuaTable ^ptbl) {
		GraphPane3D^ g3 = nullptr;
		g3 = dynamic_cast<GraphPane3D^>(PaneList[GRAPH3D]);

		Entity ^ent = nullptr;
		String ^type = (String ^)ptbl["Type"];
		if (type == nullptr) {
			MessageBox::Show("ERROR: Invalid parameter tree (missing type)");
		}
		else if (type->Equals("PDB")) {
			bool centered = LuaItemToBoolean(ptbl["Centered"]);
			ent = g3->RegisterPDB((String ^)ptbl["Filename"], (String ^)ptbl["AnomFilename"], GetLevelOfDetail(), centered, false);
		}
		else if (type->Equals("AMP")) {
			bool centered = LuaItemToBoolean(ptbl["Centered"]);
			ent = g3->RegisterAMPGrid((String ^)ptbl["Filename"], GetLevelOfDetail(), centered);
		}
		else if (type->ToLower() == "scripted geometry") {
			ent = RegisterLuaEntity((String ^)ptbl["Filename"], 1001);
		}
		else if (type->ToLower() == "scripted model") {
			ent = RegisterLuaEntity((String ^)ptbl["Filename"], 1002);
		}
		else if (type->ToLower() == "scripted symmetry") {
			ent = RegisterLuaEntity((String ^)ptbl["Filename"], 1003);
		}
		else {
			String ^container = nullptr;
			if (type->IndexOf(",") > 0)
				container = type->Substring(0, type->IndexOf(","));
			Int32 modelIndex = -1;
			if (!Int32::TryParse(type->Substring(type->IndexOf(",") + 1), modelIndex)) {
				MessageBox::Show("Invalid model index");
				return nullptr;
			}

			std::wstring cwstr;
			if (container != nullptr)
				cwstr = clrToWstring(container);
			const wchar_t *cptr = (container == nullptr) ? NULL : cwstr.c_str();

			ent = CreateEntityFromID(cptr, modelIndex, nullptr);
		}

		// Set model parameters, update the parameters in the entity
		ent->SetParameters(ParamStructFromLuaTable(ptbl), GetLevelOfDetail());
		ent->displayName = (String ^)ptbl["Name"];
		// Avi: I think this is superfluous and dangerous, as it rewrites the parameters..., no?
		// Avi(later): Without it, it doesn't load the saved parameters. However, it also ignores
		//		existing parameters, such as the extra parameter of a scripted symmetry!
	
		if (ptbl["Children"] != nullptr) {
			LuaTable ^chl = (LuaTable ^)ptbl["Children"];
			for (int i = 0; i < chl->Keys->Count; i++) {
				Entity ^subent = InnerSetParameterTree(pt->GetSubModel(i), (LuaTable ^)chl[i + 1]);
				if (subent != nullptr)
					ent->Nodes->Add(subent);
				if (subent->IsParentUseGrid() && !subent->GetParameters().bSpecificUseGrid)
				{
					MessageBox::Show("Invalid parameters tree: parent node 'use grid' is true while child node 'use grid' is false");
					return nullptr;
				}
			}
		}

		return ent;
	}

	static void InnerUpdateParameters(Entity ^ent, ParameterTree *ptree, LevelOfDetail lod) {
		if (!ptree)// || ptree->GetNodeModel() == 0) --> Did we need this? It interferes with modifying the models in the GUI from a script.
			return;

		// Modify the node
		paramStruct oldPS = ent->GetParameters();
		int nlp = oldPS.nlp, nep = oldPS.nExtraParams; // Some things never change...

		ent->SetParameters(ptree->GetNodeParameters(nlp, nep), lod);

		// Then, modify its children
		if (ptree->GetNumSubModels() != ent->Nodes->Count) {
			MessageBox::Show("Incorrect number of sub-models for " + ent->ToString() + " while updating tree");
			return;
		}

		int i = 0;
		for each (Entity ^subent in ent->Nodes) {
			InnerUpdateParameters(subent, ptree->GetSubModel(i), lod);
			i++;
		}
	}


	delegate void UPFunc(ParameterTreeCLI ^ptref);
	void MainWindow::UpdateParameters(ParameterTreeCLI ^ptref) {
		if (this->InvokeRequired) {
			Invoke(gcnew UPFunc(this, &MainWindow::UpdateParameters), gcnew array<Object ^> { ptref });
			return;
		}

		if (ptref == nullptr)
			return;

		ParameterTree *pt = ptref->pt;

		if (!pt)
			return;

		if (pt->GetNodeModel() == 0)
			return;
		if (pt->GetNodeModel() != compositeModel)
		{
//			MessageBox::Show("A weird error occurred");
			Console::WriteLine("\nA weird error occurred (pt->GetNodeModel() != compositeModel; {0} != {1}", pt->GetNodeModel(), compositeModel);
			return;
		}

		Controls3D^ c3 = dynamic_cast<Controls3D^>(PaneList[CONTROLS]);
		SymmetryView ^sv = dynamic_cast<SymmetryView ^>(PaneList[SYMMETRY_VIEWER]);

		// Modify domain scale
		paramStruct compositeps = pt->GetNodeParameters(2 + populationTrees->Count, 0);
		domainScale = compositeps.params[0][0].value;
		domainConstant = compositeps.params[1][0].value;
		sv->scaleBox->Text = "" + domainScale;
		sv->constantBox->Text = "" + domainConstant;

		// Re-create entity tree from parameter tree
		// Modify population sizes
		for (int i = 0; i < populationTrees->Count; i++)
			populationSizes[i] = compositeps.params[i + 2][0].value;
		sv->avgpopsizeText->Text = "" + populationSizes[sv->populationTabs->SelectedIndex];

		if (pt->GetNumSubModels() != populationTrees->Count) {
			MessageBox::Show("Incorrect number of subdomains while updating tree");
			return;
		}

		// Modify the parameters of all populations
		for (int i = 0; i < populationTrees->Count; i++)
		{
			Aga::Controls::Tree::TreeModel ^tree = populationTrees[i];
			ParameterTree *domain = pt->GetSubModel(i);

			if (domain->GetNumSubModels() != tree->Nodes->Count) {
				MessageBox::Show("Incorrect number of sub-models while updating tree");
				return;
			}

			int j = 0;
			for each (Entity ^ent in tree->Nodes) {
				InnerUpdateParameters(ent, domain->GetSubModel(j), GetLevelOfDetail());
				j++;
			}
		}
		

		sv->tvInvalidate();
	}

	bool MainWindow::ClearParameterTree(ParameterTree *pt) {
		bool res = true;
		for (int i = 0; i < pt->GetNumSubModels(); i++) {
			res = (res && ClearParameterTree(pt->GetSubModel(i)));
		}

		if (pt->GetNodeModel() > 0) {
			return (res && OK == frontend->DestroyModel(job, pt->GetNodeModel(), false));
		}

		return true;
	}


	ModelPtr MainWindow::ModelFromLuaString(String ^str, LuaTable ^mtbl, bool bAmp) {
		// Creates a model in the backend. If bAmp is true, another model is created - a GeometricAmplitude model.
		// The GemoetricAmplitude model contains a reference to the first model - this reference is maintained in the Backend
		// The UI receives the GeometricAmplitude and is not aware of the first model.

		if (str == nullptr) {
			MessageBox::Show("Invalid model, no type (model.Type) defined!");
			return NULL;
		}

		int comma = str->IndexOf(",");

		if (comma < 0) {
			if (str->ToLower()->Equals("scripted model")) {
				String ^scr = (String ^)mtbl["Filename"];
				if (scr == nullptr) {
					MessageBox::Show("Invalid scripted model, no script (model.Filename) defined!");
					return NULL;
				}

				msclr::interop::marshal_context context;
				std::string fn = context.marshal_as<std::string>(scr);

				std::string cscript = clrToString(System::IO::File::ReadAllText(scr));

				return frontend->CreateScriptedModel(job, cscript.c_str(), fn.c_str(), unsigned int(cscript.size()));

			}
			else if (str->ToLower()->Equals("scripted geometry")) {
				String ^scr = (String ^)mtbl["Filename"];
				if (scr == nullptr) {
					MessageBox::Show("Invalid scripted geometry, no script (model.Filename) defined!");
					return NULL;
				}

				std::string cscript = clrToString(System::IO::File::ReadAllText(scr));

				// TODO::Lua
				ModelPtr geometry = NULL; // frontend->CreateScriptedAmplitude(job, cscript.c_str(), cscript.size());
				return frontend->CreateGeometricAmplitude(job, geometry);

			}
			else if (str->ToLower()->Equals("scripted symmetry")) {
				String ^scr = (String ^)mtbl["Filename"];
				if (scr == nullptr) {
					MessageBox::Show("Invalid scripted symmetry, no script (model.Filename) defined!");
					return NULL;
				}

				msclr::interop::marshal_context context;
				std::string fn = context.marshal_as<std::string>(scr);

				std::string cscript = clrToString(System::IO::File::ReadAllText(scr));

				return frontend->CreateScriptedSymmetry(job, cscript.c_str(), fn.c_str(), unsigned int(cscript.size()));

			}
			else if (str->Equals("PDB")) {
				if (!bAmp) {
					MessageBox::Show("A PDB file cannot be chosen as a geometry, only amplitude.");
					return NULL;
				}

				String ^fname = (String ^)mtbl["Filename"];
				if (fname == nullptr) {
					MessageBox::Show("Invalid PDB, no file (model.Filename) defined!");
					return NULL;
				}

				String ^anomfname = (String ^)mtbl["AnomFilename"];

				bool centered = LuaItemToBoolean(mtbl["Centered"]);

				std::vector<std::wstring> filenames;
				std::vector<const wchar_t*> filenamePointers;
				filenames.reserve(2);
				filenamePointers.reserve(2);
				filenames.push_back(clrToWstring(fname));
				if (anomfname->Length > 0)
					filenames.push_back(clrToWstring(anomfname));
				for (const auto & f: filenames)
					filenamePointers.push_back(f.c_str());

				return frontend->CreateFileAmplitude(job, AF_PDB, filenamePointers.data(), int(filenames.size()), centered);

			}
			else if (str->ToLower()->Equals("amp")) {
				if (!bAmp) {
					MessageBox::Show("An Amplitude Grid cannot be chosen as a geometry, only amplitude.");
					return NULL;
				}

				String ^fname = (String ^)mtbl["Filename"];
				if (fname == nullptr) {
					MessageBox::Show("Invalid Amplitude Grid, no file (model.Filename) defined!");
					return NULL;
				}

				bool centered = LuaItemToBoolean(mtbl["Centered"]);

				std::wstring wst = clrToWstring(fname);
				std::vector<const wchar_t*> fn{ wst.c_str() };

				return frontend->CreateFileAmplitude(job, AF_AMPGRID, fn.data(), int(fn.size()), centered);

			}
			else { // Invalid model type
				MessageBox::Show("Invalid model type: " + str);
				return NULL;
			}
		}

		if (bAmp) { // Create a geometric amplitude
			ModelPtr geometry = NULL;

			if (comma == 0) // Default models
				geometry = frontend->CreateModel(job, NULL, Int32::Parse(str->Substring(comma + 1)), EDProfile());
			else           // With container
				geometry = frontend->CreateModel(job, clrToWstring(str->Substring(0, comma)).c_str(), Int32::Parse(str->Substring(comma + 1)), EDProfile());

			if (geometry)
				return frontend->CreateGeometricAmplitude(job, geometry);

			// If it is not a geometry, it may be a symmetry
			if (comma == 0) // Default models
				geometry = frontend->CreateSymmetry(job, NULL, Int32::Parse(str->Substring(comma + 1)));
			else           // With container
				geometry = frontend->CreateSymmetry(job, clrToWstring(str->Substring(0, comma)).c_str(), Int32::Parse(str->Substring(comma + 1)));

			return geometry;
		}
		else {
			if (comma == 0) // Default models
				return frontend->CreateModel(job, NULL, Int32::Parse(str->Substring(comma + 1)), EDProfile());
			else           // With container
				return frontend->CreateModel(job, clrToWstring(str->Substring(0, comma)).c_str(), Int32::Parse(str->Substring(comma + 1)), EDProfile());
		}
	}

	void MainWindow::InnerParamTreeFromTable(LuaTable ^tbl, ParameterTree *pt, bool bAmplitude) {
		InnerParamTreeFromTable(tbl, pt, bAmplitude, false);
	}

	void MainWindow::InnerParamTreeFromTable(LuaTable ^tbl, ParameterTree *pt, bool bAmplitude, bool bSkipModelCreationInBackend) {
		for (int i = 1; i <= tbl->Keys->Count; i++) {
			LuaTable ^mtbl = (LuaTable ^)tbl[i];

			ModelPtr mdl = NULL;
			if (!bSkipModelCreationInBackend) {
				mdl = ModelFromLuaString((String ^)mtbl["Type"], mtbl, bAmplitude);
				if (mdl == NULL) {
					MessageBox::Show("ERROR: Invalid model type specified at " + Int32(i).ToString() + ": " + (String ^)mtbl["Type"]);
					throw gcnew LuaException("Invalid model type");
				}
			}

			pt->AddSubModel(mdl, ParamStructFromLuaTable(mtbl));


			// Add children
			if (mtbl["Children"] != nullptr)
				InnerParamTreeFromTable((LuaTable ^)mtbl["Children"], pt->GetSubModel(i - 1),
				bAmplitude, bSkipModelCreationInBackend);
		}
	}

	ParameterTree MainWindow::ParamTreeFromTable(LuaTable ^tbl) {
		return ParamTreeFromTable(tbl, false);
	}

	ParameterTree MainWindow::ParamTreeFromTable(LuaTable ^tbl, bool bSkipModelCreationInBackend) {
		ParameterTree pt;
		bool bAmplitude = false;

		String ^type = (String ^)tbl["Geometry"];
		if (type == nullptr) {
			MessageBox::Show("Invalid parameter tree! Root incorrect");
			return pt;
		}

		double domscale = 1.0;
		bool domscalemut = false;
		if (tbl["Scale"] != nullptr)
			domscale = LuaItemToDouble(tbl["Scale"]);
		if (tbl["ScaleMut"] != nullptr)
			domscalemut = LuaItemToBoolean(tbl["ScaleMut"]);

		pt.SetNodeModel(compositeModel);

		ParameterTree *domain = &pt;

		if (type->Equals("Domain")) {
			// EASE-OF-USE / BACKWARD COMPATIBILITY: Allow a single domain in Lua
			bAmplitude = true;

			// Resize number of domain models
			ResizeNumDomainModels(1);

			domain = pt.AddSubModel(domainModels[0]);
			AddModelsToParamTreeFromTable(domain, tbl, bAmplitude, bSkipModelCreationInBackend);		  // Creates the models in the backend

			// Set scale and population size parameters (default values)
			paramStruct ps;
			ps.nlp = 1 + 1; // THE NUMBER OF PARAMS + NUMBER OF SUBDOMAINS
			ps.layers = 1;
			ps.params.resize(ps.nlp);
			ps.params[0].push_back(Parameter(domscale, domscalemut)); // SCALE
			ps.params[1].push_back(Parameter(1.0, false)); // AVERAGE POPULATION SIZE

			pt.SetNodeParameters(ps);
		}
		else if (type->Equals("Domains")) {
			bAmplitude = true;

			std::vector<double> popsizes;
			std::vector<bool> popsizemut;

			// Get populations
			LuaTable ^pops = (LuaTable ^)tbl["Populations"];

			// No populations
			if (pops == nullptr || pops->Length < 1) {
				MessageBox::Show("Invalid parameter tree! No populations");
				return pt;
			}

			// Modify domain count in GUI and backend
			ResizeNumDomainModels(pops->Length);

			for (int i = 0; i < pops->Length; i++) {
				LuaTable ^poptbl = (LuaTable ^)pops[i + 1];	  // poptbl is the i'th population (STATE)		

				domain = pt.AddSubModel(domainModels[i]);  // domain is the i'th population (IN MEMORY)
				AddModelsToParamTreeFromTable(domain, poptbl, bAmplitude, bSkipModelCreationInBackend);  // Fill the MEMORY population from the STATE and creates model in the backend
				popsizes.push_back(LuaItemToDouble(poptbl["PopulationSize"]));
				popsizemut.push_back(LuaItemToBoolean(poptbl["PopulationSizeMut"]));
			}

			// Set scale and population size parameters.
			paramStruct ps;
			ps.nlp = 1 + pops->Length; // THE NUMBER OF PARAMS + NUMBER OF SUBDOMAINS
			ps.layers = 1;
			ps.params.resize(ps.nlp);
			ps.params[0].push_back(Parameter(domscale, domscalemut)); // SCALE
			for (int i = 0; i < pops->Length; i++)
				ps.params[i + 1].push_back(Parameter(popsizes[i], popsizemut[i])); // AVERAGE POPULATION SIZE

			pt.SetNodeParameters(ps);
		}
		else {
			MessageBox::Show("Invalid parameter tree! Invalid root type");
			return pt;
		}

		return pt;
	}

	void MainWindow::ResizeNumDomainModels(int populations)
	{
		if (populations < 1)
		{
			MessageBox::Show("Invalid number of populations requested (RNDM), canceling.");
			return;
		}

		// Remove extras (if necessary)
		while (populations < domainModels->Count)
		{
			frontend->DestroyModel(job, domainModels[domainModels->Count - 1], false);
			domainModels->RemoveAt(domainModels->Count - 1);
		}

		// Add new domains (if necessary)
		while (populations > domainModels->Count)
		{
			domainModels->Add(frontend->CreateDomainModel(job));
		}

	}

	void MainWindow::ClearAndSetPopulationCount(int populations)
	{
		if (populations < 1)
		{
			MessageBox::Show("Invalid number of populations requested, canceling.");
			return;
		}

		int oldPopulations = populationTrees->Count;

		// Clear all population entity trees
		for (int i = 0; i < oldPopulations; i++)
		{
			Aga::Controls::Tree::TreeModel ^tree = populationTrees[i];
			for each(Entity ^ent in tree->Nodes)
				delete ent;
			tree->Nodes->Clear();
		}

		SymmetryView ^sv = dynamic_cast<SymmetryView ^>(PaneList[SYMMETRY_VIEWER]);


		// Add new, empty populations
		for (int i = 0; i < populations; i++)
			sv->AddPopulation();

		// Remove old populations
		for (int i = 0; i < oldPopulations; i++)
			sv->RemovePopulation(0);
	}

	void MainWindow::AddModelsToParamTreeFromTable(ParameterTree * domain, LuaTable ^ tbl, bool bAmplitude, bool bSkipModelCreationInBackend)
	{
		// No models
		if (tbl["Models"] == nullptr)
			return;

		LuaTable ^mods = (LuaTable ^)tbl["Models"];  // Models in STATE

		for (int i = 1; i <= mods->Length; i++) {  // mods[i] - the i'th model in STATE
			paramStruct ps = ParamStructFromLuaTable((LuaTable ^)mods[i]);  // ps reflects the parameters in STATE

			LuaTable ^mtbl = (LuaTable ^)mods[i];  // mtbl - the i'th model in STATE

			ModelPtr newModel = NULL;
			if (!bSkipModelCreationInBackend)
				newModel = ModelFromLuaString((String ^)mtbl["Type"], mtbl, bAmplitude);
			domain->AddSubModel(newModel, ps);  // Adds the GeometricAmplitude to the Parameter Tree IN MEMORY.

			// Add children
			if (mtbl["Children"] != nullptr)  // Go over all submodels
				InnerParamTreeFromTable((LuaTable ^)mtbl["Children"], domain->GetSubModel(i - 1),
				bAmplitude, bSkipModelCreationInBackend);
		}

		if (!bAmplitude) {
			if (domain->GetNumSubModels() == 0) // Add an empty form factor, if required
				domain->AddSubModel(NULL);

			// Add SF and BG
			domain->AddSubModel(NULL);
			domain->AddSubModel(NULL);
		}
	}

	void MainWindow::EnableStopButton() {
		if (this->InvokeRequired) {
			Invoke(gcnew FuncNoParams(this, &MainWindow::EnableStopButton));
			return;
		}

		ControlButtonsPane ^gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		gf->stopButton->Enabled = true;
		EnableGenDFitButton(false);

	}
	void MainWindow::EnableGenDFitButton(bool newStatus)
	{
		ControlButtonsPane ^gf = (ControlButtonsPane ^)PaneList[CALC_BUTTONS];
		gf->generateButton->Enabled = newStatus;
		gf->fitButton->Enabled = newStatus;
	}

	System::Void MainWindow::Figure_FormClosing(System::Object^ sender, System::Windows::Forms::FormClosingEventArgs^ e) {
		// Called whenever a figure is closed
		GraphFigure ^fig = (GraphFigure ^)sender;
		openFigures->Remove(fig->figure);

		if (lastFigure == fig->figure) {
			while (!openFigures->ContainsKey(lastFigure)) {
				if (lastFigure == -1)
					break;

				lastFigure--;
			}
		}
	}

	void MainWindow::OpenFigureHelper() {
		openFigures[lastFigure]->Show();

		Application::Run(openFigures[lastFigure]);
	}

	delegate String ^ SaveTable(LuaTable ^orig);
	delegate String ^ TableToString(LuaTable ^orig, String ^dName);
	System::Void MainWindow::saveParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (!luaState) {
			MessageBox::Show("Lua not loaded!!", "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);		return;
		}
		SaveFileDialog ^sfd = gcnew SaveFileDialog();
		sfd->Filter = "Saved States (*.state)|*.state|All Files (*.*)|*.*";
		sfd->FileName = "";
		sfd->Title = "Save State As...";
		sfd->OverwritePrompt = true;
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		SaveParametersToFile(sfd->FileName);
	}

	delegate String ^GetNameDelegate(int index);
	delegate Double GetDefaultParamValueDelegate(int layer, int layerParam);
	delegate Boolean IsParamApplicableDelegate(int layer, int layerParam);


	Entity ^MainWindow::RegisterLuaEntity(String ^filename, int type) {
		if (type < 1001 || type > 1003)
			return nullptr;
		if (luaState == nullptr) // was not loaded
			return nullptr;

		Entity ^ent = gcnew Entity();
		ent->frontend = frontend;
		ent->job = job;
		ent->filename = filename;

		String ^script = "";
		ModelInformation mi;
		GetNameFunc layernameFunc = NULL, layerparamnameFunc = NULL;
		GetDefaultParamValueFunc dpvFunc = NULL;
		GetParamApplicableFunc ipaFunc = NULL;
		{
			Lua ^modelContext = gcnew Lua();
			modelContext->DoString("os = nil"); // Deny access to OS

			// Read file
			try {
				script = System::IO::File::ReadAllText(filename);
			}
			catch (Exception ^ex) {
				MessageBox::Show("Error reading script: " + ex->Message, "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return nullptr;
			}

			// Compile for information and methods
			try {
				modelContext->DoFile(filename);
			}
			catch (Exception ^ex) {
				MessageBox::Show("Error compiling script: " + ex->Message, "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return nullptr;
			}

			// Obtain model information
			mi = LuaParseInformationTable(modelContext->GetTable("Information"));
			if (mi.nlp < 0) {
				MessageBox::Show("Error parsing information table", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return nullptr;
			}

			// Verify "Type" field and resulting backend functions		
			String ^mtype = dynamic_cast<String ^>(modelContext->GetTable("Information")["Type"]);
			if (mtype == "Geometry") {
				if (type != 1001) {
					MessageBox::Show("Script type mismatch", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}

				// MANDATORY: function CalculateFF(qvec, p, nlayers)
				// OPTIONAL:  function PreCalculate(p, nlayers)			
				// OPTIONAL:  function Derivative(qvec, p, nlayers, index)
				if (modelContext->GetFunction("CalculateFF") == nullptr) {
					MessageBox::Show("Scripted geometry must define a \"CalculateFF\" function", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}

				// TODO::OneStepAhead: Load GPU strings
			}
			else if (mtype == "Model") {
				if (type != 1002) {
					MessageBox::Show("Script type mismatch", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}

				// MANDATORY: function Calculate(qvec, p, nlayers)
				// OPTIONAL:  function PreCalculate(p, nlayers)			
				// OPTIONAL:  function Derivative(qvec, p, nlayers, index)
				if (modelContext->GetFunction("Calculate") == nullptr) {
					MessageBox::Show("Scripted model must define a \"Calculate\" function", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}


				// TODO::OneStepAhead: Load GPU strings
			}
			else if (mtype == "Symmetry") {
				if (type != 1003) {
					MessageBox::Show("Script type mismatch", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}

				// MANDATORY: function Populate(p, nlayers)
				if (modelContext->GetFunction("Populate") == nullptr) {
					MessageBox::Show("Scripted symmetry must define a \"Populate\" function", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return nullptr;
				}
				if (mi.nExtraParams == 0) {
					mi.nExtraParams = 1;
				}
			}
			else {
				MessageBox::Show("Invalid script type", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return nullptr;
			}

			// Obtain UI functions
			Delegate ^del;
			// function GetLayerName(index)				
			del = modelContext->GetFunction(GetNameDelegate::typeid, "GetLayerName");
			if (del != nullptr) {
				ent->hlnf = GCHandle::Alloc(del);
				layernameFunc = (GetNameFunc)Marshal::GetFunctionPointerForDelegate(del).ToPointer();
			}

			// function GetLayerParameterName(index)
			del = modelContext->GetFunction(GetNameDelegate::typeid, "GetLayerParameterName");
			if (del != nullptr) {
				ent->hlpnf = GCHandle::Alloc(del);
				layerparamnameFunc = (GetNameFunc)Marshal::GetFunctionPointerForDelegate(del).ToPointer();
			}

			// function IsParamApplicable(layer, layerParam)
			del = modelContext->GetFunction(IsParamApplicableDelegate::typeid, "IsParamApplicable");
			if (del != nullptr) {
				ent->hipaf = GCHandle::Alloc(del);
				ipaFunc = (GetParamApplicableFunc)Marshal::GetFunctionPointerForDelegate(del).ToPointer();
			}

			// function GetDefaultValue(layer, layerParam)
			del = modelContext->GetFunction(GetDefaultParamValueDelegate::typeid, "GetDefaultValue");
			if (del != nullptr) {
				ent->hdpvf = GCHandle::Alloc(del);
				dpvFunc = (GetDefaultParamValueFunc)Marshal::GetFunctionPointerForDelegate(del).ToPointer();
			}

			//////////////////////////////////////////////////////////////////////////

			ent->modelContext = modelContext;
		}

		std::string cscript = clrToString(script);

		ScriptedModelUI *mui = new ScriptedModelUI();
		mui->setModel(NULL, NULL, mi, 0);
		mui->SetHandlerCallbacks(layernameFunc, layerparamnameFunc, dpvFunc, ipaFunc);
		ent->modelUI = mui;
		ent->modelName = gcnew String(mi.name);
		ent->displayName = "";

		if (type == 1001) { // Scripted geometry
			ent->type = EntityType::TYPE_PRIMITIVE;

			// TODO::Lua: Implement
			//ent->BackendModel = frontend->CreateScriptedAmplitude(job, cscript.c_str(), cscript.size());
			ent->BackendModel = NULL;
			ent->BackendModel = frontend->CreateGeometricAmplitude(job, ent->BackendModel);
			// TODO::Lua: Renderer

		}
		else if (type == 1002) { // Scripted model
			ent->type = EntityType::TYPE_PRIMITIVE;
			msclr::interop::marshal_context context;
			std::string fn = context.marshal_as<std::string>(filename);
			ent->BackendModel = frontend->CreateScriptedModel(job, cscript.c_str(), fn.c_str(), unsigned int(cscript.size()));
			// TODO::Lua: Renderer

		}
		else if (type == 1003) { // Scripted symmetry
			ent->type = EntityType::TYPE_SYMMETRY;
			msclr::interop::marshal_context context;
			std::string fn = context.marshal_as<std::string>(filename);
			ent->BackendModel = frontend->CreateScriptedSymmetry(job, cscript.c_str(), fn.c_str(), unsigned int(cscript.size()));
		}

		// Set default model parameters
		//////////////////////////////////////////////////////////////////////////
		paramStruct ps(mi);
		ps.params.resize(mi.nlp);

		// If there are layers to add
		if (mi.minLayers > 0) {
			ps.layers = mi.minLayers;

			// Set default parameters
			for (int k = 0; k < mi.nlp; k++) {
				ps.params[k].resize(mi.minLayers);
				for (int i = 0; i < mi.minLayers; i++)
					ps.params[k][i] = Parameter(mui->GetDefaultParamValue(i, k));
			}
		}

		// Set extra parameters' default values
		ps.extraParams.resize(mi.nExtraParams);
		for (int i = 0; i < mi.nExtraParams; i++) {
			ExtraParam ep = mui->GetExtraParameter(i);
			ps.extraParams[i] = Parameter(ep.defaultVal, false, false, ep.rangeMin, ep.rangeMax);
		}

		// Finally, update the parameters in the entity
		ent->SetParameters(ps, GetLevelOfDetail());

		return ent;
	}

	System::Void MainWindow::importParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (!luaState) {
			MessageBox::Show("Lua not loaded!!", "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);		return;
		}
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Filter = "Saved States (*.state)|*.state|All Files (*.*)|*.*";
		ofd->FileName = "";
		ofd->Title = "Import State...";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		ImportParameterFile(ofd->FileName);
	}

	System::Void MainWindow::saveVantagePointToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (!luaState) {
			MessageBox::Show("Lua not loaded!!", "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);		return;
		}
		SaveFileDialog ^sfd = gcnew SaveFileDialog();
		sfd->Filter = "Saved States (*.state)|*.state|All Files (*.*)|*.*";
		sfd->FileName = "";
		sfd->Title = "Save State As...";
		sfd->OverwritePrompt = true;
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		LuaTable ^pt = GetParameterTree();
		System::IO::File::WriteAllText(sfd->FileName,
			((Controls3D ^)PaneList[CONTROLS])->SerializePreferences() + "\n" +
			"Domain = " + LuaTableToString(pt), System::Text::Encoding::ASCII);

	}

	System::Void MainWindow::loadVantagePointToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (!luaState) {
			MessageBox::Show("Lua not loaded!!", "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);		return;
		}
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Filter = "Saved States (*.state)|*.state|All Files (*.*)|*.*";
		ofd->FileName = "";
		ofd->Title = "Import Vantage Point...";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		try {
			Lua ^loadState = gcnew Lua();
			loadState->DoString("os=nil;", "Load Saved State Stuff");
			loadState->DoFile(ofd->FileName);

			if (loadState->GetTable("Viewport") != nullptr)
				((Controls3D ^)PaneList[CONTROLS])->DeserializePreferences(
				(LuaTable ^)loadState->GetTable("Viewport"));
		}
		catch (Exception ^ex) {
			MessageBox::Show("Failed to load domain: " + ex->Message, "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);
		}
	}



	void MainWindow::SaveParametersToFile(String ^filename) {

		List<String ^>^ _tableNames = gcnew List<String^>();
		_tableNames->Add("DomainPreferences");
		_tableNames->Add("FittingPreferences");
		_tableNames->Add("Viewport");
		_tableNames->Add("Domain");

		std::string path = LuaToJSON::saveJsonFile();
		String ^ jsonlua = gcnew String(path.c_str());

		Lua ^ paramLua = gcnew Lua();
		paramLua->DoString("JSON = assert(loadfile \"" + jsonlua + "\")()");
		try
		{
			paramLua->DoString(GetLuaScript()); //load up the existing parameters into the paramLua
		}
		catch (UserInputException ^e)
		{
			MessageBox::Show("Error in input fields:\n" + e->Message + "\nCorrect field and try again.", "Error", MessageBoxButtons::OK,
				MessageBoxIcon::Error);
			return;

		}
		String ^ json = "{";

		for each (String ^ table in _tableNames)
		{
			Object ^ res = paramLua->DoString("raw_json_text= JSON:encode_pretty(" + table + ")");
			String ^ str = (String ^)paramLua["raw_json_text"];
			json += "\"" + table + "\": " + str + ",";

		}


		String ^ jsonstring = json->Substring(0, json->LastIndexOf(",")) + "}";
		System::IO::File::WriteAllText(filename, jsonstring, System::Text::Encoding::ASCII);
	}


	void MainWindow::ImportParameterFile(String ^filename) {
		if (this->InvokeRequired) {
			this->Invoke(gcnew FuncString(this, &MainWindow::ImportParameterFile), gcnew array<Object ^> { filename });
			return;
		}

		try {

			std::string path = LuaToJSON::saveJsonFile();
			String ^ jsonlua = gcnew String(path.c_str());
			Lua ^loadState = gcnew Lua();

			try{ //try parsing json:
				String ^ json = System::IO::File::ReadAllText(filename);
				JObject ^ json_items = JObject::Parse(json);
				
				loadState->DoString("JSON = assert(loadfile \"" + jsonlua +"\")()");


				List<String ^>^ _tableNames = gcnew List<String^>();
				_tableNames->Add("DomainPreferences");
				_tableNames->Add("FittingPreferences");
				_tableNames->Add("Viewport");
				_tableNames->Add("Domain");

				for each (String ^ table in _tableNames)
				{
					JToken ^ value;
					if (json_items->TryGetValue(table, value))
					{
						loadState["raw_json_text"] = value->ToString();//json_items[table]->ToString();
						loadState->DoString(table + "= JSON:decode(raw_json_text)");
					}
				}
			}
			catch (Exception ^)
			{
				//check if its the old Lua format state file
				loadState->DoFile(filename);
			}
			//remove old signal
			closeSignalToolStripMenuItem_Click(gcnew System::Object(), gcnew System::EventArgs());
			if (loadState->GetTable("Domain") != nullptr)
				SetParameterTree((LuaTable ^)loadState->GetTable("Domain"));

			if (loadState->GetTable("DomainPreferences") != nullptr)
				((PreferencesPane ^)PaneList[PREFERENCES])->DeserializePreferences(
				(LuaTable ^)loadState->GetTable("DomainPreferences"));

			if (loadState->GetTable("FittingPreferences") != nullptr)
				((FittingPrefsPane^)PaneList[FITTINGPREFS])->DeserializePreferences(
				(LuaTable ^)loadState->GetTable("FittingPreferences"));


			if (loadState->GetTable("Viewport") != nullptr)
				((Controls3D ^)PaneList[CONTROLS])->DeserializePreferences(
				(LuaTable ^)loadState->GetTable("Viewport"));


		}
		catch (Exception ^ex) {
			MessageBox::Show("Failed to load domain: " + ex->Message, "ERROR", MessageBoxButtons::OK,
				MessageBoxIcon::Error);
		}
	}



	System::Void MainWindow::undoToolStrip_Click(System::Object^ sender, System::EventArgs^ e) {

	}

	System::Void MainWindow::redoToolStrip_Click(System::Object^ sender, System::EventArgs^ e) {

	}

	System::Void MainWindow::MainWindow_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
		// Layout:		saveLayoutToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
		//				loadLayoutToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
		// Parameters:	saveParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
		//				importParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)

		// Save parameters
		if ((e->KeyCode == Keys::S) && e->Control) {
			if (e->Shift)
				;//this->save_Click(saveParametersAsToolStripMenuItem, gcnew EventArgs());
			else
				;//this->save_Click(saveParametersToolStripMenuItem, gcnew EventArgs());

			e->Handled = true;
		}
		// Open parameters
		if ((e->KeyCode == Keys::O) && e->Control) {
			if (e->Shift)
				;//this->save_Click(saveParametersAsToolStripMenuItem, gcnew EventArgs());
			else
				;//this->save_Click(saveParametersToolStripMenuItem, gcnew EventArgs());

			e->Handled = true;
		}
		// Export 1D graph
		if ((e->KeyCode == Keys::E) && e->Control) {

			e->Handled = Save1DGraph();
		}

	}

	ModelInformation MainWindow::LuaParseInformationTable(LuaTable ^infotbl) {
		ModelInformation mi;
		mi.nlp = -1;

		if (infotbl == nullptr)
			return mi;

		if (infotbl["Name"] != nullptr && dynamic_cast<String ^>(infotbl["Name"]) != nullptr) {
			std::string cname = clrToString(dynamic_cast<String ^>(infotbl["Name"]));
			strncpy_s(mi.name, cname.c_str(), cname.size());
		}
		else {
			std::string cname = "Unnamed Scripted Symmetry";
			strncpy_s(mi.name, cname.c_str(), cname.size());
		}

		if (infotbl["Type"] == nullptr)
			return mi;

		if (infotbl["MinLayers"] != nullptr)
			mi.minLayers = Int32(LuaItemToDouble(infotbl["MinLayers"]));
		else
			mi.minLayers = -1;
		if (infotbl["MaxLayers"] != nullptr)
			mi.maxLayers = Int32(LuaItemToDouble(infotbl["MinLayers"]));
		else
			mi.maxLayers = -1;
		if (infotbl["ExtraParameters"] != nullptr)
			mi.nExtraParams = Int32(LuaItemToDouble(infotbl["ExtraParameters"]));
		else
			mi.nExtraParams = 0;
		if (infotbl["DisplayParameters"] != nullptr)
			mi.nDispParams = Int32(LuaItemToDouble(infotbl["DisplayParameters"]));
		else
			mi.nDispParams = 0;

		mi.isGPUCompatible = LuaItemToBoolean(infotbl["GPUCompatible"]);


		// NLP is last because it determines whether the model information was
		// loaded correctly
		if (infotbl["NLP"] != nullptr)
			mi.nlp = Int32(LuaItemToDouble(infotbl["NLP"]));
		else
			mi.nlp = -1;


		return mi;
	}

	System::Void MainWindow::openSignalToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Filter = "Signal files (*.dat, *.chi, *.out)|*.dat;*.chi;*.out|All Files (*.*)|*.*";
		ofd->FileName = "";
		ofd->Title = "Choose a signal to work with";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		LoadSignal(ofd->FileName);
	}


	void MainWindow::LoadSignal(String ^filename) {
		if (!System::IO::File::Exists(filename)) {
			MessageBox::Show("File " + filename + " does not exist!", "Warning",
				MessageBoxButtons::OK, MessageBoxIcon::Warning);
			return;
		}

		std::wstring fname = clrToWstring(filename);
		std::vector<double> xv, yv;
		ReadDataFile(fname.c_str(), xv, yv);
		if (xv.size() == 0 || yv.size() == 0) {
			MessageBox::Show("Error loading file", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		int negative_values = (Eigen::Map<Eigen::ArrayXd>(yv.data(), yv.size()) < 0).count();
		if (negative_values > 0)
		{
			String^ message = gcnew String("");
			message += "There are " + Int32(negative_values).ToString() + " negative intensity values. These will be removed when fitting.";
			MessageBox::Show(message);
		}
		
		// Load signal
		qvec = vectortoarray(xv);
		loadedSignal = vectortoarray(yv);
		signalFilename = filename;
		frontend->AddFileToExistingModel(-1, clrToWstring(signalFilename).c_str()); //this adds the signal file name to the filemap, important for remote d+. used -1 as modelptr to avoid conflicting with anything else
		//TODO: check that -1 doesn't cause bugs anywhere

		// Add graph to 2D pane
		((GraphPane2D ^)PaneList[GRAPH2D])->SetSignalGraph(qvec, loadedSignal);

		// Modify title
		if (this->Text->IndexOf(" - ") >= 0)
			this->Text = this->Text->Substring(0, this->Text->IndexOf(" - "));
		this->Text += " - " + filename->Substring(filename->LastIndexOf("\\") + 1);

		// Hide qMax
		PreferencesPane ^pr = (PreferencesPane ^)PaneList[PREFERENCES];
		pr->validate_qs = false;
		if (pr->qMaxLabel->Enabled)
		{
			staticQMaxString = pr->qMaxTextBox->Text;
		}
		if (pr->qMinLabel->Enabled)
		{
			staticQMinString = pr->qMinTextBox->Text;
		}

		pr->qMaxTextBox->Text = Double(qvec[qvec->Length - 1]).ToString();
		pr->qMaxLabel->Enabled = false;
		pr->qMaxTextBox->Enabled = false;

		pr->qMinTextBox->Text = Double(qvec[0]).ToString();
		pr->qMinLabel->Enabled = false;
		pr->qMinTextBox->Enabled = false;

		pr->genResLabel->Enabled = false;
		pr->genResTextBox->Enabled = false;

		pr->validate_qs = true;
	}


	System::Void MainWindow::closeSignalToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
		if (loadedSignal == nullptr)
			return;

		// Remove signal
		qvec = nullptr;
		loadedSignal = nullptr;
		frontend->RemoveFileFromExistingModel(-1, clrToWstring(signalFilename).c_str()); //TODO: this assumes the signal was added and may crash if it was not...
		signalFilename = "";

		// Remove graph from 2D pane
		((GraphPane2D ^)PaneList[GRAPH2D])->ClearSignalGraph();

		// Show qMax
		PreferencesPane ^pr = (PreferencesPane ^)PaneList[PREFERENCES];
		pr->validate_qs = false;
		pr->qMaxTextBox->Text = staticQMaxString;
		pr->qMaxLabel->Enabled = true;
		pr->qMaxTextBox->Enabled = true;

		pr->qMinTextBox->Text = staticQMinString;
		pr->qMinLabel->Enabled = true;
		pr->qMinTextBox->Enabled = true;

		pr->genResLabel->Enabled = true;
		pr->genResTextBox->Enabled = true;

		// Modify title
		this->Text = this->Text->Substring(0, this->Text->IndexOf(" - "));
		pr->validate_qs = false;
	}

	System::Double MainWindow::LuaParseExpression(String ^ val) {
		// Parses a Lua expression from "=expr" string

		// Try to evaluate the expression
		String ^evalExpr = val->Trim();

		// As a global
		try {
			Object ^obj = luaState[evalExpr];
			if (obj != nullptr) { // If evaluation
				if (dynamic_cast<Double ^>(obj) == nullptr) {
					MessageBox::Show("Expression is not a number");
					return 0.0;
				}

				return (Double)obj;
			}
		}
		catch (Exception ^) {}

		// As an expression
		try {
			luaState->DoString("_TEMPEXPR = (" + evalExpr + ")");

			Object ^obj = luaState["_TEMPEXPR"];
			if (obj != nullptr) { // If evaluation
				if (dynamic_cast<Double ^>(obj) == nullptr) {
					MessageBox::Show("Expression is not a number");
					return 0.0;
				}

				return (Double)obj;
			}
		}
		catch (LuaInterface::LuaCompileException ^ex) {
			// If evaluation of a nonexistent global
			if (ex->Message->EndsWith("'=' expected near '<eof>'"))
				MessageBox::Show("Expression value is nil or does not exist");
			else
				MessageBox::Show("Expression compile error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1));
		}
		catch (LuaInterface::LuaScriptException ^ex) {
			MessageBox::Show("Expression runtime error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1));
		}
		catch (LuaInterface::LuaException ^ex) {
			MessageBox::Show("Expression execution error: " + ex->Message->Substring(ex->Message->IndexOf(":") + 1));
		}
		catch (Exception ^ex) {
			MessageBox::Show("Expression general error: " + ex->Message);
		}

		return 0.0;
	}

	LevelOfDetail MainWindow::GetLevelOfDetail() {
		return LevelOfDetail(((Controls3D ^)PaneList[CONTROLS])->lodTrackbar->Value);
	}

	System::Void MainWindow::export1DGraphToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		this->Save1DGraph();
	}

	System::Void MainWindow::exportAmplitudeFileToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		SymmetryView ^sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
		//ParameterEditor ^pe = (ParameterEditor ^)PaneList[PARAMETER_EDITOR];
		Entity ^ent = sv->GetSelectedEntity();
		ParameterTree pt;
		if (!ent) {
			// TODO::HEADER - I'm not being successfull when nothing is selected... Should export the entire tree's amplitude.
			if (sv->treeViewAdv1->SelectedNodes->Count != 1) {
				MessageBox::Show("A single element in the tree must be selected!");
				return;
			}
			/************************************************************************/
			// Create the parameter tree from the entity tree
			// 		pt.SetNodeModel(domainModel);
			// 
			// 		// Get and set the parameters for the domain
			// 		paramStruct ps = ((PreferencesPane ^)(this->PaneList[PREFERENCES]))->GetDomainPreferences();
			// 		pt.SetNodeParameters(ps);
			// 
			// 		int ind = 0;
			// 		for each(Entity ^enti in entityTree->Nodes) {			
			// 			// Add the sub-model and set its parameters
			// 			pt.AddSubModel(enti->BackendModel, enti->GetParameters());
			// 
			// 			// Add its children, if applicable
			// 			if(enti->Nodes->Count > 0)
			// 				InsertEntitiesToParamTree(pt.GetSubModel(ind), enti);
			// 
			// 			ind++;
			// 		}
			// 		/************************************************************************/
			// 		if(!pt.GetNodeModel()) {
			// 			return;
			// 		}
		}

		SaveFileDialog ^sfd = gcnew SaveFileDialog();
		sfd->Filter = "Saved Amplitude (*.ampj)|*.ampj|All Files (*.*)|*.*";
		sfd->FileName = "";
		sfd->Title = "Save Amplitude As...";
		sfd->OverwritePrompt = true;
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		auto filename = clrToWstring(sfd->FileName);
		if (this->frontend->ExportAmplitude(job, ((ent) ? ent->BackendModel : domainModels[sv->populationTabs->SelectedIndex]), filename.c_str())) {
			return;
		}

		MessageBox::Show("Failed to write an amplitude file. \r\nPossible reasons :\r\n1. The Integration method was Adaptive(VEGAS) Monte Carlo \r\n2. The Hybrid or Direct methods were used.");


	}

	bool MainWindow::Save1DGraph() {

		int gSize = frontend->GetGraphSize(job);
		if (gSize == 0) {
			MessageBox::Show("No graph returned");
			return false;
		}
		std::vector<double> xv, graph;

		array<double> ^xa = gcnew array<double>(gSize), ^ya = gcnew array<double>(gSize);

		GraphPane2D ^gr = ((GraphPane2D ^)this->PaneList[GRAPH2D]);
		gr->GetModelGraph(xa, ya);
		xv = arraytovector(xa);
		graph = arraytovector(ya);

		SaveFileDialog ^sfd = gcnew SaveFileDialog();
		sfd->Filter = "Saved 1D Model (*.out)|*.out|All Files (*.*)|*.*";
		sfd->FileName = "";
		sfd->Title = "Save Model As...";
		sfd->OverwritePrompt = true;
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return false;

		std::stringstream ss;
		ss << "# Integration parameters:\n";
		ss << "#\tqmax\t" << xv[xv.size() - 1] << "\n";
		ss << "#\tOrientation Method\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->integrationMethodComboBox->Text) << "\n";
		ss << "#\tOrientation Iterations\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->integIterTextBox->Text) << "\n";
		ss << "#\tConvergence\t" << clrToString(((PreferencesPane^)PaneList[PREFERENCES])->convTextBox->Text) << "\n";
		ss << "# \n";
		// TODO::HEADER Get the domain header and add it to the file.
		char *grry;
		int len = 999999;
		grry = new char[len];
		ErrorCode err = frontend->GetDomainHeader(job, compositeModel, grry, len);
		if (err != OK)
		{
			int test = 1;
		}
		else
		{
			ss << grry;
			delete grry;
		}
		WriteDataFileWHeader(clrToWstring(sfd->FileName).c_str(), xv, graph, ss);

		return true;
	}

	cli::array<ModelPtr> ^MainWindow::LuaInnerGenerateTable(LuaTable ^ ptree) {
		if (this->InvokeRequired) {
			return (cli::array<ModelPtr> ^)Invoke(gcnew FuncLuaTableArr(this, &MainWindow::LuaInnerGenerateTable), gcnew array<Object ^> { ptree });
		}

		FittingProperties fp;

		ParameterTree pt = PrepareForWork(fp);

		if (ptree != nullptr) {
			pt = ParamTreeFromTable(ptree);

			// Get and set the parameters for the domains
			paramStruct ps = ((PreferencesPane ^)(this->PaneList[PREFERENCES]))->GetDomainPreferences();
			for (int i = 0; i < pt.GetNumSubModels(); i++)
				pt.GetSubModel(i)->SetNodeParameters(ps);

			if (pt.GetNodeModel() == compositeModel)
				statusLabel->Text = "Generating(Script): 0%";
			else
			{
				MessageBox::Show("ERROR: Root model must be Domain!");
				return nullptr;
			}

			bIsScriptComputing = true;

			// Get all sub-models from all populations
			int numModels = 0;
			for (int i = 0; i < pt.GetNumSubModels(); i++)
				numModels += pt.GetSubModel(i)->GetNumSubModels();

			cli::array<ModelPtr> ^modelstodel = gcnew cli::array<ModelPtr>(numModels);

			// Designate models to delete
			int kk = 0;
			for (int i = 0; i < pt.GetNumSubModels(); i++)
			{
				ParameterTree *population = pt.GetSubModel(i);
				for (int j = 0; j < population->GetNumSubModels(); j++)
					modelstodel[kk++] = population->GetSubModel(j)->GetNodeModel();
			}

			if (kk != numModels)
			{
				MessageBox::Show("Sanity check failed! Models and numModels mismatch!");
			}

			ErrorCode err = SaveStateAndGenerate();
			//ErrorCode err = frontend->Generate(job, pt, arraytovector(qvec), fp);
			if (err) {
				bIsScriptComputing = false;
				HandleErr("ERROR initializing generation", err);
				return modelstodel;
			}

			EnableStopButton();

			return modelstodel;
		}
		else {
			// Call the usual generate (same as pressing the button)
			Generate();

			return nullptr;
		}
	}

	System::Void MainWindow::toggleConsoleToolStripButton_Click(System::Object^ sender, System::EventArgs^ e) {
		(DPlus::Scripting::IsConsoleOpen()) ? DPlus::Scripting::CloseConsole() : DPlus::Scripting::OpenConsole();
	}

	System::Void MainWindow::exportPDBRepresentationToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		SymmetryView ^sv = (SymmetryView ^)PaneList[SYMMETRY_VIEWER];
		//ParameterEditor ^pe = (ParameterEditor ^)PaneList[PARAMETER_EDITOR];
		Entity ^ent = sv->GetSelectedEntity();
		ParameterTree pt;
		if (!ent) {
			if (sv->treeViewAdv1->SelectedNodes->Count > 1) {
				MessageBox::Show("A single element in the tree must be selected!");
				return;
			}
		}

		SaveFileDialog ^sfd = gcnew SaveFileDialog();
		sfd->Filter = "Saved PDB (*.pdb)|*.pdb|All Files (*.*)|*.*";
		sfd->FileName = "";
		sfd->Title = "Save PDB File As...";
		sfd->OverwritePrompt = true;
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		if (this->frontend->SavePDB(job, ((ent) ? ent->BackendModel : domainModels[sv->populationTabs->SelectedIndex]), clrToWstring(sfd->FileName).c_str())) {
			return;
		}

		MessageBox::Show("Failed to write PDB to file!");

	}

	paramStruct MainWindow::GetCompositeDomainParameters()
	{
		Controls3D ^c3d = dynamic_cast<Controls3D ^>(PaneList[CONTROLS]);
		SymmetryView ^ sv = dynamic_cast<SymmetryView ^>(PaneList[SYMMETRY_VIEWER]);

		// Get and set the parameters for the composite
		paramStruct ps;
		ps.nlp = 2 + domainModels->Count; // THE NUMBER OF PARAMS + NUMBER OF SUBDOMAINS
		ps.layers = 1;
		ps.params.resize(ps.nlp);
		ps.params[0].push_back(Parameter(domainScale, sv->scaleMut->Checked)); // SCALE
		ps.params[1].push_back(Parameter(domainConstant, sv->constantMut->Checked)); // Constant
		// Change the constraints if/when we create a way for the user to modify them
		for (int i = 0; i < domainModels->Count; i++)
			ps.params[i + 1].push_back(Parameter(populationSizes[i], populationSizeMutable[i], true, 0.)); // AVERAGE POPULATION SIZE

		return ps;
	}

	System::Void MainWindow::save3DViewToFileToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		GraphPane3D^ grp = (GraphPane3D^)PaneList[GRAPH3D];

		grp->SaveViewGraphicToFile(sender, nullptr);
	}

	void centerTrackBar(TrackBar^ tb) {
		tb->Value = int((double)(tb->Minimum + tb->Maximum) / 2.0);
	}

	Generic::List<Control^>^ GetAllSubControls(Control^ form) {
		Generic::List<Control^>^ outL = gcnew Generic::List<Control^>();
		for (int i = 0; i < form->Controls->Count; i++) {
			outL->AddRange(GetAllSubControls(form->Controls[i]));
			outL->Add(form->Controls[i]);
		}
		return outL;
	}

	Double LuaItemToDouble(Object ^item) {
		if (item == nullptr)
			return std::numeric_limits<double>::signaling_NaN();

		// If it's already a double, return it
		if (dynamic_cast<Double ^>(item) != nullptr)
			return (Double)item;

		// If it is otherwise not a string, there is nothing to do
		if (dynamic_cast<String ^>(item) == nullptr)
			return std::numeric_limits<double>::signaling_NaN();

		String ^str = (String ^)item;
		double db;
		if (str == "inf")
			return POSINF;
		else if (str == "-inf")
			return NEGINF;
		else if (Double::TryParse(str, db)) {
			return db;
		}

		return std::numeric_limits<double>::signaling_NaN();
	}

	Boolean LuaItemToBoolean(Object ^item) {
		if (item == nullptr)
			return false;

		// If it's already a boolean, return it
		if (dynamic_cast<Boolean ^>(item) != nullptr)
			return (Boolean)item;

		// If it is otherwise not a string, there is nothing to do
		if (dynamic_cast<String ^>(item) == nullptr)
			return false;

		String ^str = ((String ^)item)->ToLower();
		if (str == "true")
			return true;

		return false;
	}

	String ^ LuaTableToString(LuaTable ^lt, int curLevel) {
		return LuaTableToString(lt, curLevel, -1);

	}

	String ^ LuaTableToString(LuaTable ^lt) {
		return LuaTableToString(lt, 1) + ";\n";
	}

	String ^ LuaTableToString(LuaTable ^lt, int curLevel, int maxLevel) {
		String ^res = gcnew String("");
#define DELIM "\t"
		if (curLevel == maxLevel) {
			return res;
		}

		res += "{";

		bool bFirstItem = true;
		bool bEndWithNewline = false;
		for each (DictionaryEntry ^d in lt) {
			if (bFirstItem) {
				// If the first item is a table or has a string key, start with a newline and indent
				if (dynamic_cast<Double ^>(d->Key) == nullptr || d->Value->ToString() == "table") {
					res += "\n";
					bEndWithNewline = true;
				}
				else {
					res += " ";
				}
				bFirstItem = false;
			}

			if (bEndWithNewline)
				for (int i = 0; i < curLevel; i++)
					res += DELIM;

			if (dynamic_cast<Double ^>(d->Key) == nullptr)
				res += d->Key + " = ";
			if (d->Value->ToString() == "table") {
				res += LuaTableToString((LuaTable ^)(d->Value), curLevel + 1, maxLevel) + ",\n";
				bEndWithNewline = true;
			}
			else {
				if (dynamic_cast<Double ^>(d->Value) != nullptr) {
					Double val = ((Double)d->Value);
					if (Double::IsPositiveInfinity(val))
						res += "[[inf]],";
					else if (Double::IsNegativeInfinity(val))
						res += "[[-inf]],";
					else if (Double::IsNaN(val))
						res += "[[NaN]],";
					else
						res += val.ToString() + ",";
				}
				else if (dynamic_cast<Boolean ^>(d->Value) != nullptr) {
					if ((Boolean)d->Value == true)
						res += "true,";
					else
						res += "false,";
				}
				else
					res += "[[" + d->Value + "]],";


				// Determine whether to end the value with \n or space
				if (dynamic_cast<Double ^>(d->Key) == nullptr) {
					res += "\n";
					bEndWithNewline = true;
				}
				else {
					res += " ";
					bEndWithNewline = false;
				}
			}
		}
		if (bEndWithNewline) {
			for (int i = 0; i < curLevel - 1; i++)
				res += DELIM;
		}
		res += "}";
		return res;
	}

	ErrorCode MainWindow::SaveStateAndGenerate()
	{
		System::String ^script = GetLuaScript();
		std::wstring wstring = clrToWstring(script);
		std::vector<double> vec = arraytovector(qvec);
		qvec_may_be_cropped = qvec;

		return frontend->Generate(job, wstring.c_str(), vec, UseGPU);
	}

	ErrorCode MainWindow::SaveStateAndFit(std::vector<int> nomask, std::string& message)
	{
		checkFitCPU = true;
		FitJobStartTime = (unsigned long long)time(NULL);
		System::String ^script = GetLuaScript();
		auto qs = arraytovector(qvec);
		auto ys = arraytovector(loadedSignal);
		for (int i = ys.size() - 1; i >= 0; --i)
		{
			if (ys[i] < 0.)
			{
				ys.erase(ys.begin() + i);
				qs.erase(qs.begin() + i);
				nomask.erase(nomask.begin() + i);
			}
		}
		qvec_may_be_cropped = vectortoarray(qs);
		return frontend->Fit(job, clrToWstring(script).c_str(), qs, ys, nomask, UseGPU, &message);
	}

	void MainWindow::PollStatus(Object^ source, System::Timers::ElapsedEventArgs^ args)
	{
		if (responseWaiting) return;
		responseWaiting = true;
		auto *frontend = this->frontend;
		if (frontend)
		{
			try
			{
				frontend->CheckJobProgress(job);
			}
			catch (Exception ^)
			{
				frontend->MarkCheckJobProgressDone(job); //otherwise lose ability to reconnect and try again
			}
			if (checkFitCPU && /* just in the first time we want to ask the user*/
				!frontend->CheckJobOvertime(job, FitJobStartTime, 180) && /*It's been 3 minutes since the beginning of fit*/
				!UseGPU ) { /* the job is runnig on CPU*/
				fitMessage();
			}
		}
		responseWaiting = false;
	}
	void MainWindow::fitMessage() {
		if (InFitMessage)
			return;
		InFitMessage = true;
		
		try
		{
			System::Windows::Forms::DialogResult result;
			result = MessageBox::Show(gcnew String("You are running Fit without a GPU, this may take a long time. Do you want to stop the job?"), "Warning", MessageBoxButtons::YesNo, MessageBoxIcon::Question);
			if (result == ::DialogResult::Yes)
			{
				frontend->Stop(job);
			}
			if ( result == ::DialogResult::No) {
				checkFitCPU = false;
			}
		}
		catch (const std::exception&){ }
		
		InFitMessage = false;
	}
	System::Void MainWindow::timer1_Tick(System::Object^ sender, System::EventArgs^ e)
	{
		fflush(stdout);
	}

	System::Void MainWindow::suggestParametersToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
		char buf[1024] = { 0 };
		DWORD ret = GetModuleFileNameA(NULL, buf, sizeof(buf));
		if (ret == 0 || ret == sizeof(buf))
			return;
		std::string path = std::string(buf);
		path = path.substr(0, path.find_last_of('\\'));
		// We need the \"Suggest Parameters\" so that start thinks that's the title of the command,
		// and therefore path + "\\Suggest Parameters.exe\"" is the command and not the title.
		std::string sg = "start \"Suggest Parameters\" /D \"" + path + "\" \"" + path + "\\Suggest Parameters.exe\"";
		
		system(sg.c_str());
		return;
	}
	System::Void MainWindow::pdbUnitsToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
		char buf[1024] = { 0 };
		DWORD ret = GetModuleFileNameA(NULL, buf, sizeof(buf));
		if (ret == 0 || ret == sizeof(buf))
			return;
		std::string path = std::string(buf);
		path = path.substr(0, path.find_last_of('\\'));
		// We need the \"PDBUnits\" so that start thinks that's the title of the command,
		// and therefore path + "\\PDBUnits.exe\"" is the command and not the title.
		std::string sg = "start \"PDBUnits\" /D \"" + path + "\" \"" + path + "\\PDBUnits.exe\"";

		system(sg.c_str());
		return;
	}
	System::Void MainWindow::aboutDToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
	{
		AboutDPlus a;
		a.ShowDialog();
	}
	System::Void MainWindow::manualToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
	{
		System::Diagnostics::Process::Start("https://scholars.huji.ac.il/sites/default/files/uriraviv/files/dmanual.pdf");
	}

	System::Void MainWindow::visitHomePageToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
	{
		System::Diagnostics::Process::Start("https://scholars.huji.ac.il/uriraviv/book/d-0");
	}
	void MainWindow::checkCapabilitiesMainWind() {
		ErrorCode err = frontend->CheckCapabilities(UseGPU);
		if (err != OK) {
			if (err == ERROR_TDR_LEVEL) {
				TdrLevelInfo ^cw = gcnew TdrLevelInfo();
				cw->ShowDialog();
				while (cw->retry) {
					err = frontend->CheckCapabilities(UseGPU);
					if (err == ERROR_TDR_LEVEL)
						cw->ShowDialog();
					else
						cw->retry = false;
				}
				UseGPU = !cw->useCPU;
				//changing the display of UseGPU in setting menu
				changeUseGPUDisplayValue(UseGPU);
			}
			else {
				if (err == ERROR_NO_GPU)
				{
					changeUseGPUDisplayValue(false);
					this->useGPUToolStripMenuItem->Enabled = false;
				}
				else
				{
					HandleErr("ERROR check capabilities", err);
					this->Close();
					return;
				}
			}
		}
	}
};