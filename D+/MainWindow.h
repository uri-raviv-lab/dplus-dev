#pragma once

#include "Entity.h"
#include "MainWindowAux.h"
#include "ManagedHTTPCallerForm.h"
#include "ServerConfigForm.h"
#include "ManagePythonPreCaller.h"
class FrontendComm;
class BackendCaller;

namespace DPlus {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Reflection;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace WeifenLuo::WinFormsUI::Docking;
	using namespace LuaInterface;
	using namespace System::Runtime::InteropServices;

	/// <summary>
	/// Summary for MainWindow
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class MainWindow : public System::Windows::Forms::Form
	{

	public:
		Generic::List<Aga::Controls::Tree::TreeModel ^> ^populationTrees;
		Aga::Controls::Tree::TreeModel ^entityTree;		
		Generic::List<double> ^populationSizes;
		Generic::List<bool> ^populationSizeMutable;
		double domainScale;
		double domainConstant;

		Aga::Controls::Tree::TreeModel ^fittingPrefsTree;

		FrontendComm *frontend;
		BackendCaller *backendCaller;

		JobPtr job;
		Generic::List<String ^> ^loadedContainers;

		ModelPtr compositeModel;
		Generic::List<ModelPtr> ^domainModels;

		Lua ^luaState;
		bool bIsScriptComputing;
		Generic::SortedDictionary<int, Form ^> ^openFigures;
		int lastFigure;

		array<double> ^qvec;
		array<double> ^qvec_may_be_cropped;
		array<double> ^loadedSignal;
		String ^signalFilename;

		String ^staticQMaxString;
		String ^staticQMinString;
		String ^rootWriteDir;

	public: System::Collections::Generic::List<DockContent^>^ PaneList;
	private: System::Windows::Forms::ToolStripMenuItem^  quitToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  controls3DToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  dEBUGItemsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  parameterEditorToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  luaTestToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  commandWindowToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  GenerateFitToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  saveParametersToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator1;
	private: System::Windows::Forms::StatusStrip^  statusStrip1;
	private: System::Windows::Forms::ToolStripStatusLabel^  statusLabel;

	private: System::Windows::Forms::ToolStripProgressBar^  progressBar;


	private: System::Windows::Forms::ToolStrip^  toolStrip1;
	private: System::Windows::Forms::ToolStripButton^  saveToolStrip;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
	private: System::Windows::Forms::ToolStripButton^  undoToolStrip;
	private: System::Windows::Forms::ToolStripButton^  redoToolStrip;
	private: System::Windows::Forms::Panel^  panel1;
	private: System::Windows::Forms::ToolStripButton^  openToolStrip;
	private: System::Windows::Forms::ToolStripMenuItem^  settingsToolStripMenuItem;
	//public: System::Windows::Forms::ToolStripMenuItem^  liveGenerationToolStripMenuItem; //has been removed as obsolete
	public: System::Windows::Forms::ToolStripMenuItem^  updateFitGraphToolStripMenuItem;
	public: System::Windows::Forms::ToolStripMenuItem^  configureServerToolStripMenuItem;
	public: System::Windows::Forms::ToolStripMenuItem^  updateFitDomainToolStripMenuItem;
	public: System::Windows::Forms::ToolStripMenuItem^  useGPUToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  importParametersToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveVantagePointToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  loadVantagePointToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  scriptWindowToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  openSignalToolStripMenuItem;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator3;
	private: System::Windows::Forms::ToolStripButton^  loadFileToolstrip;

	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator4;
	private: System::Windows::Forms::ToolStripMenuItem^  closeSignalToolStripMenuItem;
	private: System::Windows::Forms::ToolStripButton^  closeSignalToolstrip;
	private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator5;
	private: System::Windows::Forms::ToolStripMenuItem^  export1DGraphToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  exportAmplitudeFileToolStripMenuItem;
	//private: System::Windows::Forms::ToolStripButton^  toggleConsoleToolStripButton;
	private: System::Windows::Forms::ToolStripMenuItem^  exportPDBRepresentationToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  fittingPreferencesToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  save3DViewToFileToolStripMenuItem;


	private: System::Timers::Timer^ _statusPollingTimer;
	private: System::Windows::Forms::Timer^  timer1;
	private: System::Windows::Forms::ToolStripMenuItem^  suggestParametersToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  pdbUnitsToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  helpToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  aboutDToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  manualToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  visitHomePageToolStripMenuItem;


	private: System::Collections::Generic::List<ToolStripMenuItem^>^ MenuPaneList;
	public:
		MainWindow(void)
		{
			// Must be created before the Symmetry View pane
			populationTrees = gcnew Generic::List<Aga::Controls::Tree::TreeModel ^>();
			populationSizes = gcnew Generic::List<double>();
			populationSizeMutable = gcnew Generic::List<bool>();
			populationTrees->Add(gcnew Aga::Controls::Tree::TreeModel());
			populationSizes->Add(1.0);
			populationSizeMutable->Add(false);
			entityTree = populationTrees[0];

			domainScale = 1.0;
			domainConstant = 0.0;

			fittingPrefsTree = gcnew Aga::Controls::Tree::TreeModel();

			InitializeComponent();			

			PaneList = gcnew System::Collections::Generic::List<DockContent^>();
			MenuPaneList = gcnew System::Collections::Generic::List<ToolStripMenuItem^>();
			loadedContainers = gcnew Generic::List<String ^>();
			openFigures = gcnew Generic::SortedDictionary<int, Form ^>();
			lastFigure = -1;
			frontend = NULL;		
			job = NULL;

			progressBar->Visible = false;

			bIsScriptComputing = false;

			// Hide the Settings menu items that can cause "weird things to happen"
			this->updateFitGraphToolStripMenuItem->Visible = false;
			this->updateFitDomainToolStripMenuItem->Visible = false;

			UseGPU = true;
			InFitMessage = false;
			checkFitCPU = true;
			InSelectionChange = false;
			changeUseGPUDisplayValue(UseGPU);
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MainWindow();

	private: WeifenLuo::WinFormsUI::Docking::DockPanel^  mainDockPanel;
	private: System::Windows::Forms::MenuStrip^ mainMenuStrip;
	private: System::Windows::Forms::ToolStripMenuItem^ fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ editToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ viewToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ Graph2DToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  defaultLayoutToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveLayoutToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  loadLayoutToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  defaultParametersToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^  layoutOFD;
	private: System::Windows::Forms::SaveFileDialog^  layoutSFD;
	private: System::Windows::Forms::ToolStripMenuItem^  Graph3DToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  symmetryViewToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  symmetryEditorToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  preferencesToolStripMenuItem;
private: System::ComponentModel::IContainer^  components;

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
			this->components = (gcnew System::ComponentModel::Container());
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(MainWindow::typeid));
			WeifenLuo::WinFormsUI::Docking::DockPanelSkin^  dockPanelSkin1 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPanelSkin());
			WeifenLuo::WinFormsUI::Docking::AutoHideStripSkin^  autoHideStripSkin1 = (gcnew WeifenLuo::WinFormsUI::Docking::AutoHideStripSkin());
			WeifenLuo::WinFormsUI::Docking::DockPanelGradient^  dockPanelGradient1 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPanelGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient1 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::DockPaneStripSkin^  dockPaneStripSkin1 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPaneStripSkin());
			WeifenLuo::WinFormsUI::Docking::DockPaneStripGradient^  dockPaneStripGradient1 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPaneStripGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient2 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::DockPanelGradient^  dockPanelGradient2 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPanelGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient3 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::DockPaneStripToolWindowGradient^  dockPaneStripToolWindowGradient1 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPaneStripToolWindowGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient4 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient5 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::DockPanelGradient^  dockPanelGradient3 = (gcnew WeifenLuo::WinFormsUI::Docking::DockPanelGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient6 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			WeifenLuo::WinFormsUI::Docking::TabGradient^  tabGradient7 = (gcnew WeifenLuo::WinFormsUI::Docking::TabGradient());
			this->mainMenuStrip = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openSignalToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->closeSignalToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator3 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->importParametersToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveParametersToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->loadVantagePointToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveVantagePointToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->save3DViewToFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator5 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->export1DGraphToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exportAmplitudeFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exportPDBRepresentationToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->quitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->dEBUGItemsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->luaTestToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->editToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->defaultLayoutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->defaultParametersToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveLayoutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->loadLayoutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->viewToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->Graph2DToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->Graph3DToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->symmetryViewToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->symmetryEditorToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->preferencesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->controls3DToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->GenerateFitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->parameterEditorToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->scriptWindowToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->commandWindowToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fittingPreferencesToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->settingsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->updateFitGraphToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->updateFitDomainToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->configureServerToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->suggestParametersToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->pdbUnitsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->useGPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->helpToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->aboutDToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->manualToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->layoutOFD = (gcnew System::Windows::Forms::OpenFileDialog());
			this->layoutSFD = (gcnew System::Windows::Forms::SaveFileDialog());
			this->statusStrip1 = (gcnew System::Windows::Forms::StatusStrip());
			this->statusLabel = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->progressBar = (gcnew System::Windows::Forms::ToolStripProgressBar());
			this->toolStrip1 = (gcnew System::Windows::Forms::ToolStrip());
			this->loadFileToolstrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->closeSignalToolstrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripSeparator4 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->openToolStrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->saveToolStrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->undoToolStrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->redoToolStrip = (gcnew System::Windows::Forms::ToolStripButton());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->mainDockPanel = (gcnew WeifenLuo::WinFormsUI::Docking::DockPanel());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->visitHomePageToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->mainMenuStrip->SuspendLayout();
			this->statusStrip1->SuspendLayout();
			this->toolStrip1->SuspendLayout();
			this->panel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// mainMenuStrip
			// 
			this->mainMenuStrip->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {
				this->fileToolStripMenuItem,
					this->editToolStripMenuItem, this->viewToolStripMenuItem, this->settingsToolStripMenuItem, this->helpToolStripMenuItem
			});
			this->mainMenuStrip->Location = System::Drawing::Point(0, 0);
			this->mainMenuStrip->Name = L"mainMenuStrip";
			this->mainMenuStrip->Size = System::Drawing::Size(1004, 24);
			this->mainMenuStrip->TabIndex = 1;
			this->mainMenuStrip->Text = L"mainMenuStrip";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(15) {
				this->openSignalToolStripMenuItem,
					this->closeSignalToolStripMenuItem, this->toolStripSeparator3, this->importParametersToolStripMenuItem, this->saveParametersToolStripMenuItem,
					this->loadVantagePointToolStripMenuItem, this->saveVantagePointToolStripMenuItem, this->save3DViewToFileToolStripMenuItem, this->toolStripSeparator5,
					this->export1DGraphToolStripMenuItem, this->exportAmplitudeFileToolStripMenuItem, this->exportPDBRepresentationToolStripMenuItem,
					this->toolStripSeparator1, this->quitToolStripMenuItem, this->dEBUGItemsToolStripMenuItem
			});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// openSignalToolStripMenuItem
			// 
			this->openSignalToolStripMenuItem->Name = L"openSignalToolStripMenuItem";
			this->openSignalToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->openSignalToolStripMenuItem->Text = L"Open Signal...";
			this->openSignalToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::openSignalToolStripMenuItem_Click);
			// 
			// closeSignalToolStripMenuItem
			// 
			this->closeSignalToolStripMenuItem->Name = L"closeSignalToolStripMenuItem";
			this->closeSignalToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->closeSignalToolStripMenuItem->Text = L"Close Signal";
			this->closeSignalToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::closeSignalToolStripMenuItem_Click);
			// 
			// toolStripSeparator3
			// 
			this->toolStripSeparator3->Name = L"toolStripSeparator3";
			this->toolStripSeparator3->Size = System::Drawing::Size(211, 6);
			// 
			// importParametersToolStripMenuItem
			// 
			this->importParametersToolStripMenuItem->Name = L"importParametersToolStripMenuItem";
			this->importParametersToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->importParametersToolStripMenuItem->Text = L"Import All Parameters...";
			this->importParametersToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::importParametersToolStripMenuItem_Click);
			// 
			// saveParametersToolStripMenuItem
			// 
			this->saveParametersToolStripMenuItem->Name = L"saveParametersToolStripMenuItem";
			this->saveParametersToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->saveParametersToolStripMenuItem->Text = L"Save All Parameters...";
			this->saveParametersToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::saveParametersToolStripMenuItem_Click);
			// 
			// loadVantagePointToolStripMenuItem
			// 
			this->loadVantagePointToolStripMenuItem->Name = L"loadVantagePointToolStripMenuItem";
			this->loadVantagePointToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->loadVantagePointToolStripMenuItem->Text = L"Load Vantage Point...";
			this->loadVantagePointToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::loadVantagePointToolStripMenuItem_Click);
			// 
			// saveVantagePointToolStripMenuItem
			// 
			this->saveVantagePointToolStripMenuItem->Name = L"saveVantagePointToolStripMenuItem";
			this->saveVantagePointToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->saveVantagePointToolStripMenuItem->Text = L"Save Vantage Point...";
			this->saveVantagePointToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::saveVantagePointToolStripMenuItem_Click);
			// 
			// save3DViewToFileToolStripMenuItem
			// 
			this->save3DViewToFileToolStripMenuItem->Name = L"save3DViewToFileToolStripMenuItem";
			this->save3DViewToFileToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->save3DViewToFileToolStripMenuItem->Text = L"Save 3D view to file...";
			this->save3DViewToFileToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::save3DViewToFileToolStripMenuItem_Click);
			// 
			// toolStripSeparator5
			// 
			this->toolStripSeparator5->Name = L"toolStripSeparator5";
			this->toolStripSeparator5->Size = System::Drawing::Size(211, 6);
			// 
			// export1DGraphToolStripMenuItem
			// 
			this->export1DGraphToolStripMenuItem->Name = L"export1DGraphToolStripMenuItem";
			this->export1DGraphToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->export1DGraphToolStripMenuItem->Text = L"Export 1D Graph...";
			this->export1DGraphToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::export1DGraphToolStripMenuItem_Click);
			// 
			// exportAmplitudeFileToolStripMenuItem
			// 
			this->exportAmplitudeFileToolStripMenuItem->Name = L"exportAmplitudeFileToolStripMenuItem";
			this->exportAmplitudeFileToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->exportAmplitudeFileToolStripMenuItem->Text = L"Export Amplitude File...";
			this->exportAmplitudeFileToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::exportAmplitudeFileToolStripMenuItem_Click);
			// 
			// exportPDBRepresentationToolStripMenuItem
			// 
			this->exportPDBRepresentationToolStripMenuItem->Name = L"exportPDBRepresentationToolStripMenuItem";
			this->exportPDBRepresentationToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->exportPDBRepresentationToolStripMenuItem->Text = L"Export PDB Representation";
			this->exportPDBRepresentationToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::exportPDBRepresentationToolStripMenuItem_Click);
			// 
			// toolStripSeparator1
			// 
			this->toolStripSeparator1->Name = L"toolStripSeparator1";
			this->toolStripSeparator1->Size = System::Drawing::Size(211, 6);
			// 
			// quitToolStripMenuItem
			// 
			this->quitToolStripMenuItem->Name = L"quitToolStripMenuItem";
			this->quitToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->quitToolStripMenuItem->Text = L"Quit";
			this->quitToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::quitToolStripMenuItem_Click);
			// 
			// dEBUGItemsToolStripMenuItem
			// 
			this->dEBUGItemsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->luaTestToolStripMenuItem });
			this->dEBUGItemsToolStripMenuItem->Name = L"dEBUGItemsToolStripMenuItem";
			this->dEBUGItemsToolStripMenuItem->Size = System::Drawing::Size(214, 22);
			this->dEBUGItemsToolStripMenuItem->Text = L"DEBUG Items";
			// 
			// luaTestToolStripMenuItem
			// 
			this->luaTestToolStripMenuItem->Name = L"luaTestToolStripMenuItem";
			this->luaTestToolStripMenuItem->Size = System::Drawing::Size(117, 22);
			this->luaTestToolStripMenuItem->Text = L"Lua Test";
			this->luaTestToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::luaTestToolStripMenuItem_Click);
			// 
			// editToolStripMenuItem
			// 
			this->editToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
				this->defaultLayoutToolStripMenuItem,
					this->defaultParametersToolStripMenuItem, this->saveLayoutToolStripMenuItem, this->loadLayoutToolStripMenuItem
			});
			this->editToolStripMenuItem->Name = L"editToolStripMenuItem";
			this->editToolStripMenuItem->Size = System::Drawing::Size(39, 20);
			this->editToolStripMenuItem->Text = L"Edit";
			// 
			// defaultLayoutToolStripMenuItem
			// 
			this->defaultLayoutToolStripMenuItem->Name = L"defaultLayoutToolStripMenuItem";
			this->defaultLayoutToolStripMenuItem->Size = System::Drawing::Size(174, 22);
			this->defaultLayoutToolStripMenuItem->Text = L"Default Layout";
			this->defaultLayoutToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::defaultLayoutToolStripMenuItem_Click);
			// 
			// defaultParametersToolStripMenuItem
			// 
			this->defaultParametersToolStripMenuItem->Name = L"defaultParametersToolStripMenuItem";
			this->defaultParametersToolStripMenuItem->Size = System::Drawing::Size(174, 22);
			this->defaultParametersToolStripMenuItem->Text = L"Default Parameters";
			this->defaultParametersToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::defaultParametersToolStripMenuItem_Click);
			// 
			// saveLayoutToolStripMenuItem
			// 
			this->saveLayoutToolStripMenuItem->Name = L"saveLayoutToolStripMenuItem";
			this->saveLayoutToolStripMenuItem->Size = System::Drawing::Size(174, 22);
			this->saveLayoutToolStripMenuItem->Text = L"Save Layout...";
			this->saveLayoutToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::saveLayoutToolStripMenuItem_Click);
			// 
			// loadLayoutToolStripMenuItem
			// 
			this->loadLayoutToolStripMenuItem->Name = L"loadLayoutToolStripMenuItem";
			this->loadLayoutToolStripMenuItem->Size = System::Drawing::Size(174, 22);
			this->loadLayoutToolStripMenuItem->Text = L"Load Layout...";
			this->loadLayoutToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::loadLayoutToolStripMenuItem_Click);
			// 
			// viewToolStripMenuItem
			// 
			this->viewToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(11) {
				this->Graph2DToolStripMenuItem,
					this->Graph3DToolStripMenuItem, this->symmetryViewToolStripMenuItem, this->symmetryEditorToolStripMenuItem, this->preferencesToolStripMenuItem,
					this->controls3DToolStripMenuItem, this->parameterEditorToolStripMenuItem, this->scriptWindowToolStripMenuItem, this->commandWindowToolStripMenuItem,
					this->fittingPreferencesToolStripMenuItem, this->GenerateFitToolStripMenuItem
			});
			this->viewToolStripMenuItem->Name = L"viewToolStripMenuItem";
			this->viewToolStripMenuItem->Size = System::Drawing::Size(44, 20);
			this->viewToolStripMenuItem->Text = L"View";

			// 
			// Graph2DToolStripMenuItem
			// 
			this->Graph2DToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"Graph2DToolStripMenuItem.Image")));
			this->Graph2DToolStripMenuItem->Name = L"Graph2DToolStripMenuItem";
			this->Graph2DToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->Graph2DToolStripMenuItem->Text = L"2D Graph";
			this->Graph2DToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// Graph3DToolStripMenuItem
			// 
			this->Graph3DToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"Graph3DToolStripMenuItem.Image")));
			this->Graph3DToolStripMenuItem->Name = L"Graph3DToolStripMenuItem";
			this->Graph3DToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->Graph3DToolStripMenuItem->Text = L"3D Graph";
			this->Graph3DToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// symmetryViewToolStripMenuItem
			// 
			this->symmetryViewToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"symmetryViewToolStripMenuItem.Image")));
			this->symmetryViewToolStripMenuItem->Name = L"symmetryViewToolStripMenuItem";
			this->symmetryViewToolStripMenuItem->Size = System::Drawing::Size(178, 60);
			this->symmetryViewToolStripMenuItem->Text = L"Symmetry View";
			this->symmetryViewToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// symmetryEditorToolStripMenuItem
			// 
			this->symmetryEditorToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"symmetryEditorToolStripMenuItem.Image")));
			this->symmetryEditorToolStripMenuItem->Name = L"symmetryEditorToolStripMenuItem";
			this->symmetryEditorToolStripMenuItem->Size = System::Drawing::Size(178, 25);
			this->symmetryEditorToolStripMenuItem->Text = L"Symmetry Editor";
			this->symmetryEditorToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// preferencesToolStripMenuItem
			// 
			this->preferencesToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"preferencesToolStripMenuItem.Image")));
			this->preferencesToolStripMenuItem->Name = L"preferencesToolStripMenuItem";
			this->preferencesToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->preferencesToolStripMenuItem->Text = L"Preferences";
			this->preferencesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// controls3DToolStripMenuItem
			// 
			this->controls3DToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"controls3DToolStripMenuItem.Image")));
			this->controls3DToolStripMenuItem->Name = L"controls3DToolStripMenuItem";
			this->controls3DToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->controls3DToolStripMenuItem->Text = L"Controls";
			this->controls3DToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			//
			//GenerateFitToolStripMenuItem
			//
			this->GenerateFitToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"GenerateFitToolStripMenuItem.Image")));
			this->GenerateFitToolStripMenuItem->Name = L"GenerateFitToolStripMenuItem";
			this->GenerateFitToolStripMenuItem->Size = System::Drawing::Size(178, 15);
			this->GenerateFitToolStripMenuItem->Text = L"Controls";
			this->GenerateFitToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// parameterEditorToolStripMenuItem
			// 
			this->parameterEditorToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"parameterEditorToolStripMenuItem.Image")));
			this->parameterEditorToolStripMenuItem->Name = L"parameterEditorToolStripMenuItem";
			this->parameterEditorToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->parameterEditorToolStripMenuItem->Text = L"Parameter Editor";
			this->parameterEditorToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// scriptWindowToolStripMenuItem
			// 
			this->scriptWindowToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"scriptWindowToolStripMenuItem.Image")));
			this->scriptWindowToolStripMenuItem->Name = L"scriptWindowToolStripMenuItem";
			this->scriptWindowToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->scriptWindowToolStripMenuItem->Text = L"Script Editor";
			this->scriptWindowToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// commandWindowToolStripMenuItem
			// 
			this->commandWindowToolStripMenuItem->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"commandWindowToolStripMenuItem.Image")));
			this->commandWindowToolStripMenuItem->Name = L"commandWindowToolStripMenuItem";
			this->commandWindowToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->commandWindowToolStripMenuItem->Text = L"Command Window";
			this->commandWindowToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// fittingPreferencesToolStripMenuItem
			// 
			this->fittingPreferencesToolStripMenuItem->Name = L"fittingPreferencesToolStripMenuItem";
			this->fittingPreferencesToolStripMenuItem->Size = System::Drawing::Size(178, 22);
			this->fittingPreferencesToolStripMenuItem->Text = L"Fitting Preferences";
			this->fittingPreferencesToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::MenuPaneStripMenuItem_Click);
			// 
			// settingsToolStripMenuItem
			// 
			this->settingsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {
				this->updateFitGraphToolStripMenuItem,
					this->updateFitDomainToolStripMenuItem, this->configureServerToolStripMenuItem, this->suggestParametersToolStripMenuItem, this->pdbUnitsToolStripMenuItem,
					this->useGPUToolStripMenuItem
			});
			this->settingsToolStripMenuItem->Name = L"settingsToolStripMenuItem";
			this->settingsToolStripMenuItem->Size = System::Drawing::Size(61, 20);
			this->settingsToolStripMenuItem->Text = L"Settings";
			// 
			// updateFitGraphToolStripMenuItem
			// 
			this->updateFitGraphToolStripMenuItem->Checked = true;
			this->updateFitGraphToolStripMenuItem->CheckOnClick = true;
			this->updateFitGraphToolStripMenuItem->CheckState = System::Windows::Forms::CheckState::Checked;
			this->updateFitGraphToolStripMenuItem->Name = L"updateFitGraphToolStripMenuItem";
			this->updateFitGraphToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->updateFitGraphToolStripMenuItem->Text = L"Update Fitting Results";
			// 
			// updateFitDomainToolStripMenuItem
			// 
			this->updateFitDomainToolStripMenuItem->CheckOnClick = true;
			this->updateFitDomainToolStripMenuItem->Name = L"updateFitDomainToolStripMenuItem";
			this->updateFitDomainToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->updateFitDomainToolStripMenuItem->Text = L"Show Fitting Results in Viewport";
			// 
			// configureServerToolStripMenuItem
			// 
			this->configureServerToolStripMenuItem->Name = L"configureServerToolStripMenuItem";
			this->configureServerToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->configureServerToolStripMenuItem->Text = L"Configure Server";
			this->configureServerToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::configureServerToolStripMenuItem_Click);
			// 
			// suggestParametersToolStripMenuItem
			// 
			this->suggestParametersToolStripMenuItem->Name = L"suggestParametersToolStripMenuItem";
			this->suggestParametersToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->suggestParametersToolStripMenuItem->Text = L"Suggest Parameters...";
			this->suggestParametersToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::suggestParametersToolStripMenuItem_Click);
			// 
			// pdbUnitsToolStripMenuItem
			// 
			this->pdbUnitsToolStripMenuItem->Name = L"pdbUnitsToolStripMenuItem";
			this->pdbUnitsToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->pdbUnitsToolStripMenuItem->Text = L"PDB Units...";
			this->pdbUnitsToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::pdbUnitsToolStripMenuItem_Click);
			// 
			// useGPUToolStripMenuItem
			// 
			this->useGPUToolStripMenuItem->Name = L"useGPUToolStripMenuItem";
			this->useGPUToolStripMenuItem->Size = System::Drawing::Size(243, 22);
			this->useGPUToolStripMenuItem->Text = L"Use GPU";
			this->useGPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::useGPUToolStripMenuItem_Click);
			// 
			// helpToolStripMenuItem
			// 
			this->helpToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->aboutDToolStripMenuItem,
					this->manualToolStripMenuItem, this->visitHomePageToolStripMenuItem
			});
			this->helpToolStripMenuItem->Name = L"helpToolStripMenuItem";
			this->helpToolStripMenuItem->Size = System::Drawing::Size(44, 20);
			this->helpToolStripMenuItem->Text = L"Help";
			// 
			// aboutDToolStripMenuItem
			// 
			this->aboutDToolStripMenuItem->Name = L"aboutDToolStripMenuItem";
			this->aboutDToolStripMenuItem->Size = System::Drawing::Size(159, 22);
			this->aboutDToolStripMenuItem->Text = L"About D+...";
			this->aboutDToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::aboutDToolStripMenuItem_Click);
			// 
			// manualToolStripMenuItem
			// 
			this->manualToolStripMenuItem->Name = L"manualToolStripMenuItem";
			this->manualToolStripMenuItem->Size = System::Drawing::Size(159, 22);
			this->manualToolStripMenuItem->Text = L"Manual";
			this->manualToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::manualToolStripMenuItem_Click);
			// 
			// layoutOFD
			// 
			this->layoutOFD->Filter = L"\"Layout files|*.dlayout|All files|*.*\"";
			// 
			// layoutSFD
			// 
			this->layoutSFD->Filter = L"\"Layout files|*.dlayout|All files|*.*\"";
			// 
			// statusStrip1
			// 
			this->statusStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) { this->statusLabel, this->progressBar });
			this->statusStrip1->LayoutStyle = System::Windows::Forms::ToolStripLayoutStyle::HorizontalStackWithOverflow;
			this->statusStrip1->Location = System::Drawing::Point(0, 700);
			this->statusStrip1->Name = L"statusStrip1";
			this->statusStrip1->Size = System::Drawing::Size(1004, 22);
			this->statusStrip1->TabIndex = 2;
			this->statusStrip1->Text = L"statusStrip1";
			// 
			// statusLabel
			// 
			this->statusLabel->Name = L"statusLabel";
			this->statusLabel->Size = System::Drawing::Size(26, 17);
			this->statusLabel->Text = L"Idle";
			// 
			// progressBar
			// 
			this->progressBar->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
			this->progressBar->Margin = System::Windows::Forms::Padding(1, 3, 16, 3);
			this->progressBar->Name = L"progressBar";
			this->progressBar->RightToLeft = System::Windows::Forms::RightToLeft::Yes;
			this->progressBar->Size = System::Drawing::Size(150, 16);
			// 
			// toolStrip1
			// 
			this->toolStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(8) {
				this->loadFileToolstrip,
					this->closeSignalToolstrip, this->toolStripSeparator4, this->openToolStrip, this->saveToolStrip, this->toolStripSeparator2, this->undoToolStrip,
					this->redoToolStrip
			});
			this->toolStrip1->Location = System::Drawing::Point(0, 24);
			this->toolStrip1->Name = L"toolStrip1";
			this->toolStrip1->Size = System::Drawing::Size(1004, 25);
			this->toolStrip1->TabIndex = 3;
			this->toolStrip1->Text = L"toolStrip1";
			// 
			// loadFileToolstrip
			// 
			this->loadFileToolstrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->loadFileToolstrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"loadFileToolstrip.Image")));
			this->loadFileToolstrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->loadFileToolstrip->Name = L"loadFileToolstrip";
			this->loadFileToolstrip->Size = System::Drawing::Size(23, 22);
			this->loadFileToolstrip->Text = L"Open Signal...";
			this->loadFileToolstrip->Click += gcnew System::EventHandler(this, &MainWindow::openSignalToolStripMenuItem_Click);
			// 
			// closeSignalToolstrip
			// 
			this->closeSignalToolstrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->closeSignalToolstrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"closeSignalToolstrip.Image")));
			this->closeSignalToolstrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->closeSignalToolstrip->Name = L"closeSignalToolstrip";
			this->closeSignalToolstrip->Size = System::Drawing::Size(23, 22);
			this->closeSignalToolstrip->Text = L"Close Signal";
			this->closeSignalToolstrip->ToolTipText = L"Close Signal";
			this->closeSignalToolstrip->Click += gcnew System::EventHandler(this, &MainWindow::closeSignalToolStripMenuItem_Click);
			// 
			// toolStripSeparator4
			// 
			this->toolStripSeparator4->Name = L"toolStripSeparator4";
			this->toolStripSeparator4->Size = System::Drawing::Size(6, 25);
			// 
			// openToolStrip
			// 
			this->openToolStrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->openToolStrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"openToolStrip.Image")));
			this->openToolStrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->openToolStrip->Name = L"openToolStrip";
			this->openToolStrip->Size = System::Drawing::Size(23, 22);
			this->openToolStrip->Text = L"Open Parameter File (Ctrl+O)";
			this->openToolStrip->Click += gcnew System::EventHandler(this, &MainWindow::importParametersToolStripMenuItem_Click);
			// 
			// saveToolStrip
			// 
			this->saveToolStrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->saveToolStrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"saveToolStrip.Image")));
			this->saveToolStrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->saveToolStrip->Name = L"saveToolStrip";
			this->saveToolStrip->Size = System::Drawing::Size(23, 22);
			this->saveToolStrip->Text = L"Save Parameters (Ctrl+S)";
			this->saveToolStrip->Click += gcnew System::EventHandler(this, &MainWindow::saveParametersToolStripMenuItem_Click);
			// 
			// toolStripSeparator2
			// 
			this->toolStripSeparator2->Name = L"toolStripSeparator2";
			this->toolStripSeparator2->Size = System::Drawing::Size(6, 25);
			// 
			// undoToolStrip
			// 
			this->undoToolStrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->undoToolStrip->Enabled = false;
			this->undoToolStrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"undoToolStrip.Image")));
			this->undoToolStrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->undoToolStrip->Name = L"undoToolStrip";
			this->undoToolStrip->Size = System::Drawing::Size(23, 22);
			this->undoToolStrip->Text = L"Undo (Ctrl+Z)";
			this->undoToolStrip->Click += gcnew System::EventHandler(this, &MainWindow::undoToolStrip_Click);
			// 
			// redoToolStrip
			// 
			this->redoToolStrip->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->redoToolStrip->Enabled = false;
			this->redoToolStrip->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"redoToolStrip.Image")));
			this->redoToolStrip->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->redoToolStrip->Name = L"redoToolStrip";
			this->redoToolStrip->Size = System::Drawing::Size(23, 22);
			this->redoToolStrip->Text = L"Redo (Ctrl+Y)";
			this->redoToolStrip->Click += gcnew System::EventHandler(this, &MainWindow::redoToolStrip_Click);
			// 
			// panel1
			// 
			this->panel1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->panel1->Controls->Add(this->mainDockPanel);
			this->panel1->Location = System::Drawing::Point(0, 52);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(1004, 645);
			this->panel1->TabIndex = 4;
			// 
			// mainDockPanel
			// 
			this->mainDockPanel->AutoSize = true;
			this->mainDockPanel->Dock = System::Windows::Forms::DockStyle::Fill;
			this->mainDockPanel->DockBackColor = System::Drawing::SystemColors::AppWorkspace;
			this->mainDockPanel->DocumentStyle = WeifenLuo::WinFormsUI::Docking::DocumentStyle::DockingWindow;
			this->mainDockPanel->Location = System::Drawing::Point(0, 0);
			this->mainDockPanel->Name = L"mainDockPanel";
			this->mainDockPanel->ShowDocumentIcon = true;
			this->mainDockPanel->Size = System::Drawing::Size(1004, 645);
			dockPanelGradient1->EndColor = System::Drawing::SystemColors::ControlLight;
			dockPanelGradient1->StartColor = System::Drawing::SystemColors::ControlLight;
			autoHideStripSkin1->DockStripGradient = dockPanelGradient1;
			tabGradient1->EndColor = System::Drawing::SystemColors::Control;
			tabGradient1->StartColor = System::Drawing::SystemColors::Control;
			tabGradient1->TextColor = System::Drawing::SystemColors::ControlDarkDark;
			autoHideStripSkin1->TabGradient = tabGradient1;
			autoHideStripSkin1->TextFont = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			dockPanelSkin1->AutoHideStripSkin = autoHideStripSkin1;
			tabGradient2->EndColor = System::Drawing::SystemColors::ControlLightLight;
			tabGradient2->StartColor = System::Drawing::SystemColors::ControlLightLight;
			tabGradient2->TextColor = System::Drawing::SystemColors::ControlText;
			dockPaneStripGradient1->ActiveTabGradient = tabGradient2;
			dockPanelGradient2->EndColor = System::Drawing::SystemColors::Control;
			dockPanelGradient2->StartColor = System::Drawing::SystemColors::Control;
			dockPaneStripGradient1->DockStripGradient = dockPanelGradient2;
			tabGradient3->EndColor = System::Drawing::SystemColors::ControlLight;
			tabGradient3->StartColor = System::Drawing::SystemColors::ControlLight;
			tabGradient3->TextColor = System::Drawing::SystemColors::ControlText;
			dockPaneStripGradient1->InactiveTabGradient = tabGradient3;
			dockPaneStripSkin1->DocumentGradient = dockPaneStripGradient1;
			dockPaneStripSkin1->TextFont = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			tabGradient4->EndColor = System::Drawing::SystemColors::ActiveCaption;
			tabGradient4->LinearGradientMode = System::Drawing::Drawing2D::LinearGradientMode::Vertical;
			tabGradient4->StartColor = System::Drawing::SystemColors::GradientActiveCaption;
			tabGradient4->TextColor = System::Drawing::SystemColors::ActiveCaptionText;
			dockPaneStripToolWindowGradient1->ActiveCaptionGradient = tabGradient4;
			tabGradient5->EndColor = System::Drawing::SystemColors::Control;
			tabGradient5->StartColor = System::Drawing::SystemColors::Control;
			tabGradient5->TextColor = System::Drawing::SystemColors::ControlText;
			dockPaneStripToolWindowGradient1->ActiveTabGradient = tabGradient5;
			dockPanelGradient3->EndColor = System::Drawing::SystemColors::ControlLight;
			dockPanelGradient3->StartColor = System::Drawing::SystemColors::ControlLight;
			dockPaneStripToolWindowGradient1->DockStripGradient = dockPanelGradient3;
			tabGradient6->EndColor = System::Drawing::SystemColors::InactiveCaption;
			tabGradient6->LinearGradientMode = System::Drawing::Drawing2D::LinearGradientMode::Vertical;
			tabGradient6->StartColor = System::Drawing::SystemColors::GradientInactiveCaption;
			tabGradient6->TextColor = System::Drawing::SystemColors::InactiveCaptionText;
			dockPaneStripToolWindowGradient1->InactiveCaptionGradient = tabGradient6;
			tabGradient7->EndColor = System::Drawing::Color::Transparent;
			tabGradient7->StartColor = System::Drawing::Color::Transparent;
			tabGradient7->TextColor = System::Drawing::SystemColors::ControlDarkDark;
			dockPaneStripToolWindowGradient1->InactiveTabGradient = tabGradient7;
			dockPaneStripSkin1->ToolWindowGradient = dockPaneStripToolWindowGradient1;
			dockPanelSkin1->DockPaneStripSkin = dockPaneStripSkin1;
			this->mainDockPanel->Skin = dockPanelSkin1;
			this->mainDockPanel->TabIndex = 0;
			// 
			// timer1
			// 
			this->timer1->Enabled = true;
			this->timer1->Interval = 300;
			this->timer1->Tick += gcnew System::EventHandler(this, &MainWindow::timer1_Tick);
			// 
			// visitHomePageToolStripMenuItem
			// 
			this->visitHomePageToolStripMenuItem->Name = L"visitHomePageToolStripMenuItem";
			this->visitHomePageToolStripMenuItem->Size = System::Drawing::Size(159, 22);
			this->visitHomePageToolStripMenuItem->Text = L"Visit home page";
			this->visitHomePageToolStripMenuItem->Click += gcnew System::EventHandler(this, &MainWindow::visitHomePageToolStripMenuItem_Click);
			// 
			// MainWindow
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->AutoSizeMode = System::Windows::Forms::AutoSizeMode::GrowAndShrink;
			this->ClientSize = System::Drawing::Size(1004, 722);
			this->Controls->Add(this->panel1);
			this->Controls->Add(this->statusStrip1);
			this->Controls->Add(this->toolStrip1);
			this->Controls->Add(this->mainMenuStrip);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->MainMenuStrip = this->mainMenuStrip;
			this->Name = L"MainWindow";
			this->Text = L"D+";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &MainWindow::MainWindow_FormClosing);
			this->Load += gcnew System::EventHandler(this, &MainWindow::MainWindow_Load);
			this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &MainWindow::MainWindow_KeyDown);
			this->mainMenuStrip->ResumeLayout(false);
			this->mainMenuStrip->PerformLayout();
			this->statusStrip1->ResumeLayout(false);
			this->statusStrip1->PerformLayout();
			this->toolStrip1->ResumeLayout(false);
			this->toolStrip1->PerformLayout();
			this->panel1->ResumeLayout(false);
			this->panel1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

		private: 
			void MainWindow_Load(System::Object^  sender, System::EventArgs^  e);
		
		// Some public methods for saving and loading the layout
		public:
			void SaveLayout();
			void LoadDefaultLayout();
			bool LoadLayout(System::String ^fileName);
			void SetDefaultParams();

			// Functional methods
			void Generate();
			void Fit();
			void Stop();
			bool InSelectionChange;
			IDockContent ^GetContentFromPersistString(String ^persistString);
		private: 
			bool UseGPU;
			bool InFitMessage;
			bool checkFitCPU;
			unsigned long long FitJobStartTime;
private: System::Void configureServerToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void useGPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: void changeUseGPUDisplayValue(bool newVal);
private: System::Void openServerWindow();
private: System::Void defaultLayoutToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void saveLayoutToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void loadLayoutToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void defaultParametersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void MenuPaneStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
void PaneVisibilityToggled(System::Object^  sender, System::EventArgs^  e);
private: System::Void quitToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		 
		 System::Void graph3DChanged(System::Object^ sender);
private: System::Void MainWindow_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e);
public:		 void takeFocus(System::Object^  sender, System::EventArgs^  e);

private: System::Void luaTestToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		 
		 void BindLuaFunctions();		 

public:
		 void RepopulateEntities(System::Windows::Forms::ComboBox^ entityCombo, bool ffOnly);

		 // Modify parameter tree in UI
		 void UpdateParameters(ParameterTreeCLI ^ptref);
		 void UpdateParametersLua(LuaTable ^ptree);

		 // Functions accessible from scripts
		 void LuaMessage(String ^str);
		 LuaTable ^GetParameterTree();
		 void SetParameterTree(LuaTable ^ptree);
		 LuaTable ^LuaGenerateTable(LuaTable ^ptree, String ^saveFileName, bool saveAmp);
		 bool ClearParameterTree(ParameterTree *pt);

		 cli::array<ModelPtr> ^LuaInnerGenerateTable(LuaTable ^ ptree);

		 LuaTable ^LuaGenerateCurrent();
		 LuaTable ^LuaReadData(String ^filename); // x, y
		 LuaTable ^LuaFit(LuaTable ^data, LuaTable ^ptree, LuaTable ^fittingProps/*, [Out] LuaTable ^% outPTree*/);
		 LuaTable ^MainWindow::LuaFitCurrent();
		 bool LuaWriteData(String ^filename, LuaTable ^data); // x, y		 
		 String ^LuaFindModel(String ^modelName, String ^container);
		 String ^LuaFindAmplitude(String ^ampmodelName, String ^container);

		 // If fignum is -1, adds to latest figure (or opens a new one if none are open)
		 int LuaOpenFigure(String ^title, String ^xLabel, String ^yLabel);
		 void LuaShowGraph(int fignum, LuaTable ^data, String ^color);
		 void LuaShowFileGraph(int fignum, String ^filename, String ^color);

		 static LuaTable ^ GetParamTree(Entity ^ root, Lua ^ luaState);

		 static String ^ LuaFindModelHelper(FrontendComm * frontend, String ^ container, String ^ modelName, bool bAmp);

		 Entity ^RegisterLuaEntity(String ^filename, int type);
		 Entity ^RegisterManualSymmetry(String ^fname);

		 void ImportParameterFile(String ^filename);
		 void SaveParametersToFile(String ^filename);


		 // Helper methods
		 delegate void FuncNoParams();
		 delegate void FuncString(String ^input);
		 delegate LuaTable ^FuncLuaTable(LuaTable ^input);
		 delegate LuaTable ^FuncReturnLuaTable();
		 delegate array<ModelPtr> ^FuncLuaTableArr(LuaTable ^input);
		 delegate bool BoolFuncStringLuatable(String ^, LuaTable ^);
		 void InnerParamTreeFromTable(LuaTable ^tbl, ParameterTree *pt, bool bAmplitude);
		 void InnerParamTreeFromTable(LuaTable ^tbl, ParameterTree *pt, bool bAmplitude, bool bSkipModelCreationInBackend);
		 ParameterTree ParamTreeFromTable(LuaTable ^tbl, bool bSkipModelCreationInBackend);
		 ParameterTree ParamTreeFromTable(LuaTable ^tbl);
		 
		 // UNUSED CODE
		 //void InnerParamTreeToTable(LuaTable ^% tbl, const ParameterTree *pt, bool bAmplitude);
		 //LuaTable ^ParamTreeToTable(const ParameterTree& pt, LuaTable ^origtbl);

		 ModelPtr ModelFromLuaString(String ^str, LuaTable ^mtbl, bool bAmp);
		 LuaTable ^DataToLuaTable(const std::vector<double>& x, const std::vector<double>& y);
		 void LuaTableToData(LuaTable ^data, std::vector<double>& x, std::vector<double>& y);
		 Entity ^InnerSetParameterTree(ParameterTree *pt, LuaTable ^ptbl);
		 void EnableStopButton();
		 void EnableGenDFitButton(bool newStatus);
		 void OpenFigureHelper();
		 System::Void Figure_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e);
		 Entity ^CreateEntityFromID(const wchar_t *container, int id, String ^name);



		 // Progress and notify completion callbacks
		 
		 delegate void ProgressCallbackFunc(IntPtr args, double progress);
		 delegate void CompletionCallbackFunc(IntPtr args, int error);
		 ProgressCallbackFunc ^pcf;
		 CompletionCallbackFunc ^ccf;


		 void ProgressCallback(IntPtr args, double progress);
		 void CompletionCallback(IntPtr args, int error);		 		 		 		 		 		 
private: System::Void saveParametersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void importParametersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);

private: System::Void saveVantagePointToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void loadVantagePointToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void undoToolStrip_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void redoToolStrip_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void MainWindow_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);		 
		 ModelInformation LuaParseInformationTable(LuaTable ^infotbl);
private: System::Void openSignalToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void closeSignalToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		 ParameterTree PrepareForWork(FittingProperties& fp);
		 ErrorCode SaveStateAndGenerate();
		 ErrorCode SaveStateAndFit(std::vector<int> mask, std::string& message);
		 System::String ^GetLuaScript(void);
		 bool Save1DGraph();
public:
		 void LoadSignal( String ^filename );

		 // Parses a Lua expression from "=expr" string
		 Double LuaParseExpression(String ^ val);
		 LevelOfDetail GetLevelOfDetail();		 
private: System::Void export1DGraphToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void exportAmplitudeFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void toggleConsoleToolStripButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void exportPDBRepresentationToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
		 void ClearAndSetPopulationCount(int populations);
		 void AddModelsToParamTreeFromTable(ParameterTree * domain, LuaTable ^ tbl, bool bAmplitude, bool bSkipModelCreationInBackend);
		 paramStruct GetCompositeDomainParameters();
public:  void ResizeNumDomainModels(int populations);
private: System::Void save3DViewToFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: void PollStatus(Object^ source, System::Timers::ElapsedEventArgs^ args);
private: void fitMessage();
private: String ^ serverAddress;
private: String ^ validationCode;
		 //The HTTP caller:
private: ManagedHTTPCallerForm ^ httpCallForm;
private: ManagedPythonPreCaller ^ pythonCall;
private:
		 System::Void OnserverLabelClicked(System::Object^  sender, System::EventArgs^  e);
		 System::Void OncancelButtonClicked(System::Object^  sender, System::EventArgs^  e);
		 System::Void OnRestartButtonClicked(System::Object^  sender, System::EventArgs^  e);
		 void savelayoutonclose();
		 void HandleErr(String ^ title, ErrorCode err);

		 private: bool responseWaiting = false;
private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e);
private: System::Void suggestParametersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void pdbUnitsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: void checkCapabilitiesMainWind();
private: System::Void aboutDToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void manualToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void visitHomePageToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e);
};

[Serializable]
public ref class UserInputException : public System::Exception
{
public:
	UserInputException() : System::Exception() {}
	UserInputException(System::String^ message) : System::Exception(message) {}
	UserInputException(System::String^ message, System::Exception^ inner) : Exception(message, inner) {}
protected:
	UserInputException(System::Runtime::Serialization::SerializationInfo^ info, System::Runtime::Serialization::StreamingContext context) : System::Exception(info, context) {}
};

}

