#include "FormFactor.h"


#include "SmoothWindow.h"
#include "ResultsWindow.h"
#include "FitRange.h"
#include "ExtractBaseline.h"
#include "ExternalModelDialog.h"

#include "edprofile.h"

#include "ErrorTableWindow.h"

#include <limits>


using namespace System::Windows::Forms;

template <typename T> inline T sq(T x) { return x * x; }

namespace GUICLR {  
	/*
		ListviewFF indices for the nth layer parameter (out of nlp):
		Value                - 2 * n + 1
		Mutability           - 2 * n + 2
		Constraint minimum   - 2 * nlp + 7 * n + 1
		Constraint maximum   - 2 * nlp + 7 * n + 2
		Constraint min index - 2 * nlp + 7 * n + 3
		Constraint max index - 2 * nlp + 7 * n + 4
		Constraint link      - 2 * nlp + 7 * n + 5
		Use constraint field - 2 * nlp + 7 * n + 6
		Standard deviation   - 2 * nlp + 7 * n + 7
	*/
#ifndef LV_POSITIONS
#define LV_POSITIONS

#define LV_PAR_SUBITEMS       (7)
#define LV_NAME               (0)
#define LV_VALUE(n)           (2 * n + 1)
#define LV_MUTABLE(n)         (2 * n + 2)
#define LV_CONSMIN(n, nlp)    (2 * nlp + LV_PAR_SUBITEMS * n + 1)
#define LV_CONSMAX(n, nlp)    (2 * nlp + LV_PAR_SUBITEMS * n + 2)
#define LV_CONSIMIN(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 3)
#define LV_CONSIMAX(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 4)
#define LV_CONSLINK(n, nlp)   (2 * nlp + LV_PAR_SUBITEMS * n + 5)
#define LV_CONS(n, nlp)       (2 * nlp + LV_PAR_SUBITEMS * n + 6)
#define LV_SIGMA(n, nlp)      (2 * nlp + LV_PAR_SUBITEMS * n + 7)
#endif
	/*
		listView_Extraparams indices
		Name               - 0
		Value              - 1
		Mutable ("N"/"Y")  - 2
		Constraint minimum - 3
		Constraint maximum - 4
		Infinite ("0"/"1") - 5
		Use constraint     - 6
		Standard deviation - 7
	*/
#ifndef ELV_POSITIONS
#define ELV_POSITIONS
#define ELV_NAME     0
#define ELV_VALUE    1
#define ELV_MUTABLE  2
#define ELV_CONSMIN  3
#define ELV_CONSMAX  4
#define ELV_INFINITE 5
#define ELV_CONS     6
#define ELV_SIGMA    7
#endif
	// Debug function that displays a listView in a message box
	void DisplayLVFF(System::Windows::Forms::ListView^ LV) {
		/** DEBUG LINES **/
		std::stringstream s;
		s << " \t";
		for(int j = 0; j < LV->Items[0]->SubItems->Count; j++)
			s << "j=[" << j << "]\t";
		s << "\n";
		for(int i = 0; i < LV->Items->Count; i++) {
			s << "i=[" << i << "] ";
			for(int j = 0; j < LV->Items[i]->SubItems->Count; j++) {
				s << clrToString(LV->Items[i]->SubItems[j]->Text).substr(0, 7);
				s << '\t';
			}
			s << '\n';
		}
		MessageBoxA(NULL, s.str().c_str(), clrToString(LV->Name).c_str(), NULL);
		/** END DEBUG LINES **/
	}

	FormFactor::FormFactor(const wchar_t *filename, bool bGenerate, OpeningWindow^ parent, String^ pCont) {
		_state = FWS_IDLE;		

		_bSaved			= false;
		_bAllocatedLF	= false;
		_bUseFF			= false;
		_peakPicker		= false;
		_bChanging		= false;
		_bFromFitter	= false;
		_curWssr = -1.0;
		_curFFPar = NULL;
		oldIndex = -1;

		// We have to allocate another int because of CLR and managed types
		_pShouldStop = new int; *_pShouldStop = 0;

		// Undo queue
		undoQueue = new std::deque< std::pair<ModelType, paramStruct> >();
		undoQueueIndex = -1;

		// Delegate handler handles
		progressrep = gcnew CLRProgressFunc(this, &GUICLR::FormFactor::ProgressReport);
		notifycomp  = gcnew CLRNotifyCompletionFunc(this, &GUICLR::FormFactor::NotifyCompletion);

		_bGenerateModel = bGenerate;
		if(!_bGenerateModel)
			_dataFile = gcnew System::String(filename);

		InitializeComponent();
		if(!_bGenerateModel)
			this->Text += " -  [" + CLRBasename(_dataFile) + "]";

		_data = new graphTable;
		_ff = new graphTable;
		_sf = new graphTable;
		_bg = new graphTable;
		_baseline = new graphTable;
		_storage = new graphTable;

		// For some reason, if the visibility is set to false in the designer then the tool strip is not
		// visible in the designer.  So, I set it to visible and will change it here.
		this->maskToolStrip->Visible = false;
		this->wssr->Text = UnicodeChars::chisqr + this->wssr->Text;
		this->rsquared->Text = UnicodeChars::rsqr + this->rsquared->Text;
		SetMinimumSig(5.0);

		_mask = new std::vector<int>;

		_loadedFF = new std::wstring;

		graphType = new std::vector<GraphSource>;
		_ph = new std::vector<double>;
		_generatedPhaseLocs = new std::vector<double>;
		phaseSelected = new PhaseType;
		indicesLoc = new std::vector<std::string>;
		FFparamErrors = new std::vector<double>;
		SFparamErrors = new std::vector<double>;
		BGparamErrors = new std::vector<double>;
		PhaseparamErrors = new std::vector<double>;
		FFmodelErrors = new std::vector<double>;
		SFmodelErrors = new std::vector<double>;
		BGmodelErrors = new std::vector<double>;
		_copiedIndicesFF = new std::vector<int>;
		Globalization::CultureInfo ^American;
		American = gcnew Globalization::CultureInfo(L"en-US");
		Thread::CurrentThread->CurrentUICulture = American;//System::Globalization::CultureInfo::NumberFormat InvariantCulture; //CultureInfo("en") ;
		Thread::CurrentThread->CurrentCulture = American;//System::Globalization::CultureInfo::NumberFormat InvariantCulture; //CultureInfo("en") ;

		_parent = parent;
		_miFF = _parent->_currentModel;
		_lf = _parent->lf;
		_containers = gcnew array<System::String^>{L"MISTAKE"};
		Array::Resize(_containers, _parent->files->Length);
		Array::Copy(_parent->files, 0, _containers, 0, _parent->files->Length);
		_mioFF = gcnew modelInfoObject(_miFF->category, _miFF->modelIndex, -1, _miFF->name, pCont);
		renderScene = _parent->renderScene;
	}


	FormFactor::~FormFactor() {
		delete _pShouldStop;

		delete _data;
		delete _ff;
		delete _sf;
		delete _bg;
		delete _baseline;
		delete _storage;
		delete _mask;
		delete FFparamErrors;
		delete SFparamErrors;
		delete BGparamErrors;
		delete PhaseparamErrors;
		delete FFmodelErrors;
		delete SFmodelErrors;
		delete BGmodelErrors;

		delete undoQueue;
		
		delete _loadedFF;
		delete _copiedIndicesFF;

		if(_bAllocatedLF && _lf) {
			delete _lf;
			_lf = NULL;
		}

		if(graphType)
			delete graphType;
		graphType = NULL;

		if(_curFFPar)
			delete _curFFPar;
		_curFFPar = NULL;

		if(_curSFPar)
			delete _curSFPar;
		_curSFPar = NULL;

		if(_curBGPar)
			delete _curBGPar;
		_curBGPar = NULL;

		if(_curSFPar)
			delete _curSFPar;
		_curSFPar = NULL;
		
		if(_ph)
			delete _ph;
		_ph = NULL;

		if(_generatedPhaseLocs)
			delete _generatedPhaseLocs;
		_generatedPhaseLocs = NULL;

		if(_modelSF) {
			delete _modelSF;
			_modelSF = NULL;
		}

		if(_modelBG) {
			delete _modelBG;
			_modelBG = NULL;
		}

		if(_miSF) {
			delete _miSF;
			_miSF = NULL;
		}

		if(_miBG) {
			delete _miBG;
			_miBG = NULL;
		}

		if(phaseSelected)
			delete phaseSelected;
		phaseSelected = NULL;

		if(indicesLoc)
			delete indicesLoc;
		indicesLoc = NULL;
	
		if(iniFile)
			delete iniFile;
		iniFile = NULL;

		if (components)
		{
			delete components;
		}
	}

	void FormFactor::InitializeEDProfile() {
		this->wgtPreview = (gcnew GUICLR::WGTControl());
		this->edpBox->Controls->Add(this->wgtPreview);
		// 
		// wgtPreview
		// 
		this->wgtPreview->Cursor = System::Windows::Forms::Cursors::Cross;
		this->wgtPreview->Dock = System::Windows::Forms::DockStyle::Fill;
		this->wgtPreview->Location = System::Drawing::Point(0, 0);
		this->wgtPreview->Name = L"wgtPreview";
		this->wgtPreview->Size = System::Drawing::Size(343, 349);
		this->wgtPreview->TabIndex = 0;
		
		struct graphLine graphs[3];
		std::vector<std::vector<Parameter> > p;
		std::vector<Parameter> r, ed;

		r.push_back(Parameter(0.0));
		ed.push_back(Parameter(333.0));

		p.push_back(r);
		p.push_back(ed);

		// TODO::EDP
		//generateEDProfile(p, graphs, _modelFF->GetEDProfile());

		RECT area;
		area.top = 0;
		area.left = 0;
		area.right = wgtPreview->Size.Width;
		area.bottom = wgtPreview->Size.Height;
		wgtPreview->graph = gcnew Graph(
							area, 
							graphs[0].color, 
							DRAW_LINES, graphs[0].x, 
							graphs[0].y, 
							false,
							false);
		wgtPreview->graph->Add(graphs[1].color, 
							   DRAW_LINES, 
							   graphs[1].x, graphs[1].y);
		wgtPreview->graph->Add(graphs[2].color, 
							   DRAW_LINES, 
							   graphs[2].x, graphs[2].y);

		// No ticks
		wgtPreview->graph->ToggleXTicks();
		wgtPreview->graph->ToggleYTicks();
		wgtPreview->graph->Resize(area);
	}

	void FormFactor::logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if(wgtFit && wgtFit->graph) {
				 wgtFit->graph->SetScale((sender == logScale ? 0 : 1), ((CheckBox ^)sender)->Checked ? SCALE_LOG : SCALE_LIN);
				 wgtFit->Invalidate();
			 }
		 }

	void FormFactor::UpdateEDPreview() {
		//std::vector<double> r, ed, z0;
		struct graphLine graphs[3];
		//std::string str;

		if(_bChanging)
			return;
	
		if(!this->Visible || !wgtPreview->Visible)
			return;
		paramStruct p = *_curFFPar;

		if(_modelFF->GetEDProfile().type != NONE)
			generateEDProfile(p.params, graphs, _modelFF->GetEDProfile());

		wgtPreview->graph->Modify(0, graphs[0].x, graphs[0].y);
		wgtPreview->graph->Modify(1, graphs[1].x, graphs[1].y);
		wgtPreview->graph->Modify(2, graphs[2].x, graphs[2].y);
		wgtPreview->graph->FitToAllGraphs();

		wgtPreview->Invalidate();
	}

	void FormFactor::FormFactor_Load(System::Object^  sender, System::EventArgs^  e) {
		_bLoading = true;
		
		/*********************************************************************
		/* New area for communication, job and stuff. May be prone to memory *	
		/* leaks.	                                     	                 *
		/*********************************************************************/
		if(!_lf) {
			_lf = new LocalFrontend();
			_bAllocatedLF = true;
		}
				
		wchar_t jobtitle[MAX_PATH];
		if(_bGenerateModel)
			clrToWchar_tp("Generate", jobtitle);
		else
			clrToWchar_tp(_dataFile, jobtitle);
		_job = _lf->CreateJob(jobtitle, 
							  static_cast<progressFunc>(Marshal::GetFunctionPointerForDelegate(progressrep).ToPointer()), 
							  static_cast<notifyCompletionFunc>(Marshal::GetFunctionPointerForDelegate(notifycomp).ToPointer()));
		
		if(!_lf->IsValid()) {
			Windows::Forms::MessageBox::Show("No valid backend.", "ERROR",
										MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		ErrorCode setModel = _lf->SetModel(_job, MT_FORMFACTOR/* TODO::implementAll*/, clrToWstring(_mioFF->contName).c_str(), _miFF->modelIndex, EDProfile());
		if(setModel)
			Windows::Forms::MessageBox::Show("Unable to set form factor model. Error code: " + ((Int32)setModel).ToString(), "ERROR", MessageBoxButtons::OK,
												MessageBoxIcon::Error);
		this->Text += _job;

		if(!_modelFF) {
			_modelFF = new ModelUI();
			_modelFF->setModel(_lf, clrToWstring(_mioFF->contName).c_str(), *_miFF, 10);
		}

		if(!_modelBG) {
			_modelBG = new ModelUI();
			// TODO::BG _modelBG->setModel(_lf, clrToWstring(_mioBG->contName).c_str(), _job, MT_BACKGROUND, *_miBG, 10);
		}

		/*********************************************************************/
		/*              END OF NEW ZONE	                                     */
		/*********************************************************************/

		//reportButton->Visible = false;

		this->KeyPreview = true;

		gPUToolStripMenuItem->Enabled = hasGPUBackend();
		gPUToolStripMenuItem->Checked = isGPUBackend();
		cPUToolStripMenuItem->Checked = !isGPUBackend();

		edpResolution->Text = Int32(DEFAULT_EDRES).ToString();

		InitializeEDProfile();

		edpBox->Visible = _modelFF->IsLayerBased();

		// The big "disable everything related to fitting" if
		if(_bGenerateModel) {
			generationToolStripMenuItem->Visible = true;
			genRangeBox->Visible = true;
			calculate->Text = "Generate";
			liveFittingToolStripMenuItem->Text = "Live Generation";
			label6->Text = "Generating...";
			calculate->Enabled = true;
			fitphase->Enabled = false;
			manipBox->Visible = false;
			label1->Visible = false;
			minim->Enabled	= false;
			maxim->Enabled	= false;
			LocOnGraph->Visible = true;
			consGroupBox->Visible = false;
			wssr->Visible = false;
			rsquared->Visible = false;
			exportBackgroundToolStripMenuItem->Visible = false;
			fittingMethodToolStripMenuItem->Visible = false;
			maskButton->Visible = false;
			reportButton->Visible = false;

			thresholdBox1->Visible = false;
			thresholdBox2->Visible = false;
			Threshold_label1->Visible = false;
			Threshold_label2->Visible = false;
			automaticPeakFinderButton->Visible = false;
			PeakPicker->Visible = false;
			PeakFinderCailleButton->Visible = false;
			
			// Phases
			label22->Enabled	= false;
			label23->Enabled	= false;
			MinPhases->Enabled	= false;
			MaxPhases->Enabled	= false;
			fitphase->Text = L"Generate Phase";
			fitphase->Width = 92;
			undoPhases->Width = 50;
			clearPositionsButton->Width = 50;
			undoPhases->Location = System::Drawing::Point(110, 43);
			clearPositionsButton->Location = System::Drawing::Point(163, 43);
			listView_phases->Columns[1]->Width *= 2;
			listView_phases->Columns[2]->Width = 0;
			listView_phases->Columns[3]->Width = 0;
			listView_phases->Columns[4]->Width = 0;
			listView_phases->Columns[5]->Width *= 2;

			exmin->Enabled = false;
			exmax->Enabled = false;

			exportSignalToolStripMenuItem->Visible = false;
			exportDecomposedToolStripMenuItem->Visible = false;
			exportSigModBLToolStripMenuItem->Visible = false;
			importBaselineToolStripMenuItem->Enabled = false;

			liveRefreshToolStripMenuItem->Checked = false;

			changeData->Visible = false;
			PeakPicker->Visible = false;
			PeakFinderCailleButton->Visible = false;

			accurateDerivativeToolStripMenuItem->Visible = false;
			accurateFittingToolStripMenuItem->Visible = false;
			chiSquaredBasedFittingToolStripMenuItem->Visible = false;
			logScaledFittingParamToolStripMenuItem->Visible = false;
			minimumSignalToolStripMenuItem->Visible = false;

			// Background tab
			baseMut->Visible	= false;
			baseMaxBox->Visible	= false;
			baseMinBox->Visible	= false;
			decayMut->Visible	= false;
			decMaxBox->Visible	= false;
			decMinBox->Visible	= false;
			xCenterMut->Visible	= false;
			xcMaxBox->Visible	= false;
			xcMinBox->Visible	= false;
			maxLabel->Visible	= false;
			minLabel->Visible	= false;
			BGListview->Columns[3]->Width = 0;
			BGListview->Columns[5]->Width = 0;
			BGListview->Columns[7]->Width = 0;
		}
		// End of the "big generation if"

		// Prepares the GUI for the chosen model (adding parameters, extra 
		// parameters and so on)
		PrepareModelUI();

		ExtractBaseline::bUsingOld = false;

		// Load the rest of the UIs
		StructureFactor_Load();
		Background_Load();


		// Read radii and EDs
		paramStruct par(_modelFF->GetModelInformation());
		std::string type = _modelFF->GetName();
		std::wstring filename;
	
		if(_bGenerateModel) {
			filename = L".\\XModelFitter.ini";
		} else {
			std::wstring res, dir;
			std::wstring dataFile;
			clrToString(_dataFile, dataFile);
			
			dir = clrToWstring(CLRDirectory(_dataFile));
			res = clrToWstring(CLRBasename(_dataFile));
			
			res = dir + res + L"-params.ini";

			filename = res;
		}

		iniFile = NewIniFile();

		if(CheckSizeOfFile(filename.c_str()) > 0 && IniHasModelType(filename.c_str(), type, iniFile)) {
			// Read ED Profile configuration (prior to parameters)
			{
				// Get the default electron density profile
				EDProfile defaultEDP = _modelFF->GetEDProfile();
				// If not discrete, disable other options
				if(defaultEDP.shape != DISCRETE) {
					electronDensityProfileToolStripMenuItem->Visible = false;
				} else {
					electronDensityProfileToolStripMenuItem->Visible = true;

					// Read profile shape
					ProfileShape psh = (ProfileShape)GetIniInt(filename, type, 
															   "EDProfileShape", 
															   iniFile, DISCRETE);
/* TODO::EDP
					EDProfile op = _modelFF->GetEDProfile();
					_modelFF->SetEDProfile(EDProfile(op.type, psh));
*/


					// Read profile resolution
					int res = GetIniInt(filename, type, "EDProfileResolution", 
										iniFile, DEFAULT_EDRES);
/* TODO::EDP
					if(_modelFF->GetEDProfileFunction())
						_modelFF->GetEDProfileFunction()->SetResolution(res);
*/

					int absres = (res < 0) ? -res : res;

					// Resolution GUI modification
					edpResolution->Enabled = true;
					edpResolution->Text = Int32(absres).ToString();
					adaptiveToolStripMenuItem->Checked = (res < 0) ? true : false;
				}
			}

			// Set ED profile menu items
			{
				ProfileShape psh = _modelFF->GetEDProfile().shape;

				discreteStepsToolStripMenuItem->Checked = (psh == DISCRETE);
				gaussiansToolStripMenuItem->Checked = (psh == GAUSSIAN);
				hyperbolictangentSmoothStepsToolStripMenuItem->Checked = (psh == TANH);
			}
				
			bool bValidFile = ReadParameters(filename.c_str(), type, &par, *_modelFF, iniFile);

			if(bValidFile) {
				ParametersToUI(&par, _modelFF, listViewFF, listView_Extraparams);
				UItoParameters(_curFFPar, _modelFF, listViewFF, listView_Extraparams);

				if(_bGenerateModel) {
					std::string tmpstr;
					GetIniString(filename, type, "GenRangeStart", tmpstr, iniFile);
					if(! (strtod(tmpstr.c_str(), NULL) > 0.0))
						startGen->Text = gcnew System::String("0.100000");
					else
						startGen->Text = gcnew System::String(tmpstr.c_str());
					GetIniString(filename, type, "GenRangeEnd", tmpstr, iniFile);
					if(! (strtod(tmpstr.c_str(), NULL) > 0.0))
						endGen->Text = gcnew System::String("5.000000");
					else
						endGen->Text = gcnew System::String(tmpstr.c_str());
					int tmpInt = GetIniInt(filename, type, "GenResolution", iniFile);
					toolStripTextBox2->Text = (tmpInt > 1) ? tmpInt.ToString() : gcnew System::String("500");
				}

				// The reason the following (quadrature, extra parameters) are
				// here is to make sure we have a saved model before applying settings
				// (the defaults are different)

				// Quadrature
				if(integrationToolStripMenuItem->Visible) {
					toolStripTextBox1->Text = GetIniInt(filename, type, "quadratureres", iniFile).ToString();

					switch(int(GetIniInt(filename, type, "quadraturemethod", iniFile))) {
						default:
							break;

						case QUAD_MONTECARLO:
							gaussLegendreToolStripMenuItem->Checked = false;
							monteCarloToolStripMenuItem->Checked = true;
							break;

						case QUAD_SIMPSON:
							gaussLegendreToolStripMenuItem->Checked = false;
							simpsonsRuleToolStripMenuItem->Checked = true;
							break;
					}
					ClassifyQuadratureMethod((QuadratureMethod)(int((GetIniInt(filename, type, "quadraturemethod", iniFile)))));
				}
			}
			
			// Loading peaks from INI
			paramStruct parsf(_modelSF->GetModelInformation());
			bValidFile = ReadParameters(filename.c_str(), type, &parsf, *_modelSF, iniFile);
			ParametersToUI(&parsf, _modelSF, listView_peaks, nullptr);
			UItoParameters(_curSFPar, _modelSF, listView_peaks, nullptr);
/*
			peakStruct peaks;
			ReadPeaks(filename, type, &peaks, iniFile);
			//peakfit->SelectedIndex = GetPeakType();
			for(unsigned int i = 0; i < peaks.amp.size(); i++)
				AddPeak(peaks.amp[i].value, peaks.amp[i].mut, peaks.fwhm[i].value, peaks.fwhm[i].mut,
				peaks.center[i].value, peaks.center[i].mut);
*/			

			// Loading BG functions from INI
			// TODO::BG

			AddPhasesParam("a",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[0]
			AddPhasesParam("b",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[1]
			AddPhasesParam(UnicodeChars::gammaUnicode,MODE_ABSOLUTE,90.0,0.0, 180.0);	// Items[2]
			AddPhasesParam("c",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[3]
			AddPhasesParam(UnicodeChars::alphaUnicode,MODE_ABSOLUTE,90.0,0.0,180.0);	// Items[4]
			AddPhasesParam(UnicodeChars::betaUnicode,MODE_ABSOLUTE,90.0,0.0,180.0);	// Items[5]

			// Loading Phases from INI			
			int pt;
			phaseStruct ps;
			// TODO::Phases
// 			ReadPhases(filename, type, &ps, &pt, iniFile);
// 			order->SelectedIndex = pt;
// 			SetPhases(&ps);

			// Calculate Reciprocal values
			calculateRecipVectors();

			// Polydispersity settings
			int pdType, pdRes;
			pdType = GetIniInt(filename, type, "PDFunc", iniFile);
			pdRes = GetIniInt(filename, type, "PDResolution", iniFile);
			SetPDFunc((PeakType)pdType);
			switch((PeakType)pdType) {
				default:
				case SHAPE_GAUSSIAN:
					uniformPDToolStripMenuItem->Checked = false;
					gaussianPDToolStripMenuItem->Checked = true;
					lorentzianPDToolStripMenuItem->Checked = false;
					break;

				case SHAPE_LORENTZIAN:
					uniformPDToolStripMenuItem->Checked = false;
					gaussianPDToolStripMenuItem->Checked = false;
					lorentzianPDToolStripMenuItem->Checked = true;
					break;

				case SHAPE_LORENTZIAN_SQUARED:
					uniformPDToolStripMenuItem->Checked = true;
					gaussianPDToolStripMenuItem->Checked = false;
					lorentzianPDToolStripMenuItem->Checked = false;
					break;
			}
			
			// General Settings
			logScale->Checked = (GetIniChar(filename, "Settings", "logscale", iniFile) == 'Y');
			logScaledFittingParamToolStripMenuItem->Checked = (GetIniChar(filename, "Settings", "LogFitting", iniFile) == 'Y');
			liveRefreshToolStripMenuItem->Checked = (GetIniChar(filename, "Settings", "liverefresh", iniFile) == 'Y');
			liveFittingToolStripMenuItem->Checked = (GetIniChar(filename, "Settings", "livefit", iniFile) == 'Y');
			sigmaToolStripMenuItem->Checked = (GetIniChar(filename, "Settings", "Sigma", iniFile) == 'Y' ||GetIniChar(filename, "Settings", "Sigma", iniFile) == '-');
			fWHMToolStripMenuItem->Checked = !sigmaToolStripMenuItem->Checked;
			sigmaToolStripMenuItem_CheckedChanged(sender,e);
			sigmaFWHMToolStripMenuItem->Visible = GetPeakType() == SHAPE_GAUSSIAN;

			std::string tmpStr;
			GetIniString(filename, "Settings", "MinSignal", tmpStr, iniFile);
			if(! (strtod(tmpStr.c_str(), NULL) > 0.0))
				minimumSignalTextbox->Text = gcnew System::String("5.000000");
			else
				minimumSignalTextbox->Text = gcnew System::String(tmpStr.c_str());

		} else {	// Enter default values for Caille and Phases

			// Prepares the GUI for the chosen model (adding parameters, extra 
			// parameters and so on)
			//PrepareModelUI();

			AddPhasesParam("a",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[0]
			AddPhasesParam("b",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[1]
			AddPhasesParam("gamma",MODE_ABSOLUTE,90.0,0.0, 180.0);	// Items[2]
			AddPhasesParam("c",MODE_ABSOLUTE,6.0,0.0,100.0);		// Items[3]
			AddPhasesParam("alpha",MODE_ABSOLUTE,90.0,0.0,180.0);	// Items[4]
			AddPhasesParam("beta",MODE_ABSOLUTE,90.0,0.0,180.0);	// Items[5]
		}

		CloseIniFile(iniFile);
		iniFile = NULL;
		
		// Load the preview window
		if(!timer1->Enabled) {
			if(oglPreview) {
				delete oglPreview;
			}
 			oglPreview = gcnew OpenGLWidget(oglPanel, 
 											gcnew pRenderScene(this, &GUICLR::FormFactor::RenderPreviewScene));

			timer1->Enabled = true;
			oglPreview->Render();
			oglPreview->SwapOpenGLBuffers();
		}


		// Update E.D. Preview
		UpdateEDPreview();

		initPhasesParams();


		// Start in BGTab so baseline can be removed
		if(!_bGenerateModel)
			tabControl1->SelectTab("BGTab");
		
		EDAreaGroup->Visible = true;

		slowModelGroupbox->Visible = _modelFF->IsSlow();

		if(listView_Extraparams->Items->Count > 0)
			exParamGroupbox->check->Enabled = false;

		SFParameterUpdateHandler();
		BGParameterUpdateHandler();
		_bChanging = true;	// triggers the UpdateGraph(true) in ParameterUpdateHandler
		FFParameterUpdateHandler();
		_bLoading = false;
		_bChanging = false;
	
	}

	//Adds each corresponding value of a and b
	void FormFactor::AddVectors(vector<double> &result, const vector<double> &a, 
								const vector<double> &b) {
		int size = min(a.size(), b.size());

		result.resize(size);

		for(int i = 0; i < size; i++)
			result[i] = a[i] + b[i];
	}

	//Subtracts each corresponding value of b from a
	void FormFactor::SubtractVectors(vector<double> &result, const vector<double> &a, 
									 const vector<double> &b) {
		int size = min(a.size(), b.size());

		result.resize(size);
		
		for(int i = 0; i < size; i++)
			result[i] = a[i] - b[i];
	}

	//Multiplies each corresponding value of a and b
	void FormFactor::MultiplyVectors(vector<double> &result, const vector<double> &a, 
									 const vector<double> &b) {
		int size = min(a.size(), b.size());

		result.resize(size);
		for(int i = 0; i < size; i++)
			result[i] = a[i] * b[i];
	}
	//Divides each corresponding value of a by b
	void FormFactor::DivideVectors(vector<double> &result, const vector<double> &a, 
									 const vector<double> &b) {
		int size = min(a.size(), b.size());

		result.resize(size);
		for(int i = 0; i < size; i++)
			if(fabs(b[i]) > 0.0)
				result[i] = a[i] / b[i];
			else
				result[i] = a[i] / 1e-9;

	}

	void FormFactor::multiplyVectorByValue(std::vector<double> &vec, double val) {
		for(int i = 0; i < (int)vec.size(); i++)
			vec.at(i) *= val;
	}

	void FormFactor::InitializeFitGraph() {
		vector<double> x, y;
		InitializeFitGraph(x, y);
	}

	void FormFactor::InitializeFitGraph(vector<double>& x, vector<double>& y) {
		delete wgtFit;
		graphType->clear();
		this->wgtFit = (gcnew GUICLR::WGTControl());
		this->wgtFit->setDragToZoom(g_bDragToZoom);
		this->groupBox1->Controls->Add(this->wgtFit);

		// 
		// wgtFit
		// 
		this->wgtFit->Cursor = System::Windows::Forms::Cursors::Cross;
		this->wgtFit->Dock = System::Windows::Forms::DockStyle::Fill;
		this->wgtFit->Location = System::Drawing::Point(0, 0);
		this->wgtFit->Name = L"wgtFit";
		this->wgtFit->Size = System::Drawing::Size(343, 349);
		this->wgtFit->TabIndex = 0;
		this->wgtFit->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &FormFactor::wgtFit_MouseMove);
		this->wgtFit->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &FormFactor::wgtFit_MouseDown);
		this->wgtFit->MouseUp   += gcnew System::Windows::Forms::MouseEventHandler(this, &FormFactor::wgtFit_MouseUp);
		this->wgtFit->Visible = true;

		RECT area;
		area.top = 0;
		area.left = 0;
		area.right = wgtFit->Size.Width;
		area.bottom = wgtFit->Size.Height;
		wgtFit->graph = gcnew Graph(
							area, 
							RGB(250, 0, 0), 
							DRAW_LINES, x, y, 
							logXCheckBox->Checked,
							logScale->Checked);
		graphType->push_back(GRAPH_DATA);

		wgtFit->graph->SetXLabel("q [nm" + UnicodeChars::minusOne + "]");
		wgtFit->graph->SetYLabel("Intensity [a.u.]");

		wgtFit->graph->Resize(area);

		std::vector<std::string> legendNames;

		legendNames.push_back("Model");

		if(!_bGenerateModel) {
			std::vector<double> my;
			paramStruct par(_modelFF->GetModelInformation());
			peakStruct peaks;
			phaseStruct phase;
			bgStruct background;
			par = *_curFFPar;

			GetPhasesFromListView(&phase); 
			GetBGFromGUI(&background);

			UItoParameters(_curSFPar, _modelSF, listView_peaks, nullptr);


			_ff->x = x;
			_sf->x = x;
			_bg->x = x;
			_ff->y.clear();
			_sf->y.clear();	
			_bg->y.clear();	

			// Filling our separate form factor, structure factor and background graphs
			//GenerateStructureFactor(x, _sf->y, &peaks);	// TODO::SF
			//GenerateBackground(x, _bg->y, &background);	// TODO::BG

			GenerateProperties gp;
			gp.bProgressReport	= false;
			gp.liveGenerate		= false;

			//GenerateModel(x, _ff->y, &par, _pShouldStop);		
			// TODO::NewFitter
			_lf->Generate(_job, MT_FORMFACTOR, _curFFPar, NULL, NULL, x, gp);
			_lf->WaitForFinish(_job);
			int numPoints = _lf->GetGraphSize(_job);
			my.resize(numPoints);
			_ff->y.resize(numPoints);
			_sf->y.resize(numPoints);
			_bg->y.resize(numPoints);
			_lf->GetGraph(_job, &my[0], numPoints);
			_lf->GetGraphMoiety(_job, MT_FORMFACTOR, &(_ff->y[0]), numPoints);
			_lf->GetGraphMoiety(_job, MT_STRUCTUREFACTOR, &(_sf->y[0]), numPoints);
			_lf->GetGraphMoiety(_job, MT_BACKGROUND, &(_bg->y[0]), numPoints);
			my = MachineResolutionF(_ff->x, my, GetResolution());

			if(_ff->y.empty())	// Something not so good happened
				_ff->y.resize(_ff->x.size(), 0.0);

			//TODO::phases create a condition to select between structure factor and phases.
			//MultiplyVectors(my, _ff->y, _sf->y);
			//AddVectors(my, my, _bg->y);
			//my = MachineResolutionF(_ff->x, my, GetResolution());

			graphType->push_back(GRAPH_MODEL);
			wgtFit->graph->Add(RGB(54,13,187), DRAW_LINES, _ff->x, my);

			// Calculate Chi squared (WSSR)
			UpdateChisq(WSSR(y, my));
			// Calculate R squared
			UpdateRSquared(RSquared(y, my));

			legendNames.insert(legendNames.begin(), "Signal");

			wgtFit->graph->Legend(legendNames);

			PeakPicker->Enabled = true;
		} else {
			int res = int(clrToDouble(toolStripTextBox2->Text));
			std::vector<double> x (res, 0.0);

			double s = clrToDouble(startGen->Text), end = clrToDouble(endGen->Text);
			for(int i = 0; i < int(x.size()); i++)
				x[i] = s + (double(i + 1) * (end - s) / (double(res)));

			_ff->x = x;
			_sf->x = x;

			_ff->y.resize(x.size(), 1.0);
			_sf->y.resize(x.size(), 1.0);

			wgtFit->graph->Legend(legendNames);
		}

		wgtFit->Invalidate();
	}

	void FormFactor::UpdateExtraParamBox() {
		ListViewItem ^lvi = 
			listView_Extraparams->Items[paramBox->SelectedIndex];

		if(listView_Extraparams->SelectedItems->Count == 0)
			listView_Extraparams->SelectedIndices->Add(0);
		exParamGroupbox->Text = listView_Extraparams->SelectedItems[0]->SubItems[ELV_NAME]->Text;
		exParamGroupbox->text->Text = lvi->SubItems[exParamGroupbox->rStddev->Checked ? ELV_SIGMA : ELV_VALUE]->Text;
		exmin->Text   = lvi->SubItems[ELV_CONSMIN]->Text;
		exmax->Text   = lvi->SubItems[ELV_CONSMAX]->Text;
	
		// Extra parameter specification modifications
		ExtraParam ep = _modelFF->GetExtraParameter(paramBox->SelectedIndex);
		
		infExtraParam->Visible = ep.canBeInfinite;
		infExtraParam->Checked = lvi->SubItems[ELV_INFINITE]->Text->Equals("1");

		if(infExtraParam->Checked) {
				 exParamGroupbox->Enabled = false;
				 exmin->Enabled   = false;
				 exmax->Enabled   = false;
		} else {
				exParamGroupbox->Enabled = true;
				exmin->Enabled   = !_bGenerateModel;
				exmax->Enabled   = !_bGenerateModel;
				exParamGroupbox->check->Enabled   = !_bGenerateModel;
		}

		if(lvi->SubItems[ELV_MUTABLE]->Text->Equals("-"))
			exParamGroupbox->check->Enabled = false;

		if(!_bGenerateModel)
			exParamGroupbox->check->Checked = (lvi->SubItems[ELV_MUTABLE]->Text->Equals("Y"));
	}

	void FormFactor::UItoParameters(paramStruct *p, ModelUI *mui, ListView^ lvp, ListView^ lvep) {
		p->nlp = mui->GetNumLayerParams();
		p->nExtraParams = mui->GetNumExtraParams();
		p->layers = lvp->Items->Count;
		int nlp = mui->GetNumLayerParams();
		p->params.resize(nlp);
		for(int jaja = 0; jaja < nlp; jaja++)
			p->params[jaja].resize(p->layers);
		
		for(int i = 0; i < lvp->Items->Count; i++) {
			std::string str;
			ListViewItem ^lvi = lvp->Items[i];
		
			for (int j = 0; j < nlp; j++) {
				Parameter parame (clrToDouble(lvi->SubItems[LV_VALUE(j)]->Text),
					lvi->SubItems[LV_MUTABLE(j)]->Text->Equals("Y") ? true : false,
					(lvi->SubItems[LV_CONS(j, nlp)]->Text->Equals("Y")),
					clrToDouble(lvi->SubItems[LV_CONSMIN(j, nlp)]->Text),
					clrToDouble(lvi->SubItems[LV_CONSMAX(j, nlp)]->Text),
					(int)clrToDouble(lvi->SubItems[LV_CONSIMIN(j, nlp)]->Text),
					(int)clrToDouble(lvi->SubItems[LV_CONSIMAX(j, nlp)]->Text),
					(int)clrToDouble(lvi->SubItems[LV_CONSLINK(j, nlp)]->Text),
					clrToDouble(lvi->SubItems[LV_SIGMA(j, nlp)]->Text));

				p->params[j][i] = parame;
			}
		}
		if(lvep) {
			p->extraParams.resize(lvep->Items->Count);
			for(int i = 0; i < lvep->Items->Count; i++) {
				std::string str;
				ListViewItem ^lvi2 = lvep->Items[i];

				Parameter para(clrToDouble(lvi2->SubItems[ELV_VALUE]->Text), 
						lvi2->SubItems[ELV_MUTABLE]->Text->Equals("Y")? true : false, 
						(constraints->Checked && 
						 lvi2->SubItems[ELV_CONS]->Text->Equals("Y")) || raindrop->Checked,
						clrToDouble(lvi2->SubItems[ELV_CONSMIN]->Text),
						clrToDouble(lvi2->SubItems[ELV_CONSMAX]->Text),
						-1, -1, -1,
						clrToDouble(lvi2->SubItems[ELV_SIGMA]->Text));
				p->extraParams[i] = para;

				// If this parameter is infinite
				if(lvi2->SubItems[ELV_INFINITE]->Text->Equals("1"))
					p->extraParams[i].value = std::numeric_limits<double>::infinity();
			}
		}

		p->bConstrain = constraints->Checked;
	}

	void FormFactor::ParametersToUI(const paramStruct *p, ModelUI *mui, ListView ^lvp, ListView ^lvep) {	

		// Regular parameters
		int nlp = mui->GetNumLayerParams();
		AddToLV ^addItem;

		if(mui == _modelSF)
			addItem = gcnew AddToLV(this, &GUICLR::FormFactor::AddPeak);
		if(mui == _modelFF)
			addItem = gcnew AddToLV(this, &GUICLR::FormFactor::AddParamLayer);

		while(lvp->Items->Count < p->layers)
			addItem();
		while(lvp->Items->Count > p->layers)
			lvp->Items->RemoveAt(lvp->Items->Count - 1);

		for(int i = 0; i < lvp->Items->Count; i++) {
			ListViewItem ^lvi = lvp->Items[i];

			for(int j = 0; j < nlp; j++) {
				Parameter param = p->params[j][i];

				if(mui->IsParamApplicable(i, j)) {
					// Mutability
					lvi->SubItems[LV_MUTABLE(j)]->Text = param.isMutable ? "Y" : 
													 "N";

					// Parameter link constraint
					if(param.linkIndex >= 0) {
						lvi->SubItems[LV_MUTABLE(j)]->Text = "L";
						lvi->SubItems[LV_CONSLINK(j, nlp)]->Text = 
										param.linkIndex.ToString();
						// Override linked parameters
						param.value = p->params[j][param.linkIndex].value;
					}

					// Value
					lvi->SubItems[LV_VALUE(j)]->Text = 
						param.value.ToString("0.000000");

					// Absolute constraints
					lvi->SubItems[LV_CONSMIN(j, nlp)]->Text =
										param.consMin.ToString("0.000000");
					lvi->SubItems[LV_CONSMAX(j, nlp)]->Text =
										param.consMax.ToString("0.000000");
					lvi->SubItems[LV_CONS(j, nlp)]->Text =
										param.isConstrained ? "Y" : "N";

					// Relative constraints
					lvi->SubItems[LV_CONSIMIN(j, nlp)]->Text =
										param.consMinIndex.ToString();
					lvi->SubItems[LV_CONSIMAX(j, nlp)]->Text =
										param.consMaxIndex.ToString();
					
					// Model modifiers
					lvi->SubItems[LV_SIGMA(j, nlp)]->Text =
										param.sigma.ToString();
				}
			}
		}
		
		if(lvep) {
			for(int i = 0; i < lvep->Items->Count; i++) {
				ListViewItem ^lvi2 = lvep->Items[i];
				int decpoints = mui->GetExtraParameter(i).decimalPoints;

				Parameter param = p->extraParams[i];

				char a[64] = {0};
				sprintf(a, "%.*f", mui->GetExtraParameter(i).decimalPoints,
						param.value);

				// If this parameter is infinite
				if(mui->GetExtraParameter(i).canBeInfinite && 
					!_finite(param.value)) {
						if(!(lvi2->SubItems[ELV_INFINITE]->Text->Equals("1"))) {
							// If the parameter wasn't infinite, add an "(inf)"
							lvi2->Text += " (inf)";
							lvi2->SubItems[ELV_INFINITE]->Text = "1";
						}
						sprintf(a, "%.*f", decpoints, 0.0);
				} else if(_finite(param.value) && lvi2->SubItems[ELV_INFINITE]->Text->Equals("1")) {
					// Removing the "(inf)" if necessary
					lvi2->Text = lvi2->Text->Substring(0, lvi2->Text->Length - 6);
					lvi2->SubItems[ELV_INFINITE]->Text = "0";
				}

				// Value
				lvi2->SubItems[ELV_VALUE]->Text = gcnew String(a);
				
				// Mutability
				lvi2->SubItems[ELV_MUTABLE]->Text = param.isMutable ? "Y" : "N";

				// Constraints
				sprintf(a, "%.*f", decpoints, param.consMin);
				lvi2->SubItems[ELV_CONSMIN]->Text = gcnew String(a);
				sprintf(a, "%.*f", decpoints, param.consMax);
				lvi2->SubItems[ELV_CONSMAX]->Text = gcnew String(a);
				lvi2->SubItems[ELV_CONS]->Text = param.isConstrained ? "Y" : "N";

				// Modifiers
				sprintf(a, "%.*f", decpoints, param.sigma);
				lvi2->SubItems[ELV_SIGMA]->Text = gcnew String(a);		
			}

			if(lvep->Items->Count > 0)
				UpdateExtraParamBox();	// TODO::SF FF ONLY
		}		
			
		FFParameterUpdateHandler();	// TODO::SF FF ONLY
	}

	void FormFactor::InitializeGraph(bool bZero, std::vector<double>& bgx, std::vector<double>& bgy,
									 std::vector<double>& ffy) {

		std::vector <double> x, y;
		ReadCLRFile(_dataFile, x, y);

		while(!x.empty() && x[0] < 1.0e-9) {
			x.erase(x.begin());
			y.erase(y.begin());
		}

		_data->x = x;
		_data->y = y;

		_baseline->x = bgx;
		_baseline->y = bgy;

		exportBackgroundToolStripMenuItem->Enabled = true;
		calculate->Enabled = true;

		if(listView_PeakPosition->Items->Count > 0)
			fitphase->Enabled = true;
		else
			fitphase->Enabled = false;
		
		smooth->Enabled = true;

		InitializeFitGraph(bZero ? x : bgx, bZero ? y : ffy);
	}

	void FormFactor::ExtraParameter_Enter(System::Object^ sender, System::EventArgs^  e) {
		if(listView_Extraparams->SelectedItems->Count == 0)
			listView_Extraparams->SelectedIndices->Add(paramBox->SelectedIndex);
	}

	void FormFactor::ExtraParameter_TextChanged(System::Object^ sender, System::EventArgs^  e) {
		// This function transforms a written extra parameter in a textbox (sender)
		// to the correct form (using ExtraParameter description)
		// and updates the corresponding listview
		TextBox ^textbox = (TextBox ^)sender;

		if(listView_Extraparams->SelectedItems->Count == 0)
			return;

		int exindex = listView_Extraparams->SelectedIndices[0];
		// This will define the extra parameter
		ExtraParam def = _modelFF->GetExtraParameter(exindex);

		double res;
		std::string str;
		char f[128] = {0};
		
		// Parse the value to a string
		clrToString(textbox->Text, str);
		res = strtod(str.c_str(), NULL);

		// Modify the value according to the definition

		// Range
		if(def.isRanged) {
			if(res < def.rangeMin)
				res = def.rangeMin;
			if(res > def.rangeMax)
				res = def.rangeMax;
		}

		// Absolute values
		if(def.isAbsolute && res < 0.0)
			res = -res;

		// Format the double as the modified value
		sprintf(f, "%.*f", def.decimalPoints, res);

					
		textbox->Text = gcnew String(f);
		
		if(sender == exParamGroupbox->text)
			listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[exParamGroupbox->rValue->Checked ? ELV_VALUE : ELV_SIGMA]->Text = exParamGroupbox->text->Text;
		if(sender == exmin)
			listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_CONSMIN]->Text = exmin->Text;
		if(sender == exmax)
			listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_CONSMAX]->Text = exmax->Text;

		FFParameterUpdateHandler();
		
		save->Enabled = true;
		AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
	}


	void FormFactor::Parameter_TextChanged(System::Object^  sender, System::EventArgs^  e) {
		double res;
		std::string str;
		char f[128] = {0};
		GroupBoxList ^gbl;
		ModelUI *mui = NULL;
		ListView ^lv;
		ParamGroupBox ^parnt = nullptr;
		parnt = dynamic_cast<ParamGroupBox^>(((TextBox^)sender)->Parent);
		if(parnt) {
			if(FFGroupBoxList->Contains(parnt)) {
				gbl = FFGroupBoxList;
				lv = listViewFF;
				mui = _modelFF;
			} else if(SFGroupBoxList->Contains(parnt)) {
				gbl = SFGroupBoxList;
				lv = listView_peaks;
				mui = _modelSF;
			} else if(BGGroupBoxList->Contains(parnt)) {
				gbl = BGGroupBoxList;
				lv = BGListview;
				mui = _modelBG;
			}
		} else
			return;


		if(_bChanging || _bLoading) return;

		if(lv->SelectedIndices->Count == 0) return;

		clrToString(((TextBox ^)(sender))->Text, str);

		res = fabs(strtod(str.c_str(), NULL));
		
		sprintf(f, "%.6f", res);		
		((TextBox ^)(sender))->Text = gcnew String(f);

		for (int i = 0; i < mui->GetNumLayerParams(); i++)
			if(sender == gbl[i]->text)
				for(int j = 0; j < lv->SelectedIndices->Count; j++)
					lv->SelectedItems[j]->SubItems[
						gbl[i]->rValue->Checked
						? LV_VALUE(i)
						: LV_SIGMA(i, mui->GetNumLayerParams())
						]->Text = gbl[i]->text->Text;	
			

		linkedParameterChangedCheck(lv->Items, lv->SelectedIndices[0]);

		if(mui == _modelFF) {
			FFParameterUpdateHandler();
			AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
		} else if(mui == _modelSF) {
			SFParameterUpdateHandler();
			AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);
		} else if(mui == _modelBG) {
			BGParameterUpdateHandler();
			AddToUndoQueue(MT_BACKGROUND, _curBGPar);
		}

		save->Enabled = true;
	}
	
	void FormFactor::PDRadioChanged(System::Object ^ sender, System::EventArgs^ e) {
		ParamGroupBox^ snd = nullptr;
		snd = dynamic_cast<ParamGroupBox^>(((RadioButton^)sender)->Parent);
		if(!snd)
			return;
		if(snd == exParamGroupbox) {
			listView_Extraparams_SelectedIndexChanged(sender, e);
			exParamGroupbox->text->Text = listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[exParamGroupbox->rValue->Checked ? ELV_VALUE : ELV_SIGMA]->Text;
		} else	// TODO::PD
			parameterListView_SelectedIndexChanged(listViewFF, e);
	}

	void FormFactor::double_TextChanged(System::Object^  sender, System::EventArgs^  e) {
		double res;
		std::string str;
		char f[128] = {0};
		
		clrToString(((TextBox ^)(sender))->Text, str);

		res = strtod(str.c_str(), NULL);
		
		sprintf(f, "%.6f", res);		
		((TextBox ^)(sender))->Text = gcnew String(f);
		
		FFParameterUpdateHandler();

		save->Enabled = true;
		AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
	}


	void FormFactor::AddParamLayer(std::vector<Parameter> layer) {
		AddParamLayer(layer, _modelFF, listViewFF);
	}

	void FormFactor::AddParamLayer(std::vector<Parameter> layer, ModelUI *mui, ListView ^lv) {
		ListViewItem ^lvi;
		char f[128] = {0};
		int currentLayer = lv->Items->Count;
		

		String ^param = gcnew String(mui->GetLayerName(
							currentLayer).c_str());

		lvi = gcnew ListViewItem(param);
	
		for(int i = 0; i < (int)layer.size(); i++) {
			if(!mui->IsParamApplicable(currentLayer, i)) {
				lvi->SubItems->Add("N/A");
				lvi->SubItems->Add("-");
			} else {		
				lvi->SubItems->Add(layer[i].value.ToString("0.000000"));
				lvi->SubItems->Add(layer[i].isMutable ? "Y" : "N");
			}
		}

		for(int i = 0; i < (int)layer.size(); i++) {
			lvi->SubItems->Add(layer[i].consMin.ToString("0.000000"));
			lvi->SubItems->Add(layer[i].consMax.ToString("0.000000"));
			lvi->SubItems->Add(layer[i].consMinIndex.ToString());
			lvi->SubItems->Add(layer[i].consMaxIndex.ToString());
			lvi->SubItems->Add(layer[i].linkIndex.ToString());
			lvi->SubItems->Add(layer[i].isConstrained ? "Y" : "N");
			lvi->SubItems->Add(layer[i].sigma.ToString("0.000000"));
		}
	
		lv->Items->Add(lvi);
		

		// Making sure we don't pass maximal layer count
		if(lv->Items->Count == mui->GetMaxLayers())
			addLayer->Enabled = false;

		if(mui == _modelFF) {
			FFParameterUpdateHandler();
			
		} else if(mui == _modelSF) {
			SFParameterUpdateHandler();			
		} else if(mui == _modelBG) {
			BGParameterUpdateHandler();			
		} else
			MessageBox::Show("mui not set right...");

		save->Enabled = true;		
	}

	void FormFactor::parameterListView_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		ModelUI *mui = NULL;
		ListView ^lv;
		Button ^rmv;
		GroupBoxList ^gbl;
		Label ^lb = nullptr;
		if(sender == listViewFF) {
			mui = _modelFF;
			lv = listViewFF;
			lb = paramLabel;
			gbl = FFGroupBoxList;
			rmv = removeLayer;
		}
		if(sender == listView_peaks) {
			mui = _modelSF;
			lv = listView_peaks;
			lb = sfParamLabel;
			gbl = SFGroupBoxList;		
			rmv = removePeak;
		}
		if(sender == BGListview) {
			mui = _modelBG;
			lv = BGListview;
			gbl = BGGroupBoxList;
			rmv = removeFuncButton;
		}

		int paramNum = mui->GetNumLayerParams();
		int minLayers = mui->GetMinLayers();

		if(lv->SelectedIndices->Count == 0) {
			// No Items
			if(lb)
				lb->Text = "<None>";
			removeLayer->Enabled = false;
			
			for (int i = 0; i < paramNum; i++){
				gbl[i]->Enabled = false;
				gbl[i]->track->Value = int((gbl[i]->track->Minimum + gbl[i]->track->Maximum) / 2.0);
			}	
				
			fitRange->Enabled = false;
			
		} else if(lv->SelectedIndices->Count > 1) {
			// Multiple Items
			if(lb)
				lb->Text = "<Multiple>";

			for (int i = 0; i < paramNum; i++){
				gbl[i]->Enabled = true;
				gbl[i]->text->Enabled = false;
				gbl[i]->track->Enabled = true;
				gbl[i]->check->Enabled = false;

				if(!_bGenerateModel)  
					gbl[i]->check->Enabled = true;				
			}		
				
			fitRange->Enabled = false;
			
			
			rmv->Enabled = true;
			for(int i = 0; i < lv->SelectedIndices->Count; i++)
				if(lv->SelectedIndices[i] < minLayers)
						rmv->Enabled = false;
				

			
			for(int i = 0; i < lv->SelectedItems->Count; i++) 
				for (int j = 0; j < paramNum ; j++){
					if(lv->SelectedItems[i]->SubItems[LV_MUTABLE(j)]->Text->Equals("-"))
						gbl[j]->check->Enabled = false;
					if(lv->SelectedItems[i]->SubItems[LV_VALUE(j)]->Text->Equals("N/A"))
						gbl[j]->track->Enabled = false;
			}

		} else {
			// One Item
			ListViewItem ^lvi = lv->SelectedItems[0];
		
			lb->Text = lvi->Text;
			
			if(lv->SelectedIndices[0] < minLayers)
				rmv->Enabled = false;
			else
				rmv->Enabled = true;
			
			for (int i = 0; i < paramNum; i++){
				gbl[i]->Enabled = true;
				gbl[i]->text->Enabled = true;
				gbl[i]->check->Enabled = false;
				gbl[i]->track->Enabled = true; 
				if(!_bGenerateModel)  
					gbl[i]->check->Enabled = true;
			}		

			if(constraints->Checked)
				fitRange->Enabled = true;


			for (int j = 0; j < paramNum ; j++){
				if(lvi->SubItems[LV_MUTABLE(j)]->Text->Equals("-"))
					gbl[j]->check->Enabled = false;
				if(lvi->SubItems[LV_VALUE(j)]->Text->Equals("N/A") || 
				   Int32::Parse(lvi->SubItems[LV_CONSLINK(j, paramNum)]->Text) > 0) {
					gbl[j]->track->Enabled = false;
					gbl[j]->text->Enabled = false;
				}
			}
			
			for (int i = 0; i < paramNum; i++){
				gbl[i]->text->Text = lvi->SubItems[
					gbl[i]->rValue->Checked
					? LV_VALUE(i)
					: LV_SIGMA(i, mui->GetNumLayerParams())
					]->Text;
				gbl[i]->check->Checked = (lvi->SubItems[LV_MUTABLE(i)]->Text->Equals("Y") ? true : false);
			}

			if(sender == listViewFF)
				EDU();
			if(sender == listView_peaks) {
				//highlight the peaks
				if(wgtFit && wgtFit->graph) {
					for(int i = 0; i < listView_peaks->SelectedIndices->Count; i++) {
						int srt = -1, nd = -1;
						double lft, rgt;
						lft = clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[5]->Text) - ((5.0 / ((double)peakfit->SelectedIndex + 2) * clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[3]->Text)));
						rgt = clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[5]->Text) + ((5.0 / ((double)peakfit->SelectedIndex + 2) * clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[3]->Text)));
						for (int hrt = 1; hrt < (int)wgtFit->graph->x->size(); hrt++) {
							if (srt < 0 && wgtFit->graph->x->at(hrt) > lft)
								srt = hrt - 1;
							if (wgtFit->graph->x->at(hrt) > rgt) {
								nd = hrt;
								break;
							}
						}
						wgtFit->graph->Highlight(1, srt, nd, RGB(0, 200, 0));
					}
					wgtFit->Invalidate();
				}

			}
			
		}
	}

	void FormFactor::listView_Extraparams_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if(listView_Extraparams->SelectedIndices->Count > 0 &&
			paramBox->SelectedIndex != listView_Extraparams->SelectedIndices[0])
			paramBox->SelectedIndex = listView_Extraparams->SelectedIndices[0];
	}

	void FormFactor::UpdateChisq(double chisq) {
		if(chisq < 0.0) return;
		wssr->Text = wssr->Text->Substring(0, wssr->Text->LastIndexOf('=') + 1);
		wssr->Text += chisq.ToString();
		_curWssr = chisq;
	}

	void FormFactor::UpdateRSquared(double rsq) {
		if(rsq < 0.0)
			rsq = 0.0;
		if(rsq > 1.0)
			rsq = 1.0;
		rsquared->Text = rsquared->Text->Substring(0, rsquared->Text->LastIndexOf('=') + 1);
		rsquared->Text += rsq.ToString("0.000000");
		_curRSquared = rsq;
	}

#pragma region Undo and Redo
	//////////////////////////////////////////////////////////////////////////
	// UNDO/REDO
	//////////////////////////////////////////////////////////////////////////
	void FormFactor::ParamStructToUI(ModelType type, const paramStruct& par) {
		switch(type) {
			default:
				return;

			case MT_FORMFACTOR:
				ParametersToUI(&par, _modelFF, listViewFF, listView_Extraparams);
				return;

			case MT_STRUCTUREFACTOR:
				ParametersToUI(&par, _modelSF, listView_peaks, nullptr);
				return;

			case MT_BACKGROUND:
				ParametersToUI(&par, _modelBG, BGListview, nullptr);
				return;
		}
	}

	void FormFactor::AddToUndoQueue(ModelType type, const paramStruct *par) {			
		if(!par)
			return;
		// NOTE: When undoQueueIndex is -1, it means that there is no undo/redo
		//       information (undoQueueIndex points to the current location in the queue,
		//       or size()-1 if we haven't undone anything)

		// Remove items from back of queue
		int curSize = (int)undoQueue->size();
		for(int i = undoQueueIndex + 1; i < curSize; i++)
			undoQueue->pop_back();

		// If queue is too full, remove an item from the front (oldest undo item)
		if(undoQueue->size() == UNDO_HISTORY_SIZE)
			undoQueue->pop_front();

		// Add the latest item to back of queue
		undoQueue->push_back(std::pair<ModelType, paramStruct>(type, *par));

		// Enable undo
		if(undoQueue->size() > 1)
			undo->Enabled = true;	

		// Disable redo
		undoQueueIndex = undoQueue->size() - 1;
		redo->Enabled = false;
	}

	void FormFactor::undo_Click(System::Object^  sender, System::EventArgs^  e) {
		// Invalid undo (should never happen)
		if(undoQueueIndex <= 0)
			return;

		if(undoQueueIndex == 1)
			undo->Enabled = false;

		undoQueueIndex--;

		// Set the parameters
		std::pair<ModelType, paramStruct>& queueItem = undoQueue->at(undoQueueIndex);
		ParamStructToUI(queueItem.first, queueItem.second);

		// Enable redo
		redo->Enabled = true;
	}

	void FormFactor::redo_Click(System::Object^  sender, System::EventArgs^  e) {
		// Invalid redo (should never happen)
		if(undoQueueIndex >= ((int)undoQueue->size() - 1))
			return;

		if(undoQueueIndex == ((int)undoQueue->size() - 2))
			redo->Enabled = false;

		undoQueueIndex++;

		// Set the parameters
		std::pair<ModelType, paramStruct>& queueItem = undoQueue->at(undoQueueIndex);
		ParamStructToUI(queueItem.first, queueItem.second);

		// Enable undo
		undo->Enabled = true;
	}
	//////////////////////////////////////////////////////////////////////////
	// END OF UNDO/REDO
	//////////////////////////////////////////////////////////////////////////
#pragma endregion
	
	//////////////////////////////////////////////////////////////////////////
	// The moment the calculate button is clicked, the following *should* happen:
	// If we are generating (Generate):
	// Code:
	//   1. Fitter window's state = GENERATING
	//   2. Determine what tab needs to be generated
	//   3. Prepare parameter structures from UI (UIToParameters)	
	//   4. Call the frontend's Generate() function
	// UI Events:
	//   1. Disable all relevant controls (EnableFittingControls(false))
	//   2. "Generate" --> "Stop" (and the button's functionality should change)
	//////////////////////////////////////////////////////////////////////////
	// If we are fitting (Fit):
	// Code:
	//   1. Fitter window's state = FITTING
	//   2. Determine what tabs are fit (using currentTab and the combobox)
	//   3. Check constraints in the UI (VerifyConstraints) (with possible highlighting of invalid parameters TODO::Later)
	//   4. Prepare parameter structures from UI (UIToParameters)	
	//   5. Call the frontend's Fit() function
	// UI Events:
	//   1. Disable all relevant controls (EnableFittingControls(false))
	//   2. "(Re)Calculate" --> "Stop" (and the button's functionality should change)
	//////////////////////////////////////////////////////////////////////////

	void FormFactor::calculate_Click(System::Object^  sender, System::EventArgs^  e) {
		if(calculate->Text->Equals("Stop")) {
			Stop();
			return;
		}

		// Initialize fitting graph panel
		if(!wgtFit)
			InitializeFitGraph();
		if(!_bGenerateModel && !wgtFit->Visible) {
			wgtFit->Visible = true;
			LocOnGraph->Visible = true;
			wssr->Visible = true;
			rsquared->Visible = true;
		}
		// End of graph panel

		bool result = false;
		if(_bGenerateModel)
			result = Generate();
		else
			result = Fit();

		if(result)
			calculate->Text = "Stop";
	}

	//////////////////////////////////////////////////////////////////////////
	// If the Stop button has been pressed:
	// 1. Call stop signal, disable Stop button
	// 2. Don't wait for the job to finish, it will call NotifyCompletion on its
	//    own (see below)
	//////////////////////////////////////////////////////////////////////////

	void FormFactor::Stop() {
		_lf->Stop(_job);
		
		calculate->Enabled = false;
	}

	ModelType FormFactor::DetermineFitType() {
		ModelType res;
		switch(tabControl1->SelectedIndex) {
			case 0:
				res = MT_FORMFACTOR;
				break;
			case 1:
				res = MT_STRUCTUREFACTOR;
				break;
			case 2:
				res = MT_BACKGROUND;
				break;
		}

		// Combined fitting
		switch(CalcComboBox->SelectedIndex) {
			default:
			case 0:
				break;

			case 1:
				res = MT_FFSF;
				break;
			case 2:
				res = MT_FFBG;
				break;
			case 3:
				res = MT_SFBG;
				break;
			case 4:
				res = MT_ALL;
				break;
		}

		return res;
	}

	bool FormFactor::Generate() {
		ErrorCode success = OK;

		// Determine tab to use in generation
		_fitType = DetermineFitType();
		
		
		// Set up generation parameters
		GenerateProperties gp;
		gp.bProgressReport	= true;
		gp.liveGenerate		= liveFittingToolStripMenuItem->Checked;
		gp.msUpdateInterval = 500; // TODO::updateInterval

		// Set up the X axis
		int res = int(clrToDouble(toolStripTextBox2->Text));
		std::vector<double> x (res, 0.0);

		double s = clrToDouble(startGen->Text), end = clrToDouble(endGen->Text);
		for(int i = 0; i < int(x.size()); i++)
			x[i] = s + (double(i + 1) * (end - s) / (double(res)));

		_ff->x = x;
		_sf->x = x;
		_bg->x = x;
		// End of X axis setup

		// Send the generate command to the backend
		success = _lf->Generate(_job, _fitType, 
								(_fitType & MT_FORMFACTOR ? _curFFPar : NULL), 
								(_fitType & MT_STRUCTUREFACTOR ? _curSFPar : NULL), 
								(_fitType & MT_BACKGROUND ? _curBGPar : NULL), 
								_ff->x, gp);
		
		if(success != OK) {
			int sucint = (int)success;
			MessageBox::Show("Generation failure: " + sucint);
			return false;
		}
		
		_state = FWS_GENERATING;

		// Disable relevant controls and show fitting/generation related controls
		EnableFittingControls(false);

		return true;
	}

	bool FormFactor::Fit() {
		ErrorCode success = OK;

		// Determine tab to use in generation
		_fitType = DetermineFitType();

		// Check constraints in the UI (with possible highlighting of invalid parameters)
		if(!VerifyConstraints(_fitType)) {
			MessageBox::Show("One or more of your constraints do not match the "
							 "input parameters. Correct the parameters and try again.", "Invalid Constraints", 
							 MessageBoxButtons::OK, MessageBoxIcon::Warning);

			return false;
		}

		// Set up fitting parameters
		FittingProperties fp;
		
		fp.bProgressReport	= true;
		fp.liveFitting = liveFittingToolStripMenuItem->Checked;
		fp.msUpdateInterval = 500; // TODO::updateInterval

		fp.accurateFitting	= accurateFittingToolStripMenuItem->Checked;
		fp.accurateDerivative	= accurateDerivativeToolStripMenuItem->Checked;
		fp.wssrFitting		= !(chiSquaredBasedFittingToolStripMenuItem->Checked) || (_curRSquared <= 1e-6);
		fp.logScaleFitting	= logScaledFittingParamToolStripMenuItem->Checked;

		fp.fitIterations	= int(clrToDouble(_bFitPhase ? phaseIterations->Text : fittingIter->Text));
		fp.method			= _bFitPhase ? GetPhaseFitMethod() : GetFitMethod();
		fp.resolution		= clrToDouble(ExperimentResToolStripMenuItem->Text);
		fp.minSignal		= clrToDouble(minimumSignalTextbox->Text);
		fp.usingGPUBackend	= false;

		// Set default mask
		if(_mask->size() != wgtFit->graph->x[0].size())
			_mask->resize(wgtFit->graph->x[0].size(), false);

		// Send the fit command to the backend
		success = _lf->Fit(_job, _fitType, 
						   (_fitType & MT_FORMFACTOR ? _curFFPar : NULL), 
						   (_fitType & MT_STRUCTUREFACTOR ? _curSFPar : NULL), 
						   (_fitType & MT_BACKGROUND ? _curBGPar : NULL),
						   wgtFit->graph->x[0], wgtFit->graph->y[0], *_mask, fp);
		if(success != OK) {
			int sucint = (int)success;
			MessageBox::Show("Fitting failure: " + sucint);
			return false;
		}
	
		_state = FWS_FITTING;

		// Disable relevant controls and show fitting/generation related controls
		EnableFittingControls(false);

		return true;
	}

	bool FormFactor::VerifyConstraints(ModelType type) {
		// TODO::NewFitter
		return true;
	}

	void FormFactor::EnableFittingControls(bool enable) {
		// This is the function responsible for the disabling
		// of all controls in the fitter window, and then
		// re-enabling them (if "enable" is true)

		// Clearing selected list items from the window first
		listViewFF->SelectedIndices->Clear();
		listView_PeakPosition->SelectedIndices->Clear();
		listView_peaks->SelectedIndices->Clear();
		BGListview->SelectedIndices->Clear();

		// Showing and re-setting the progress bar
		progressBar1->Visible = !enable;
		progressBar1->Value = progressBar1->Minimum;

		label6->Visible = !enable;

		this->Cursor = (enable ? System::Windows::Forms::Cursors::Default : 
								 System::Windows::Forms::Cursors::AppStarting);

		for (int i = 0; i < Controls->Count; i++) { 
			if(enable) {
				Controls[i]->Enabled = Controls[i]->AllowDrop;
			} else {
				Controls[i]->AllowDrop = Controls[i]->Enabled;
				Controls[i]->Enabled = false;
			}
		}		

		tableLayoutPanel1->Enabled = true;
		panel3->Enabled = true;
		
		if(enable)
			changeData->Enabled = changeData->AllowDrop;
		else {
			changeData->AllowDrop = changeData->Enabled;
			changeData->Enabled = false;
		}

		save->Enabled = enable;
		fitphase->Enabled = enable;
		changeModel->Enabled = enable;
		panel2->Enabled = enable;
		undo->Enabled = enable;
		maskButton->Enabled = enable;
	}

	//////////////////////////////////////////////////////////////////////////
	// Progress Report:
	//		UI Events:
	//			1. Update progress bar
	//		Code:
	//			1. If live generation/fitting is checked:
	//				1.1. Request the latest graph from frontend
	//				1.2. Display the new graph
	//////////////////////////////////////////////////////////////////////////


	void FormFactor::ProgressReport(void *args, double progress) {
		// Invoke from inside UI thread, if required
		if(this->InvokeRequired) {
			array<Object^> ^fparams = { IntPtr(args), progress };
			this->Invoke(progressrep, fparams);
			return;
		}

		// Update progress bar value
		progressBar1->Value = int(progress * progressBar1->Maximum);

		// If live generation/fitting is checked, retrieve the latest graph and display it
		if(liveFittingToolStripMenuItem->Checked) {
			// Update the graph
			int graphsz = _lf->GetGraphSize(_job);
			if(graphsz <= 0)
				MessageBox::Show("ERROR retrieving resulting graph (" + graphsz + ")");
			std::vector<double> resy (graphsz), resx = _ff->x;
			resx.resize(graphsz);

			if(_lf->GetGraph(_job, &resy[0], graphsz)) {
				_ff->y.resize(graphsz);
				_sf->y.resize(graphsz);
				_bg->y.resize(graphsz);
				_lf->GetGraphMoiety(_job, MT_FORMFACTOR, &(_ff->y[0]), graphsz);
				_lf->GetGraphMoiety(_job, MT_STRUCTUREFACTOR, &(_sf->y[0]), graphsz);
				_lf->GetGraphMoiety(_job, MT_BACKGROUND, &(_bg->y[0]), graphsz);
				wgtFit->graph->Modify(int(!_bGenerateModel), resx, resy);
				wgtFit->Invalidate();
			} else
				MessageBox::Show("ERROR retrieving resulting graph");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Notify Completion:
	//		UI Events:
	//			1. Revert disabling of relevant controls (EnableFittingControls(true))
	//			2. "Stop" -> "Recalculate"/"Generate"
	//		Code (HandleSuccessfulFit):
	//			1. Request the latest graph from frontend
	//			2. Display the new graph
	//			3. If fitting: Request the latest parameters, display them (ParametersToUI)
	//			4. If fitting: Update Chi^2 and R^2
	//////////////////////////////////////////////////////////////////////////

	void FormFactor::HandleSuccessfulFit() {
		// Update the graph
		int graphsz = _lf->GetGraphSize(_job);
		if(graphsz <= 0)
			MessageBox::Show("ERROR retrieving resulting graph (" + graphsz + ")");
		std::vector<double> resy (graphsz);
		if(_lf->GetGraph(_job, &resy[0], graphsz)) {
			_ff->y.resize(graphsz);
			_sf->y.resize(graphsz);
			_bg->y.resize(graphsz);
			_lf->GetGraphMoiety(_job, MT_FORMFACTOR, &(_ff->y[0]), graphsz);
			_lf->GetGraphMoiety(_job, MT_STRUCTUREFACTOR, &(_sf->y[0]), graphsz);
			_lf->GetGraphMoiety(_job, MT_BACKGROUND, &(_bg->y[0]), graphsz);
			wgtFit->graph->Modify(int(!_bGenerateModel), _ff->x, resy);
			wgtFit->Invalidate();
		} else
			MessageBox::Show("ERROR retrieving resulting graph");

		// Modify the parameters in the UI and update chi^2/R^2
		if(_state == FWS_FITTING) {			
			// Obtain parameters
			if(_fitType & MT_FORMFACTOR) {
				if(!_lf->GetResults(_job, MT_FORMFACTOR, *_curFFPar))
					MessageBox::Show("ERROR retrieving form factor fitting results");
				ParamStructToUI(MT_FORMFACTOR, *_curFFPar);
				AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
			}
			if(_fitType & MT_STRUCTUREFACTOR) {
				if(!_lf->GetResults(_job, MT_STRUCTUREFACTOR, *_curSFPar))
					MessageBox::Show("ERROR retrieving structure factor fitting results");
				ParamStructToUI(MT_STRUCTUREFACTOR, *_curSFPar);
				AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);
			}
			if(_fitType & MT_BACKGROUND) {
				if(!_lf->GetResults(_job, MT_BACKGROUND, *_curBGPar))
					MessageBox::Show("ERROR retrieving background fitting results");
				ParamStructToUI(MT_BACKGROUND, *_curBGPar);
				AddToUndoQueue(MT_BACKGROUND, _curBGPar);
			}
			// END of parameter modification

			// Update chi^2 and R^2
			_curWssr = WSSR(wgtFit->graph->y[0], wgtFit->graph->y[1]);
			UpdateChisq(_curWssr);
			UpdateRSquared(RSquared(wgtFit->graph->y[0], wgtFit->graph->y[1]));
		}

		// UI stuff
		progressBar1->Value = progressBar1->Maximum;
		Sleep(100); // Showing the full progress for some time
	}

	void FormFactor::NotifyCompletion(void *args, int error) {
		// Invoke from inside UI thread, if required
		if(this->InvokeRequired) {
			array<Object^> ^fparams = { IntPtr(args), error };
			this->Invoke(notifycomp, fparams);
			return;
		}

		if(error != OK && error != ERROR_STOPPED)
			MessageBox::Show("Error while fitting. Error code: " + error, "ERROR",
							 MessageBoxButtons::OK, MessageBoxIcon::Error);
		
		// If there was no error
		if(error == OK || error == ERROR_STOPPED) 
			HandleSuccessfulFit();

		// Revert calculate button's state
		calculate->Enabled = true;
		if(_bGenerateModel)
			calculate->Text = "Generate";
		else
			calculate->Text = "(Re)Calculate";

		// Revert disabling of relevant controls
		EnableFittingControls(true);		

		_state = FWS_IDLE;
	}

	void FormFactor::UpdateGraph() {
		UpdateGraph(false);
	}

	void FormFactor::UpdateGraph(bool calcAll) {
		// TODO::NewFitting
		// TODO::NewFitting
		// TODO::NewFitting

		if(_bLoading || !wgtFit)
			return;
		if(!wgtFit->graph)
			return;
		std::vector<double> mx, my;
		paramStruct par		= *_curFFPar;
		paramStruct peaks	= *_curSFPar;
		bgStruct background;
		ErrorCode err;

		UItoParameters(&peaks, _modelSF, listView_peaks, nullptr);

		GetBGFromGUI(&background);

		mx = wgtFit->graph->x[0];

		if(tabControl1->SelectedTab->Name == "SFTab" || calcAll) { //sender is the SF tab
			if(sfUseCheckBox->Checked) {
				GenerateProperties gp;
				gp.bProgressReport	= false;
				gp.liveGenerate		= false;

				_sf->y.clear(); //use existing form factor
				// TODO::NewFitting
				err = _lf->Generate(_job, MT_STRUCTUREFACTOR, NULL, &peaks, NULL, mx, gp);

				if(_peakPicker)  // We're looking only at the SF...
					my = _sf->y;
			}
			if(!_peakPicker) {
				MultiplyVectors(my, _ff->y, _sf->y);
				AddVectors(my, my, _bg->y);
			}
		}
		if(tabControl1->SelectedTab->Name == "FFTab" || calcAll) { // sender is in FF tab
			if(ffUseCheckBox->Checked) {
				GenerateProperties gp;
				gp.bProgressReport	= false;
				gp.liveGenerate		= false;

				_ff->y.clear();	//use existing structure factor

				// TODO::NewFitting
				_lf->Generate(_job, MT_FORMFACTOR, &par, NULL, NULL,  mx, gp);
			}
			MultiplyVectors(my, _ff->y, _sf->y);
			AddVectors(my, my, _bg->y);
		}
		if(tabControl1->SelectedTab->Name == "BGTab" || calcAll) { //sender is in the BG tab
			if(bgUseCheckBox->Checked) {
				_bg->y.clear();
				//GenerateBackground(mx, _bg->y, &background); // TODO::BG

				if(_fitToBaseline)
					my = _bg->y;
			}
			
			if(!_fitToBaseline) {
				MultiplyVectors(my, _ff->y, _sf->y);
				AddVectors(my, my, _bg->y);
			}
		}

		_lf->WaitForFinish(_job);
		int numPoints = _lf->GetGraphSize(_job);
		my = MachineResolutionF(mx, my, GetResolution());
		my.resize(numPoints);
		_ff->y.resize(numPoints);
		_sf->y.resize(numPoints);
		_bg->y.resize(numPoints);
		_lf->GetGraph(_job, &my[0], numPoints);
		_lf->GetGraphMoiety(_job, MT_FORMFACTOR, &(_ff->y[0]), numPoints);
		_lf->GetGraphMoiety(_job, MT_STRUCTUREFACTOR, &(_sf->y[0]), numPoints);
		_lf->GetGraphMoiety(_job, MT_BACKGROUND, &(_bg->y[0]), numPoints);

		if(!_bGenerateModel) {
			UpdateChisq(WSSR(wgtFit->graph->y[0], my));
			UpdateRSquared(RSquared(wgtFit->graph->y[0], my));
		}

		wgtFit->graph->ToggleYTicks();
		wgtFit->graph->ToggleYTicks();
		if(my.size() == mx.size())
			wgtFit->graph->Modify(_bGenerateModel ? 0 : 1, mx, my);
		wgtFit->Invalidate();
	}

	void FormFactor::zeroBG_Click(System::Object^  sender, System::EventArgs^  e) {
		if(_peakPicker) PeakPicker_Click(sender, e);
		if(_fitToBaseline) fitToBaseline_Click(sender, e);
		label1->Visible = false;
		exportSignalToolStripMenuItem->Enabled = true;
		exportModelToolStripMenuItem1->Enabled = true;
		plotFittingResultsToolStripMenuItem->Enabled = true;
		exportGraphToolStripMenuItem->Enabled = true;
		exportFormFactorToolStripMenuItem->Enabled = true;
		exportSigModBLToolStripMenuItem->Enabled = true;
		exportDecomposedToolStripMenuItem->Enabled = true;
		exportStructureFactorToolStripMenuItem->Enabled = true;
		SetMinimumSig(0.0);
		minimumSignalTextbox->Text = GetMinimumSig().ToString();
		
		fitToBaseline->Enabled = false;

		ffUseCheckBox->Checked = true;
		sfUseCheckBox->Checked = true;
		bgUseCheckBox->Checked = true;

		_ff->tmpY.clear();
		_sf->tmpY.clear();
		_bg->tmpY.clear();

		std::vector <double> x, y, ffy; 
		InitializeGraph(true, x, y, ffy);

		ffUseCheckBox->Enabled = true;
		sfUseCheckBox->Enabled = true;
		bgUseCheckBox->Enabled = true;

		tabControl1->SelectTab("FFTab");
	}
	
	FitMethod FormFactor::GetFitMethod() {
		if(levmar->Checked)
			return FIT_LM;
		if(diffEvo->Checked)
			return FIT_DE;
		if(raindrop->Checked)
			return FIT_RAINDROP;

		return FIT_LM;
	}

	FitMethod FormFactor::GetPhaseFitMethod() {
		if(pLevmar->Checked)
			return FIT_LM;
		if(pDiffEvo->Checked)
			return FIT_DE;
		if(pRaindrop->Checked)
			return FIT_RAINDROP;

		return FIT_LM;
	}
	bool RequalsZero(String ^s) {
		//R² = ERROR
		wchar_t *w = new wchar_t;
		//w[0] = 'R'; ; w[1] ='²'; w[2]=' '; w[3]='=';
		//= ("R² =");
		w = L"R² =";
		for (int i=0; i < 4; i++)
			s = (s->TrimStart(w[i]));
		double a = clrToDouble(s);
		return abs(a)<1e-9;		
	}

	void FormFactor::save_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring filename;
		std::string type = _modelFF->GetName();

		if(_bGenerateModel) {
			filename = L".\\XModelFitter.ini";
		} else {
			std::wstring res;

			res = clrToWstring(CLRDirectory(_dataFile)) + 
				clrToWstring(CLRBasename(_dataFile)) + L"-params.ini";

			filename = res;
		}

		if (sender == this->saveParametersAsToolStripMenuItem) {
			sfd->FileName = "";
			sfd->Title = "Save parameter file as... ";
			sfd->Filter = "Parameter Files (*.ini)|*.ini|All Files (*.*)|*.*";
			if(sfd->ShowDialog() == 
				System::Windows::Forms::DialogResult::Cancel)
				return;
			clrToString(sfd->FileName, filename);
		}
		
		if(sender == exportSigModBLToolStripMenuItem)
			filename = clrToWstring(CLRDirectory(_dataFile)) + clrToWstring(CLRBasename(_dataFile)) + L"-auto-params.ini";



		paramStruct p = *_curFFPar;

		iniFile = NewIniFile();

		// Write parameters		
		WriteParameters(filename, type, &p, *_modelFF, iniFile);
		
		// Write ED Profile configuration
		SetIniInt(filename, type, "EDProfileShape", iniFile, _modelFF->GetEDProfile().shape);
		if(adaptiveToolStripMenuItem->Checked)
			SetIniInt(filename, type, "EDProfileResolution", iniFile, -Int32::Parse(edpResolution->Text));
		else
			SetIniInt(filename, type, "EDProfileResolution", iniFile, Int32::Parse(edpResolution->Text));

		// Write Polydispersity configuration
		//SetIniInt(filename, type, "PDFunc", iniFile, GetPDFunc());		// TODO::INI
//		SetIniInt(filename, type, "PDResolution", iniFile, GetPDResolution());	// TODO::INI

		// Write quadrature configuration
		if(integrationToolStripMenuItem->Visible) {
			SetIniString(filename, type, "QuadratureRes", iniFile, clrToString(toolStripTextBox1->Text));
			if(monteCarloToolStripMenuItem->Checked)
				SetIniString(filename, type, "QuadratureMethod", iniFile, "1");
			else if(simpsonsRuleToolStripMenuItem->Checked)
				SetIniString(filename, type, "QuadratureMethod", iniFile, "2");
			else // if(gaussLegendreToolStripMenuItem->Checked)
				SetIniString(filename, type, "QuadratureMethod", iniFile, "0");
		}

		if(_bGenerateModel) {
			SetIniString(filename, type, "GenRangeStart", iniFile, clrToString(startGen->Text));
			SetIniString(filename, type, "GenRangeEnd", iniFile, clrToString(endGen->Text));
			SetIniString(filename, type, "GenResolution", iniFile, clrToString(toolStripTextBox2->Text));
		}

		// Saving Structure Factor
		peakStruct peaks;
		UItoParameters(_curSFPar, _modelSF, listView_peaks, nullptr);
		WriteParameters(filename, type, _curSFPar, *_modelSF, iniFile);
/*
		WritePeaks(filename, type, _curSFPar, iniFile);
		if(_modelFF->HasSpecializedSF()) {
			graphTable caille;
			cailleParamStruct cailleP;
			if(cailleParamListView->Items->Count > 0) {
				GetCailleFromGUI(&caille, &cailleP);
				WriteCaille(filename, type, &caille, &cailleP, iniFile);
			}
		}
*/
		// Save phases
		//phaseStruct ps;
		//GetPhasesFromListView(&ps);
		//WritePhases(filename, type, &ps, order->SelectedIndex, iniFile);	// TODO::Phases / TODO::INI

		// Saving background
		bgStruct BGs;
		//GetBGFromGUI(&BGs);
		//WriteBG(filename, type, &BGs, iniFile);	// TODO::BG / TODO::INI

		// Saving General Settings
		SetIniChar(filename, "Settings", "LogScale", iniFile, (logScale->Checked ? 'Y' : 'N'));
		SetIniChar(filename, "Settings", "LogFitting", iniFile, (logScaledFittingParamToolStripMenuItem->Checked ? 'Y' : 'N'));
		SetIniChar(filename, "Settings", "LiveRefresh", iniFile, (liveRefreshToolStripMenuItem->Checked ? 'Y' : 'N'));
		SetIniChar(filename, "Settings", "LiveFit", iniFile, (liveFittingToolStripMenuItem->Checked ? 'Y' : 'N'));
		SetIniChar(filename, "Settings", "sigma", iniFile, (sigmaToolStripMenuItem->Checked ? 'Y' : 'N'));
		SetIniString(filename, "Settings", "MinSignal", iniFile, clrToString(minimumSignalTextbox->Text));

		
		SaveAndCloseIniFile(filename, "Settings", iniFile);
		iniFile = NULL;

		save->Enabled = false;
		_bSaved = true;
	}

	void FormFactor::plotElectronDensityProfileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		struct graphLine graphs[3];
		paramStruct p = *_curFFPar;

		generateEDProfile(p.params, graphs,_modelFF->GetEDProfile());
		std::pair<double, double> in = calcEDIntegral(p.params[0], p.params[0]);
		
		ResultsWindow rw(graphs, 3, "Positive one-sided area: " + Double(in.first).ToString("#.######") + 
			", Negative area: " + Double(in.second).ToString("#.######"));

		rw.ShowDialog();
	}

	void FormFactor::importParametersToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		bool bValidFile = false;
		ofd->Title = "Choose a parameter file to import";
		ofd->Filter = "Parameter Files (*.ini)|*.ini|All Files (*.*)|*.*";
		if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		paramStruct par(_modelFF->GetModelInformation());

		iniFile = NewIniFile();

		bValidFile = ReadParameters(clrToWstring(ofd->FileName), _modelFF->GetName(), &par, *_modelFF, iniFile);
		if(!bValidFile) {
			for(int i = 0; i < _modelFF->GetNumRelatedModels(); i++) {
				ModelUI tmpUI; wchar_t tmpDes[] = L"MI Query";
				ModelInformation tmpMI = _lf->QueryModel(clrToWstring(_mioFF->contName).c_str(), _modelFF->GetRelatedModelIndex(i));

				JobPtr tmpJob = _lf->CreateJob(tmpDes);
				if(_lf->SetModel(tmpJob, MT_FORMFACTOR /*TODO::implementAll make this a variable?*/, clrToWstring(_mioFF->contName).c_str(), i, EDProfile()))
					continue;
				tmpUI.setModel(_lf, clrToWstring(_mioFF->contName).c_str(), tmpMI, 5);
				bValidFile = ReadParameters(clrToWstring(ofd->FileName), _modelFF->GetRelatedModelName(i), &par, tmpUI, iniFile);
				if(bValidFile) {
					// Change model to first available one
					//FFModel *ffm = dynamic_cast<FFModel *>(_modelFF->CreateRelatedModel(i));
					*_miFF = tmpMI;
					handleModelChange(tmpUI);
					break;
				}
			}
		}

		if(!bValidFile) {
			MessageBox::Show("No model of this sort in this parameter file",
							 "No Such Model",
							 MessageBoxButtons::OK,
							 MessageBoxIcon::Error);
			CloseIniFile(iniFile);
			iniFile = NULL;
			return;
		}

		System::String ^filename;
		if(_bGenerateModel) {
			filename = ".\\XModelFitter.ini";
		} else {
			filename = CLRDirectory(_dataFile) + 
				CLRBasename(_dataFile) + "-params.ini";
	 	}
		
		if(!ofd->FileName->Equals(filename))
			if(!_bGenerateModel || !CLRBasename(ofd->FileName)->Equals(CLRBasename(filename)))
				System::IO::File::Copy(ofd->FileName, filename, true);

		// TODO::EDP Think of ED profile type here
		
		// Remove Layers
		while(listViewFF->Items->Count > 1)
			listViewFF->Items->RemoveAt(1);

		listViewFF->Items->Clear();

		// Remove Peaks
		listView_peaks->Items->Clear();

		// Remove BG Functions
		BGListview->Items->Clear();

		// Remove Extra Parameters
		listView_Extraparams->Items->Clear();
		paramBox->Items->Clear();

		// Remove Phase listView
		listView_phases->Items->Clear();

		delete wgtPreview;
		wgtPreview = nullptr;
		FormFactor_Load(sender, e);

		CloseIniFile(iniFile);
		iniFile = NULL;

		paramStruct p = *_curFFPar;

		UpdateGraph(true);	//Avi: Already called in FF_Load, but not "true"
	}

	void FormFactor::importBaselineToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring fname;
		if(!openDataFile(ofd, "Choose a baseline file to import", fname, true))
			return;

		std::vector <double> x, y, ffy;
		// Copy externalfile to <data>-baseline.out
		System::String ^filename, ^dir = CLRDirectory(_dataFile);

		filename = dir + CLRBasename(_dataFile) + "-baseline.out";

		if(!filename->Equals(ofd->FileName))
			System::IO::File::Copy(ofd->FileName, filename, true);

		GenerateBGLinesandFormFactor(clrToWstring(_dataFile).c_str(), clrToWstring(filename).c_str(), x, y, ffy,false);	// TODO::DUMMY
		label1->Visible = false;
		exportSignalToolStripMenuItem->Enabled = true;
		exportModelToolStripMenuItem1->Enabled = true;
		plotFittingResultsToolStripMenuItem->Enabled = true;
		exportGraphToolStripMenuItem->Enabled = true;
		exportFormFactorToolStripMenuItem->Enabled = true;
		exportSigModBLToolStripMenuItem->Enabled = true;
		exportDecomposedToolStripMenuItem->Enabled = true;
		exportStructureFactorToolStripMenuItem->Enabled = true;

		// Don't ask if we want to use the existing baseline
		ExtractBaseline::DontAsk(); 

 	    InitializeGraph(false, x, y, ffy);
		
		ffUseCheckBox->Enabled = true;
		sfUseCheckBox->Enabled = true;
		bgUseCheckBox->Enabled = true;
	}


	void FormFactor::RenderPreviewScene() {

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
		glEnable(GL_DEPTH_TEST);							

		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
		glLoadIdentity();

		glTranslatef(0.0f,0.0f,-5.0f);
		
		if(renderScene)
			renderScene(*_curFFPar, _modelFF->GetEDProfile(), false);

		glDisable(GL_SMOOTH);

		glPopMatrix();
		
	}

	void FormFactor::exportDataFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			std::wstring file;

			if(!_bGenerateModel) {
				sfd->FileName = CLRBasename(_dataFile);
				sfd->FileName = sfd->FileName->Substring(1, sfd->FileName->Length - 1);
			}
			sfd->Filter = L"Output Files (*.out)|*.out|Data Files (*.dat, *.chi)|*.dat;*.chi|All files|*.*";
			if(sender == this->exportSignalToolStripMenuItem || sender == this->baseline) sfd->FileName += "-sig";
			else if(sender == this->exportBackgroundToolStripMenuItem) sfd->FileName += "_BG";
			else if(sender == this->exportModelToolStripMenuItem1) sfd->FileName += "_modelFF";
			else sfd->FileName = "";

			if(sender == this->exportSigModBLToolStripMenuItem) {
				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-sigA.dat", file);
				WriteDataFile(file.c_str(), _data->x, _data->y);

				clrToString(sfd->FileName + "_BGA.dat", file);
				if(fitToBaseline->Enabled && fitToBaseline->Text->StartsWith("Fit Back")) // Make sure there is a baseline
					WriteDataFile(file.c_str(), _baseline->x, _baseline->y);

				clrToString(sfd->FileName + "_modelA.dat", file);
				WriteDataFile(file.c_str(), wgtFit->graph->x[1], wgtFit->graph->y[1]);

				save_Click(sender, e);
				return;
			}

			if(sender == exportDecomposedToolStripMenuItem) {
				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-FF.dat", file);
				WriteDataFile(file.c_str(), _ff->x, ffUseCheckBox->Checked ? _ff->y : _ff->tmpY);

				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-SF.dat", file);
				WriteDataFile(file.c_str(), _sf->x, sfUseCheckBox->Checked ? _sf->y : _sf->tmpY);

				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-BG.dat", file);
				WriteDataFile(file.c_str(), _bg->x, bgUseCheckBox->Checked ? _bg->y : _bg->tmpY);

				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-model.dat", file);
				std::vector<double> model;
				MultiplyVectors(model, ffUseCheckBox->Checked ? _ff->y : _ff->tmpY, sfUseCheckBox->Checked ? _sf->y : _sf->tmpY);
				AddVectors(model, model, bgUseCheckBox->Checked ? _bg->y : _bg->tmpY);
				WriteDataFile(file.c_str(), _ff->x, MachineResolutionF(_ff->x, model, GetResolution()));

				sfd->FileName = CLRDirectory(_dataFile) + CLRBasename(_dataFile);
				clrToString(sfd->FileName + "-signal.dat", file);
				WriteDataFile(file.c_str(), wgtFit->graph->x[0], wgtFit->graph->y[0]);

				return;
			}

			sfd->Title = "Choose a signal output file";
			if(sfd->ShowDialog() == 
				System::Windows::Forms::DialogResult::Cancel)
				return;

			clrToString(sfd->FileName, file);

			if(sender == this->exportFormFactorToolStripMenuItem) {			
				WriteDataFile(file.c_str(), _ff->x, ffUseCheckBox->Checked ? _ff->y : _ff->tmpY);
				return;
			}

			if(sender == this->exportStructureFactorToolStripMenuItem) {
				WriteDataFile(file.c_str(), _sf->x, sfUseCheckBox->Checked ? _sf->y : _sf->tmpY);
				return;
			}

			if(sender == this->exportBackgroundToolStripMenuItem) {
				WriteDataFile(file.c_str(), _bg->x, (bgUseCheckBox->Checked) ? _bg->y : _bg->tmpY);
				return;
			}

			WriteDataFile(file.c_str(), wgtFit->graph->x[0], wgtFit->graph->y[0]);
		 }

	void FormFactor::exportModelToolStripMenuItem1_Click(System::Object^  sender, System::EventArgs^  e) {
			std::wstring file;
			std::vector<double> err(_ff->x.size());
			sfd->FileName = "";
			sfd->Title = "Choose a model output file";
			sfd->Filter = L"Output Files (*.out)|*.out|Data Files (*.dat, *.chi)|*.dat;*.chi|All files|*.*";
			if(sfd->ShowDialog() == 
				System::Windows::Forms::DialogResult::Cancel)
				return;

			clrToString(sfd->FileName, file);
			int pos = (_bGenerateModel ? 0 : 1);
			
			// Calculate Error Vector
			for(int i = 0; i < (int)err.size(); i++) {
				double dff = ((int)FFmodelErrors->size() > i) ? FFmodelErrors->at(i) : 0.0;
				double dsf = ((int)SFmodelErrors->size() > i) ? SFmodelErrors->at(i) : 0.0;
				double dbg = ((int)BGmodelErrors->size() > i) ? BGmodelErrors->at(i) : 0.0;

				err.at(i) = sqrt(sq(dff*_sf->y.at(i)) + sq(dsf*_ff->y.at(i))  + sq(dbg) );
			}

			if(_peakPicker){
				vector<double> res(_ff->y.size(), 0.0);
				MultiplyVectors(res, _ff->y, _sf->y);
				AddVectors(res, res, _bg->y);
				Write3ColDataFile(file.c_str(), wgtFit->graph->x[pos], MachineResolutionF(_ff->x, res, GetResolution()), err);
			} else
				Write3ColDataFile(file.c_str(), wgtFit->graph->x[pos], MachineResolutionF(_ff->x, wgtFit->graph->y[pos], GetResolution()), err);
		 }

	void FormFactor::changeData_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring fname;
		if(!openDataFile(ofd, "Choose a data file", fname, false))
			return;

		if(_fitToBaseline) fitToBaseline_Click(sender, e);
		_dataFile = ofd->FileName;

		if(wgtFit && wgtFit->graph)
			delete wgtFit->graph;
		delete wgtFit;
		wgtFit = nullptr;
		graphType->clear();

		this->Text = this->Text->Substring(0, this->Text->IndexOf('['));

		this->Text += "[" + CLRBasename(_dataFile) + "]";

		calculate->Enabled = false;
		fitphase->Enabled = false;
		smooth->Enabled = false;
		exportBackgroundToolStripMenuItem->Enabled = false;
		label1->Visible = true;
		tabControl1->SelectTab("BGTab");

		SFParameterUpdateHandler();
		BGParameterUpdateHandler();
		FFParameterUpdateHandler();
	}

	void FormFactor::FormFactor_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		// Save parameters
		if((e->KeyCode == Keys::S) && e->Control && save->Enabled) {
			if(e->Shift)
				this->save_Click(saveParametersAsToolStripMenuItem, gcnew EventArgs());
			else
				this->save_Click(saveParametersToolStripMenuItem, gcnew EventArgs());
			
			e->Handled = true;
		}

	}

	void FormFactor::listViewFF_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		ListView ^lv = (ListView ^)sender;
		//If the user selected indices and pressed delete/backspace, remove indicies
		if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back) {
			while(lv->SelectedItems->Count > 0) {
				// Check to make sure the layer is removable
				for(int i = 0; i < listViewFF->SelectedIndices->Count; i++)
					if(listViewFF->SelectedIndices[i] < _modelFF->GetMinLayers())
						return;
				removeLayer_Click(this, e);
			}

			save->Enabled = true;
			e->Handled = true;
		}

		// Copy selected listViewItems
		if((e->KeyCode == Keys::C) && (System::Windows::Forms::Control::ModifierKeys == Keys::Control)) {
			if(lv->SelectedItems->Count > 0)
				_copiedIndicesFF->clear();

			for(int i = 0; i < lv->SelectedItems->Count; i++)
				_copiedIndicesFF->push_back(lv->SelectedIndices[i]);
			e->Handled = true;
		}

		// Paste selected items (default is linked to the original items
		if((e->KeyCode == Keys::V) && (System::Windows::Forms::Control::ModifierKeys == Keys::Control)) {
			int nlp = _modelFF->GetNumLayerParams();
			for(int i = 0; (i < (int)_copiedIndicesFF->size()) && 
					(listViewFF->Items->Count < (_modelFF->GetMaxLayers() > 0 ? _modelFF->GetMaxLayers() : listViewFF->Items->Count + 50));
					i++) {
				// Add the item
				listViewFF->Items->Add((ListViewItem^)(listViewFF->Items[_copiedIndicesFF->at(i)]->Clone()));
				int cnt = listViewFF->Items->Count - 1;
				int ind = listViewFF->Items[cnt]->SubItems[LV_NAME]->Text->LastIndexOf(" ");
				System::String ^str = listViewFF->Items[cnt]->SubItems[LV_NAME]->Text->Remove(ind + 1);
				listViewFF->Items[cnt]->SubItems[LV_NAME]->Text = str->Insert(ind + 1, cnt.ToString());

				// Link all the parameters to the _copiedIndicesFF[i]th layer
				for(int n = 0; n < nlp; n++) {
					listViewFF->Items[cnt]->SubItems[LV_CONSLINK(n, nlp)]->Text = _copiedIndicesFF->at(i).ToString();
					listViewFF->Items[cnt]->SubItems[LV_MUTABLE(n)]->Text = "L";
				}
			}

			FFParameterUpdateHandler();
			save->Enabled = true;
			AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
			e->Handled = true;
		}
	}

	void FormFactor::EDU() {
		if(_modelFF->GetEDProfile().type == NONE)
			return;
		double solvent = clrToDouble(listViewFF->Items[0]->SubItems[LV_VALUE(1)]->Text);
		double area = 0.0;

		// Computing total ED profile area
		if(false/*TODO::EDP _modelFF->GetEDProfileFunction()*/) {;
/*
			_modelFF->GetEDProfileFunction()->MakeSteps(0.0, 0.0, 0.0, area);
			area -= (_modelFF->GetEDProfileFunction()->GetUpperLimit() - 
			         _modelFF->GetEDProfileFunction()->GetLowerLimit()) * solvent;
*/
		} else { 
			// Discrete layers, compute manually			
			for(int i = 0; i < listViewFF->Items->Count; i++) {
				double layerWidth, layerHeight;
				layerWidth  = clrToDouble(listViewFF->Items[i]->SubItems[LV_VALUE(0)]->Text);
				layerHeight = clrToDouble(listViewFF->Items[i]->SubItems[LV_VALUE(1)]->Text);

				area += layerWidth * (layerHeight - solvent);
			}				
		}
		if(_modelFF->GetEDProfile().type == SYMMETRIC)
			area *= 2.0;

		AreaText->Text = area.ToString("0.000");		
	}

	void FormFactor::plotGenerateResults() {
		if(!wgtFit || !wgtFit->graph || wgtFit->graph->numGraphs == 0)
			return;

		struct graphLine graphs[1];
		graphs[0].color = RGB(255, 0, 0);

		graphs[0].legendKey = "Model";

		graphs[0].x = wgtFit->graph->x[0];
		graphs[0].y = wgtFit->graph->y[0];

		ResultsWindow rw(graphs, 1);

		rw.ShowDialog();
	}

	void FormFactor::plotFittingResults() {
		if(!wgtFit || !wgtFit->graph || wgtFit->graph->numGraphs == 0)
			return;

		struct graphLine graphs[2];
		graphs[0].color = RGB(255, 0, 0);
		graphs[1].color = RGB(0, 0, 255);

		graphs[0].legendKey = "Signal";
		graphs[1].legendKey = "Model";

		graphs[0].x = wgtFit->graph->x[0];
		graphs[0].y = wgtFit->graph->y[0];

		graphs[1].x = wgtFit->graph->x[1];
		graphs[1].y = wgtFit->graph->y[1];
		
		ResultsWindow rw(graphs, 2);

		rw.ShowDialog();
	}

	void FormFactor::automaticPeakFinderButton_Click(System::Object^  sender, System::EventArgs^  e) {
			 /**
			  * Collect the two thresholds from the fields
			  *	Send the signal (data), FormFactor and StructureFactor vectors along with
			  *	the thresholds to an automatic FindPeaks function.
			 **/
			 if(!wgtFit || !wgtFit->graph) return;
			 threshold1 = clrToDouble(thresholdBox1->Text);
			 threshold2 = clrToDouble(thresholdBox2->Text);
			 
			 graphTable *signal;
			 signal = new graphTable;
			 signal->x = wgtFit->graph->x[0];
			 signal->y = wgtFit->graph->y[0];
			 AutoFindPeaks();
			 delete signal;
		 }

	void FormFactor::exportGraphToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if(_bGenerateModel)
			plotGenerateResults();
		else
			plotFittingResults();
	}

	void FormFactor::minimumSignalTextbox_Leave(System::Object^  sender, System::EventArgs^  e) {
		double prevMin = GetMinimumSig(), newMin;
		newMin = clrToDouble(minimumSignalTextbox->Text);
		
		if((newMin == 0.0 && !minimumSignalTextbox->Text->StartsWith("0"))	//Starts with text
			|| fabs(newMin - prevMin) < 1e-12) {	// The number hasn't changed
			minimumSignalTextbox->Text = prevMin.ToString("0.000000");
			return;
		}
		
		if(!wgtFit || _fitToBaseline) {
			SetMinimumSig(newMin);
			minimumSignalTextbox->Text = newMin.ToString();
			return;
		}

		// As we don't change graph->x/y directly (it's computed inside 
		// graphtoolkit), we have to call the Modify method
		std::vector<double> newy, newx = wgtFit->graph->x[0];
		for (int i = 0; i < (int)wgtFit->graph->y[0].size(); i++)
			newy.push_back(wgtFit->graph->y[0].at(i) - prevMin + newMin);

		wgtFit->graph->Modify(0, newx, newy);

		SetMinimumSig(newMin);
		minimumSignalTextbox->Text = newMin.ToString("0.000000");
		
		if(liveRefreshToolStripMenuItem->Checked) {
			RedrawGraph();
			wgtFit->graph->FitToAllGraphs();
		}
	}

	void FormFactor::minimumSignalTextbox_Enter(System::Object^  sender, System::EventArgs^  e) {
		minimumSignalTextbox->Text = GetMinimumSig().ToString("0.000000");
	}

	void FormFactor::addValueToVector(std::vector<double> &vec, double val) {
		for(int i = 0; i < (int)vec.size(); i++)
			vec.at(i) += val;
	}

	void FormFactor::AddParamLayer() {
		AddParamLayer(_modelFF);
	}
	void FormFactor::AddParamLayer(ModelUI *mui) {
		int nlp = mui->GetNumLayerParams();
		ListView ^lv = nullptr;
		if(mui == _modelFF) {
			lv = listViewFF;
		} else if(mui == _modelSF) {
			lv = listView_peaks;
		} else if(mui == _modelBG) {
			lv = BGListview;
		}
		std::vector<Parameter> layer (nlp);
		for(int i = 0; i < nlp; i++)
			layer[i].value = mui->GetDefaultParamValue(i, 
									lv->Items->Count);
		
		AddParamLayer(layer, mui, lv);
	}

	void FormFactor::smooth_Click(System::Object^  sender, System::EventArgs^  e) {
		if(!wgtFit || !wgtFit->graph)
			return;

		// Write temporary file with current signal
		String ^tempFile = System::IO::Path::GetTempFileName();
		std::wstring tempStr = clrToWstring(tempFile);

		WriteDataFile(tempStr.c_str(), wgtFit->graph->x[0],
			wgtFit->graph->y[0]);

		// Show smoothing dialog
		SmoothWindow sw (tempStr.c_str(), true);
		sw.ShowDialog();

		// Read back the results
		std::vector<double> resx, resy;
		ReadDataFile(tempStr.c_str(), resx, resy);

		// Modify the graphs
		wgtFit->graph->Modify(0, resx, resy);

		wgtFit->Invalidate();

		// Delete the temporary file
		System::IO::File::Delete(tempFile);
	}

	void FormFactor::WriteCSVParamsFile() {
		// Get the name/dir of the file (+ dialog)
		std::wstring file;
		if(!_bGenerateModel) {
			sfd->FileName = CLRBasename(_dataFile)->Remove(0, 1);
			sfd->FileName += "_parameters";
		}

		sfd->Filter = "TSV Files (*.tsv)|*.tsv|All Files (*.*)|*.*";
		sfd->Title = "Choose a filename";
		if(sfd->ShowDialog() == 
			System::Windows::Forms::DialogResult::Cancel)
			return;
		
		clrToString(sfd->FileName, file);


		// Open a file for writing
		FILE *fp;

		if ((fp = _wfopen(file.c_str(), L"w, ccs=UTF-8")) == NULL) {
			fprintf(stderr, "Error opening file %s for writing\n",
							file);
			
			MessageBox::Show("Please make sure that the file is not open.", "Error opening file for writing", MessageBoxButtons::OK,
									 MessageBoxIcon::Error);
			return;
		}

		// Write each tab with its parameters and titles (+chisqr/Rsqr)
		// Collect titles (header field names)
		System::Collections::Generic::List<ListView^>^ LV = gcnew System::Collections::Generic::List<ListView^>();
		LV->Add(listViewFF);
		LV->Add(listView_Extraparams);
		LV->Add(listView_peaks);
		LV->Add(listView_PeakPosition);
		LV->Add(listView_phases);
		LV->Add(BGListview);

		for(int cnt = 0; cnt < LV->Count; cnt++) {
			if(LV[cnt]->Items->Count > 0) {
				for(int i = 0; i < LV[cnt]->Columns->Count; i++) {
					if(!(LV[cnt]->Columns[i]->Text->Equals("M") || LV[cnt]->Columns[i]->Text->Length == 0)) {
						fwprintf(fp, L"%s\t ", clrToWstring(LV[cnt]->Columns[i]->Text).c_str());
					}
				}
				fwprintf(fp, L"\t ");
			}
		}

		fwprintf(fp, L"\n");
		
		// Fill in the data for the tables
		int maxRows = 0;
		maxRows = max(listViewFF->Items->Count, max(listView_Extraparams->Items->Count, 
			max(listView_peaks->Items->Count, max(listView_PeakPosition->Items->Count,
			max(listView_phases->Items->Count, listView_phases->Items->Count)))));

		for(int row = 0; row < maxRows; row++) {
			for(int cnt = 0; cnt < LV->Count; cnt++) {
				if(LV[cnt]->Items->Count > 0) {
					for(int itm = 0; itm < LV[cnt]->Columns->Count; itm++) {
						if(row < LV[cnt]->Items->Count) {
							if(!(LV[cnt]->Columns[itm]->Text->Equals("M") || LV[cnt]->Columns[itm]->Text->Length == 0)) {
							//make sure not to write muts, and stuff with no title... (mn, mx etc)
								fwprintf(fp, L"%s", clrToWstring(LV[cnt]->Items[row]->SubItems[itm]->Text).c_str());
							}
						}
						if(!(LV[cnt]->Columns[itm]->Text->Equals("M") || LV[cnt]->Columns[itm]->Text->Length == 0))
							fwprintf(fp, L"\t ");
					}
					if(!(cnt == LV->Count - 1))
						fwprintf(fp, L"\t ");
				}
			}
			fwprintf(fp, L"\n");
		}


		fwprintf(fp, L"\n\n\n%s \t %s", clrToWstring(UnicodeChars::chisqr).c_str(), clrToWstring(wssr->Text).c_str());
		if(phaseErrorTextBox->Text != "-")
			fwprintf(fp, L"\t\t\tPhase Error \t %s", clrToWstring(phaseErrorTextBox->Text).c_str());
		fwprintf(fp, L"\n%s \t %s", clrToWstring(UnicodeChars::rsqr).c_str(), clrToWstring(rsquared->Text).c_str());
		if(Volume->Text != "-")
			fwprintf(fp, L"\t\t\tUnit cell volume \t %s\n", clrToWstring(Volume->Text).c_str());

		// Close file
		fclose(fp);

	}

	void FormFactor::exportAllParametersAsCSVFileToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		WriteCSVParamsFile();
	}

	void FormFactor::General_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return) {
			//Change the focus so that any field that has a new value will be updated
			this->tabControl1->Focus();
			e->Handled = true; // This is so Windows won't beep (i.e., try to handle the key press)
		}
	}

	void FormFactor::fitRange_Click(System::Object^  sender, System::EventArgs^  e) {
			FitRange fr (listViewFF->Items, listViewFF->SelectedIndices[0], 
						 _modelFF->GetNumLayerParams(), _modelFF);

			fr.ShowDialog();

			linkedParameterCheck(listViewFF, listViewFF->SelectedIndices[0]);

			// Enable/Disable appropriate Textboxes
			parameterListView_SelectedIndexChanged(sender, e);

			UItoParameters(_curFFPar, _modelFF, listViewFF, listView_Extraparams);

	}

	void FormFactor::linkedParameterCheck(System::Windows::Forms::ListView ^lv, int layer) {
		ListViewItem ^lvi = lv->Items[layer];
		int curLayer, finalLayer;
		int paramNum = _modelFF->GetNumLayerParams();

		for(int i = 0; i < paramNum; i++) {
			bool bLink;
			curLayer = layer;
			int col = LV_CONSLINK(i, paramNum);
			bLink = Int32::Parse(lv->Items[curLayer]->SubItems[col]->Text) > -1;
			while(curLayer >= 0) {
				finalLayer = curLayer;
				curLayer = Int32::Parse(lv->Items[curLayer]->SubItems[col]->Text);
				if(curLayer == layer) { // Circular
					// Remove link
					char aa[255] = {0};
					sprintf(aa, "You have made a circular set of linked %s parameters",  
							lv->Columns[2 * i + 1]->Text);
					System::String ^text = gcnew System::String(aa);
					MessageBox::Show(text, "ERROR", MessageBoxButtons::OK,
									 MessageBoxIcon::Error);

					lvi->SubItems[col]->Text = "-1";
					curLayer = -2;
					bLink = false;
				}
			}
			if(bLink && finalLayer > -1) {
				// change mutability to 'L'
				// change value to match linked item
				// change the constraints to match the linked item
				
				lvi->SubItems[LV_VALUE(i)]->Text = lv->Items[finalLayer]->SubItems[LV_VALUE(i)]->Text;
				lvi->SubItems[LV_SIGMA(i,_modelFF->GetNumLayerParams())]->Text = lv->Items[finalLayer]->SubItems[LV_SIGMA(i,_modelFF->GetNumLayerParams())]->Text;
				FFGroupBoxList[i]->check->Checked = false;

				//Causes a bug: all parameters are linked to another, even if not told to be linked.
				//lvi->SubItems[LV_CONSLINK(i, nlp)]->Text = finalLayer.ToString();

				lvi->SubItems[LV_MUTABLE(i)]->Text = "L";
				if(Int32::Parse(lvi->SubItems[col]->Text) > 0) {
					for( int j = col - 4; j < col; j++)
						lvi->SubItems[j]->Text = lv->Items[/*Int32::Parse(lvi->SubItems[col]->Text)*/ finalLayer]->SubItems[j]->Text;
								
				}
			}
			// If the linking was removed, change the mutability of the item to mutable
			if((lvi->SubItems[LV_MUTABLE(i)]->Text == "L") && (Int32::Parse(lvi->SubItems[col]->Text) == -1))
				lvi->SubItems[LV_MUTABLE(i)]->Text = "Y";
		}	//end for(i < paramNum)
	}

	void FormFactor::linkedParameterChangedCheck(System::Windows::Forms::ListView::ListViewItemCollection ^lv, int layer) {
		ListViewItem ^lvi;
		if(_bGenerateModel) // No links
			return;

		//TODO::SF enable linked parameters
		if(lv != listViewFF->Items)
			return;
		
		int nlp = _modelFF->GetNumLayerParams();

		// Go over other linked items.  If they are linked to the changed value, change their value
		for(int i = 0; i < lv->Count; i++) {
			lvi = lv[i];
			for (int j = 0; j < nlp; j++) {
				int col = LV_CONSLINK(j, nlp);

				System::String^ ter = lvi->SubItems[col]->Text;

				if(Int32::Parse(lvi->SubItems[col]->Text) == layer)	{
					lvi->SubItems[LV_VALUE(j)]->Text = lv[layer]->SubItems[LV_VALUE(j)]->Text;
					linkedParameterChangedCheck(lv, i);
				}
			}

		}
	}

	void FormFactor::addLayer_Click(System::Object^  sender, System::EventArgs^  e) {
			AddParamLayer();
			
			AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
	}

	void FormFactor::removeLayer_Click(System::Object^  sender, System::EventArgs^  e) {
			if(listViewFF->SelectedIndices->Count == 0) return;

			// This function works the following way:
			// 1. Relink all linked layers (or remove links to the removed layer/s)
			// 2. Remove the layer
			// 3. Rename the layers

			int nlp = _modelFF->GetNumLayerParams();

			while(listViewFF->SelectedIndices->Count > 0) {
				int index = listViewFF->SelectedIndices[0];
								
				for(int i = 0; i < listViewFF->Items->Count; i++) {
					// Looping over all layer parameters for linked indices
					for(int j = 0; j < nlp; j++) {
						int linkInd = LV_CONSLINK(j, nlp);

						// If a linked parameter/index constraint point to the current item, 
						// remove that link/index
						if(Int32::Parse(listViewFF->Items[i]->SubItems[linkInd]->Text) == index) {
							listViewFF->Items[i]->SubItems[linkInd]->Text = "-1";
							listViewFF->Items[i]->SubItems[LV_MUTABLE(j)]->Text = "N";
						}

						// All linked/index constraints that point to subsequent items should 
						// be reduced by one
						if(Int32::Parse(listViewFF->Items[i]->SubItems[linkInd]->Text) > index)
							listViewFF->Items[i]->SubItems[linkInd]->Text = (Int32::Parse(listViewFF->Items[i]->SubItems[linkInd]->Text) - 1).ToString();
					}
				}

				// 2. Removing the item from the list
				listViewFF->Items->RemoveAt(index);

				// 3. Renaming all the layers
				for(int i = index; i < listViewFF->Items->Count; i++) {
					Windows::Forms::ListViewItem ^lvi = listViewFF->Items[i];

					// Changing layer name
					lvi->Text = stringToClr(_modelFF->GetLayerName(i));

					// Checking for applicability of the layer params
					for(int j = 0; j < nlp; j++) {
						if(!_modelFF->IsParamApplicable(i, j)) {
							lvi->SubItems[LV_VALUE(j)]->Text = "N/A";
							lvi->SubItems[LV_MUTABLE(j)]->Text = "-";
						}
					}
				}
				
			}

			// Maximal layer count check
			if(listViewFF->Items->Count < _modelFF->GetMaxLayers())
				addLayer->Enabled = true;

			_copiedIndicesFF->clear();

			FFParameterUpdateHandler();
			save->Enabled = true;
			AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
	}

	void FormFactor::Mut_CheckedChanged(System::Object^  sender, System::EventArgs^  e){
		ParamGroupBox^ snd = nullptr;
		snd = dynamic_cast<ParamGroupBox^>(((CheckBox^)sender)->Parent);
		if(!snd)
			return;
		if(snd == exParamGroupbox) {
			listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_MUTABLE]->Text = exParamGroupbox->check->Checked ? "Y" : "N";
			_curFFPar->extraParams[paramBox->SelectedIndex].isMutable = exParamGroupbox->check->Checked;
		} else {
			ModelUI *mui = NULL;
			ListView ^lv = nullptr;
			GroupBoxList ^gbl = nullptr;
			paramStruct *ps = NULL; // _curFFPar
			if(FFGroupBoxList->Contains(snd)) {
				mui = _modelFF;			lv = listViewFF;
				gbl = FFGroupBoxList;	ps = _curFFPar;
			} else if(SFGroupBoxList->Contains(snd)) {
				mui = _modelSF;			lv = listView_peaks;
				gbl = SFGroupBoxList;	ps = _curSFPar;
			} else if(BGGroupBoxList->Contains(snd)) {
				mui = _modelBG;			lv = BGListview;
				gbl = BGGroupBoxList;	ps = _curBGPar;
			}

			int nlp = mui->GetNumLayerParams();
			for(int i  = 0; i < lv->SelectedItems->Count; i++) {
				for (int j = 0; j < nlp; j++) {
					if (sender == gbl[j]->check){
						lv->SelectedItems[i]->SubItems[LV_MUTABLE(j)]->Text = gbl[j]->check->Checked ? "Y" : "N";
						lv->SelectedItems[i]->SubItems[LV_CONSLINK(j, nlp)]->Text = "-1";
						ps->params[j][lv->SelectedIndices[i]].isMutable
							= gbl[j]->check->Checked;
					}
				}
			}
		}
		save->Enabled = true;
	}

	void FormFactor::exmut_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_MUTABLE]->Text = exParamGroupbox->check->Checked ? "Y" : "N";
		_curFFPar->extraParams[paramBox->SelectedIndex].isMutable = exParamGroupbox->check->Checked;
		save->Enabled = true;
	}

	void FormFactor::useCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(sender == ffUseCheckBox) {
			_bUseFF = ffUseCheckBox->Checked;
			FFGroupbox->Enabled = ffUseCheckBox->Checked;
			if(_ff->tmpY.size() < _ff->y.size())
				_ff->tmpY.resize(_ff->y.size(), 1.0);
			std::swap(_ff->tmpY, _ff->y);

		} else if(sender == sfUseCheckBox) {
			phasefitter->Enabled = sfUseCheckBox->Checked;
			Peakfitter->Enabled = sfUseCheckBox->Checked;
			if(_peakPicker)
				PeakPicker_Click(sender, e);
			if(wgtFit && wgtFit->graph) {
				for(int i = 0; i < wgtFit->graph->numGraphs; i++) {
					if (graphType->at(i) == GRAPH_PEAK || graphType->at(i) == GRAPH_PHASEPOS)
						wgtFit->graph->SetGraphVisibility(i, sfUseCheckBox->Checked);
				}
				if(!(sfUseCheckBox->Checked))
					listView_peaks->SelectedItems->Clear();
			}
			if(_sf->tmpY.size() < _sf->y.size())
				_sf->tmpY.resize(_sf->y.size(), 1.0);
			std::swap(_sf->tmpY, _sf->y);

		} else if(sender == bgUseCheckBox) {
			functionsGroupBox->Enabled = bgUseCheckBox->Checked;
			if(_bg->tmpY.size() < _bg->y.size())
				_bg->tmpY.resize(_bg->y.size(), 0.0);
			std::swap(_bg->tmpY, _bg->y);

		}

		if(wgtFit && wgtFit->graph && wgtFit->graph->x)
			if(liveRefreshToolStripMenuItem->Checked)
				UpdateGraph();

	}

	void FormFactor::paramBox_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if(_bLoading || _bChanging)
			return;
		// Saving old parameters
		if(listView_Extraparams->SelectedIndices->Count == 0 || 
		   paramBox->SelectedIndex != listView_Extraparams->SelectedIndices[0]) {
			   exParamGroupbox->track->Value = int((exParamGroupbox->track->Minimum + exParamGroupbox->track->Maximum) / 2.0);
			listView_Extraparams->SelectedIndices->Clear();
			listView_Extraparams->SelectedIndices->Add(paramBox->SelectedIndex);
		}

		ListViewItem ^lvi;

		if(oldIndex > -1) {
			lvi = listView_Extraparams->Items[oldIndex];

			lvi->SubItems[exParamGroupbox->rValue->Checked ? ELV_VALUE : ELV_SIGMA]->Text = exParamGroupbox->text->Text;
			if(!_bGenerateModel)
				lvi->SubItems[ELV_MUTABLE]->Text = exParamGroupbox->check->Checked ? "Y" : "N";
			lvi->SubItems[ELV_CONSMIN]->Text = exmin->Text;
			lvi->SubItems[ELV_CONSMAX]->Text = exmax->Text;
		}
		
		// Loading new parameters
		UpdateExtraParamBox();
		

		oldIndex = paramBox->SelectedIndex;
	}

	void FormFactor::exportElectronDensityProfileToolStripMenuItem1_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring file;
		sfd->FileName = "";
		sfd->Title = "Choose an E.D. profile data file";
		sfd->Filter = L"Output Files (*.out)|*.out|Data Files (*.dat, *.chi)|*.dat;*.chi|All files|*.*";
		if(sfd->ShowDialog() == 
			System::Windows::Forms::DialogResult::Cancel)
			return;

		clrToString(sfd->FileName, file);
		WriteDataFile(file.c_str(), wgtPreview->graph->x[0], wgtPreview->graph->y[0]);
	}

	void FormFactor::FormFactor_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
		this->Visible = false;

		// We need to delete all the vectors, peaks, layers, etc.!
		if(wgtFit != nullptr)
			wgtFit->graph = nullptr;

		// TODO::NewFitter: If Job is still alive, ask (and close it)

		SetMinimumSig(5.0);
	}

	void FormFactor::FormFactor_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e) {
		this->Visible = false;

		// We need to delete all the vectors, peaks, layers, etc.!
		if(wgtFit != nullptr)
			wgtFit->graph = nullptr;

		// TODO::NewFitter: If Job is still alive, ask (and close it)

		SetMinimumSig(5.0);
	}

	void FormFactor::logScaledFittingParamToolStripMenuItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(_bLoading)
			return;
		if(logScaledFittingParamToolStripMenuItem->Checked == false) {
			for (unsigned int i=0; i<wgtFit->graph->y[0].size(); i++) {
				if( wgtFit->graph->y[0][i] <= 0.0) {
					MessageBox::Show("Negative values found: log scaled fitting is not allowed", "Negative values found:" );
					logScaledFittingParamToolStripMenuItem->Checked = false;
					break;
				}
			}
		}

		if(wgtFit && wgtFit->graph) {
			UpdateChisq(WSSR(wgtFit->graph->y[0], wgtFit->graph->y[1]));
			UpdateRSquared(RSquared(wgtFit->graph->y[0], wgtFit->graph->y[1]));
		}
		return;
	}

	void FormFactor::infExtraParam_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(listView_Extraparams->SelectedItems->Count == 0)
			return;

		ListViewItem ^lvi = listView_Extraparams->SelectedItems[0];
		
		// Modify parameter name so that the user knows it's infinite
		if(infExtraParam->Checked && lvi->SubItems[ELV_INFINITE]->Text->Equals("0")) {
			lvi->Text += " (inf)";
			lvi->SubItems[ELV_INFINITE]->Text = "1";
		} else if(!infExtraParam->Checked && lvi->SubItems[ELV_INFINITE]->Text->Equals("1")) {
			lvi->Text = lvi->Text->Substring(0, lvi->Text->Length - 6);
			lvi->SubItems[ELV_INFINITE]->Text = "0";
		}

		slowModelGroupbox->Visible = _modelFF->IsSlow();

		// Enable/disable controls according to infinity status
		if(infExtraParam->Checked) {
			exParamGroupbox->Enabled = false;
			exmin->Enabled   = false;
			exmax->Enabled   = false;
		} else {
			exParamGroupbox->Enabled = true;
			exmin->Enabled   = !_bGenerateModel;
			exmax->Enabled   = !_bGenerateModel;
			exParamGroupbox->check->Enabled   = !_bGenerateModel;
			exParamGroupbox->track->Enabled = true;
		}

		FFParameterUpdateHandler();

		save->Enabled = true;
		AddToUndoQueue(MT_FORMFACTOR, _curFFPar);
	}

	/**
	 * Makes the wgtFit graph be redrawn without recalculating any models.
	**/
	void FormFactor::RedrawGraph() {
		if(!wgtFit || !wgtFit->graph)
			return;
		// These are the only relevant parts of UpdateGraph
		if(!_bGenerateModel) {
			UpdateChisq(WSSR(wgtFit->graph->y[0], wgtFit->graph->y[1]));
			UpdateRSquared(RSquared(wgtFit->graph->y[0], wgtFit->graph->y[1]));
		}

		wgtFit->graph->ToggleYTicks();
		wgtFit->graph->ToggleYTicks();
		wgtFit->Invalidate();
	}

	void FormFactor::useModelFFButton_Click(System::Object^  sender, System::EventArgs^  e) {
		useModelFFButton->Visible = false;
		relatedModelsToolStripMenuItem->Enabled = true;

		_loadedFF->clear();

		std::swap(_storage->y, _ff->y);
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();
	}

	void FormFactor::maskButton_Click(System::Object^  sender, System::EventArgs^  e) {
		if(!wgtFit || !wgtFit->graph)
			return;

		maskToolStrip->Visible = !maskToolStrip->Visible;
		_bMasking = maskToolStrip->Visible;
		_bAddMask = _bMasking;

		if(_mask->size() != wgtFit->graph->x[0].size())
			_mask->resize(wgtFit->graph->x[0].size(), false);
		
		if(!_bMasking)
			_pressX = _pressY = -1;
	}

	void FormFactor::maskPanel_Click(System::Object^  sender, System::EventArgs^  e) {
		if(sender == addMaskButton) {
			_bAddMask = true;
			return;
		}

		if(sender == removeMaskButton) {
			_bAddMask = false;
			return;
		}

		if(sender == invertMaskButton)
			for(int i = 0; i < (int)_mask->size(); i++)
				_mask->at(i) = !(_mask->at(i));

		if(sender == clearMaskButton) {
			_mask->clear();
			_mask->resize(_mask->size(), false);
		}

		wgtFit->graph->RemoveMask();
		if(wgtFit && wgtFit->graph) {
			int ind = 0;
			while (ind < (int)_mask->size()) {
				int srt = -1, nd = -1;

				for(; ind < (int)_mask->size() && !_mask->at(ind); ind++);

				srt = ind;
				for(nd = srt + 1; nd < (int)_mask->size(); nd++)
					if(!(_mask->at(nd)))
						break;

				ind = nd--;

				if(nd < (int)_mask->size())
					wgtFit->graph->Mask(0, srt, nd, RGB(50, 50, 50));
		
				wgtFit->Invalidate();
			}
			RedrawGraph();

		}
	} //maskPanel_Click

	void FormFactor::expResToolStripTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e) {
		double prevRes = GetResolution(), newRes;
		newRes = clrToDouble(expResToolStripTextBox->Text);
		
		if((newRes == 0.0 && !expResToolStripTextBox->Text->StartsWith("0"))	//Starts with text
			|| fabs(newRes - prevRes) < 1e-12) {	// The number hasn't changed
			expResToolStripTextBox->Text = prevRes.ToString("0.000000");
			return;
		}
		
		if(!wgtFit) {
			//SetResolution(newRes);	// TODO::ResChanged
			expResToolStripTextBox->Text = newRes.ToString();
			return;
		}

		//SetResolution(newRes);	// TODO::ResChanged
		expResToolStripTextBox->Text = newRes.ToString("0.000000");
		
		if(liveRefreshToolStripMenuItem->Checked) {
			UpdateGraph();
		}
	}
	void FormFactor::expResToolStripTextBox_Enter(System::Object^  sender, System::EventArgs^  e) {
		expResToolStripTextBox->Text = GetResolution().ToString("0.000000");
	}

	void FormFactor::reportButton_Click(System::Object^  sender, System::EventArgs^  e) {
		std::vector <std::vector <double>> errorsVector;
		System::Collections::Generic::List<ListView^>^ LV = gcnew System::Collections::Generic::List<ListView^>();
		LV->Add(listViewFF);
		LV->Add(listView_peaks);
		LV->Add(listView_phases);
		LV->Add(BGListview);
		LV->Add(listView_Extraparams);

		errorsVector.push_back(*FFparamErrors);
		errorsVector.push_back(*SFparamErrors);
		errorsVector.push_back(*PhaseparamErrors);
		errorsVector.push_back(*BGparamErrors);

		ErrorTableWindow errrr(LV, errorsVector, _dataFile);

		errrr.ShowDialog();
	}

	// Helper function to change the current model in a paramStruct
	void FormFactor::ChangeModel(paramStruct *p, ModelUI *newModel) {
		// The real deal
		// Main parameters
		int first = p->params.size(),
			second = p->params[0].size();
		p->params.resize(newModel->GetNumLayerParams());
		for(int i = 0; i < newModel->GetNumLayerParams(); i++) {
			p->params[i].resize(max(p->layers, newModel->GetMinLayers()));
			for(int j = (i < first) ? second : 0; j < max(p->layers, newModel->GetMinLayers()); j++) {
				// Prepare Parameter using default values
				Parameter param(newModel->GetDefaultParamValue(i, j));
				// Insert into p
				p->params[i][j] = param;
			}
		}
		int maxLayers = newModel->GetMaxLayers();
		if(maxLayers > -1) {
			for(int i = 0; i < maxLayers; i++)
				p->params[i].resize(maxLayers);
			p->layers = maxLayers;
		}

		// Extra parameters
		first = p->extraParams.size();
		p->extraParams.resize(newModel->GetNumExtraParams());
		for(int i = first; i < newModel->GetNumExtraParams(); i++) {
			ExtraParam gr = newModel->GetExtraParameter(i);
			Parameter pr(gr.defaultVal);
			p->extraParams[i] = pr;
		}		
	}

	void FormFactor::changeModel_Click(System::Object^  sender, System::EventArgs^  e) {
		if(!_modelFF) {
			MessageBox::Show("Error changing model: no model");
			return;
		}

		// Modify model type
		ExternalModelDialog^ emd = _parent->emd;
		emd->ClearModelSelection();

		emd->LoadDefaultModels();
		System::Windows::Forms::DialogResult result = emd->ShowDialog();
		if(result != System::Windows::Forms::DialogResult::OK)
			return;		// No model was selected

		ModelInformation ffm = emd->GetSelectedModel();

		ModelUI newMoUI;
		// TODO::ChangeModel(?) Should this be the other way around? (retrieve before SetModel) so that we know its
		//       EDP?
		if(_lf->SetModel(_job, MT_FORMFACTOR, clrToWstring(_mioFF->contName).c_str(), ffm.modelIndex, EDProfile() /*TODO::EDP*/ ) )
			return;	// Error with SetModel. Maybe display it to the user

		*_miFF = ffm;
		newMoUI.setModel(_lf, clrToWstring(_mioFF->contName).c_str(), ffm, listViewFF->Items->Count);
		handleModelChange(newMoUI);
	}

	void FormFactor::relatedModelToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if(!_modelFF) {
			MessageBox::Show("Error changing model: no model");
			return;
		}

		// Determine which model was selected
		int ind = relatedModelsToolStripMenuItem->DropDownItems->IndexOf((ToolStripItem^)(sender));

		// Create selected model
		ModelInformation ffm = _lf->QueryModel(clrToWstring(_mioFF->contName).c_str(), _modelFF->GetRelatedModelIndex(ind));
		ModelUI newMoUI;
		if(_lf->SetModel(_job, MT_FORMFACTOR, clrToWstring(_mioFF->contName).c_str(), ffm.modelIndex, EDProfile() /*TODO::EDP*/ ) )
			return;	// Error with SetModel. Maybe display it to the user

		*_miFF = ffm;
		newMoUI.setModel(_lf, clrToWstring(_mioFF->contName).c_str(), ffm, listViewFF->Items->Count);
		handleModelChange(newMoUI);
	}

	void FormFactor::handleModelChange(ModelUI &ffm) {
		delete _modelFF;		
		_modelFF = &ffm;
		_parent->_currentModel = _miFF;
		
		ReloadModelUI();
	}

	void FormFactor::ReloadModelUI() {
		//TODO::ChangeModel this whole method is buggy now.

		// Load current parameters
		paramStruct p = *_curFFPar;

		_bChanging = true;

		// Replace renderer
		wchar_t con[MAX_PATH] = {0};
		_modelFF->GetContainer(con);
		ModelRenderer ren (con, _modelFF->GetModelInformation().modelIndex);
		_parent->renderScene = renderScene = ren.GetRenderer();
		_parent->previewScene = ren.GetPreview();
		// END of renderer substitution

		// Resize parameter struct
		ChangeModel(&p, _modelFF);

		// Reload UI
		PrepareModelUI();

		// Reload parameters
		ParametersToUI(&p, _modelFF, listViewFF, listView_Extraparams);

		_bChanging = false;

		FFParameterUpdateHandler();
	}

	void FormFactor::PrepareModelUI() {
#pragma region Form Factor tab
		// Add extra parameters to the interface
		listView_Extraparams->Items->Clear();
		paramBox->Items->Clear();
		if(_modelFF->GetNumExtraParams() > 0) {
			for(int i = 0; i < _modelFF->GetNumExtraParams(); i++) {
				ListViewItem ^lvi = gcnew ListViewItem();
				
				char valstr[128] = {0};
				ExtraParam ep = _modelFF->GetExtraParameter(i);
				String ^nameCLRstr = gcnew String(stringToClr(std::string(ep.name)));

				lvi->Text = nameCLRstr;
				paramBox->Items->Add(nameCLRstr);

				// Format the default value
				sprintf(valstr, "%.*f", ep.decimalPoints, 
					_finite(ep.defaultVal) ? ep.defaultVal : 0.0);
				
				lvi->SubItems->Add(gcnew String(valstr));
				
				lvi->SubItems->Add("N");           // Mutable
				
				// Constraints (default constraints are the range values)
				sprintf(valstr, "%.*f", ep.decimalPoints, ep.rangeMin);
				lvi->SubItems->Add(gcnew String(valstr));  // Min constraint
				sprintf(valstr, "%.*f", ep.decimalPoints, ep.rangeMax);
				lvi->SubItems->Add(gcnew String(valstr));  // Max constraint

				// Infinite
				lvi->SubItems->Add(_finite(ep.defaultVal) ? "0" : "1");
				if(!_finite(ep.defaultVal))
					lvi->SubItems[ELV_NAME]->Text += " (inf)";

				lvi->SubItems->Add("Y");  // isConstained -> Fix this if we ever want to add another checkbox

				lvi->SubItems->Add("0.000000"); // Standard deviation

				listView_Extraparams->Items->Add(lvi);
			}
			if(paramBox->Items->Count > 0)
				paramBox->SelectedIndex = 0;
			if(exParamGroupbox)
				delete exParamGroupbox;
			exParamGroupbox = gcnew ParamGroupBox(listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_NAME]->Text, ELV_VALUE, true);
			exParamGroupbox->text->Text = listView_Extraparams->Items[paramBox->SelectedIndex]->SubItems[ELV_VALUE]->Text;

			// Group box Events
			exParamGroupbox->text->Enter += gcnew System::EventHandler(this, &FormFactor::ExtraParameter_Enter);
			exParamGroupbox->text->Leave += gcnew System::EventHandler(this, &FormFactor::ExtraParameter_TextChanged);
			exParamGroupbox->rStddev->CheckedChanged += gcnew System::EventHandler(this, &FormFactor::PDRadioChanged);
			exParamGroupbox->track->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &FormFactor::centerTrackBar);
			exParamGroupbox->check->Click += gcnew System::EventHandler(this, &FormFactor::Mut_CheckedChanged);

			globalParamtersGroupBox->Controls->Add(exParamGroupbox);
			exParamGroupbox->Location = Point(6, 42);
			exParamGroupbox->Enabled = true;
			// END of extra parameters

		}		

		// Initialize param groupbox list
		while(FFGroupBoxList && FFGroupBoxList->Count > 0) {
			delete FFGroupBoxList[0];
			FFGroupBoxList->RemoveAt(0);
		}
		FFGroupBoxList = gcnew GroupBoxList();

		// Initialize related models list
		while(relatedModelsList && relatedModelsList->Count > 0)
		{
			delete relatedModelsList[0];
			relatedModelsList->RemoveAt(0);
		}
		relatedModelsList = gcnew System::Collections::Generic::List<ToolStripMenuItem^>();
		
		PrepareUISection(listViewFF, FFGroupBoxList, _modelFF);

		// Model display values
		// If there are no display parameters, don't show the list
		if (_modelFF->GetNumDisplayParams() == 0)
			listView_display->Visible = false;
		else
			listView_display->Visible = true;

		// Custom Electron Density profile functions
		if(_modelFF->IsLayerBased()) {
			// Get the default electron density profile
			EDProfile defaultEDP = _modelFF->GetEDProfile();
			// TODO::EDP EDPFunction *edp = _modelFF->GetEDProfileFunction();
			// If not discrete, disable other options
			if(defaultEDP.shape != DISCRETE /*&& TODO::EDP !edp*/) {
				electronDensityProfileToolStripMenuItem->Visible = false;
			} else {
				electronDensityProfileToolStripMenuItem->Visible = true;

				discreteStepsToolStripMenuItem->Enabled = true;			
				gaussiansToolStripMenuItem->Enabled = true;
				hyperbolictangentSmoothStepsToolStripMenuItem->Enabled = true;
				stepResolutionToolStripMenuItem->Enabled = true;				
			}

			edpBox->Visible = true;			

			// Reset model ED profile type
			discreteStepsToolStripMenuItem->Checked = true;
			gaussiansToolStripMenuItem->Checked = false;
			hyperbolictangentSmoothStepsToolStripMenuItem->Checked = false;
		} else {
			discreteStepsToolStripMenuItem->Enabled = false;			
			gaussiansToolStripMenuItem->Enabled = false;
			hyperbolictangentSmoothStepsToolStripMenuItem->Enabled = false;
			stepResolutionToolStripMenuItem->Enabled = false;
			edpBox->Visible = false;
		}

		// Getting paramStruct from GUI
		paramStruct p(_modelFF->GetModelInformation());
		UItoParameters(&p, _modelFF, listViewFF, listView_Extraparams);
		
		if(!_curFFPar)
			_curFFPar = new paramStruct(_modelFF->GetModelInformation());
		*_curFFPar = p;

		// Update ED profile
		UpdateEDPreview();

		// If there are display parameters, add the parameters names and 
		// values to the list
		listView_display->Items->Clear();
		for (int i = 0 ; i < _modelFF->GetNumDisplayParams(); i++){
			ListViewItem ^lvi = gcnew ListViewItem();

			std::string nameStr = _modelFF->GetDisplayParamName(i);
			String ^nameCLRstr = gcnew String(nameStr.c_str());

			lvi->Text = nameCLRstr;
/* TODO::DisplayParams
			lvi->SubItems->Add(_modelFF->GetDisplayParamValue(i, &p).
				ToString("0.000000"));
*/
			listView_display->Items->Add(lvi);
		}
		// END of display values

		// This is silly. The only time this (PrepareModelUI) method is called is during loading or changing.
		// The first thing that ParameterUpdateHandler does is return if it is one of those...
		FFParameterUpdateHandler();

		// Fill the related models menu
		relatedModelsToolStripMenuItem->DropDownItems->Clear();
		for(int i = 0; i < _modelFF->GetNumRelatedModels(); i++) {
			relatedModelsList->Add(gcnew System::Windows::Forms::ToolStripMenuItem());
			relatedModelsList[i]->Name = stringToClr(_modelFF->GetRelatedModelName(i));
			relatedModelsList[i]->Text = stringToClr(_modelFF->GetRelatedModelName(i));
			relatedModelsList[i]->Click += gcnew System::EventHandler(this, &FormFactor::relatedModelToolStripMenuItem_Click);
			relatedModelsToolStripMenuItem->DropDownItems->Add(relatedModelsList[i]);
			if(_modelFF->GetRelatedModelName(i).compare(_modelFF->GetName()) == 0)
				relatedModelsList[i]->Visible = false;
		}
		relatedModelsToolStripMenuItem->Visible = (relatedModelsToolStripMenuItem->DropDownItems->Count > 0);

#pragma endregion
#pragma region Structure Factor tab
		if(!_modelSF)
			_modelSF = new ModelUI();
		if(!SFGroupBoxList)
			SFGroupBoxList = gcnew GroupBoxList();

		PrepareUISection(listView_peaks, SFGroupBoxList, _modelSF);

		if(!_curSFPar)
			_curSFPar = new paramStruct(_modelSF->GetModelInformation());

#pragma endregion
#pragma region Background tab
		if(!_modelBG)
			_modelBG = new ModelUI();
		//PrepareUISection(BGListview, BGGroupBoxList, _modelBG);	// TODO::BG
		if(!_curBGPar)
			_curBGPar = new paramStruct(_modelBG->GetModelInformation());
#pragma endregion

	}

	void FormFactor::PrepareUISection(ListView ^lv, GroupBoxList ^gbl, ModelUI *mui) {
		ModelType type;
		ComboBox ^cb = nullptr;
		FlowLayoutPanel ^flp = nullptr;
		modelInfoObject ^mio = nullptr;
		paramStruct *ps = NULL;
		int mi = -1;

		if(mui == _modelFF) {
			type = MT_FORMFACTOR;
			flp = FFparamPanel;
			ps = _curFFPar;
			mi = _parent->_currentModel->modelIndex;
			mio = _mioFF;
		} else if(mui == _modelSF) {
			type = MT_STRUCTUREFACTOR;
			cb = peakfit;
			flp = SFparamPanel;
			ps = _curSFPar;
		} else if(mui == _modelBG) {
			type = MT_BACKGROUND;
			;	// TODO::BG find the combobox and add st
			;	// TODO::BG Make FLP for BG
			;	// TODO::BG Determine how to deal with enumeration of BG types.
				//			Issues: 1) A parameter in the listView needs to be shown as
				//						a string while representing a function and cannot
				//						be changed as a numerical parameter (by both the
				//						fitter and the user).
				//					2) If we use an enumeration, we will be unable
				//						to add more BG models at whim.
				//					3) When this is figured out, consider applying to SF.
			ps = _curBGPar;
		} else
			return;

		// Query all structure factor models
		for(int k = 0; k < _containers->Length; k++) {
			std::wstring str = clrToWstring(_containers[k]);
			int cats = _lf->QueryCategoryCount(str.c_str());
			int count = 0;
			for(int i = 0; i < cats; i++) {
				ModelCategory mc = _lf->QueryCategory(str.c_str(), i);
				if(mc.type == type) {
					for(int j = 0; j < 16; j++) {
						if(mc.models[j] == -1)
							break;

						ModelInformation grr = _lf->QueryModel(str.c_str(), mc.models[j]);
						modelInfoObject^ st = gcnew modelInfoObject(grr.category,
							grr.modelIndex, count++, grr.name, str.c_str());
						if(cb) {
							bool contained = false;
							for(int c = 0; c < cb->Items->Count; c++)
								if(cb->Items[c]->ToString()->Equals(st->ToString()))
									contained = true;
							if(!contained)
								cb->Items->Add(st);
						}
					}
				}
			}
		}
		if(type == MT_STRUCTUREFACTOR)
			peakfit_SelectedIndexChanged(nullptr, gcnew System::EventArgs());

		// Initialize SF param groupbox list
		while(gbl && gbl->Count > 0) {
			delete gbl[0];
			gbl->RemoveAt(0);
		}

		while(lv->Columns->Count > 1)
			lv->Columns->RemoveAt(1);
		lv->Items->Clear();

		if(cb) {
			mio = (modelInfoObject ^)(cb->SelectedItem);
			mi = mio->modelIndex;
		}

		ErrorCode setModel = _lf->SetModel(_job, type, clrToWstring(mio->contName).c_str(), mio->modelIndex, EDProfile());
		if(setModel)
			Windows::Forms::MessageBox::Show("Unable to set model. Error code: " + ((Int32)setModel).ToString(),
												"ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);

		ModelInformation grr = _lf->QueryModel(clrToWstring(mio->contName).c_str(), mio->modelIndex);
		bool succ = mui->setModel(_lf, clrToWstring(mio->contName).c_str(), grr, 10);

		if(!ps)
			ps = new paramStruct(mui->GetModelInformation());

		for (int i = 0; i < mui->GetNumLayerParams(); i++) {
			String ^lpName = stringToClr(mui->GetLayerParamName(i));
			int lvIndex;

			// Handle layer listview columns
			lvIndex = lv->Columns->Count;			
			lv->Columns->Add(lpName);			
			lv->Columns->Add("M");

			// Default column widths
			lv->Columns[lvIndex]->Width = 80;

			// Don't show mutability columns if we're generating
			lv->Columns[lvIndex + 1]->Width = _bGenerateModel ? 0 : 25;
			lv->Columns[lvIndex + 1]->TextAlign = HorizontalAlignment::Center;
			// END of listview columns

			// Handle parameter groupboxes
			gbl->Add(gcnew ParamGroupBox(lpName, lvIndex, true));

			// Group box Events
			gbl[i]->text->Leave += gcnew System::EventHandler(this, &FormFactor::Parameter_TextChanged);
			gbl[i]->rStddev->CheckedChanged += gcnew System::EventHandler(this, &FormFactor::PDRadioChanged);
			gbl[i]->track->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &FormFactor::centerTrackBar);
			gbl[i]->check->Click += gcnew System::EventHandler(this, &FormFactor::Mut_CheckedChanged);

			if(_bGenerateModel)
				gbl[i]->check->Enabled = false;

			flp->Controls->Add(gbl[i]);	// 
			// END of groupboxes
		}

		// Initial layers
		for(int i = 0; i < mui->GetMinLayers(); i++)
			AddParamLayer(mui);

		// TODO::PS default values/sizes or whatever
		if(!ps)
			ps = new paramStruct();
	}

	void FormFactor::FFParameterUpdateHandler() {
		if(_bLoading ^ _bChanging)
			return;
		// This function should:
		// 0. Update _curFFPar
		UItoParameters(_curFFPar, _modelFF, listViewFF, listView_Extraparams);

		// 1. If necessary, redraw the graph
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph(_bLoading || _bChanging);		

		// 2. Modify display values
		// 3. Redraw 3d preview and ED profile		

		// Modifying display values
		for( int i = 0; i < listView_display->Items->Count; i++) {
			ListViewItem ^lvi = listView_display->Items[i];
/* TODO::DisplayParams
			lvi->SubItems[1]->Text = _modelFF->GetDisplayParamValue(i, _curFFPar).
													ToString("0.000000");	
*/
		}

		if(!_bFromFitter) {
			FFparamErrors->clear();
			FFmodelErrors->clear();
		}
		// Disable/enable + button according to number of layers
		if(_modelFF->GetMaxLayers() < 0) // The infinite case
			addLayer->Enabled = true;
		else
			addLayer->Enabled = (listViewFF->Items->Count < _modelFF->GetMaxLayers());

		// Updating ED profile-related stuff
		UpdateEDPreview();
		EDU();

	}

	void FormFactor::listViewFF_DoubleClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		if(listViewFF->SelectedItems->Count < 1)
			return;
		fitRange_Click(sender, e);
	}

	void FormFactor::discreteStepsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		// Change ED Profile in model, if necessary
		if(discreteStepsToolStripMenuItem->Checked && _modelFF) {
/* TODO::EDP
			_modelFF->SetEDProfile(EDProfile(SYMMETRIC, DISCRETE));
*/

			ReloadModelUI();
		}

		discreteStepsToolStripMenuItem->Checked = true;
		gaussiansToolStripMenuItem->Checked = false;
		hyperbolictangentSmoothStepsToolStripMenuItem->Checked = false;		
	}

	void FormFactor::gaussiansToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		// Change ED Profile in model, if necessary
		if(gaussiansToolStripMenuItem->Checked && _modelFF) {
/* TODO::EDP
			_modelFF->SetEDProfile(EDProfile(SYMMETRIC, GAUSSIAN));

			// Modify profile resolution
			int res = atoi(clrToString(edpResolution->Text).c_str());
			if(adaptiveToolStripMenuItem->Checked)
				_modelFF->GetEDProfileFunction()->SetResolution(-res);
			else
				_modelFF->GetEDProfileFunction()->SetResolution(res);
*/
			// Reload UI
			ReloadModelUI();
		}

		discreteStepsToolStripMenuItem->Checked = false;
		gaussiansToolStripMenuItem->Checked = true;
		hyperbolictangentSmoothStepsToolStripMenuItem->Checked = false;
	}

	void FormFactor::hyperbolictangentSmoothStepsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		// Change ED Profile in model, if necessary
		if(hyperbolictangentSmoothStepsToolStripMenuItem->Checked && _modelFF) {
/* TODO::EDP
			_modelFF->SetEDProfile(EDProfile(SYMMETRIC, TANH));

			// Modify profile resolution
			int res = atoi(clrToString(edpResolution->Text).c_str());
			if(adaptiveToolStripMenuItem->Checked)
				_modelFF->GetEDProfileFunction()->SetResolution(-res);
			else
				_modelFF->GetEDProfileFunction()->SetResolution(res);
*/
			// Reload UI
			ReloadModelUI();
		}

		discreteStepsToolStripMenuItem->Checked = false;
		gaussiansToolStripMenuItem->Checked = false;
		hyperbolictangentSmoothStepsToolStripMenuItem->Checked = true;
	}

	void FormFactor::edpResolution_TextChanged(System::Object^  sender, System::EventArgs^  e) {
		int res = atoi(clrToString(edpResolution->Text).c_str());
		if(res < 0)
			res = -res;
		else if(res == 0)
			res = 1;
		edpResolution->Text = Int32(res).ToString();

/* TODO::EDP
		if(_modelFF && _modelFF->GetEDProfileFunction())
			_modelFF->GetEDProfileFunction()->SetResolution(
							adaptiveToolStripMenuItem->Checked ? -res : res);
*/
		FFParameterUpdateHandler();
	}

	void FormFactor::adaptiveToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		// TODO::EDP
/*
		if(_modelFF && _modelFF->GetEDProfileFunction()) {
			int res = Int32::Parse(edpResolution->Text);

			if(adaptiveToolStripMenuItem->Checked) {
				res = DEFAULT_EDEPS;				
				_modelFF->GetEDProfileFunction()->SetResolution(-res);
			} else {
				res = DEFAULT_EDRES;
				_modelFF->GetEDProfileFunction()->SetResolution(res);
			}

			edpResolution->Text = Int32(res).ToString();

			FFParameterUpdateHandler();
		}*/
	}

}