#include "FormFactor.h"

#include <string>
#include <cstdlib>
#include <cmath>

//#include "calculation_external.h"
//#include "genbackground.h"

//#include "clrfunctionality.h"
//#include "FrontendExported.h"

using namespace System::Windows::Forms;

namespace GUICLR {

	void FormFactor::Background_Load() {
		funcTypeList->SelectedIndex = 0;
		funcTypeList->Enabled = false;
	}

	/**
	 * Transfers the newly entered value to the relevant field in the list
	**/
	void FormFactor::BGTextBox_Leave(System::Object^  sender, System::EventArgs^  e) {
		double res;
		std::string str;
		char f[128] = {0};
		
		clrToString(((TextBox ^)(sender))->Text, str);

		if(!(sender == baseMaxBox || sender == baseMinBox || sender == baseTextbox))
			res = strtod(str.c_str(), NULL);
		else //base or modelAmplitude
			res = fabs(strtod(str.c_str(), NULL));
		
		sprintf(f, "%.6f", res);		
		((TextBox ^)(sender))->Text = gcnew String(f);

		if(BGListview->SelectedItems->Count > 0) {
		
			if(sender == baseTextbox)
				BGListview->SelectedItems[0]->SubItems[2]->Text = baseTextbox->Text;

			if(sender == decayTextbox)
				BGListview->SelectedItems[0]->SubItems[4]->Text = decayTextbox->Text;

			if(sender == xCenterTextbox)
				BGListview->SelectedItems[0]->SubItems[6]->Text = xCenterTextbox->Text;

			if(sender == baseMinBox)
				BGListview->SelectedItems[0]->SubItems[8]->Text = baseMinBox->Text;
			if(sender == baseMaxBox)
				BGListview->SelectedItems[0]->SubItems[9]->Text = baseMaxBox->Text;

			if(sender == decMinBox)
				BGListview->SelectedItems[0]->SubItems[10]->Text = decMinBox->Text;
			if(sender == decMaxBox)
				BGListview->SelectedItems[0]->SubItems[11]->Text = decMaxBox->Text;

			if(sender == xcMinBox)
				BGListview->SelectedItems[0]->SubItems[12]->Text = xcMinBox->Text;
			if(sender == xcMaxBox)
				BGListview->SelectedItems[0]->SubItems[13]->Text = xcMaxBox->Text;
		}

		BGparamErrors->clear();
		BGmodelErrors->clear();

		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();
		save->Enabled = true;
		AddToUndoQueue(MT_BACKGROUND, _curBGPar);
	}


	/**
	 * Returns the trackbar to the center when no longer being held with the mouse
	**/
	void FormFactor::BGTrackBar_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			TrackBar ^track = (TrackBar ^)sender;
			track->Value = int((track->Maximum - track->Minimum) / 2.0);

			if(liveRefreshToolStripMenuItem->Checked)
				UpdateGraph();

			save->Enabled = true;
			AddToUndoQueue(MT_BACKGROUND, _curBGPar);
	}

	/**
	 * Adds a background function to the list of functions
	**/
	void FormFactor::addFuncButton_Click(System::Object^  sender, System::EventArgs^  e) {
		//Default BG function
		AddBGFunction(BG_EXPONENT, 1.0, 'N', 0.1, 'N', 0.0, 'N', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();
		save->Enabled = true;
		AddToUndoQueue(MT_BACKGROUND, _curBGPar);
	}

	/**
	 * Removes the selected background functions from the list
	**/
	void FormFactor::removeFuncButton_Click(System::Object^  sender, System::EventArgs^  e) {
		while(BGListview->SelectedItems->Count > 0)
				BGListview->Items->Remove(BGListview->SelectedItems[0]);

		for(int i = 0; i < BGListview->Items->Count; i++)
			BGListview->Items[i]->Text = (i + 1).ToString();

		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();

		save->Enabled = true;
		AddToUndoQueue(MT_BACKGROUND, _curBGPar);
	}

	/**
	 * Controls the behavior of the GUI based on the number of background functions selected
	**/
	void FormFactor::BGListview_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			if(BGListview->SelectedIndices->Count == 0) {
				// No Items
				// 20 items must be disabled 3 * 6(amp, dec, Xc stuff) + "-" button + dropdown list
				removeFuncButton->Enabled = false;
				funcTypeList->Enabled = false;
				
				baseLabel->Enabled = false;
				decayLabel->Enabled = false;
				xCenterLabel->Enabled = false;
				baseTextbox->Enabled = false;
				decayTextbox->Enabled = false;
				xCenterTextbox->Enabled = false;
				baseMut->Enabled = false;
				decayMut->Enabled = false;
				xCenterMut->Enabled = false;
				baseMinBox->Enabled = false;
				baseMaxBox->Enabled = false;
				decMinBox->Enabled = false;
				decMaxBox->Enabled = false;
				xcMinBox->Enabled = false;
				xcMaxBox->Enabled = false;
				baseTrackBar->Value = int((baseTrackBar->Minimum + baseTrackBar->Maximum) / 2.0);
				baseTrackBar->Enabled = false;
				decayTrackBar->Value = int((decayTrackBar->Minimum + decayTrackBar->Maximum) / 2.0);
				decayTrackBar->Enabled = false;
				xCenterTrackBar->Value = int((xCenterTrackBar->Minimum + xCenterTrackBar->Maximum) / 2.0);
				xCenterTrackBar->Enabled = false;
			} else if(BGListview->SelectedIndices->Count > 1) {
				// Multiple Items
				baseLabel->Enabled = true;
				decayLabel->Enabled = true;
				xCenterLabel->Enabled = true;
				baseTextbox->Enabled = false;
				decayTextbox->Enabled = false;
				xCenterTextbox->Enabled = false;
				baseMut->Enabled = true;
				decayMut->Enabled = true;
				xCenterMut->Enabled = true;
				baseMinBox->Enabled = false;
				baseMaxBox->Enabled = false;
				decMinBox->Enabled = false;
				decMaxBox->Enabled = false;
				xcMinBox->Enabled = false;
				xcMaxBox->Enabled = false;
				baseTrackBar->Enabled = true;
				decayTrackBar->Enabled = true;
				xCenterTrackBar->Enabled = true;
				funcTypeList->Enabled = true;
				funcTypeList->SelectedIndex = GetFuncType(BGListview->SelectedItems[0]->SubItems[1]->Text);

				if(!_bGenerateModel) { 
					baseMut->Enabled = true;
					decayMut->Enabled = true;
					xCenterMut->Enabled = true;
				}
			} else {
				// One item selected
				// Everything should be enabled
				removeFuncButton->Enabled = true;
				funcTypeBox->Enabled = true;
				baseLabel->Enabled = true;
				decayLabel->Enabled = true;
				xCenterLabel->Enabled = true;
				baseTextbox->Enabled = true;
				decayTextbox->Enabled = true;
				xCenterTextbox->Enabled = true;
				baseMinBox->Enabled = true;
				baseMaxBox->Enabled = true;
				decMinBox->Enabled = true;
				decMaxBox->Enabled = true;
				xcMinBox->Enabled = true;
				xcMaxBox->Enabled = true;
				baseTrackBar->Enabled = true;
				decayTrackBar->Enabled = true;
				xCenterTrackBar->Enabled = true;

				if(!_bGenerateModel) { 
					baseMut->Enabled = true;
					decayMut->Enabled = true;
					xCenterMut->Enabled = true;
				}
				
				ListViewItem ^lvi = BGListview->SelectedItems[0];
				baseTextbox->Text = lvi->SubItems[2]->Text;
				decayTextbox->Text = lvi->SubItems[4]->Text;
				xCenterTextbox->Text = lvi->SubItems[6]->Text;

				baseMut->Checked = (lvi->SubItems[3]->Text->Equals("Y") ? true : false);
				decayMut->Checked = (lvi->SubItems[5]->Text->Equals("Y") ? true : false);
				xCenterMut->Checked = (lvi->SubItems[7]->Text->Equals("Y") ? true : false);

				baseMinBox->Text = lvi->SubItems[8]->Text;
				baseMaxBox->Text = lvi->SubItems[9]->Text;
				decMinBox->Text = lvi->SubItems[10]->Text;
				decMaxBox->Text = lvi->SubItems[11]->Text;
				xcMinBox->Text = lvi->SubItems[12]->Text;
				xcMaxBox->Text = lvi->SubItems[13]->Text;

				funcTypeList->Enabled = true;
				funcTypeList->SelectedIndex = GetFuncType(lvi->SubItems[1]->Text);

			}

			bool blinear = false;
			for(int i = 0; i < BGListview->SelectedIndices->Count; i++) {
				BGFuncType selectedFunc = GetFuncType(BGListview->SelectedItems[i]->SubItems[1]->Text);
				if(selectedFunc == BG_LINEAR) {
					blinear = true;
				}
			}

			if(blinear) {
				// Deativate the xcenter box
				if(!_bGenerateModel) {
					xcMaxBox->Visible			= false;
					xcMinBox->Visible			= false;
				}
				xCenterLabel->Visible		= false;
				xCenterMut->Visible			= false;
				xCenterTextbox->Visible		= false;
				xCenterTrackBar->Visible	= false;
			} else {
				// Activate the xcenter box
				if(!_bGenerateModel) {
					xcMaxBox->Visible			= true;
					xcMinBox->Visible			= true;
				}
				xCenterLabel->Visible		= true;
				xCenterMut->Visible			= true;
				xCenterTextbox->Visible		= true;
				xCenterTrackBar->Visible	= true;
			}
	}

	/**
	 * Deletes the selected background functions if delete/backspace is pressed
	**/
	void FormFactor::BGListview_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		//If the user selected indices and pressed delete/backspace, remove indicies
		if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back) {
			while(BGListview->SelectedItems->Count > 0)
				removeFuncButton_Click(this, e);

			save->Enabled = true;
			AddToUndoQueue(MT_BACKGROUND, _curBGPar);
		}
	}
	
	/**
	 * Changes the mutabilities in the list based on the checkboxes
	**/
	void FormFactor::BGCheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if (sender == baseMut)
			for(int i  = 0; i < BGListview->SelectedItems->Count; i++)
				BGListview->SelectedItems[i]->SubItems[3]->Text = baseMut->Checked ? "Y" : "N";
		if (sender == decayMut)
			for(int i  = 0; i < BGListview->SelectedItems->Count; i++)
				BGListview->SelectedItems[i]->SubItems[5]->Text = decayMut->Checked ? "Y" : "N";
		if (sender == xCenterMut)
			for(int i  = 0; i < BGListview->SelectedItems->Count; i++)
				BGListview->SelectedItems[i]->SubItems[7]->Text = xCenterMut->Checked ? "Y" : "N";
		//if (sender == modelAmpMut)

		save->Enabled = true;
	}

	/**
	 * Changes the type of background function of the selected items in the list
	**/
	void FormFactor::funcTypeList_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		//roi
		BGFuncType index = (BGFuncType)funcTypeList->SelectedIndex;
		System::String ^text = GetBGFuncString(index);

		for(int i = 0; i < BGListview->SelectedItems->Count; i++)
			BGListview->SelectedItems[i]->SubItems[1]->Text = text;

		switch (index) {
			case BG_LINEAR: // Deativate the xcenter box
				xcMaxBox->Visible			= false;
				xcMinBox->Visible			= false;
				xCenterLabel->Visible		= false;
				xCenterMut->Visible			= false;
				xCenterTextbox->Visible		= false;
				xCenterTrackBar->Visible	= false;
				break;
			default:
			case BG_EXPONENT: // Activate the xcenter box
			case BG_POWER:
				if(!_bGenerateModel) {
					xcMaxBox->Visible			= true;
					xcMinBox->Visible			= true;
				}
				xCenterLabel->Visible		= true;
				xCenterMut->Visible			= true;
				xCenterTextbox->Visible		= true;
				xCenterTrackBar->Visible	= true;
				break;
		}
	
		UpdateGraph();
	}
	void FormFactor::AddBGFunction(BGFuncType type, double amp, char ampMut, 
								   double dec, char decMut, double xc, char xcMut,
								   double ampmin, double ampmax, double decmin, 
								   double decmax, double centermin, double centermax) {
		System::String ^text = GetBGFuncString(type);
		ListViewItem ^lvi;
		BGListview->Items->Add((BGListview->Items->Count + 1).ToString());
		lvi = BGListview->Items[BGListview->Items->Count - 1];
		BGparamErrors->clear();
		BGmodelErrors->clear();


		lvi->SubItems->Add(text);
		lvi->SubItems->Add(amp.ToString("0.000000"));
		lvi->SubItems->Add(gcnew System::String(ampMut, 1));
		lvi->SubItems->Add(dec.ToString("0.000000"));
		lvi->SubItems->Add(gcnew System::String(decMut, 1));
		lvi->SubItems->Add( xc.ToString("0.000000"));
		lvi->SubItems->Add(gcnew System::String(xcMut, 1));

		lvi->SubItems->Add(ampmin.ToString("0.000000"));
		lvi->SubItems->Add(ampmax.ToString("0.000000"));
		lvi->SubItems->Add(decmin.ToString("0.000000"));
		lvi->SubItems->Add(decmax.ToString("0.000000"));
		lvi->SubItems->Add(centermin.ToString("0.000000"));
		lvi->SubItems->Add(centermax.ToString("0.000000"));

	}
	
	System::String ^FormFactor::GetBGFuncString(BGFuncType type) {
		switch (type) {
			default:
			case BG_EXPONENT: // Exponent
				return "Exponent";
			case BG_LINEAR: // Linear Function
				return "Linear";
			case BG_POWER: //Power Function
				return "Power";
		}
	}

	BGFuncType FormFactor::GetFuncType(System::String ^str) {
		if(str->Equals("Exponent"))
			return BG_EXPONENT;
		else if(str->Equals("Linear"))
			return BG_LINEAR;
		else if(str->Equals("Power"))
			return BG_POWER;

		return BG_EXPONENT;
	}

	// Self informative name
	void FormFactor::GetBGFromGUI(bgStruct *BGs) {
		return;	// TODO::Background
		BGs->type.clear();
		BGs->base.clear();
		BGs->decay.clear();
		BGs->center.clear();
		BGs->baseMutable.clear();
		BGs->decayMutable.clear();
		BGs->centerMutable.clear();

		for(int i = 0; i < BGListview->Items->Count; i++) {
			BGs->type.push_back(GetFuncType(BGListview->Items[i]->SubItems[1]->Text));

			BGs->base.push_back(clrToDouble(BGListview->Items[i]->SubItems[2]->Text));
			BGs->decay.push_back(clrToDouble(BGListview->Items[i]->SubItems[4]->Text));
			BGs->center.push_back(clrToDouble(BGListview->Items[i]->SubItems[6]->Text));

			BGs->baseMutable.push_back((char)BGListview->Items[i]->SubItems[3]->Text[0]);
			BGs->decayMutable.push_back((char)BGListview->Items[i]->SubItems[5]->Text[0]);
			BGs->centerMutable.push_back((char)BGListview->Items[i]->SubItems[7]->Text[0]);

			BGs->basemin.push_back(clrToDouble(BGListview->Items[i]->SubItems[8]->Text));
			BGs->basemax.push_back(clrToDouble(BGListview->Items[i]->SubItems[9]->Text));
			BGs->decmin.push_back(clrToDouble(BGListview->Items[i]->SubItems[10]->Text));
			BGs->decmax.push_back(clrToDouble(BGListview->Items[i]->SubItems[11]->Text));
			BGs->centermin.push_back(clrToDouble(BGListview->Items[i]->SubItems[12]->Text));
			BGs->centermax.push_back(clrToDouble(BGListview->Items[i]->SubItems[13]->Text));
		}
	}

	// Self informative name
	void FormFactor::SetBGtoGUI(bgStruct *BGs) {
		
		if(_bLoading) {
			BGListview->Items->Clear();
			for(unsigned int i = 0; i < BGs->base.size(); i++)
				AddBGFunction(BGs->type[i], BGs->base[i], BGs->baseMutable[i], BGs->decay[i],
						  BGs->decayMutable[i], BGs->center[i], BGs->centerMutable[i], 
						  BGs->basemin[i], BGs->basemax[i], BGs->decmin[i], BGs->decmax[i],
						  BGs->centermin[i], BGs->centermax[i]);
		}
		else {
			for(int i = 0; i < BGListview->Items->Count; i++) {
				ListViewItem ^lvi = BGListview->Items[i];
				lvi->SubItems[2]->Text = BGs->base[i].ToString("0.000000");
				lvi->SubItems[4]->Text = BGs->decay[i].ToString("0.000000");
				lvi->SubItems[6]->Text = BGs->center[i].ToString("0.000000");
			}
		}
		BGParameterUpdateHandler();

	}

	void FormFactor::baseline_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring st;
		std::vector <double> x, y, ffy; 
		clrToString(_dataFile, st);

		ffUseCheckBox->Checked = true;
		sfUseCheckBox->Checked = true;
		bgUseCheckBox->Checked = true;

		_ff->tmpY.clear();
		_sf->tmpY.clear();
		_bg->tmpY.clear();

		if(_peakPicker) PeakPicker_Click(sender, e);
		if(_fitToBaseline) fitToBaseline_Click(sender, e);
		// TODO::Baseline
/*
		if(GenerateBackground(st, x, y, ffy, a1nm1ToolStripMenuItem->Checked)) {
			label1->Visible = false;
			exportSignalToolStripMenuItem->Enabled = true;
			exportBackgroundToolStripMenuItem->Enabled = true;
			exportModelToolStripMenuItem1->Enabled = true;
			plotFittingResultsToolStripMenuItem->Enabled = true;
			exportGraphToolStripMenuItem->Enabled = true;
			exportFormFactorToolStripMenuItem->Enabled = true;
			exportSigModBLToolStripMenuItem->Enabled = true;
			exportDecomposedToolStripMenuItem->Enabled = true;
			exportStructureFactorToolStripMenuItem->Enabled = true;
			fitToBaseline->Enabled = true;
			_mask->clear();

	 		InitializeGraph(false, x, y, ffy);
			tabControl1->SelectTab("FFTab");

			ffUseCheckBox->Enabled = true;
			sfUseCheckBox->Enabled = true;
			bgUseCheckBox->Enabled = true;
		}
*/
	}

	void FormFactor::fitToBaseline_Click(System::Object^  sender, System::EventArgs^  e) {
		static std::vector<double> tmpy;
		if(!wgtFit || !wgtFit->graph  || !wgtFit->Visible) return;	//Only works if a model has been fitted
		
		if(_fitToBaseline) { // Was clicked on while fitting to baseline
			std::vector<double> my (_ff->y.size());			

			// Make phase graphs visible
			for(int p = 0; p < (int)graphType->size(); p++)
				if(graphType->at(p) == GRAPH_PEAK)
					wgtFit->graph->SetGraphVisibility(p, true);
			
			fitToBaseline->Text = L"Fit Background to Baseline";
			
			// Change display back to signal and model			
			wgtFit->graph->Modify(0, _ff->x, tmpy);

			MultiplyVectors(my, _ff->y, _sf->y);
			AddVectors(my, my, _bg->y);
			my = MachineResolutionF(_ff->x, my, GetResolution());
			wgtFit->graph->Modify(1, _ff->x, my);
		} else { // Was clicked on while showing model
			// Make phase graphs invisible
			for(int p = 0; p < (int)graphType->size(); p++)
				if(graphType->at(p) == GRAPH_PEAK)
					wgtFit->graph->SetGraphVisibility(p, false);

			
			this->fitToBaseline->Text = L"Fit to Signal";
			tmpy = wgtFit->graph->y[0];

			// Changes signal to baseline
			wgtFit->graph->Modify(0, _ff->x, _baseline->y);
			wgtFit->graph->Modify(1, _bg->x, _bg->y);
		}

		UpdateChisq(WSSR(wgtFit->graph->y[0], wgtFit->graph->y[1]));
		UpdateRSquared(RSquared(wgtFit->graph->y[0], wgtFit->graph->y[1]));

		wgtFit->graph->FitToAllGraphs();

		//Cause the y-axis to be redrawn
		wgtFit->graph->ToggleYTicks();
		wgtFit->graph->ToggleYTicks();

		_fitToBaseline = !_fitToBaseline;
		bgUseCheckBox->Enabled = !_fitToBaseline;

		wgtFit->Invalidate();
	}

	void FormFactor::BGParameterUpdateHandler() {
		if(!_curBGPar)
			return;

		// This function should:
		// 0. Update _curBG

		// TODO::BG What? should be _curBGPar
		//GetBGFromGUI(_curBG);
		
		UItoParameters(_curBGPar, _modelBG, BGListview, nullptr);

		if(_bLoading ^ _bChanging)
			return;
		// 1. If necessary, redraw the graph
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph(_bLoading || _bChanging);		


		if(!_bFromFitter) {
			BGparamErrors->clear();
			BGmodelErrors->clear();
		}
	}

};
