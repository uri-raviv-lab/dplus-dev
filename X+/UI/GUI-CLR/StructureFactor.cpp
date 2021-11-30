#include "FormFactor.h"
#include "ExtractBaseline.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include <sstream>

#ifndef PI
	#define PI 3.141592653589793
#endif

// This file handles the structure factor tab

using namespace System::Windows::Forms;

namespace GUICLR {

	PhaseType _SelectedPhaseType;

	void FormFactor::SetPhaseType( PhaseType a){_SelectedPhaseType=a;}
	
	PhaseType getPhase() { return _SelectedPhaseType;}

	void FormFactor::AddPeak(double amp, char ampMut, double fwhm, char fwhmMut, double center, char centerMut) {
		int nlp = _modelSF->GetNumLayerParams();
		std::vector<Parameter> layer (nlp);
		for(int i = 0; i < nlp; i++)
			layer[i].value = _modelSF->GetDefaultParamValue(i, 
			listView_peaks->Items->Count);

		layer[0].value = amp;
		layer[0].isMutable = ampMut == 'Y';

		layer[1].value = fwhm;
		layer[1].isMutable = fwhmMut == 'Y';

		layer[2].value = center;
		layer[2].isMutable = centerMut == 'Y';

		AddParamLayer(layer, _modelSF, listView_peaks);

		SFparamErrors->clear();
		SFmodelErrors->clear();
	}

	void FormFactor::AddPeak() {
		int nlp = _modelSF->GetNumLayerParams();
		std::vector<Parameter> layer (nlp);
		for(int i = 0; i < nlp; i++)
			layer[i].value = _modelSF->GetDefaultParamValue(i, 
			listView_peaks->Items->Count);

		AddParamLayer(layer, _modelSF, listView_peaks);
	}

	void FormFactor::StructureFactor_Load() {
		order->SelectedIndex = 0;
		if(peakfit->Items->Count > 0)
			peakfit->SelectedIndex = (GetPeakType() > 2) ? 0 : GetPeakType();
		_pressX = _pressY = -1;
	}

	/**
	 * Changes the title of the window based on the current tab
	**/
	void FormFactor::tabControl1_TabIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if(tabControl1->SelectedIndex == 0) { //Moving into FF tab
			if(_fitToBaseline)
				fitToBaseline_Click(sender, e);
			if(_peakPicker) // If we're in PeakPicker mode, leave mode
				PeakPicker_Click(sender, e);
			if(_fitToBaseline) fitToBaseline_Click(sender, e);
			if(this->Text->Contains("Structure Factor"))
				this->Text = this->Text->Replace("Structure Factor", "Form Factor");
			else if(this->Text->Contains("Background"))
				this->Text = this->Text->Replace("Background", "Form Factor");
			if(wgtFit) {
				for(int i = 0; i < (int)graphType->size(); i++)
					if(graphType->at(i) == GRAPH_PEAK)
						wgtFit->graph->SetGraphVisibility(i, false);
				wgtFit->Invalidate();
			}
		
		} else if(tabControl1->SelectedIndex == 1) { //Moving into SF tab
			if(_fitToBaseline)
				fitToBaseline_Click(sender, e);
			if(this->Text->Contains("Form Factor"))
				this->Text = this->Text->Replace("Form Factor", "Structure Factor");
			else if(this->Text->Contains("Background"))
				this->Text = this->Text->Replace("Background", "Structure Factor");
			if(wgtFit) {
				for(int i = 0; i < (int)graphType->size(); i++)
					if(graphType->at(i) == GRAPH_PEAK)
						wgtFit->graph->SetGraphVisibility(i, true);
				wgtFit->Invalidate();
			}
		
		} else if(tabControl1->SelectedIndex == 2) { //Moving into BG tab
			if(_peakPicker) // If we're in PeakPicker mode, leave mode
				PeakPicker_Click(sender, e);
			if(this->Text->Contains("Form Factor"))
				this->Text = this->Text->Replace("Form Factor", "Background");
			else if(this->Text->Contains("Structure Factor"))
				this->Text = this->Text->Replace("Structure Factor", "Background");
			if(wgtFit) {
				for(int i = 0; i < (int)graphType->size(); i++)
					if(graphType->at(i) == GRAPH_PEAK)
						wgtFit->graph->SetGraphVisibility(i, false);
				wgtFit->Invalidate();
			}
		}
	}

	void FormFactor::addPeak_Click(System::Object^  sender, System::EventArgs^  e) {
		AddPeak();
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();

		save->Enabled = true;
		AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);
	}

	void FormFactor::peaksForPhases_Click(System::Object^  sender, System::EventArgs^  e) {
		ListViewItem ^lvi = listView_peaks->Items[listView_peaks->Items->Count - 1];
		phaseStruct p;
		std::vector<double> peakPositions;

		for(int i = 0; i < listView_PeakPosition->Items->Count; i++)
			peakPositions.push_back(clrToDouble(listView_PeakPosition->Items[i]->Text));

		GetPhasesFromListView(&p);

		AddPeak(clrToDouble(lvi->SubItems[1]->Text), 'Y', 
			    clrToDouble(lvi->SubItems[3]->Text), 'Y', 
				1.0, 'Y');

		listView_PeakPosition->Items->Add(listView_peaks->Items[listView_peaks->Items->Count - 1]->SubItems[5]->Text);
	}

	void FormFactor::removePeak_Click(System::Object^  sender, System::EventArgs^  e) {
		while(listView_peaks->SelectedItems->Count > 0)
			listView_peaks->Items->Remove(listView_peaks->SelectedItems[0]);

		for(int i = 0; i < listView_peaks->Items->Count; i++)
			listView_peaks->Items[i]->Text = "Peak " + (i + 1).ToString();

		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph();

		save->Enabled = true;
		AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);
	}

	void FormFactor::PeakPicker_Click(System::Object^  sender, System::EventArgs^  e) {
		static std::vector<double> signaly;
		vector<double> my;
		
		if(!wgtFit || !wgtFit->graph  || !wgtFit->Visible) return;	//Only works if a model has been fitted
		
		if(_peakPicker) { // Was clicked on while using peak picker
			// Make phase graphs visible
			for(int p = 0; p < (int)graphType->size(); p++)
				if(graphType->at(p) == GRAPH_PEAK)
					wgtFit->graph->SetGraphVisibility(p, true);
			/**
			 * Steps:	1) the form factor vector (?)
			 *			2) re-instate the signal data vector
			 *			3) update the graph
			**/
			this->PeakPicker->Text = L"Peak Finder";
			PeakFinderCailleButton->Text = "Peak Finder";
			/*
			change display back to signal and model
			*/
			
			MultiplyVectors(my, _ff->y, _sf->y);
			AddVectors(my, my, _bg->y);
			my = MachineResolutionF(_ff->x, my, GetResolution());
			
			wgtFit->graph->Modify(0, _sf->x, signaly);
			wgtFit->graph->Modify(1, _sf->x, my);
		}
		else { // Was clicked on while showing model
			// Make phase graphs invisible
			for(int p = 0; p < (int)graphType->size(); p++)
				if(graphType->at(p) == GRAPH_PEAK)
					wgtFit->graph->SetGraphVisibility(p, false);

			/**
			 * Steps:	1) backup the vector containing the signal data
			 *			2) create a vector containg the signal data divided by the form factor
			 *			3) display the new vector on the graph
			**/
			this->PeakPicker->Text = L"Show Signal";
			PeakFinderCailleButton->Text = L"Show Signal";
			signaly = wgtFit->graph->y[0];
			
			SubtractVectors(my, signaly, _bg->y);
			DivideVectors(my, my, _ff->y);
			my = MachineResolutionF(_ff->x, my, GetResolution());
			
			wgtFit->graph->Modify(0, _sf->x, my);
			wgtFit->graph->Modify(1, _sf->x, _sf->y);
		}

		UpdateChisq(WSSR(wgtFit->graph->y[0], wgtFit->graph->y[1]));
		UpdateRSquared(RSquared(wgtFit->graph->y[0], wgtFit->graph->y[1]));

		//Cause the y-axis to be redrawn
		wgtFit->graph->ToggleYTicks();
		wgtFit->graph->ToggleYTicks();

		_peakPicker = !_peakPicker;
		wgtFit->Invalidate();
	}

	void FormFactor::wgtFit_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		if(!wgtFit || !wgtFit->graph)
			return;

		std::pair<double, double> loc;
		if(e->Button == System::Windows::Forms::MouseButtons::Left && (_bMasking || tabControl1->SelectedIndex == 1)) {
			loc = wgtFit->graph->PointToData(e->X-wgtFit->graph->xoff, e->Y-wgtFit->graph->yoff);

			if(wgtFit->graph->LogScaleX())
				_pressX = pow(10, loc.first);
			else
				_pressX = loc.first;

			if(wgtFit->graph->LogScaleY())
				_pressY = pow(10, loc.second);
			else
				_pressY = loc.second;
		}

		if(tabControl1->SelectedIndex == 1 && !_bMasking) {
			if((_modelSF->GetMaxLayers() > -1 && _curSFPar->layers >= _modelSF->GetMaxLayers()) || !sfUseCheckBox->Checked) {
				_pressY = -1.0;
				_pressX = -1.0;
				return;
			}
			if(!listView_peaks->Enabled)
				return;

			_Xdown = e->X;
			_Ydown = e->Y;

		} else if(_bMasking) {
			_Xdown = e->X;
			_Ydown = e->Y;
		}
	}


	void FormFactor::wgtFit_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		if(_state != FWS_IDLE)
			return;
		if(!wgtFit || !wgtFit->graph)
			return;

		// When drawing peaks for the SF, indicate the size of the peaks
		if(tabControl1->SelectedIndex == 1 && _pressX > 0 && !_bMasking && sfUseCheckBox->Checked) {
			wgtFit->graph->SetDrawPeak(true);
			wgtFit->graph->DrawPeakOutline(_Xdown, _Ydown, e->X, e->Y);
		}

		// When masking, draw the "crop" lines
		if(_bMasking && _pressX > 0) {
			if(_bMasking, _Xdown > 0)
				wgtFit->graph->SetCropping(true, true);
			else {
				wgtFit->graph->SetCropping(false, false);
				return;
			}
			wgtFit->graph->DrawPeakOutline(_Xdown, 2, e->X, 3);
		}
		 
		std::pair<double, double> loc;
		loc = wgtFit->graph->PointToData(e->X - wgtFit->graph->xoff, e->Y - wgtFit->graph->yoff);

		std::vector<double> x = wgtFit->graph->x[0];
		std::vector<double> y = wgtFit->graph->y[0];

		if(y.empty() || x.empty()) return;

		if(wgtFit->graph->LogScaleX())
			loc.first = pow(10.0, loc.first);

		int pos = x.size() - 1;
		for (unsigned int i = 1; i < x.size(); i++)
			if (x[i] > loc.first) { pos = i - 1; break; }

		LocOnGraph->Text = "("+ Double(x[pos]).ToString() + ","+ Double(y[pos]).ToString()+ ")";
		wgtFit->Invalidate();

	}


	void FormFactor::wgtFit_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		if(!listView_peaks->Enabled && !_bMasking)
				return;
		if(wgtFit->graph->DrawPeak())
			wgtFit->graph->SetDrawPeak(false);

		if(tabControl1->SelectedIndex == 1 && !_bMasking) {
			if(!wgtFit || !wgtFit->graph || !(sfUseCheckBox->Checked))
				 return;
			if(_modelSF->GetMaxLayers() > -1 && _curSFPar->layers >= _modelSF->GetMaxLayers())
				return;
			if(e->Button == System::Windows::Forms::MouseButtons::Left && _pressX >= 0) {
				std::pair<double, double> loc;
				loc = wgtFit->graph->PointToData(e->X - wgtFit->graph->xoff, e->Y - wgtFit->graph->yoff);

				if(wgtFit && wgtFit->graph) {
					if(wgtFit->graph->LogScaleX())
						loc.first = pow(10, loc.first);
					if(wgtFit->graph->LogScaleY())
						loc.second = pow(10, loc.second);
				}

				double deltaX = _pressX - loc.first;
				double deltaY = _pressY - loc.second;

				std::vector<double> x = wgtFit->graph->x[0];
				//std::vector<double> y = wgtFit->graph->y[1];

				if(/*y.empty() || */x.empty()) return;

				int pos=x.size()-1;

				for (unsigned int i=1; i<x.size(); i++)
					if (x[i]>_pressX) { pos=i-1; break; }
			
				double fw = fabs(deltaX); //full width

				if(!(fw > 0.0) || !(fabs(deltaY) > 0.0)) { //Shape cannot be drawn and leads to SF = 0
					_pressX = _pressY = -1;
					wgtFit->graph->SetDrawPeak(false);
					return;
				}


				if(!_peakPicker)  //If we need to divide by the FF...
					deltaY /= (_ff->y.at(pos) > 0.0 ? _ff->y.at(pos) : 1.0e-9);

				double rtPln2 = sqrt(PI / log(2.0));
				switch(GetPeakType()){
					default:
					case SHAPE_GAUSSIAN:
						if(sigmaToolStripMenuItem->Checked)
							AddPeak(fabs(deltaY) * (0.5 * fw * rtPln2), 'Y', fw / 2.355, 'Y', _pressX, 'Y');
						else
							AddPeak(fabs(deltaY) * (0.5 * fw * rtPln2), 'Y', fw, 'Y', _pressX, 'Y');
						break;
					case SHAPE_LORENTZIAN:
						AddPeak(fabs(deltaY) * (fw * PI / 2.0 ), 'Y', fw, 'Y', _pressX, 'Y');
						break;
					case SHAPE_LORENTZIAN_SQUARED:
						AddPeak(fabs(deltaY)  * (fw * PI / 4.0 ), 'Y', fw, 'Y', _pressX, 'Y');
						break;
				}

				
				if(liveRefreshToolStripMenuItem->Checked)
					UpdateGraph();

				wgtFit->graph->SetDrawPeak(false);
				wgtFit->Invalidate();

				save->Enabled = true;
				
				_pressX = _pressY = -1;
			}
		} else if(_bMasking) {
			// Stop drawing "crop" lines
			wgtFit->graph->SetCropping(false, false);
			
			//the point to which we are pointing at in the q & I coordinates.
			std::pair <double, double> point = wgtFit->graph->PointToData(e->X - wgtFit->graph->xoff, e->Y - wgtFit->graph->yoff);
			point.first = (wgtFit->graph->LogScaleX()) ? pow(10, point.first) : point.first;
			point.second = (wgtFit->graph->LogScaleY()) ? pow(10, point.second) : point.second;
			int Xup = GUICLR::ExtractBaseline::PosToDataIndex(point.first, wgtFit->graph->x[0]);

			point = wgtFit->graph->PointToData(_Xdown - wgtFit->graph->xoff, _Ydown - wgtFit->graph->yoff);
			point.first = (wgtFit->graph->LogScaleX()) ? pow(10, point.first) : point.first;
			point.second = (wgtFit->graph->LogScaleY()) ? pow(10, point.second) : point.second;
			int Xdown = GUICLR::ExtractBaseline::PosToDataIndex(point.first, wgtFit->graph->x[0]);

			// we find the cropping margins and zone
			int left = min(Xdown, Xup), right = max(Xdown, Xup);
			if(_mask->empty())
				_mask->resize(_ff->x.size(), false);
			// TODO::Mask crop _mask with the other _vectors
			for(int p = left; p < right; p++)
				_mask->at(p) = _bAddMask;

			maskPanel_Click(sender, e);

			_Xdown = -1;   

			_pressX = _pressY = -1;
		}
	}

	void FormFactor::centerTrackBar(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		TrackBar ^track = (TrackBar ^)sender;
		track->Value = int((track->Maximum - track->Minimum) / 2.0);
	}

	// Copies the *peaks to listView_peaks
	void FormFactor::SetPeaks(paramStruct *peaks) {
		for(int i = 0; i < listView_peaks->Items->Count; i++) {
			for(int j = 0; j < peaks->nlp; j++)
				listView_peaks->Items[i]->SubItems[LV_VALUE(j)]->Text = (peaks->params[j][i]).value.ToString("0.000000");		
		}
		SFParameterUpdateHandler();
	}

	void FormFactor::peakfit_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if(peakfit->SelectedIndex < 0 && peakfit->Items->Count > 0) {
			peakfit->SelectedIndexChanged -= gcnew System::EventHandler(this, &FormFactor::peakfit_SelectedIndexChanged);
			peakfit->SelectedIndex = 0;
			peakfit->SelectedIndexChanged += gcnew System::EventHandler(this, &FormFactor::peakfit_SelectedIndexChanged);
		}
		modelInfoObject^ ob = (modelInfoObject^)(peakfit->SelectedItem);
		_mioSF = ob;

		// TODO::SF handle model change fully
		if(sender != nullptr) {
			delete _modelSF;
			_modelSF = new ModelUI();

			PrepareUISection(listView_peaks, SFGroupBoxList, _modelSF);
		}

/*
		switch(peakfit->SelectedIndex) {
			default:
			case 0:
				SetPeakType(SHAPE_GAUSSIAN);
				break;
			case 1:
				SetPeakType(SHAPE_LORENTZIAN);
				break;
			case 2:
				SetPeakType(SHAPE_LORENTZIAN_SQUARED);
				break;
		}
		sigmaFWHMToolStripMenuItem->Visible = GetPeakType() == SHAPE_GAUSSIAN;
		if(GetPeakType() != SHAPE_GAUSSIAN) {
			listView_peaks->Columns[3]->Text = "FWHM";
			label12->Text = "FWHM";
		}
		else
			sigmaToolStripMenuItem_CheckedChanged(sender, e);
*/
		//parameterListView_SelectedIndexChanged(sender, e); TODO::SF Why do we need this here?
		if(liveRefreshToolStripMenuItem->Checked && sender != nullptr)
			UpdateGraph();

		// TODO::PS: Modify in _curSFPar

		save->Enabled = true;
	}

	void FormFactor::scrollerTimer_Tick(System::Object^  sender, System::EventArgs^  e) {
		ExtraParam absDesc ("N/A", 0.0, false, true);
		if(_bLoading)
			return;
		bool bChanged = false;
		double mid = 0.0;

		if(tabControl1->SelectedIndex == 1) { // In SF tab
			if(SFGroupBoxList) {
				for(int j = 0; j < SFGroupBoxList->Count; j++) {
					TrackBar ^tb = SFGroupBoxList[j]->track;
					TextBox ^tex = SFGroupBoxList[j]->text;
					mid = (tb->Maximum - tb->Minimum) / 2.0;

					for(int i = 0; i < listView_peaks->SelectedItems->Count; i++) {
						if(tb->Value != mid){
							dealWithTrackBar(listView_peaks->SelectedItems[i], 
								SFGroupBoxList[j]->rValue->Checked
								? LV_VALUE(j)
								: LV_SIGMA(j, _modelSF->GetNumLayerParams()),
								tb, tex, 
								75.0, absDesc);						

							linkedParameterChangedCheck(listView_peaks->Items, listView_peaks->SelectedIndices[i]);
							bChanged = true;
						}
					} // end for listViewFF

				}
			}

			if(bChanged) {
				save->Enabled = true;
				bool created = !_curSFPar;
				if(!_curSFPar)
					_curSFPar = new paramStruct(_modelSF->GetModelInformation());
				UItoParameters(_curSFPar, _modelSF, listView_peaks, nullptr);
				SFParameterUpdateHandler();
				if(created) {
					delete _curSFPar;
					_curSFPar = NULL;
				}

			}

		} else if(tabControl1->SelectedIndex == 2) { // In BG tab
			for(int i = 0; i < BGListview->SelectedItems->Count; i++) {
				mid = (baseTrackBar->Maximum - baseTrackBar->Minimum) / 2.0;
				if(baseTrackBar->Value != mid) {
					dealWithTrackBar(BGListview->SelectedItems[i], 2, baseTrackBar, baseTextbox, 
						75.0, absDesc);
					bChanged = true;
				}

				mid = (decayTrackBar->Maximum - decayTrackBar->Minimum) / 2.0;
				if(decayTrackBar->Value != mid) {
					dealWithTrackBar(BGListview->SelectedItems[i], 4, decayTrackBar, decayTextbox, 
						75.0, ExtraParam("N/A"));
					bChanged = true;
				}

				mid = (xCenterTrackBar->Maximum - xCenterTrackBar->Minimum) / 2.0;
				if(xCenterTrackBar->Value != mid) {
					dealWithTrackBar(BGListview->SelectedItems[i], 6, xCenterTrackBar, xCenterTextbox, 
						75.0, ExtraParam("N/A"));
					bChanged = true;
				}

				if(bChanged)
						BGParameterUpdateHandler();
			} // end for BGListview
		} else if(tabControl1->SelectedIndex == 0) {	// FormFactor tab

			if(FFGroupBoxList) {
				for(int j = 0; j < FFGroupBoxList->Count; j++) {
					TrackBar ^tb = FFGroupBoxList[j]->track;
					TextBox ^tex = FFGroupBoxList[j]->text;
					mid = (tb->Maximum - tb->Minimum) / 2.0;

					for(int i = 0; i < listViewFF->SelectedItems->Count; i++) {
						if(tb->Value != mid){
							dealWithTrackBar(listViewFF->SelectedItems[i], 
								FFGroupBoxList[j]->rValue->Checked
								? LV_VALUE(j)
								: LV_SIGMA(j, _modelFF->GetNumLayerParams()),
								tb, tex, 
								75.0, absDesc);						

							linkedParameterChangedCheck(listViewFF->Items, listViewFF->SelectedIndices[i]);
							bChanged = true;
						}
					} // end for listViewFF

					if(bChanged) {
						save->Enabled = true;
						FFParameterUpdateHandler();
					}
				}
			}

			if(paramBox->Items->Count > 0) {
				mid = (exParamGroupbox->track->Maximum - exParamGroupbox->track->Minimum) / 2.0;
				if(exParamGroupbox->track->Value != mid) {
					dealWithTrackBar(listView_Extraparams->Items[paramBox->SelectedIndex],
						exParamGroupbox->rValue->Checked ? ELV_VALUE : ELV_SIGMA ,
						exParamGroupbox->track, exParamGroupbox->text, 
						75.0, _modelFF->GetExtraParameter(paramBox->SelectedIndex));
						
					save->Enabled = true;
					FFParameterUpdateHandler();
		
				}
			}			
		} // end else if formFactor tab
	} //end scrollerTimer_Tick

	
	void FormFactor::Move2Phases_Click(System::Object^  sender, System::EventArgs^  e) {
		
		clearPhases_Click(sender, e);
		
		//Don't add peaks to listView or generate other stuff
		if(listView_peaks->SelectedIndices->Count > 0) {
			fitphase->Enabled = (wgtFit && wgtFit->graph);
		}

		for (int i=0; i < listView_peaks->SelectedIndices->Count; i++) {
			listView_PeakPosition->Items->Add(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[5]->Text);
			
			// Add a new circles graph representing the peak
			if(wgtFit && wgtFit->graph) {
				int srt = -1, nd = -1;
				double lft, rgt;
				lft = clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[5]->Text) - 1.75 * clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[3]->Text);
				rgt = clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[5]->Text) + 1.75 * clrToDouble(listView_peaks->Items[listView_peaks->SelectedIndices[i]]->SubItems[3]->Text);
				for (int hrt = 1; hrt < (int)wgtFit->graph->x->size(); hrt++) {
					if (srt < 0 && wgtFit->graph->x->at(hrt) > lft)
						srt = hrt - 1;
					if (wgtFit->graph->x->at(hrt) > rgt) {
						nd = hrt;
						break;
					}
				}

				// In case the user has recropped the data
				if(nd < 0 || srt < 0) 
					continue;

				std::vector<double> tmx = wgtFit->graph->x[_bGenerateModel ? 0 : 1], tmy = wgtFit->graph->y[_bGenerateModel ? 0 : 1];
				tmx.erase(tmx.begin() + nd, tmx.end());
				tmx.erase(tmx.begin(), tmx.begin() + srt);
				tmy.erase(tmy.begin() + nd, tmy.end());
				tmy.erase(tmy.begin(), tmy.begin() + srt);

				graphType->push_back(GRAPH_PEAK);
				wgtFit->graph->Add(RGB(51, 10, 102), DRAW_FULL_SCATTER, tmx, tmy);
				wgtFit->Invalidate();
			}
		}
		SortListView(listView_PeakPosition,0,false);

		// Generates the phase's peaks. TODO::phases Cleanup
		phaseStruct curPhase;
		std::vector<std::string> indices_locs;
		GetPhasesFromListView(&curPhase);
		std::vector <double> genPhase = GenPhases(GetPhase(), &curPhase, indices_locs);
		int c;

		for(c = 0; c < (int)indices_locs.size(); c++)
			if(genPhase[c] > curPhase.qmax)
				break;																				
			
		genPhase.erase(genPhase.begin() + c, genPhase.end());
		indices_locs.erase(indices_locs.begin() + c, indices_locs.end());
		
		*_generatedPhaseLocs = genPhase;
		*indicesLoc = indices_locs;

		PhasesCompleted();
		// END Generate phases
	}
	
	void FormFactor::UpdateGeneratedPhasePeaks() {
		if(indicesLoc->size() > 0) {
			while(listView_PeakPosition->Items->Count < (int)indicesLoc->size()) {
				listView_PeakPosition->Items->Add("N/A");
				listView_PeakPosition->Items[listView_PeakPosition->Items->Count - 1]->SubItems->Add("");
				listView_PeakPosition->Items[listView_PeakPosition->Items->Count - 1]->SubItems->Add("");
			}	

			for(int i = indicesLoc->size() ; i< listView_PeakPosition->Items->Count; i++) {
				while (listView_PeakPosition->Items[i]->SubItems->Count > 1)  			
					listView_PeakPosition->Items[i]->SubItems->RemoveAt(1);
			}

			for (int cn = 0; cn < (int)indicesLoc->size(); cn++) {
				while(listView_PeakPosition->Items[cn]->SubItems->Count < 3)
					listView_PeakPosition->Items[cn]->SubItems->Add("");
			
				listView_PeakPosition->Items[cn]->SubItems[1]->Text = (gcnew String(indicesLoc->at(cn).c_str()));
				listView_PeakPosition->Items[cn]->SubItems[2]->Text = (_generatedPhaseLocs->at(cn).ToString("0.000000"));
			}

		}

		indicesLoc->clear();
		_generatedPhaseLocs->clear();
		_ph->clear();
		*phaseSelected = PHASE_NONE;
		undoPhases->Enabled = true;
	}


	void FormFactor::fitphase_Click(System::Object^  sender, System::EventArgs^  e) {
		// TODO::NewFitter , TODO::Phases
		MessageBox::Show("TODO::Phases");
		/*
		static phaseStruct OldPhase;
				
		_bFitPhase = true;
		if(sender == undoPhases) {
			std::vector<double> mx, my;
			SetPhases(&OldPhase);
			save->Enabled = true;
			
			// Generates the last phase's peaks. TODO::Phases Cleanup
			std::vector<std::string> indices_locs;
			std::vector <double> genPhase = GenPhases(GetPhase(), &OldPhase, indices_locs);
		
			PhasesCompleted();

			// END Generate last phases

			UpdateGraph();
			undoPhases->Enabled = false;
			return;
		}

		this->Cursor = System::Windows::Forms::Cursors::AppStarting;
		for (int i = 0 ; i < this->Controls->Count ; i++) { 
			this->Controls[i]->AllowDrop=this->Controls[i]->Enabled;
			this->Controls[i]->Enabled=false;
		}
		this->listView_phases->SelectedIndices->Clear();
		this->label6->Visible = true;
		this->progressBar1->Visible = true;
		progressBar1->Value = 0;
					
		if(!_bGenerateModel && !wgtFit->Visible) {
			wgtFit->Visible = true;
			LocOnGraph->Visible = true;
			wssr->Visible = true;
			rsquared->Visible = true;
		}
		if(!_curPhases)
			_curPhases = new phaseStruct;
		GetPhasesFromListView(_curPhases);
		OldPhase = *_curPhases;
		fitphase->Enabled = false;
		Caille_button->Enabled = false;
		calculate->Text = "Stop";
		calculate->Enabled = true;
		tableLayoutPanel1->Enabled = true;
		panel3->Enabled = true;
		std::vector <double> locs;
		for (int yy = 0; yy < listView_PeakPosition->Items->Count; yy++) 
			if(!listView_PeakPosition->Items[yy]->SubItems[0]->Text->Contains("N"))// Equals("N/A"))
				locs.push_back(clrToDouble(listView_PeakPosition->Items[yy]->SubItems[0]->Text));
		*_ph = locs;
		*phaseSelected = GetPhase();


		// Disabling what needs to be disabled
		save->AllowDrop = true;
		save->Enabled = false;
		changeData->AllowDrop = changeData->Enabled;
		changeData->Enabled = false;
		panel2->Enabled = false;
		Caille_button->Enabled = false;
		undoPhases->Enabled = false;
		fitterThread = gcnew Threading::Thread(gcnew Threading::ParameterizedThreadStart(this, &FormFactor::modelFitter_threadFunc));
		bThreadSuspended = false;
		*_pShouldStop = 0;
		fitterThread->Start(gcnew Int32(tabControl1->SelectedIndex));
		*/
	}

	void FormFactor::listView_peaks_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		//If the user selected indices and pressed delete/backspace, remove indicies
		if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back) {
			while(listView_peaks->SelectedItems->Count > 0)
				removePeak_Click(this, e);

			save->Enabled = true;
		}
	}

	void FormFactor::AutoFindPeaks() {
		/* STEPS:
		1) Obtain "SF" by dividing the signal by the calculated FF -> remaining_signal
		2) Calculate the WSSR of the remaining_signal and the existing SF (from user input)
		3) Determine segment size (minimum of one point)
		4) Find peaks
			For each segment:
			a) Sum the difference between existing SF and remaining_signal for all points in the segment 
			b) Calculate the WSSR of the segment between the remaining_signal and the existing SF
			c) If sum is not positive
				Move on to the next segment and continue from "a"
			d) If WSSR of the segment is greater or equal to threshhold2 * (the initial WSSR)
				Find peak parameters and add a peak
				i) Flag that a peak is to be added
			e) If the previous segments WSSR is greater than the current segments WSSR (we've passed the peak's peak)
				i)   Zero in on the local maximum within neighboring segments and mark as Xc
				ii)  Mark the amplitude of the peak as the difference between remaining_signal and existing SF
				iii) Find FWHM by "sliding" backwards until HM is found
				iv)  Add peak with the above parameters to existing SF
				v)   Reset flag to not add a peak
				vi)  Set previous WSSR to 0.0
				vii) Move on to the next segment and continue from "a"
			f) Set previous segment WSSR to current segemnt WSSR
			g) Set current segment WSSR to 0.0
			h) Move on to the next segment and continue from "a"
		*/
		if (wgtFit->graph->x[0].size() != _ff->y.size() || wgtFit->graph->x[0].size() != _sf->y.size())
			return;

		vector<double> remaining_signal;
		//remaining_signal becomes signal divided by the form factor
		if(!_peakPicker) {
			//addValueToVector(wgtFit->graph->y[0], -Double::Parse(listView_Extraparams->Items[1]->SubItems[1]->Text));
			//addValueToVector(_ff->y, -Double::Parse(listView_Extraparams->Items[1]->SubItems[1]->Text));
			SubtractVectors(remaining_signal, wgtFit->graph->y[0], _bg->y);
			DivideVectors(remaining_signal, remaining_signal, _ff->y);
			//addValueToVector(_ff->y, Double::Parse(listView_Extraparams->Items[1]->SubItems[1]->Text));
			//addValueToVector(wgtFit->graph->y[0], Double::Parse(listView_Extraparams->Items[1]->SubItems[1]->Text));
		}
		else
			remaining_signal = wgtFit->graph->y[0];
		smoothVector(2, remaining_signal);	
		double initialWssr = WSSR(_sf->y, remaining_signal), sum = 0.0, val = 0.0, prev_segChiSqr = 0.0, segChiSqr = 0.0;
		int segment = int(threshold1 * wgtFit->graph->x[1].size());
		if(segment == 0)
			segment = 1;
		bool b_addPeak = false;
		int cnt = 0, first_pos = 1;

		while (unsigned(cnt + first_pos) < wgtFit->graph->x[1].size()) {
			for (cnt = 1; cnt <= segment; cnt++) {
				val = remaining_signal.at(first_pos + cnt) - _sf->y.at(first_pos + cnt);
				segChiSqr += val * val / 
					((sqrt(remaining_signal.at(first_pos + cnt)) + 1.0) * (sqrt(_sf->y.at(first_pos + cnt)) + 1.0));
				sum += val;
				// TODO::SF Check if segment is mostly noise (delta-correlation)
				//	if (noise) return/continue;
			}
			if (!(sum > 0.0)) {
				first_pos += cnt;
				segChiSqr = 0.0;
				prev_segChiSqr = 0.0;
				sum = 0.0;
				//b_addPeak = false;	//AVI: I think this should (maybe) be removed...
				continue;
			}
			sum = 0.0;
			if (segChiSqr >= threshold2 * initialWssr) b_addPeak = true;
			if (b_addPeak) {
				if (prev_segChiSqr > segChiSqr) {
					//find maximum point -> peak center and height
					double height, width, center;
					int i, maxPos = first_pos - segment;
					// Find the absolute maximum
					for(i = first_pos - segment + 1; i < first_pos + segment; i++)
						if (remaining_signal.at(i) - _sf->y.at(i)> remaining_signal.at(maxPos) - _sf->y.at(maxPos))
							maxPos = i;
					center = wgtFit->graph->x[0].at(maxPos);
					height = remaining_signal.at(maxPos) - _sf->y.at(maxPos);
					//find HWHM
					for (i = first_pos - 1; (i > 0) && (remaining_signal.at(i) - _sf->y[i] > 0.5 * height); i--);
					width = center - wgtFit->graph->x[0].at(i);
					
					//Add peak
					//Ensure that the peak makes sense
					if(height > 0.0 && width > 0.0)
						AddPeak(height* (width * sqrt(2.0*PI)), 'Y', width, 'Y', center, 'Y');
					if(liveRefreshToolStripMenuItem->Checked)
						UpdateGraph();
					save->Enabled = true;
					AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);

					b_addPeak = false;
					prev_segChiSqr = 0.0;
					segChiSqr = 0.0;
					first_pos += cnt;
					continue;					
				}	// end if (prev_segChiSqr > segChiSqr)
			}	//end if (b_addPeak)
			first_pos += cnt;
			prev_segChiSqr = segChiSqr;
			segChiSqr = 0.0;
		}	//end while
	}

	
		
		
	void FormFactor::phaseorder_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e){	
		std::vector <int> enabled;
		int d;
		switch(order->SelectedIndex) {
			default:
			case 0:
				d=3;
				SetPhaseType(PHASE_NONE);
				break;
			case 1:
				d=1;
				SetPhaseType(PHASE_LAMELLAR_1D);
				enabled.push_back(0);
				break;
			case 2:
				d=2;
				SetPhaseType(PHASE_2D);
				enabled.push_back(0);
				enabled.push_back(1);
				enabled.push_back(2);
				break;
			case 3:
				d=2;
				SetPhaseType(PHASE_RECTANGULAR_2D);
				enabled.push_back(0);
				enabled.push_back(1);
				break;
			case 4:
				d=2;
				SetPhaseType(PHASE_CENTERED_RECTANGULAR_2D);
				enabled.push_back(0);
				enabled.push_back(1);
				break;
			case 5:
				d=2;
				SetPhaseType(PHASE_SQUARE_2D);
				enabled.push_back(0);
				break;
			case 6:
				d=2;
				SetPhaseType(PHASE_HEXAGONAL_2D);
				enabled.push_back(0);
				break;
			case 7:
				d=3;
				SetPhaseType(PHASE_3D);
				enabled.push_back(0);
				enabled.push_back(1);
				enabled.push_back(2);
				enabled.push_back(3);
				enabled.push_back(4);
				enabled.push_back(5);
				enabled.push_back(6);
				break;
			case 8:
				d=3;
				SetPhaseType(PHASE_RHOMBO_3D);
				enabled.push_back(0);
				enabled.push_back(2);
				break;
			case 9:
				d=3;
				SetPhaseType(PHASE_HEXA_3D);
				enabled.push_back(0);
				enabled.push_back(3);
				break;
			case 10:
				d=3;
				SetPhaseType(PHASE_MONOC_3D);
				enabled.push_back(0);
				enabled.push_back(1);
				enabled.push_back(3);
				enabled.push_back(4);
				break;
			case 11:
				d=3;
				SetPhaseType(PHASE_ORTHO_3D);
				enabled.push_back(0);
				enabled.push_back(1);
				enabled.push_back(3);
				break;
			case 12:
				d=3;
				SetPhaseType(PHASE_TETRA_3D);
				enabled.push_back(0);
				enabled.push_back(3);
				break;
			case 13:
				d=3;
				SetPhaseType(PHASE_CUBIC_3D);
				enabled.push_back(0);
				break;


		}
		
		int j = 0;
		for (int i = 0; i < listView_phases->Items->Count; i++) {
			if (j < (int)enabled.size() && enabled[j] == i) {
				listView_phases->Items[i]->SubItems[6]->Text = "1";
				if(fabs(clrToDouble(listView_phases->Items[i]->SubItems[1]->Text)) < 1e-7){
					if( i == 0 || i == 1 || i == 3) {
						listView_phases->Items[i]->SubItems[1]->Text = (1.0).ToString("0.000000");
						listView_phases->Items[i]->SubItems[3]->Text = (0.0).ToString("0.000000");
						listView_phases->Items[i]->SubItems[4]->Text = (100.0).ToString("0.000000");
					}
					else {
						listView_phases->Items[i]->SubItems[1]->Text = (90.0).ToString("0.000000");
						listView_phases->Items[i]->SubItems[3]->Text = (0.0).ToString("0.000000");
						listView_phases->Items[i]->SubItems[4]->Text = (180.0).ToString("0.000000");
					}
				}
				j++;
			}
			else {
				listView_phases->Items[i]->SubItems[6]->Text = "0";				
			}
		}
		for (int i = d * d - d + 1; i<listView_phases->Items->Count; i++ )
		{
			listView_phases->Items[i]->SubItems[1]->Text = (0.0).ToString("0.000000");
			listView_phases->Items[i]->SubItems[3]->Text = (0.0).ToString("0.000000");
			listView_phases->Items[i]->SubItems[4]->Text = (0.0).ToString("0.000000");
		}
		if (listView_phases->SelectedItems->Count > 0) {
			ValPhases->Text =  listView_phases->SelectedItems[0]->SubItems[1]->Text;
			MinPhases->Text =  listView_phases->SelectedItems[0]->SubItems[3]->Text;
			MaxPhases->Text =  listView_phases->SelectedItems[0]->SubItems[4]->Text;

		}
		listView_phases_SelectedIndexChanged(sender,e);
	}

	void FormFactor::listView_phases_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			switch(GetPhase()) {
			default:
			case PHASE_NONE:
			case PHASE_LAMELLAR_1D:
			case PHASE_2D:
			case PHASE_3D:
				break;
			case PHASE_RECTANGULAR_2D:
				listView_phases->Items[2]->SubItems[1]->Text = (90.0).ToString("0.000000"); 
				break;
			case PHASE_CENTERED_RECTANGULAR_2D:
				listView_phases->Items[2]->SubItems[1]->Text = (45.0).ToString("0.000000"); 
				break;
			case PHASE_SQUARE_2D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[2]->SubItems[1]->Text = (45.0).ToString("0.000000"); 
				break;
			case PHASE_HEXAGONAL_2D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[2]->SubItems[1]->Text = (120.0).ToString("0.000000"); 
				break;
			case PHASE_RHOMBO_3D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[3]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text;
				listView_phases->Items[4]->SubItems[1]->Text = listView_phases->Items[2]->SubItems[1]->Text; 
				listView_phases->Items[5]->SubItems[1]->Text = listView_phases->Items[2]->SubItems[1]->Text;
				break;
			case PHASE_HEXA_3D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[2]->SubItems[1]->Text = (120.0).ToString("0.000000"); 
				listView_phases->Items[4]->SubItems[1]->Text = (90.0).ToString("0.000000");  
				listView_phases->Items[5]->SubItems[1]->Text = (90.0).ToString("0.000000"); 
				break;
			case PHASE_MONOC_3D:
				listView_phases->Items[2]->SubItems[1]->Text = (90.0).ToString("0.000000");
				listView_phases->Items[5]->SubItems[1]->Text = (90.0).ToString("0.000000");
				break;
			case PHASE_ORTHO_3D:
				listView_phases->Items[2]->SubItems[1]->Text = (90.0).ToString("0.000000"); 
				listView_phases->Items[4]->SubItems[1]->Text = (90.0).ToString("0.000000");  
				listView_phases->Items[5]->SubItems[1]->Text = (90.0).ToString("0.000000");
				break;
			case PHASE_TETRA_3D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[2]->SubItems[1]->Text = (90.0).ToString("0.000000"); 
				listView_phases->Items[4]->SubItems[1]->Text = (90.0).ToString("0.000000");  
				listView_phases->Items[5]->SubItems[1]->Text = (90.0).ToString("0.000000");
				break;
			case PHASE_CUBIC_3D:
				listView_phases->Items[1]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[3]->SubItems[1]->Text = listView_phases->Items[0]->SubItems[1]->Text; 
				listView_phases->Items[2]->SubItems[1]->Text = (90.0).ToString("0.000000"); 
				listView_phases->Items[4]->SubItems[1]->Text = (90.0).ToString("0.000000");  
				listView_phases->Items[5]->SubItems[1]->Text = (90.0).ToString("0.000000");
				break;


		}
		if(listView_phases->SelectedIndices->Count > 0) {
			comboPhases->SelectedIndex = listView_phases->SelectedIndices[0];
			if(listView_phases->Items[listView_phases->SelectedIndices[0]]->SubItems[6]->Text == "0") {
				ValPhases->Enabled = false;
				MinPhases->Enabled = false;
				MaxPhases->Enabled = false;
				MutPhases->Checked = false;
				MutPhases->Enabled = false;
				
			}
			else {
				ValPhases->Enabled = true;
				MinPhases->Enabled = !_bGenerateModel;
				MaxPhases->Enabled = !_bGenerateModel;
				MutPhases->Enabled = !_bGenerateModel;
			}
			calculateRecipVectors();
		}
	}
	

	 void FormFactor::ValPhases_Leave(System::Object^  sender, System::EventArgs^  e) {
		double res;
		std::string str;
		char f[128] = {0};
		
		clrToString(((TextBox ^)(sender))->Text, str);

		res = strtod(str.c_str(), NULL);

		ParamMode mode = ParamMode(int::Parse(listView_phases->Items[comboPhases->SelectedIndex]->SubItems[7]->Text));
		switch(mode) {
			
			case MODE_PRECISION:
				sprintf(f, "%.12f", res);
				break;

			case MODE_01:
				if(res < 0.0)
					res = 0.0;
				if(res > 1.0)
					res = 1.0;

			case MODE_ABSOLUTE:
				res = fabs(res);

			default:
			case MODE_DEFAULT:
				sprintf(f, "%.6f", res);
				break;
		}
		((TextBox ^)(sender))->Text = gcnew String(f);
		
		if(sender == ValPhases) {
			listView_phases->Items[comboPhases->SelectedIndex]->SubItems[1]->Text = ValPhases->Text;
			// Recalculate the reciprocal vectors and angles
			calculateRecipVectors();

			bool zero = false;
			for(int i = 0; i < listView_phases->Items->Count; i++)
				if((clrToDouble(listView_phases->Items[i]->SubItems[1]->Text) < 1.0e-7) &&
					listView_phases->Items[i]->SubItems[6]->Text == "1")
					zero = true;

			if(liveRefreshToolStripMenuItem->Checked && !zero)
				PhasesCompleted();
		}
		if(sender == MinPhases)
			listView_phases->Items[comboPhases->SelectedIndex]->SubItems[3]->Text = MinPhases->Text;
		if(sender == MaxPhases)
			listView_phases->Items[comboPhases->SelectedIndex]->SubItems[4]->Text = MaxPhases->Text;
	
		save->Enabled = true;
		AddToUndoQueue(MT_STRUCTUREFACTOR, _curSFPar);
	 }
	 
	
	 void FormFactor::MutPhases_Click(System::Object^  sender, System::EventArgs^  e) {
		 if(comboPhases->SelectedIndex == -1) return;
		 listView_phases->Items[comboPhases->SelectedIndex]->SubItems[2]->Text = MutPhases->Checked ? "Y" : "N";
	 }

	 void FormFactor::AddPhasesParam(System::String ^str, ParamMode mode, double defaultValue, double min, double max){
		 System::String ^valuestr, ^minval, ^maxval;

		valuestr = defaultValue.ToString("0.000000");
		minval = min.ToString("0.000000");
		maxval = max.ToString("0.000000");

		listView_phases->Items->Add(str);
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add(valuestr);				// SubItems[1]	Value
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add("N");					// SubItems[2]	Mutable
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add(minval);				// SubItems[3]	MinVal
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add(maxval);				// SubItems[4]	MaxVal
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add("0.0");				// SubItems[5]	Reciprocal
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add("0");					// SubItems[6]	Enabled
		listView_phases->Items[listView_phases->Items->Count - 1]->SubItems->Add(int(mode).ToString());	// SubItems[7]	Input Mode

		comboPhases->Items->Add(str);
	 }

	 void FormFactor::GetPhasesParameters(vector<double> &exParams) {
		for(int i = 0; i < listView_phases->Items->Count; i++)
			exParams.push_back(Double::Parse(listView_phases->Items[i]->SubItems[1]->Text));
	}

	void FormFactor::SetPhasesParameters(const vector<double> &exParams) {
		int size = exParams.size();

		for(int i = 0; i < size; i++) {
			if(exParams.at(i) >= 0.0) {
				if(int::Parse(listView_phases->Items[i]->SubItems[7]->Text) == MODE_PRECISION)
					listView_phases->Items[i]->SubItems[1]->Text = exParams.at(i).ToString("0.000000");
				else
					listView_phases->Items[i]->SubItems[1]->Text = exParams.at(i).ToString("0.000000");
			}
		}

		ValPhases->Text = listView_phases->Items[comboPhases->SelectedIndex]->SubItems[1]->Text;
		MinPhases->Text = listView_phases->Items[comboPhases->SelectedIndex]->SubItems[3]->Text;
		MaxPhases->Text = listView_phases->Items[comboPhases->SelectedIndex]->SubItems[4]->Text;
	}

	void FormFactor::GetPhasesParameters(vector<double> &exParams, vector<bool>& mutex) {
		GetPhasesParameters(exParams);

		for(int i = 0; i < listView_phases->Items->Count; i++) {
			if(exParams[i] == -1.0)
				mutex.push_back(false);
			else
				mutex.push_back(listView_phases->Items[i]->SubItems[2]->Text->Equals("Y"));
		}

	}

	void FormFactor::initPhasesParams() {
		System::Windows::Forms::ListViewItem ^lvi = listView_phases->Items[0];

		ValPhases->Text    = lvi->SubItems[1]->Text;
		MinPhases->Text    = lvi->SubItems[3]->Text;
		MaxPhases->Text    = lvi->SubItems[4]->Text;
		MutPhases->Checked = lvi->SubItems[2]->Text->Equals("Y");

		if(comboPhases->SelectedIndex < 0)
			comboPhases->SelectedIndex = 0;
	}

	void FormFactor::comboPhases_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {

		System::Windows::Forms::ListViewItem ^lvi = listView_phases->Items[oldPhasesI];

		
		lvi->SubItems[1]->Text = ValPhases->Text;
		lvi->SubItems[3]->Text = MinPhases->Text;
		lvi->SubItems[4]->Text = MaxPhases->Text;
		if(!_bGenerateModel)
			lvi->SubItems[2]->Text = MutPhases->Checked ? "Y" : "N";

		lvi = listView_phases->Items[comboPhases->SelectedIndex];
		ValPhases->Text = lvi->SubItems[1]->Text;
		MinPhases->Text = lvi->SubItems[3]->Text;
		MaxPhases->Text = lvi->SubItems[4]->Text;
		if(!_bGenerateModel)
			MutPhases->Checked = (lvi->SubItems[2]->Text->Equals("Y"));
		if(lvi->SubItems[6]->Text == "0") {
			ValPhases->Enabled = false;
			MinPhases->Enabled = false;
			MaxPhases->Enabled = false;
			MutPhases->Enabled = false;
		} else {
			ValPhases->Enabled = true;
			MinPhases->Enabled = !_bGenerateModel;
			MaxPhases->Enabled = !_bGenerateModel;
			MutPhases->Enabled = !_bGenerateModel;
		}

		oldPhasesI = comboPhases->SelectedIndex;
	}

	void FormFactor::GetPhasesFromListView(phaseStruct *phase) {
		int dim = (int)order->SelectedIndex;
		if (dim == 0 || dim > 6)
			dim = 3;
		else if (dim > 1)
			dim = 2;

		phase->a	 = clrToDouble(listView_phases->Items[0]->SubItems[1]->Text);
		phase->b     = (dim >= 2) ? clrToDouble(listView_phases->Items[1]->SubItems[1]->Text) : 0.0;
		phase->gamma = (dim >= 2) ? clrToDouble(listView_phases->Items[2]->SubItems[1]->Text) * PI/180.0 : 0.0;
		phase->c	 = (dim >= 3) ? clrToDouble(listView_phases->Items[3]->SubItems[1]->Text) : 0.0;
		phase->alpha = (dim >= 3) ? clrToDouble(listView_phases->Items[4]->SubItems[1]->Text) * PI/180.0: 0.0;
		phase->beta  = (dim >= 3) ? clrToDouble(listView_phases->Items[5]->SubItems[1]->Text) * PI/180.0: 0.0;

		phase->amin		 = clrToDouble(listView_phases->Items[0]->SubItems[3]->Text);
		phase->bmin      = (dim >= 2) ? clrToDouble(listView_phases->Items[1]->SubItems[3]->Text) : 0.0;
		phase->gammamin  = (dim >= 2) ? clrToDouble(listView_phases->Items[2]->SubItems[3]->Text) * PI/180.0: 0.0;
		phase->cmin	     = (dim >= 3) ? clrToDouble(listView_phases->Items[3]->SubItems[3]->Text) : 0.0;
		phase->alphamin	 = (dim >= 3) ? clrToDouble(listView_phases->Items[4]->SubItems[3]->Text) * PI/180.0: 0.0;
		phase->betamin   = (dim >= 3) ? clrToDouble(listView_phases->Items[5]->SubItems[3]->Text) * PI/180.0: 0.0;
//		phase->ummin     = clrToDouble(listView_phases->Items[6]->SubItems[4]->Text);

		phase->amax		 = clrToDouble(listView_phases->Items[0]->SubItems[4]->Text);
		phase->bmax      = (dim >= 2) ? clrToDouble(listView_phases->Items[1]->SubItems[4]->Text) : 0.0;
		phase->gammamax  = (dim >= 2) ? clrToDouble(listView_phases->Items[2]->SubItems[4]->Text) * PI/180.0: 0.0;
		phase->cmax	     = (dim >= 3) ? clrToDouble(listView_phases->Items[3]->SubItems[4]->Text) : 0.0;
		phase->alphamax	 = (dim >= 3) ? clrToDouble(listView_phases->Items[4]->SubItems[4]->Text) * PI/180.0: 0.0;
		phase->betamax   = (dim >= 3) ? clrToDouble(listView_phases->Items[5]->SubItems[4]->Text) * PI/180.0: 0.0;
//		phase->ummax	 = clrToDouble(listView_phases->Items[6]->SubItems[5]->Text);

		

		phase->aM	          = (char)listView_phases->Items[0]->SubItems[2]->Text[0];
		if(dim >= 2) {
			phase->bM         = ((char)listView_phases->Items[1]->SubItems[6]->Text[0] == '1') ? 
									((char)listView_phases->Items[1]->SubItems[2]->Text[0]) : ('N');
			phase->gammaM     = ((char)listView_phases->Items[2]->SubItems[6]->Text[0] == '1') ? 
									((char)listView_phases->Items[2]->SubItems[2]->Text[0] ) : ('N');
			if(dim >= 3) {
				phase->cM     = ((char)listView_phases->Items[3]->SubItems[6]->Text[0] == '1') ? 
									((char)listView_phases->Items[3]->SubItems[2]->Text[0]) : ('N');;
				phase->alphaM = ((char)listView_phases->Items[4]->SubItems[6]->Text[0] == '1') ? 
									((char)listView_phases->Items[4]->SubItems[2]->Text[0]) : ('N');;
				phase->betaM  = ((char)listView_phases->Items[5]->SubItems[6]->Text[0] == '1') ? 
									((char)listView_phases->Items[5]->SubItems[2]->Text[0]) : ('N');;
			} else
				phase->cM = phase->alphaM = phase->betaM = 'N';
		} else
			phase->bM = phase->gammaM = 'N';

//		phase->umM    = (int)clrToDouble(listView_phases->Items[6]->SubItems[7]->Text) == 1 ? 'Y': 'N';

		if(wgtFit && wgtFit->graph && wgtFit->graph->x[0].size() > 0)
			phase->qmax = wgtFit->graph->x[0].back();
		else
			phase->qmax = 5.0;
	}

	PhaseType FormFactor::GetPhase() {
		switch ((int)order->SelectedIndex) {
			default:
			case 0:
				return PHASE_NONE;
				
			case 1:
				return(PHASE_LAMELLAR_1D);
				
				
			case 2:
				return(PHASE_2D);

			case 3:
				return(PHASE_RECTANGULAR_2D);
			
			case 4:
				return(PHASE_CENTERED_RECTANGULAR_2D);
			
			case 5:
				return(PHASE_SQUARE_2D);
						
			case 6:
				return(PHASE_HEXAGONAL_2D);
				
				
			case 7:
				return(PHASE_3D);

			case 8:
				return PHASE_RHOMBO_3D;
				
			case 9:
				return(PHASE_HEXA_3D);
				
				
			case 10:
				return(PHASE_MONOC_3D);
				
				
			case 11:
				return(PHASE_ORTHO_3D);
				
				
			case 12:
				return(PHASE_TETRA_3D);

			case 13:
				return(PHASE_CUBIC_3D);



			
		}	
			
		
	}

	void FormFactor::SetPhases(phaseStruct *phase) {
		// Set only if there are actually values
		if( (phase->a < 1.0e-8) &&
			(phase->b < 1.0e-8) &&
			(phase->c < 1.0e-8) )
			return;

		listView_phases->Items[0]->SubItems[1]->Text = phase->a.ToString("0.000000");	  
		listView_phases->Items[1]->SubItems[1]->Text = phase->b.ToString("0.000000");     
		listView_phases->Items[2]->SubItems[1]->Text = (phase->gamma * 180.0/PI).ToString("0.000000"); 
		listView_phases->Items[3]->SubItems[1]->Text = phase->c.ToString("0.000000");	  
		listView_phases->Items[4]->SubItems[1]->Text = (phase->alpha * 180.0/PI).ToString("0.000000");	  
		listView_phases->Items[5]->SubItems[1]->Text = (phase->beta * 180.0/PI).ToString("0.000000"); 

		listView_phases->Items[0]->SubItems[3]->Text = phase->amin.ToString("0.000000");		
		listView_phases->Items[1]->SubItems[3]->Text = phase->bmin.ToString("0.000000");      
		listView_phases->Items[2]->SubItems[3]->Text = (phase->gammamin * 180.0/PI).ToString("0.000000");  
		listView_phases->Items[3]->SubItems[3]->Text = phase->cmin.ToString("0.000000");	    
		listView_phases->Items[4]->SubItems[3]->Text = (phase->alphamin * 180.0/PI).ToString("0.000000");	
		listView_phases->Items[5]->SubItems[3]->Text = (phase->betamin * 180.0/PI).ToString("0.000000");  

		listView_phases->Items[0]->SubItems[4]->Text = phase->amax.ToString("0.000000");	
		listView_phases->Items[1]->SubItems[4]->Text = phase->bmax.ToString("0.000000");     
		listView_phases->Items[2]->SubItems[4]->Text = (phase->gammamax * 180.0/PI).ToString("0.000000"); 
		listView_phases->Items[3]->SubItems[4]->Text = phase->cmax.ToString("0.000000");	 
		listView_phases->Items[4]->SubItems[4]->Text = (phase->alphamax * 180.0/PI).ToString("0.000000");	 
		listView_phases->Items[5]->SubItems[4]->Text = (phase->betamax * 180.0/PI).ToString("0.000000"); 

		listView_phases->Items[0]->SubItems[2]->Text = phase->aM == 'Y' ? "Y" : "N"; 
		listView_phases->Items[1]->SubItems[2]->Text = phase->bM == 'Y' ? "Y" : "N";    
		listView_phases->Items[2]->SubItems[2]->Text = phase->gammaM == 'Y' ? "Y" : "N"; 
		listView_phases->Items[3]->SubItems[2]->Text = phase->cM == 'Y' ? "Y" : "N";
		listView_phases->Items[4]->SubItems[2]->Text = phase->alphaM == 'Y' ? "Y" : "N";
		listView_phases->Items[5]->SubItems[2]->Text = phase->betaM == 'Y' ? "Y" : "N";

		ValPhases->Text = listView_phases->Items[comboPhases->SelectedIndex > -1 ? comboPhases->SelectedIndex : 0]->SubItems[1]->Text;
		MinPhases->Text = listView_phases->Items[comboPhases->SelectedIndex > -1 ? comboPhases->SelectedIndex : 0]->SubItems[3]->Text;
		MaxPhases->Text = listView_phases->Items[comboPhases->SelectedIndex > -1 ? comboPhases->SelectedIndex : 0]->SubItems[4]->Text;
		
	}

	/**
	 * Sort the peak list view based on the xCenter values
	**/
	void FormFactor::SortButton_Click(System::Object^  sender, System::EventArgs^  e) {
		SortListView(listView_peaks, 5, true);
		//// You're welcome to make it more efficient if you'd like... No NIS
		//for(int i = 0; i < listView_peaks->Items->Count; i++) {
		//	for(int j = i + 1; j < listView_peaks->Items->Count; j++) {
		//		if(clrToDouble(listView_peaks->Items[i]->SubItems[5]->Text) >= clrToDouble(listView_peaks->Items[j]->SubItems[5]->Text)) {
		//			//swap
		//			for(int s = 1; s < listView_peaks->Items[i]->SubItems->Count; s++) {
		//				System::String ^tmp = listView_peaks->Items[i]->SubItems[s]->Text->ToString();
		//				listView_peaks->Items[i]->SubItems[s]->Text = listView_peaks->Items[j]->SubItems[s]->Text->ToString();
		//				listView_peaks->Items[j]->SubItems[s]->Text = tmp;
		//			}
		//		}
		//	}
		//}
	}
	void FormFactor::listView_PeakPosition_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if(_bGenerateModel)	// Avi: Do we want to implement this for Generate mode?
			return;
		if(!wgtFit || !wgtFit->graph)
			return;
		if(wgtFit->graph->numGraphs == 0)
			 return;
		 for (int i = 0; i < wgtFit->graph->numGraphs; i++) {
			 if (graphType->at(i) == GRAPH_PHASEPOS) {
				 graphType->erase(graphType->begin() + i);
				 wgtFit->graph->Remove(i);
			 }
		 }
		 				 
		 std::vector <double> px;

		 for(int i = 0; i < listView_PeakPosition->SelectedIndices->Count; i++) 
			 if (listView_PeakPosition->Items[listView_PeakPosition->SelectedIndices[i]]->SubItems->Count > 2 && listView_PeakPosition->Items[listView_PeakPosition->SelectedIndices[i]]->SubItems[2]->Text != "")
				px.push_back(clrToDouble(listView_PeakPosition->Items[listView_PeakPosition->SelectedIndices[i]]->SubItems[2]->Text));
		
		 wgtFit->graph->Add(RGB(0,100,150),DRAW_VERTICAL_LINE,px,px);
		 graphType->push_back(GRAPH_PHASEPOS);
		 wgtFit->Invalidate();
	}

	void FormFactor::PhasesCompleted () {
		std::vector <double> data;
		std::vector <std::string> indicesGen;
		phaseStruct p;
		//if(!_curPhases) GetPhasesFromListView(&p); // TODO::Phases
		//else p = *_curPhases;
		std::vector <double> normGen = GenPhases(GetPhase(),&p,indicesGen), errGen;
		for (int i = 0; i<listView_PeakPosition->Items->Count; i++)
			if(listView_PeakPosition->Items[i]->Text != "N/A")
				data.push_back(clrToDouble(listView_PeakPosition->Items[i]->Text));

	
		listView_PeakPosition->Items->Clear(); 
		//sorts the data vector
		for( int i = 0; i< (int)data.size(); i++) {
			for (int j = i; j < (int)data.size(); j++){
				if(data[i] >data [j]) {
					double c; 
					c = data[i];
					data[i] = data[j];
					data[j] = c;
				}
			}
		}
		int c = 0; 
		while (c < (int)normGen.size()														// Still have more generated peaks AND
			/*&& (normGen[c] < ((wgtFit && wgtFit->graph) ? wgtFit->graph->x[0].back() : 5.0) // ( The generated peak is below q-max
				|| (c < (int)data.size() ))*/) {												// OR we still have more data peaks)
			listView_PeakPosition->Items->Add("N/A");
			listView_PeakPosition->Items[listView_PeakPosition->Items->Count -1]->SubItems->Add(gcnew String(indicesGen[c].c_str()));
			listView_PeakPosition->Items[listView_PeakPosition->Items->Count -1]->SubItems->Add(normGen[c].ToString("0.000000"));
			c++;

		}
		SortListView(listView_PeakPosition, 2, 0);
		int mini = 0;
		for (int i = 0; i < (int)data.size(); i++) {
			if (data[i] < 0.0) continue;
			
			int j = mini;
			while(j < listView_PeakPosition->Items->Count - 1 && (fabs(data[i] - normGen[j]) > fabs(data[i] - normGen[j + 1]) ) )
				j++;

			mini = j;
			
			if(mini > listView_PeakPosition->Items->Count - 1)
				break;

			listView_PeakPosition->Items[mini]->Text = data[i].ToString("0.000000");
			listView_PeakPosition->Items[mini]->BackColor = System::Drawing::Color::LemonChiffon;
			mini++;
		}

		/* DEBUG */
		//std::stringstream strg;
		//for(int y = 0; y < listView_PeakPosition->Items->Count; y++) {
		//	strg<<"\n"<< listView_PeakPosition->Items[y]->SubItems[0]->Text->ToCharArray();
		//}
		//MessageBoxA(NULL, strg.str().c_str(), "DEBUG", NULL);
		/* END DEBUG */

		for(int y = listView_PeakPosition->Items->Count - 1; y >= 0; y--) {
			if(listView_PeakPosition->Items[y]->SubItems[0]->Text == "N/A"
				&& clrToDouble(listView_PeakPosition->Items[y]->SubItems[2]->Text) > ((wgtFit && wgtFit->graph) ? wgtFit->graph->x[0].back() : 5.0)) {
				listView_PeakPosition->Items[y]->Remove();
				continue;
			}
			break;
		}

		for (int i = 0; i < listView_PeakPosition->Items->Count; i++)
			if(listView_PeakPosition->Items[i]->Text != "N/A")
				errGen.push_back(clrToDouble(listView_PeakPosition->Items[i]->SubItems[2]->Text));

		phaseErrorTextBox->Text = calcPhaseErr(data, errGen).ToString("0.000000");
		calculateRecipVectors();
	}
	void FormFactor::SortListView(ListView ^l, int sub, bool firstCol){
			for(int i = 0; i < l->Items->Count; i++) {
				for(int j = i + 1; j < l->Items->Count; j++) {
					if(clrToDouble(l->Items[i]->SubItems[sub]->Text) >= clrToDouble(l->Items[j]->SubItems[sub]->Text)) {
						//swap
						for(int s = int(firstCol); s < l->Items[i]->SubItems->Count; s++) {
							System::String ^tmp = l->Items[i]->SubItems[s]->Text->ToString();
							l->Items[i]->SubItems[s]->Text = l->Items[j]->SubItems[s]->Text->ToString();
							l->Items[j]->SubItems[s]->Text = tmp;
						}
					}
				}
			}

	}
	
	/**
	 * Calculate the relevant reciprocal vectors
	**/
	void FormFactor::calculateRecipVectors() {
		int dim = (int)order->SelectedIndex;
		if (dim == 0 || dim > 6)
			dim = 3;
		else if (dim > 1)
			dim = 2;

			/*
			0[None]
			1Lamellar
			2General 2D
			3Rectangular 2D
			4Cent. Rect. 2D
			5Squared 2D
			6Hexagonal 2D
			7General 3D
			8Rhombohegral 3D
			9Hexagonal 3D
			10Monoclinic 3D
			11Orthorombic 3D
			12Tetragonal 3D
			13Cubic 3D
			*/

		double root, num1, num2, a, b, c, alpha, beta, gamma, sa, sb, sc, ca, cb, cc;
		a	  = clrToDouble(listView_phases->Items[0]->SubItems[1]->Text);
		b	  = clrToDouble(listView_phases->Items[1]->SubItems[1]->Text);
		c	  = clrToDouble(listView_phases->Items[3]->SubItems[1]->Text);

		alpha = clrToDouble(listView_phases->Items[4]->SubItems[1]->Text) * PI/180.0;
		beta  = clrToDouble(listView_phases->Items[5]->SubItems[1]->Text) * PI/180.0;
		gamma = clrToDouble(listView_phases->Items[2]->SubItems[1]->Text) * PI/180.0;

		sa = sin(alpha);
		sb = sin(beta);
		sc = sin(gamma);
		ca = cos(alpha);
		cb = cos(beta);
		cc = cos(gamma);

		switch(dim) { // Add unit cell volume here
			case 1:
				listView_phases->Items[0]->SubItems[5]->Text = (2.0 * PI / a).ToString("0.000000");
				for(int i = 1; i < listView_phases->Items->Count; i++)
					listView_phases->Items[i]->SubItems[5]->Text = "-";
				Volume ->Text = (a).ToString("0.000000");

				break;
			case 2:
				// qa
				root = sqrt( 1.0 + cc * cc / sc / sc);
				listView_phases->Items[0]->SubItems[5]->Text = (2.0 * PI / a * root).ToString("0.000000"); 
				// qb
				listView_phases->Items[1]->SubItems[5]->Text = (2.0 * PI / b / sc).ToString("0.000000");
				// q_gamma
				listView_phases->Items[2]->SubItems[5]->Text = (acos(1.0/(-root * fabs(sc/cc)))*180.0/PI).ToString("0.000000");
			
				//Volume
				Volume ->Text = (a*b*sc).ToString("0.000000");

				for(int i = 3; i < listView_phases->Items->Count; i++)
					listView_phases->Items[i]->SubItems[5]->Text = "-";
				
				break;
			default:
			case 3:
				// qa
				root = (ca * cc / sc - cb / sc);
				root = root * root;
				root = sa * sa - root;

				num1 = (cb * cc / sc - ca / sc) / cc;
				num1 = num1 * num1 / fabs(root);
				num1 = 1.0 + 1.0 / (sc * sc / (cc * cc)) + num1;

				listView_phases->Items[0]->SubItems[5]->Text = 
					(2.0 * PI * sqrt(num1) / a).ToString("0.000000");

				// qb
				num2 = (ca * cc / sc - cb / sc) / sc;
				num2 = num2 * num2 / fabs(root);
				num2 = 1.0 /(sc * sc) + num2;

				listView_phases->Items[1]->SubItems[5]->Text = 
					(2.0 * PI * sqrt(num2) / b).ToString("0.000000");

				// qc
				listView_phases->Items[3]->SubItems[5]->Text = 
					(2.0 * PI / (c * sqrt(fabs(root)))).ToString("0.000000");

				// q_alpha
				listView_phases->Items[4]->SubItems[5]->Text = 
					(acos(sqrt(fabs(root))*(ca * cc - cb) / 
						(-sqrt(fabs(num2)) * (ca * (ca * cc - 2.0 * cb) * cc + cb * cb - sa * sa * sc * sc)))*180.0/PI).ToString("0.000000");

				// q_gamma
				listView_phases->Items[2]->SubItems[5]->Text = 
					(acos((cc - ca * cb) / 
						((ca * (ca * cc - 2.0 * cb) * cc + cb * cb - sa * sa * sc * sc)
						* sqrt(fabs(num1 * num2))))*180.0/PI).ToString("0.000000");

				// q_beta
				listView_phases->Items[5]->SubItems[5]->Text = 
					(acos((-cb * cc + ca) / (sc * sc * sqrt(fabs(root * num1))))* 180.0/PI).ToString("0.000000");
				
				//Volume
				Volume ->Text = (a*b*c*sqrt(root)*sc).ToString("0.000000");
				break;
		}
	}
	
	double FormFactor::calcPhaseErr(std::vector<double> O, std::vector<double> E) {
		int n = 0;
		double res = 0.0;

		for(int i = 0; i < (int)O.size() && i < (int)E.size(); i++) {
			if(O[i] > 0.0) {
				n++;
				res += (E[i] - O[i]) * (E[i] - O[i]) / (E[i] * E[i]);
			}
		}
		return sqrt(res / double(n));
	}
	
	void FormFactor::clearPhases_Click(System::Object^  sender, System::EventArgs^  e) {
		listView_PeakPosition->Items->Clear();
		// Delete all circle graphs representing peaks
		if(wgtFit && wgtFit->graph) {
			for(int p = 0; p < (int)graphType->size(); p++) {
				if(graphType->at(p) == GRAPH_PEAK || graphType->at(p) == GRAPH_PHASEPOS) {
					wgtFit->graph->Remove(p);
					graphType->erase(graphType->begin() + p);
					p--;
				}
			}

			wgtFit->Invalidate();
		}

		fitphase->Enabled = false;
		undoPhases->Enabled = false;
	}

	void FormFactor::dealWithTrackBar(ListViewItem ^lvi, int subItem, TrackBar ^tb, TextBox ^txt, double factor, ExtraParam desc) {
		double mid = (tb->Maximum - tb->Minimum) / 2.0;

		// The following is meant to allow a relative change of the value and also allow the transition
		//	to negative numbers and from zero
		String ^str = lvi->SubItems[subItem]->Text;
		int G = str->LastIndexOf(".");
		double val = clrToDouble((G >= 0) ? str->Remove(G, 1) : str);
		if(fabs(val / factor) <= 1.0)
			factor = val - 0.9 * (val < 0.0 ? -1.0 : 1.0);
		double newVal = clrToDouble(lvi->SubItems[subItem]->Text);
		if(fabs(val) < 1.1)
			newVal += (newVal < 0.0 ? -1.0 : 1.0) * powf(1.0f, (float)-desc.decimalPoints);
		newVal *= (1.0 + (newVal < 0.0 ? -1.0 : 1.0) * ((tb->Value - mid) / tb->Maximum / fabs(factor)));

		

		if(desc.isRanged) {
			if(newVal < desc.rangeMin)
				newVal = desc.rangeMin;
			if(newVal > desc.rangeMax)
				newVal = desc.rangeMax;
		}
		
		char a[64] = {0};
		sprintf(a, "%.*f", desc.decimalPoints, newVal);

		txt->Text = gcnew String(a);
		lvi->SubItems[subItem]->Text = gcnew String(a);

		//FFParameterUpdateHandler();
	}
	
	void FormFactor::sigmaToolStripMenuItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(sender == sigmaToolStripMenuItem) {
			if  (sigmaToolStripMenuItem->Checked)
				fWHMToolStripMenuItem->Checked = false;
			else
				fWHMToolStripMenuItem->Checked = true;
		} else if(sender == fWHMToolStripMenuItem) {
			if  (fWHMToolStripMenuItem->Checked)
				sigmaToolStripMenuItem->Checked = false;
			else
				sigmaToolStripMenuItem->Checked = true;
		}
		if(liveRefreshToolStripMenuItem->Checked && sender != nullptr)
			UpdateGraph();
		//columnHeader4->Text = L"FWHM";
	}

	void FormFactor::SFParameterUpdateHandler() {
		if(!_curSFPar)
			return;

		// This function should:
		// 0. Update _curSFPar
		UItoParameters(_curSFPar, _modelSF, listView_peaks, nullptr);

		if(_bLoading ^ _bChanging)
			return;
		// 1. If necessary, redraw the graph
		if(liveRefreshToolStripMenuItem->Checked)
			UpdateGraph(_bLoading || _bChanging);		



		if(!_bFromFitter) {
			SFparamErrors->clear();
			SFmodelErrors->clear();
		}
	}
};

	


