#include "SignalSeries.h"
#include "UnicodeChars.h"

template <typename T> inline T sq(T x) { return x * x; }

namespace GUICLR {
	std::vector<double> MultiplyVector(const std::vector<double> &vec, double val) {
		std::vector<double> res;
		for(int i = 0; i < (int)vec.size(); i++)
			res.push_back(vec[i] * val);
		return res;
	}
	std::vector<double> AddVector(const std::vector<double> &vec, double val) {
		std::vector<double> res;
		for(int i = 0; i < (int)vec.size(); i++)
			res.push_back(vec[i] + val);
		return res;
	}
	
	void SignalSeries::draw() {
		if(wgtGraph && wgtGraph->graph) {
			wgtGraph->graph->Deselect();
			wgtGraph->Invalidate();
		}
	}

	void SignalSeries::SignalSeries_Load(System::Object^  sender, System::EventArgs^  e) {
		srand((unsigned)time(NULL));
		changingSel = false;
		changingVis = false;
		minDistance = 0.23;
		selectedIndex = -1;
		selectedIndexV = -1;
		this->tableLayoutPanel1->Size = System::Drawing::Size(splitContainer1->Panel2->Width,
														splitContainer1->Panel2->Height);
		this->splitContainer1->Panel1MinSize = 300;
		this->splitContainer1->Panel2MinSize = 275;
		this->MinimumSize = Drawing::Size(565, 300);
		this->splitContainer1->BorderStyle = Windows::Forms::BorderStyle::FixedSingle;
		// 
		// wgtGraph
		// 
		this->wgtGraph = (gcnew GUICLR::WGTControl());
		this->wgtGraph->Cursor = System::Windows::Forms::Cursors::Cross;
		this->wgtGraph->Dock = System::Windows::Forms::DockStyle::Fill;
		this->wgtGraph->Location = System::Drawing::Point(0, 0);
		this->wgtGraph->Name = L"wgtGraph";
		this->wgtGraph->Size = System::Drawing::Size(10, 10);
		this->wgtGraph->TabIndex = 0;
		this->wgtGraph->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::wgt_MouseMove);

		this->tableLayoutPanel1->Controls->Add(this->wgtGraph);
		addKeyDownEventRec(this);

		this->timer1->Enabled = true;
		this->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmUp);
	}

	void SignalSeries::addKeyDownEventRec(System::Windows::Forms::Control^ sender) {
		for(int i = 0; i < sender->Controls->Count; i++) {
			addKeyDownEventRec(sender->Controls[i]);
		}
		sender->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SignalSeries::General_KeyDown);
	}

	void SignalSeries::removeButton_Click(System::Object^  sender, System::EventArgs^  e) {
		for(int i = signalFileList->Count - 1; i >= 0; i--) {
			if(signalFileList[i]->selectedCheckBox->Checked) { // Selected for destruction
				//	Remove graph
				//	Remove panel
				//	For all objects that pointed to a higher graph, decrease the index by 1
				int remInd = signalFileList[i]->index;
				wgtGraph->graph->Remove(remInd);
				for(int j = 0; j < signalFileList->Count; j++)
					if(signalFileList[j]->index >= remInd)
						signalFileList[j]->index--;
				signalFileList[i]->~signalFile();
				signalFileList->RemoveAt(i);
			}
		}
		if(signalFileList->Count == 0) {
			delete wgtGraph->graph;
			delete wgtGraph;
			SignalSeries_Load(sender, e);
		} else
			draw();
	}

	void SignalSeries::addButton_Click(System::Object^  sender, System::EventArgs^  e) {
		bool badFileFlag =  false;
		if(!(wgtGraph)) // Ensure that this works
			return;

		ofd->Title = L"Select files to be displayed";
		ofd->Filter = "Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
		
		// If not raised from a drag and drop event
		if(!files) {
			if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
				return;
			files = ofd->FileNames;
		}

		for(int i = 0; i < files->Length; i++) {
			std::vector<double> x, y;
			int r, g, b;
			// The program cannot tolerate files with 1 point or less
			ReadDataFile(clrToWstring(files[i]).c_str(), x, y);
			if(x.size() <= 1) {
				badFileFlag = true;
				continue;
			}

			r = rand() % 256;
			g = rand() % 256;
			b = rand() % 256;

			if(!wgtGraph->graph) { // This is the first addition
				wgtGraph->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &SignalSeries::General_KeyDown);

				RECT area;		
				area.top = 0;
				area.left = 0;
				area.right = wgtGraph->Size.Width + area.left;
				area.bottom = wgtGraph->Size.Height + area.top;
				this->wgtGraph->graph = gcnew Graph(area,
									RGB(r, g, b), 
									DRAW_LINES, x, y, 
									logQCheckBox->Checked,
									logICheckBox->Checked);
				logQCheckBox->Enabled = true;
				logICheckBox->Enabled = true;
				wgtGraph->graph->SetXLabel(" [nm" + UnicodeChars::minusOne + "]");
				wgtGraph->graph->SetYLabel("Intensity [a.u.]");
			} else {
				this->wgtGraph->graph->Add(RGB(r, g, b), DRAW_LINES, x, y);
			}
			this->signalFileList->Add(gcnew signalFile(files[i], pathCheckBox->Checked, signalFileList->Count, x, y));
			this->flowLayoutPanel1->Controls->Add(signalFileList[signalFileList->Count - 1]);
			signalFileList[signalFileList->Count - 1]->setColor(r, g, b);
			signalFileList[signalFileList->Count - 1]->Width = max(400, flowLayoutPanel1->Width-2);;

			// Events																and handlers
			signalFileList[signalFileList->Count - 1]->scaleTextBox->Leave			+= gcnew System::EventHandler(this, &SignalSeries::textBox_Leave);
			signalFileList[signalFileList->Count - 1]->bgTextBox->Leave				+= gcnew System::EventHandler(this, &SignalSeries::textBox_Leave);
			signalFileList[signalFileList->Count - 1]->bgTextBox->TextChanged		+= gcnew System::EventHandler(this, &SignalSeries::textBox_Changed);
			signalFileList[signalFileList->Count - 1]->scaleTextBox->TextChanged	+= gcnew System::EventHandler(this, &SignalSeries::textBox_Changed);
			signalFileList[signalFileList->Count - 1]->scalePlusLabel->MouseDown	+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmDown);
			signalFileList[signalFileList->Count - 1]->scaleMinusLabel->MouseDown	+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmDown);
			signalFileList[signalFileList->Count - 1]->bgPlusLabel->MouseDown		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmDown);
			signalFileList[signalFileList->Count - 1]->bgMinusLabel->MouseDown		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmDown);
			signalFileList[signalFileList->Count - 1]->visCheckBox->CheckedChanged	+= gcnew System::EventHandler(this, &SignalSeries::vis_CheckChanged);
			signalFileList[signalFileList->Count - 1]->selectedCheckBox->CheckedChanged+= gcnew System::EventHandler(this, &SignalSeries::select_CheckChanged);
			signalFileList[signalFileList->Count - 1]->colorLabel->MouseClick		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::color_Clicked);
			signalFileList[signalFileList->Count - 1]->scaleMinusLabel->MouseUp		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmUp);
			signalFileList[signalFileList->Count - 1]->scalePlusLabel->MouseUp		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmUp);
			signalFileList[signalFileList->Count - 1]->bgMinusLabel->MouseUp		+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmUp);
			signalFileList[signalFileList->Count - 1]->bgPlusLabel->MouseUp			+= gcnew System::Windows::Forms::MouseEventHandler(this, &SignalSeries::pmUp);
			for(int i = 0; i < signalFileList[signalFileList->Count - 1]->Controls->Count; i++)
				signalFileList[signalFileList->Count - 1]->Controls[i]->KeyDown		+= gcnew System::Windows::Forms::KeyEventHandler(this, &SignalSeries::General_KeyDown);

		}
		ArrangeList();

		// After all is said and done, tell the user if any of the files failed
		if(badFileFlag)
			System::Windows::Forms::MessageBox::Show("One or more of the chosen files is invalid or empty and has been ignored", 
					"Invalid data file", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);

		if(files)
			delete files;

	}	// end addButton_Click

	void SignalSeries::moveUpButton_Click(System::Object^  sender, System::EventArgs^  e) {
		for(int i = 0; i < signalFileList->Count - 1; i++) {
			// at interface between checked and not checked
			if(signalFileList[i]->selectedCheckBox->Checked != signalFileList[i + 1]->selectedCheckBox->Checked) {
				// Swap the two positions and relevant indices (class index refering to graph, graph indices)
				// Swap in the flowLayoutPanel
				int iInd = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[i]);
				int i2Ind = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[i + 1]);
				flowLayoutPanel1->Controls->SetChildIndex(signalFileList[i], i2Ind);
				flowLayoutPanel1->Controls->SetChildIndex(signalFileList[i + 1], iInd);
				// Swap in the list
				signalFileList->Insert(i, signalFileList[i + 1]);
				signalFileList->RemoveAt(i + 2);
			}
		}	//end for
	}	// end moveUpButton_Click

	void SignalSeries::moveDownButton_Click(System::Object^  sender, System::EventArgs^  e) {
		for(int i = signalFileList->Count - 1; i > 0; i--) {
			// at interface between checked and not checked
			if(signalFileList[i]->selectedCheckBox->Checked != signalFileList[i - 1]->selectedCheckBox->Checked) {
				// Swap the two positions and relevant indices (class index refering to graph, graph indices)
				// Swap in the flowLayoutPanel
				int iInd = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[i]);
				int i2Ind = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[i - 1]);
				flowLayoutPanel1->Controls->SetChildIndex(signalFileList[i], i2Ind);
				flowLayoutPanel1->Controls->SetChildIndex(signalFileList[i - 1], iInd);
				// Swap in the list
				signalFileList->Insert(i - 1, signalFileList[i]);
				signalFileList->RemoveAt(i + 1);
			}
		}	//end for

	}	// end moveDownButton_Click

	void SignalSeries::sortButton_Click(System::Object^  sender, System::EventArgs^  e) {
		// Reset if the shift button was held
		if((sender == sortButton) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) ) {
			for(int i = signalFileList->Count - 1; i > 0; i--) {
				signalFileList[i - 1]->bgTextBox->Text = "0.000000";
				signalFileList[i - 1]->scaleTextBox->Text = "1.000000";
				wgtGraph->graph->Modify(signalFileList[i - 1]->index,
					*(signalFileList[i - 1]->x),
					AddVector(
					MultiplyVector(*(signalFileList[i - 1]->y), clrToDouble(signalFileList[i - 1]->scaleTextBox->Text)),
									clrToDouble(signalFileList[i - 1]->bgTextBox->Text)));
			}
			draw();
			return;
		} // Reset

		// TODO::Later-- Ignore invisible graphs?
		double avi;
		std::vector<double> xi1, yi1, xi2, yi2, der, minScale;
		for(int i = signalFileList->Count - 1; i > 0; i--) {
			xi1 = *(signalFileList[i]->x);		// Lower curve
			yi1 = *(signalFileList[i]->y);
			xi2 = *(signalFileList[i - 1]->x);	// Upper curve
			yi2 = *(signalFileList[i - 1]->y);
			minScale.resize(xi2.size(), 1.0);

			// Determine minimum scale
			avi = 0.0;
			for(int j = 0; j < (int)xi2.size(); j++) {

				// Define the limits of the search in xi1
				int start = 0, end = xi1.size() - 1;
				for(int k = 1; k < (int)xi1.size(); k++) {
					if(xi1[k - 1] + minDistance < xi2[j])
						start = k;
					else
						break;
				}
				for(int k = start; k < (int)xi1.size(); k++) {
					if(xi1[k] - minDistance < xi2[j])
						end = k;
					else
						break;
				}
				//end = min((int)yi2.size() - 1, end);

				// Determine minimum scale
				avi = 0.0; 
				if(this->logICheckBox->Checked) {	// Log scale
					for(int k = start; k < end; k++) {
						if(yi2[j] == 0.0)
							continue;
						avi = max(avi, yi1[k] / yi2[j] * (1.0 + minDistance));
					}
				} else { // Linear scale
					for(int k = start; k < end; k++) {
						avi = max(avi, (yi1[k] - yi2[j]) + minDistance);
					}
				}

				// Store scale
				minScale[j] = avi;
			}	// end for j

			for(int y = 0; y < (int)minScale.size(); y++)
				avi = max(minScale[y], avi);	// The final scale (relative to the ith graph)
			avi = max(avi, 1.0e-6);

			// Set scale to textbox
			if(this->logICheckBox->Checked) {
				signalFileList[i - 1]->scaleTextBox->Text = (clrToDouble(signalFileList[i]->scaleTextBox->Text) * avi).ToString("0.000000");
				signalFileList[i - 1]->bgTextBox->Text = "0.000000";
			}
			else {
				signalFileList[i - 1]->scaleTextBox->Text = "1.000000";
				signalFileList[i - 1]->bgTextBox->Text = (clrToDouble(signalFileList[i]->bgTextBox->Text) + avi).ToString("0.000000");
			}
			wgtGraph->graph->Modify(signalFileList[i - 1]->index,
				xi2,
				AddVector(
				MultiplyVector(yi2, clrToDouble(signalFileList[i - 1]->scaleTextBox->Text)),
								clrToDouble(signalFileList[i - 1]->bgTextBox->Text)));

		} // end for i
		draw();
	}
	
	void SignalSeries::exportButton_Click(System::Object^  sender, System::EventArgs^  e) {
		// Write to TSV file:
		// Format should be title row(s) on top, each file being 2 columns (q, I)
		// Title rows should include (filename\tfilename\t)xN\n(q\tIntensity\t)xN\n([nm-1]\t[a.u.]\t)xN

		// Get the name/dir of the file (+ dialog)
		// Open a file for writing
		// Write three lines of headers
		// Write all the data 
		// Close file

		// Make sure there are visible files
		bool vi = false;
		for(int i = 0; i < signalFileList->Count; i++) {
			if(signalFileList[i]->visCheckBox->Checked) {
				vi = true;
				break;
			}
		}
		if(!vi)
			return;


		// Get the name/dir of the file (+ dialog)
		std::wstring file;
		int maxLen = 0;
		sfd->Filter = "TSV Files (*.tsv)|*.tsv|All Files (*.*)|*.*";
		sfd->Title = "Choose a filename";
		if(sfd->ShowDialog() == 
			System::Windows::Forms::DialogResult::Cancel)
			return;
		clrToString(sfd->FileName, file);
		// Open a file for writing
		FILE *fp;
		if ((fp = _wfopen(file.c_str(), L"w")) == NULL) {
			fprintf(stderr, "Error opening file %s for writing\n",
							file);
			
			MessageBox::Show("Please make sure that the file is not open.", "Error opening file for writing", MessageBoxButtons::OK,
									 MessageBoxIcon::Error);
			return;
		}

		// Check to see if all the q's are the same
		bool sameQ = true;
		for(int i = 1; i < signalFileList->Count && sameQ; i++) {
			if(signalFileList[i]->x->size() != signalFileList[0]->x->size()) {
				sameQ = false;
				break;
			}
			for(int j = 0; j < (int)signalFileList[i]->x->size(); j++) {
				if(fabs(signalFileList[i]->x->at(j) - signalFileList[0]->x->at(j)) > 1.0e-10) {
					sameQ = false;
					break;
				}
			}
		}

		// Write three lines of headers
		if(sameQ)
			fprintf(fp, "\t");
		for(int i = 0; i < signalFileList->Count; i++) {
			if(signalFileList[i]->visCheckBox->Checked) {
				fprintf(fp, "%s\t", signalFileList[i]->file);
				if(!sameQ)
					fprintf(fp, "%s\t", signalFileList[i]->file);
			}
		}
		fprintf(fp, "\n");
		if(sameQ)
			fprintf(fp, "q\t");
		for(int i = 0; i < signalFileList->Count; i++)
			if(signalFileList[i]->visCheckBox->Checked)
				fprintf(fp, sameQ ? "Intensity\t" : "q\tIntensity\t");
		fprintf(fp, "\n");
		
		fprintf(fp, "[nm-1]\t[a.u.]\t"); // Need the first set of q always
		for(int i = 1; i < signalFileList->Count; i++)
			if(signalFileList[i]->visCheckBox->Checked)
				fprintf(fp, sameQ ? "[a.u.]\t" : "[nm-1]\t[a.u.]\t");
		fprintf(fp, "\n");

		// Write all the data 
		for(int i = 0; i < signalFileList->Count; i++) {
			if(signalFileList[i]->visCheckBox->Checked)
				maxLen = max(maxLen, (int)(signalFileList[i]->x->size()));
		}
		for(int i = 0; i < maxLen; i++) {
			for(int j = 0; j < signalFileList->Count; j++) {
				if(!(signalFileList[j]->visCheckBox->Checked))
					continue;
				if(!sameQ || j == 0)
					fprintf(fp, "%s\t",
						(i < (int)signalFileList[j]->x->size()) ? (signalFileList[j]->x->at(i)).ToString() : (" "));
				fprintf(fp, "%s\t",
					(i < (int)signalFileList[j]->y->size()) ? 
								((signalFileList[j]->y->at(i) * clrToDouble(signalFileList[j]->scaleTextBox->Text)
										+ clrToDouble(signalFileList[j]->bgTextBox->Text)).ToString())
								: (" "));
			}
			fprintf(fp, "\n");
		}

		// Close file
		fclose(fp);
	}
	
	void SignalSeries::importTSVFileToolStripMenuItem_Click(System::Object ^sender, System::EventArgs ^e) {
	}

	void SignalSeries::logICheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph && wgtGraph->graph) {
			wgtGraph->graph->SetScale(0, logICheckBox->Checked ? SCALE_LOG : SCALE_LIN);
			draw();
		}
	}
	
	void SignalSeries::logQCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph && wgtGraph->graph) {
			wgtGraph->graph->SetScale(1, logQCheckBox->Checked ? SCALE_LOG : SCALE_LIN);
			draw();
		}
	}

	void SignalSeries::ArrangeList() {
		//for(int i = 0; i < )
	}

	void SignalSeries::textBox_Leave(System::Object^  sender, System::EventArgs^  e) {
		Control^ control = ((System::Windows::Forms::TextBox^)(sender))->Parent;
		signalFile^ sig = ((signalFile^)(control));
		wgtGraph->graph->Modify(sig->index,
			*(sig->x),
			AddVector(
			MultiplyVector(*(sig->y), clrToDouble(sig->scaleTextBox->Text)),
							clrToDouble(sig->bgTextBox->Text)));
		draw();

	}

	void SignalSeries::textBox_Changed(System::Object^  sender, System::EventArgs^  e) {
	}

	void SignalSeries::pmDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		signalFile^ sig = (signalFile^)((System::Windows::Forms::Label^)(sender))->Parent;

		sig->bScaleMinus	= sender == sig->scaleMinusLabel;
		sig->bScalePlus		= sender == sig->scalePlusLabel;
		sig->bBGMinus		= sender == sig->bgMinusLabel;
		sig->bBGPlus		= sender == sig->bgPlusLabel;
	}

	void SignalSeries::pmUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		for(int i = 0; i < signalFileList->Count; i++) {
			signalFileList[i]->bScalePlus	= false;
			signalFileList[i]->bScaleMinus	= false;
			signalFileList[i]->bBGPlus		= false;
			signalFileList[i]->bBGMinus		= false;
		}
	}

	void SignalSeries::vis_CheckChanged(System::Object^  sender, System::EventArgs^  e) {
		if(changingVis)
			return;

		Control^ control = ((System::Windows::Forms::CheckBox^)(sender))->Parent;
		signalFile^ sig = ((signalFile^)(control));

		int newIndex;
		bool selection = sig->visCheckBox->Checked;
		
		for(int i = 0; i < signalFileList->Count; i++)
			if(signalFileList[i] == sig)
				newIndex = i;

		if((!( (System::Windows::Forms::Control::ModifierKeys & Keys::Shift) == Keys::Shift)) || (selectedIndexV < 0)){
			selectedIndexV = newIndex;
			wgtGraph->graph->SetGraphVisibility(sig->index, sig->visCheckBox->Checked);
			draw();
			return;
		}

		changingVis = true;

		for(int i = min(selectedIndexV, newIndex); i <= max(selectedIndexV, newIndex); i++) {
			signalFileList[i]->visCheckBox->Checked = selection;
			wgtGraph->graph->SetGraphVisibility(signalFileList[i]->index, signalFileList[i]->visCheckBox->Checked);
		}
		selectedIndexV = newIndex;
		changingVis = false;

		draw();
	}

	void SignalSeries::color_Clicked(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		int r, g, b;
		r = rand() % 256;
		g = rand() % 256;
		b = rand() % 256;

		// Right click causes manual color selection
		if(e->Button == Windows::Forms::MouseButtons::Right) {
			ColorDialog cd;
			if(cd.ShowDialog() == Windows::Forms::DialogResult::OK)	{
				r = cd.Color.R;
				g = cd.Color.G;
				b = cd.Color.B;
			} else
				return;
		}

		signalFile^ sig = ((signalFile^)((System::Windows::Forms::Label^)(sender))->Parent);

		sig->setColor(r, g, b);
		wgtGraph->graph->ChangeColor(sig->index, r, g, b);
		wgtGraph->graph->Modify(sig->index,
			wgtGraph->graph->x[sig->index],
			wgtGraph->graph->y[sig->index]);
		//wgtGraph->graph->FitToAllGraphs();
		draw();
			 
	}

	void SignalSeries::timer1_Tick(System::Object ^sender, System::EventArgs ^e) {
		double factor = 30.0;
		TextBox^ box;
		for(int i = 0; i < signalFileList->Count; i++) {
			if(signalFileList[i]->bBGMinus || signalFileList[i]->bBGPlus)
				box = signalFileList[i]->bgTextBox;
			else if(signalFileList[i]->bScaleMinus || signalFileList[i]->bScalePlus)
				box = signalFileList[i]->scaleTextBox;
			else
				continue;

			// WTF?!??!?
			String ^str = box->Text;
			double newVal = clrToDouble(str);
			int G = str->LastIndexOf(".");
			double val = clrToDouble((G >= 0) ? str->Remove(G, 1) : str);
			if(fabs(val / factor) <= 1.0)
				factor = val - 0.9 * (val < 0.0 ? -1.0 : 1.0);
			if(fabs(val) < 1.1)
				newVal += ((signalFileList[i]->bScalePlus || signalFileList[i]->bBGPlus) ? 1.0 : -1.0) * 1.0e-5;
			newVal *= (1.0 + ((signalFileList[i]->bScalePlus || signalFileList[i]->bBGPlus) ? 1.0 : -1.0) * (0.4 / fabs(factor)));
			box->Text = newVal.ToString("0.00000");

			wgtGraph->graph->Modify(signalFileList[i]->index,
				*(signalFileList[i]->x),
				AddVector(
				MultiplyVector(*(signalFileList[i]->y), clrToDouble(signalFileList[i]->scaleTextBox->Text)),
				clrToDouble(signalFileList[i]->bgTextBox->Text)));
			draw();

		}
	}
	void SignalSeries::General_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
		if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return) {
			// TODO::Later-- Make graph be redrawn before focus is moved.
			// Avi: These don't work. I would like to raise the Leave event to fix the text in the textbox
			//((TextBox^)(sender))->Leave(sender, gcnew System::EventArgs());
			//((TextBox^)(sender))->OnLeave(gcnew System::EventArgs());
			//Change the focus so that any field that has a new value will be updated
			//((TextBox^)(sender))->Invoke
			draw();
			this->wgtGraph->Focus();
		}

		if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back)
			removeButton_Click(sender, gcnew System::EventArgs());

		if((e->KeyCode == Keys::A) /*&& (System::Windows::Forms::Control::ModifierKeys == Keys::Control)*/) {
			if(System::Windows::Forms::Control::ModifierKeys == Keys::Control) {
				if(signalFileList->Count > 0)
					signalFileList[0]->selectedCheckBox->Focus();
				bool allSel = true;
				for(int i = 0; i < signalFileList->Count; i++)
					allSel &= signalFileList[i]->selectedCheckBox->Checked;
				for(int i = 0; i < signalFileList->Count; i++)
					signalFileList[i]->selectedCheckBox->Checked = !allSel;
			}
		}
	}

	void SignalSeries::pathCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		for(int i = 0; i < signalFileList->Count; i++)
			signalFileList[i]->fileText(pathCheckBox->Checked);
	}

	void SignalSeries::select_CheckChanged(System::Object ^sender, System::EventArgs ^e) {
		if(changingSel)
			return;
		Control^ control = ((System::Windows::Forms::CheckBox^)(sender))->Parent;
		signalFile^ sig = ((signalFile^)(control));

		int newIndex;
		bool selection = sig->selectedCheckBox->Checked;

		for(int i = 0; i < signalFileList->Count; i++)
			if(signalFileList[i] == sig)
				newIndex = i;

		if(selectedIndex < 0) { // No previously selected item
			selectedIndex = newIndex;
			return;
		}

		if(!( (System::Windows::Forms::Control::ModifierKeys & Keys::Shift) == Keys::Shift)) {
			selectedIndex = newIndex;
			return;
		}

		changingSel = true;

		for(int i = min(selectedIndex, newIndex); i <= max(selectedIndex, newIndex); i++)
			signalFileList[i]->selectedCheckBox->Checked = selection;
		
		selectedIndex = newIndex;
		changingSel = false;

	}

	void SignalSeries::reverseOrderToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		//
		int N = signalFileList->Count - 1;
		for(int i = 0; i < (N + 1) / 2; i++) {
			// Swap i and N - i
			int iInd = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[i]);
			int i2Ind = flowLayoutPanel1->Controls->GetChildIndex(signalFileList[N - i]);
			flowLayoutPanel1->Controls->SetChildIndex(signalFileList[i], i2Ind);
			flowLayoutPanel1->Controls->SetChildIndex(signalFileList[N - i], iInd);
			// Swap in the list
			signalFileList->Insert(N - i + 1, signalFileList[i]);
			signalFileList->RemoveAt(i);
			signalFileList->Insert(i, signalFileList[N - i - 1]);
			signalFileList->RemoveAt(N - i);

		}
	}

	void SignalSeries::minSpacingTrackBar_Scroll(System::Object^  sender, System::EventArgs^  e) {
		// Default value (in the middle) should have minDistance = 0.23
		minDistance = exp((exp(sq(0.91 * (minSpacingTrackBar->Value - minSpacingTrackBar->Minimum) / (minSpacingTrackBar->Maximum - minSpacingTrackBar->Minimum) )) - 1.0) / 0.2 - 1.0) / 10.0;
		sortButton_Click(sender, e);
	}
	
	void SignalSeries::SignalSeries_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		if(e->Data->GetDataPresent(DataFormats::FileDrop))
			e->Effect = DragDropEffects::All;
		else
			e->Effect = DragDropEffects::None;
	}

	void SignalSeries::SignalSeries_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		if(files)
			delete files;
		files = gcnew array<System::String ^>(1);
		ArrayList ^ bigList = gcnew ArrayList();
		ArrayList ^ more = gcnew ArrayList();
		bigList->AddRange(dynamic_cast<array<System::String^>^> (e->Data->GetData(DataFormats::FileDrop, false)));
		for(int i = 0; i < bigList->Count; i++) {
			System::String^ st = bigList[i]->ToString();
			if(System::IO::Directory::Exists(st)) {
				more->AddRange(System::IO::Directory::GetFiles(st, "*", System::IO::SearchOption::AllDirectories));
				bigList->RemoveAt(i--);
			}
		}

		bigList->AddRange(more);
		files = reinterpret_cast<array<System::String^>^>(bigList->ToArray(System::String::typeid));

		addButton_Click(sender, gcnew EventArgs());

		if(files)
			delete files;
			
	}

	void SignalSeries::flowLayoutPanel1_Resize(System::Object ^sender, System::EventArgs ^e) {
		for(int i = 0; i < flowLayoutPanel1->Controls->Count; i++)
			flowLayoutPanel1->Controls[i]->Width = max(400, flowLayoutPanel1->Width-2);
		splitContainer2->SplitterDistance = 43;
		this->label7->Location = System::Drawing::Point(max(270, flowLayoutPanel1->Width - 2 - 130), 30);	// Scale
		this->label4->Location = System::Drawing::Point(max(335, flowLayoutPanel1->Width - 2 - 65), 30);	// Background
	}

	void SignalSeries::wgt_MouseMove(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e) {
		if(!wgtGraph || !wgtGraph->graph)
			return;
		 
		std::pair<double, double> loc;
		loc = wgtGraph->graph->PointToData(e->X - wgtGraph->graph->xoff, e->Y - wgtGraph->graph->yoff);

		if(wgtGraph->graph->LogScaleX())
			loc.first = pow(10.0, loc.first);

		if(wgtGraph->graph->LogScaleY())
			loc.second = pow(10.0, loc.second);

		LocOnGraph->Text = "("+ Double(loc.first).ToString((loc.first < 1.0e-3 || loc.first > 1.0e5) ? ("e5") : ("0.000000")) + ",\n "
							+ Double(loc.second).ToString((loc.second < 1.0e-3 || loc.second > 1.0e5) ? ("e5") : ("0.000000")) + ")";
	}
}