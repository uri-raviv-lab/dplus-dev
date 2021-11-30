#include "TWindow.h"
#include "calculationExternal.h"
#include "filemgt.h"

#include <sstream>
#include <fstream>
#include <limits>

using std::wofstream;
using namespace Eigen;


namespace PopulationGUI {
	TWindow::TWindow(void) {
			InitializeComponent();
			FileExpressionList = gcnew System::Collections::Generic::List<FileExpressionPanel^>();

		   
			coefficients	= new Eigen::Array<double,1,26>;
			dataArray		= new Eigen::ArrayXXd;
			xDataArray		= new Eigen::ArrayXXd;
			a				= new Eigen::ArrayXd;
			pmut			= new Eigen::ArrayXi;
			pMin			= new cons(26);
			pMax			= new cons(26);
	}

	//
	void TWindow::TWindow_Load(System::Object ^sender, System::EventArgs ^e) {
		timer1->Enabled = true;

		SetLogFitting(false);

		std::string str(" ");
		for(char c = 'a'; c <= 'z'; c++) {
			str[0] = c;
			variableDataGridView->Rows->Add(false, gcnew String(str.c_str()), gcnew String("0.00000"), gcnew String("0.00000"), gcnew String("0.00000"));
		}

		fitIterationsToolStripTextBox->Text = L"20";

#ifdef _DEBUG
		AddPanel();
		AddPanel();

		FileExpressionList[1]->expressionTextBox->Text = L"a(A*2)";
		FileExpressionList[1]->expressionRadioButton->Checked = true;
		this->fitTextBox->Text = L"B";
		this->toTextBox->Text = L"A";
		variableDataGridView[0,0]->Value = "Y";
		variableDataGridView[2,0]->Value = "5";
#endif
	}
	//
	void TWindow::exprFlowLayoutPanel_Resize(System::Object^  sender, System::EventArgs^  e) {
		for(int i = 0; i < exprFlowLayoutPanel->Controls->Count; i++)
			exprFlowLayoutPanel->Controls[i]->Width = max(400, exprFlowLayoutPanel->Width - 5);

	}

	void TWindow::useLogFittingToolStripMenuItem_CheckedChanged(System::Object ^sender, System::EventArgs ^e) {
		SetLogFitting(useLogFittingToolStripMenuItem->Checked);
	}
	
#pragma region Helper Functions
	////////////////////////////////////////////
	// Helper Functions
	template <typename T>
	std::string to_string(T const& value) {
		std::stringstream sstr;
		sstr << value;
		return sstr.str();
	}

	// End Helper Functions
	////////////////////////////////////////////
#pragma endregion

	//
	void TWindow::exprTextBox_Leave(System::Object ^sender, System::EventArgs ^e) {
		FileExpressionPanel^ par = GetParent(sender);

		if(par->expressionTextBox->Text->Length < 1)
			return;

		if(par->expressionTextBox->Text->Contains(par->designationTextBox->Text)) {
			System::Windows::Forms::MessageBox::Show(L"Self reference", 
					"Invalid expression", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
			return;
		}

		// TODO: Look for circular references

		if(!loadVariables(false))
			return;

		std::vector<string> errs;
		ArrayXXd	dummyData	= *dataArray,
					dummyX		= *xDataArray;
		ArrayXd		resX;

		ArrayXd  dummyCoefs = *coefficients;
		std::string expr = clrToString(par->expressionTextBox->Text);
		ParserSolver parse(errs, expr, dummyX, dummyData, dummyCoefs);
		ArrayXd res = parse.EvaluateExpression(resX);

		if(!errs.empty()) {
			String ^erS = gcnew String(L"");
			for(int i = 0; i < (int)errs.size(); i++)
				erS += "\n" + stringToClr(errs[i]);

			System::Windows::Forms::MessageBox::Show(erS, 
					"Invalid expression", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
			return;
		}

		par->xE->clear();
		par->yE->clear();

		for(int i = 0; i < res.size(); i++) {
			par->xE->push_back(resX(i));
			par->yE->push_back(res(i));
		}
		
		if(par->expressionRadioButton->Checked)
			modifyGraph(par, true);
	}

	//
	void TWindow::RemovePanel(int ind) {
		// Remove graph
		// Modifiy other graphIndex s
		// Remove panel from list
		// Delete panel
	}

	//
	bool TWindow::AddPanel() {
		return AddPanel(nullptr, std::vector<double>(), std::vector<double>());
	}

	bool TWindow::AddPanel(String ^ fileN, std::vector<double> x, std::vector<double> y) {
		// Find first letter (designation) that's not taken (if none, return;)
		// Create new panel and add to list and to FlowPanel
		char des = 'A';
		bool notFound = true;
		int size = FileExpressionList->Count;

		if(size > 0) {
			for(des--; des <= 'Z' && notFound;) {
				notFound = false;
				des++;
				for(int i = 0; i < size; i++) {
					if(FileExpressionList[i]->designationTextBox->Text->Length > 0)
						notFound = notFound || (FileExpressionList[i]->designationTextBox->Text[0] == des);
				}
			} // for des

			if((des > 'Z') && !(notFound))
				return false;
		}

		FileExpressionPanel^ grr;
		if(fileN)
			grr = gcnew FileExpressionPanel(fileN, x, y, des);
		else
			grr = gcnew FileExpressionPanel(des);

		FileExpressionList->Add(grr);

		grr->Width = max(400, exprFlowLayoutPanel->Width - 5);
		grr->Dock = Windows::Forms::DockStyle::Top;
		grr->AllowDrop = true;

		//////////////////
		// Panel Events //
		//////////////////
		grr->expressionTextBox->Leave			+= gcnew System::EventHandler(this, &TWindow::exprTextBox_Leave);
		grr->fileRadioButton->Click				+= gcnew System::EventHandler(this, &TWindow::fileRadioButton_Click);
		grr->visCheckBox->CheckedChanged		+= gcnew System::EventHandler(this, &TWindow::visCheckedChanged);
		grr->fileRadioButton->CheckedChanged	+= gcnew System::EventHandler(this, &TWindow::radioButton_CheckChanged);
		grr->colorLabel->BackColorChanged		+= gcnew System::EventHandler(this, &TWindow::ChangeColor);
		grr->removeButton->Click				+= gcnew System::EventHandler(this, &TWindow::RemoveButtonClick);
		grr->exportButton->Click				+= gcnew System::EventHandler(this, &TWindow::exportButtonClick);
		grr->DragDrop							+= gcnew System::Windows::Forms::DragEventHandler(this, &TWindow::panel_DragDrop);
		grr->DragEnter							+= gcnew System::Windows::Forms::DragEventHandler(this, &TWindow::panel_DragEnter);
		grr->DragLeave							+= gcnew System::EventHandler(this, &TWindow::panel_DragLeave);
		
		this->exprFlowLayoutPanel->Controls->Add(grr);

		modifyGraph(grr, !(grr->expressionRadioButton->Checked));

		return true;
	}

	//
	void TWindow::addPanelToolStripMenuItem_Click(System::Object ^sender, System::EventArgs ^e) {
		AddPanel();
	}	
	//
	FileExpressionPanel^ TWindow::GetParent(System::Object ^sender) {
		try {
			return (FileExpressionPanel^)((System::Windows::Forms::Control^)(sender))->Parent;
		} catch (...) {
			int i = 0;
			i++;
			return nullptr;
		}
	}

	//
	bool TWindow::LoadFile(System::Object ^sender) {
		FileExpressionPanel^ par = GetParent(sender);

		ofd->Multiselect = false;
		ofd->Title = L"Select file to be displayed";
		ofd->Filter = "Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
		
		if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return false;
		
		return LoadFile(ofd->FileName, par);

	}

	//
	bool TWindow::LoadFile(System::String ^fileName) {
		if(!AddPanel()) {
			System::Windows::Forms::MessageBox::Show(
				"There are too many open files. This program has a limit of 26 files. Please remove some open files and try again.", 
					"Too many open files", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
			return false;

		}
		FileExpressionPanel^ par = (FileExpressionPanel^)(exprFlowLayoutPanel->Controls[exprFlowLayoutPanel->Controls->Count - 1]);
		return LoadFile(fileName, par);
	}

	bool TWindow::LoadFile(System::String ^fileName, FileExpressionPanel ^par) {
		std::wstring file = clrToWstring(fileName);

		std::vector<double> x, y;

		// The program cannot tolerate files with 1 point or less
		ReadDataFile(file.c_str(), x, y);
		if(x.size() <= 1) {
			System::Windows::Forms::MessageBox::Show("The chosen file is invalid or empty and has been ignored", 
					"Invalid data file", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
			return false;
		} else {
			*(par)->xF = x;
			*(par)->yF = y;

			modifyGraph(par, false);

			par->filenameTextBox->Text = CLRBasename(stringToClr(file))->Remove(0, 1);
			par->ttip->SetToolTip(par->filenameTextBox, stringToClr(file));

			for(int i = 0; i < FileExpressionList->Count; i++) {
				String	^exBox	= FileExpressionList[i]->expressionTextBox->Text,
						^des	= par->designationTextBox->Text;
				if(exBox->Contains(des))
					exprTextBox_Leave(FileExpressionList[i]->expressionTextBox, gcnew System::EventArgs());
			}
		} // else
		return true;
	}
	//
	void TWindow::fileRadioButton_Click(System::Object^  sender, System::EventArgs^  e) {
		FileExpressionPanel^ par = GetParent(sender);

		if(!bHandled && par->fileRadioButton->AutoCheck)
			LoadFile(sender);
		bHandled = false;

	}
	//
	void TWindow::radioButton_CheckChanged(System::Object^  sender, System::EventArgs^  e) {
		// TODO change the graph
		FileExpressionPanel^ par = GetParent(sender);
		
		bHandled = modifyGraph(par, !(par->fileRadioButton->Checked));

		par->x = (par->fileRadioButton->Checked ? par->xF : par->xE);
		par->y = (par->fileRadioButton->Checked ? par->yF : par->yE);
	}
	//
	bool TWindow::modifyGraph(FileExpressionPanel ^ grr, bool ex) {
		std::vector<double> *x, *y;
		x = ex ? grr->xE : grr->xF;
		y = ex ? grr->yE : grr->yF;

		if(x->size() > 1 && y->size() > 1) {
			if(grr->graphIndex < 0) {
				wgtGraph->Add(grr->GetColorRef(), GraphToolkit::Graph1D::GraphDrawType::DRAW_LINES, vectortoarray(*x), vectortoarray(*y));
				grr->graphIndex = wgtGraph->numGraphs - 1;
			}  else {
				// Modify existing
				wgtGraph->Modify(grr->graphIndex, vectortoarray(*x), vectortoarray(*y));
			}

			draw();

			return true;
		}

		return false;
	}

	//
	void TWindow::visCheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		FileExpressionPanel^ par = GetParent(sender);
		wgtGraph->SetGraphVisibility(par->graphIndex, par->visCheckBox->Checked);
	
		draw();
	}

	//
	void TWindow::ChangeColor(System::Object ^sender, System::EventArgs ^e) {
		FileExpressionPanel^ par = GetParent(sender);
		if(par->graphIndex < 0)
			return;

		if(wgtGraph) {
			wgtGraph->ChangeColor(par->graphIndex, par->GetColorRef());
			wgtGraph->Invalidate();
			draw();
		}
	}

	//
	void TWindow::draw() {					
		wgtGraph->ShowYTicks = (bool)wgtGraph->ShowYTicks;
		wgtGraph->Invalidate();
	}

	//
	bool TWindow::loadVariables(bool bRange) {
		int sizeD = 26;
		bool range = false;
		// Load scalar data
		for(int i = 0; i < variableDataGridView->RowCount; i++) {
			(*coefficients)(i) = clrToDouble((String^)(variableDataGridView["valColumn", i]->Value));
		}

		// Load vector data
		int maxLen = 0;
		for(int i = 0; i < (int)FileExpressionList->Count; i++) {
			try {
				FileExpressionList[i]->x = (FileExpressionList[i]->fileRadioButton->Checked ? FileExpressionList[i]->xF : FileExpressionList[i]->xE);
				FileExpressionList[i]->y = (FileExpressionList[i]->fileRadioButton->Checked ? FileExpressionList[i]->yF : FileExpressionList[i]->yE);
				maxLen = max(maxLen, (int)FileExpressionList[i]->x->size());
			} catch (...) {
				continue;
			}
		}

		if(maxLen < 2)
			return false;

		double minR = clrToDouble(minRangeTextBox->Text), maxR = clrToDouble(maxRangeTextBox->Text);
		if(bRange && minR < maxR) {
			range = true;
			sizeD++;
		}

		(*dataArray).resize(maxLen, sizeD);
		(*xDataArray).resize(maxLen, sizeD);

		(*dataArray) = (*dataArray) * 0.0;
		(*xDataArray) = (*xDataArray) * 0.0;

		for(int i = 0; i < (int)FileExpressionList->Count; i++) {
#ifdef _DEBUG
			std::vector<double> DEBX;
#endif
			FileExpressionPanel^ itm = FileExpressionList[i];
			if(!(itm->x))
				continue;
			itm->x = (itm->fileRadioButton->Checked ? itm->xF : itm->xE);
			itm->y = (itm->fileRadioButton->Checked ? itm->yF : itm->yE);
			for(int j = 0; j < (int)itm->x->size(); j++) {
#ifdef _DEBUG
				DEBX.push_back(itm->x->at(j));
#endif
				((*dataArray)(j, itm->designationTextBox->Text[0] - 'A')) = itm->y->at(j);
				((*xDataArray)(j, itm->designationTextBox->Text[0] - 'A')) = itm->x->at(j);
			}
		}

		if(range) {
			(*xDataArray)(0, sizeD - 1) = minR;
			(*xDataArray)(1, sizeD - 1) = maxR;
			(*dataArray)(0, sizeD - 1) = 1.0;
			(*dataArray)(1, sizeD - 1) = 1.0;
		}


		Eigen::ArrayXi *pm = pmut;
		int ma = 26;
		(*pmut) = Eigen::ArrayXi::Zero(ma);
		double minC = 0.0, maxC = 0.0;
		for(int i = 0; i < ma; i++) {
			// Mutability
			if(variableDataGridView["MutColumn", i]->Value->Equals("Y"))
				(*pm)(i)++;

			// Constraints
			minC = clrToDouble((String^)(variableDataGridView["MinColumn", i]->Value));
			maxC = clrToDouble((String^)(variableDataGridView["MaxColumn", i]->Value));
			if(minC < maxC) {
				pMin->num(i) = minC;
				pMax->num(i) = maxC;
			} else {
				pMin->num(i) = -numeric_limits<double>::infinity();
				pMax->num(i) = numeric_limits<double>::infinity();
			}
		}


		return true;
	}

	//
	void TWindow::RemoveButtonClick(System::Object ^sender, System::EventArgs ^e) {
		FileExpressionPanel^ par = GetParent(sender);

		int grInd = par->graphIndex;

		for(int i = 0; i < FileExpressionList->Count; i++) {
			if(FileExpressionList[i]->graphIndex > grInd)
				FileExpressionList[i]->graphIndex--;
		}

		if(wgtGraph) {
			wgtGraph->Remove(grInd);
			draw();
		}

		bool rmvd = FileExpressionList->Remove(par);

		delete par;

	}
	//
	void TWindow::exportButtonClick(System::Object ^sender, System::EventArgs ^e) {
		std::wstring file;
		FileExpressionPanel^ par = GetParent(sender);

		par->x = (par->fileRadioButton->Checked ? par->xF : par->xE);
		par->y = (par->fileRadioButton->Checked ? par->yF : par->yE);

		if(!(par->x)			||
			!(par->y)			||
			par->x->size() < 2	||
			par->y->size() < 2) {
				// TODO: Report empty vector
				;
				return;
		}

		sfd->Filter = L"Output Files (*.out)|*.out|Data Files (*.dat, *.chi)|*.dat;*.chi|All files|*.*";
		sfd->FileName = L"";
		if(!par->fileRadioButton->AutoCheck)
			sfd->FileName = par->filenameTextBox->Text;

		sfd->Title = "Choose a signal output file";
		if(sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		clrToString(sfd->FileName, file);
		try {
			if(par->fileRadioButton->Checked)
				WriteDataFile(file.c_str(), *(par->x), *(par->y));
			else {
				std::stringstream st;
				std::string strg = " ";
				st << "#\tExpression:\t" << clrToString(par->expressionTextBox->Text) << "\n";
				for(char ch = 'a'; ch <= 'z'; ch++) {
					strg[0] = ch;
					if(par->expressionTextBox->Text->Contains(stringToClr(strg)))
						st << "#\t" << ch << " = " << clrToString((String^)(variableDataGridView["ValColumn", int(ch - 'a')]->Value)) << "\n";
				}
				for(char ch = 'A'; ch <= 'Z'; ch++) {
					strg[0] = ch;
					if(par->expressionTextBox->Text->Contains(stringToClr(strg))) {
						FileExpressionPanel^ ref = par;
						
						for(int i = 0; i < FileExpressionList->Count; i++)
							if(FileExpressionList[i]->designationTextBox->Text->Contains(stringToClr(strg)))
								ref = FileExpressionList[i];
						st << "#\t" << ch << " = " << 
							clrToString(ref->fileRadioButton->Checked ? ref->filenameTextBox->Text : (ref->fileRadioButton->AutoCheck ? ref->expressionTextBox->Text : ref->filenameTextBox->Text)) << "\n";
					}
				}
				WriteDataFileWHeader(file.c_str(), *(par->x), *(par->y), st);
			}
		} catch (...) {
			// TODO: report error writing file
			return;
		}
	}
	//
	void TWindow::logCheckedChanged(System::Object ^sender, System::EventArgs ^e) {
		wgtGraph->LogScaleX = logQcheckBox->Checked;
		wgtGraph->LogScaleY = logIcheckBox->Checked;		
	}

	void TWindow::variableDataGridView_CellEndEdit(System::Object ^sender, System::Windows::Forms::DataGridViewCellEventArgs ^e) {
		int col = e->ColumnIndex, row = e->RowIndex;
		String ^str = gcnew String("0.000000");

		if(col > 1) {	// Make sure it's a number
			if((String^)(((DataGridView^)(sender))[col, row]->Value))
				str = (String^)(((DataGridView^)(sender))[col, row]->Value);
			((DataGridView^)(sender))[col, row]->Value = gcnew String(clrToDouble(str).ToString());
		}

		if(col == 2) {
			for(int j = 0; j < FileExpressionList->Count; j++) {
				String	^exBox	= FileExpressionList[j]->expressionTextBox->Text,
						^cel	= (String^)(((DataGridView^)(sender))["varColumn", row]->Value);
				if(exBox->Contains(cel))
					exprTextBox_Leave(FileExpressionList[j]->expressionTextBox, e);
			} // for
		}

	}

	void TWindow::timer1_Tick(System::Object ^sender, System::EventArgs ^e) {
		bool bChanged = false;
		double mid = (trackBar1->Maximum - trackBar1->Minimum) / 2.0;

		if(trackBar1->Value != (int)mid) {
			for(int i = 0; i < variableDataGridView->SelectedCells->Count; i++) {
				if(variableDataGridView->SelectedCells[i]->ColumnIndex < 2)
					continue;
				dealWithTrackBar(variableDataGridView->SelectedCells[i], trackBar1, 10.0);
			}
			bChanged = true;
		}

		if(bChanged) {
			// Check each expression to see if it contains the relevant variable
			// Trigger expression evaluation
			for(int i = 0; i < variableDataGridView->SelectedCells->Count; i++) {
				if(variableDataGridView->SelectedCells[i]->ColumnIndex == 2) {
					for(int j = 0; j < FileExpressionList->Count; j++) {
						String	^exBox	= FileExpressionList[j]->expressionTextBox->Text,
								^cel	= (String^)(variableDataGridView["varColumn", (variableDataGridView->SelectedCells[i]->RowIndex)]->Value);
						if(exBox->Contains(cel))
							exprTextBox_Leave(FileExpressionList[j]->expressionTextBox, e);
					} // for
				} // if
			} //for
		} // if bChanged
	}

	void TWindow::dealWithTrackBar(DataGridViewCell^ cell, TrackBar ^tb, double factor) {
		double mid = (tb->Maximum - tb->Minimum) / 2.0;

		// The following is meant to allow a relative change of the value and also allow the transition
		//	to negative numbers and from zero
		String ^str = (String^)(cell->Value);

		int G = str->LastIndexOf(".");
		double val = clrToDouble((G >= 0) ? str->Remove(G, 1) : str);
		if(fabs(val / factor) <= 1.0)
			factor = val - 0.9 * (val < 0.0 ? -1.0 : 1.0);
		double newVal = clrToDouble(str);
		if(fabs(val) < 1.1)
			newVal += (newVal < 0.0 ? -1.0 : 1.0) * powf(1.0f, (float)-6);
		newVal *= (1.0 + (newVal < 0.0 ? -1.0 : 1.0) * ((tb->Value - mid) / tb->Maximum / fabs(factor)));

		char a[64] = {0};
		sprintf(a, "%.*f", 6, newVal);

		cell->Value = gcnew String(a);
	}

	void TWindow::variableDataGridView_SelectionChanged(System::Object ^sender, System::EventArgs ^e) {
		trackBar1->Value = int((trackBar1->Maximum - trackBar1->Minimum) / 2.0);
	}

	void TWindow::trackBar1_MouseUp(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e) {
		trackBar1->Value = int((trackBar1->Maximum - trackBar1->Minimum) / 2.0);
	}

	void TWindow::fitTextBox_KeyPress(System::Object ^sender, System::Windows::Forms::KeyPressEventArgs ^e) {
		if(!Char::IsLetter(e->KeyChar) && !(e->KeyChar == Convert::ToChar(Keys::Back)))
			e->Handled = true;

		System::String ^str = gcnew System::String((Char::ToUpper(e->KeyChar)).ToString());

		for(int i = 0; i < FileExpressionList->Count; i++) {
			if(FileExpressionList[i]->designationTextBox->Text->Contains((Char::ToUpper(e->KeyChar)).ToString())) {
				((TextBox^)(sender))->Text = L"";
				e->KeyChar = (Char::ToUpper(e->KeyChar));
				return;
			}
		}

		e->Handled = true;
	}

	void TWindow::fitButton_Click(System::Object ^sender, System::EventArgs ^e) {
		String ^xStr = fitTextBox->Text, ^yStr = toTextBox->Text;
		FileExpressionPanel ^yPan, ^xPan;
		
		int iters = (int)clrToDouble(fitIterationsToolStripTextBox->Text);
		SetFitIterations(iters);
		fitIterationsToolStripTextBox->Text = iters.ToString();

		// Check to see that the chosen panels can work
		if(xStr->Length < 1 || yStr->Length < 1)
			return;
		for(int i = 0; i < FileExpressionList->Count; i++) {
			if(FileExpressionList[i]->designationTextBox->Text->Contains(xStr))
				xPan = FileExpressionList[i];
			if(FileExpressionList[i]->designationTextBox->Text->Contains(yStr))
				yPan = FileExpressionList[i];
		}

		if(!((yPan->x) && !(yPan->x->size() < 2))			||
			!(xPan->expressionRadioButton->Checked)			||
			(xPan->expressionTextBox->Text->Length < 1)
			) {
			MessageBox::Show("Check fit designations.", "Fit not initialized" ,MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}
		
		// organize relevant curves and variables
		if(!loadVariables(true))
			return;

		std::vector<string> errs;
		ArrayXXd	dummyData	= *dataArray,
					dummyX		= *xDataArray;
		ArrayXd  dummyCoefs		= *coefficients, y, x, wts, paramErrs, modelErrs, pMut;

		y.resize(yPan->y->size());
		x.resize(yPan->x->size());
		for(int i = 0; i < (int)yPan->y->size(); i++) {
			y(i) = yPan->y->at(i);
			x(i) = yPan->x->at(i);
		}

		bool sub = FitCoeffs(y, x, clrToString(xPan->expressionTextBox->Text), dummyX, dummyData, dummyCoefs,
             *pmut, pMin, pMax, paramErrs, modelErrs);


		//Hyperplane<double, 27> hyp;
//		Eigen::Hyperplane<double, > hyp(

		// Replace the old coefficients with the new ones
		string ch = "a";
		ch[0]--;
		for(int i = 0; i < 26; i++) {
			ch[0]++;
			if((*pmut)(i) < 1 || !(xPan->expressionTextBox->Text->Contains(stringToClr(ch))))
				continue;
			char a[64] = {0};
			sprintf(a, "%.*f", 6, dummyCoefs(i));

			variableDataGridView["ValColumn", i]->Value = gcnew String(a);
		}

		/** Remove the min and max **/
		loadVariables(false);
		exprTextBox_Leave(xPan->expressionTextBox, gcnew System::EventArgs());

		/** Cycle through the panels and update those that were dependant on the mutables or the fitted curve **/
		for(int i = 0; i < FileExpressionList->Count; i++) {
			bool redraw = false;

			// Check for referenced mutables
			for(int j = 0; j < (*pmut).size() && !redraw; j++) {
				if(FileExpressionList[i] == xPan)
					continue;
				string ch = "a";
				ch[0] += j;
				if(FileExpressionList[i]->expressionTextBox->Text->Contains(stringToClr(ch)))
					redraw = true;
			}

			// Check for referenced 
			if(FileExpressionList[i]->expressionTextBox->Text->Contains(xPan->designationTextBox->Text))
				redraw = true;

			if(redraw)
				exprTextBox_Leave(FileExpressionList[i]->expressionTextBox, gcnew System::EventArgs());
		}
	}

	void TWindow::fitManyButton_Click(System::Object ^sender, System::EventArgs ^e) {
		openManyFD->Title = L"Select files to be fit to";
		openManyFD->Filter = "Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";

		if(openManyFD->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;
		
		seriesDataGridView->ColumnCount	= 0;
		seriesDataGridView->RowCount	= 0;

		files = openManyFD->FileNames;

		// Validate and import files

		ArrayXXd manyFilesX, manyFilesY;

		seriesDataGridView->Columns->Add(L"fileColumn", L"File");
		seriesDataGridView->Columns->Add(L"fileIndColumn","NotVisible");
		seriesDataGridView->Columns[L"fileIndColumn"]->Visible = false;
		seriesDataGridView->RowCount = files->Length;
		for(int i = 0; i < files->Length; i++) {
			seriesDataGridView[L"fileColumn", i]->Value = gcnew String(CLRBasename(files[i]->ToString())->Remove(0,1));
			seriesDataGridView[L"fileIndColumn", i]->Value = gcnew String(i.ToString());
		}

		seriesDataGridView->AutoResizeColumns();

		fitNowButton->Enabled = true;
	}

	void TWindow::fitExpressionTextBox_Leave(System::Object^  sender, System::EventArgs^  e) {
		TextBox^ tb = (TextBox^)sender;

		fitNowButton->Enabled = false;

		if(tb->Text->Length < 1)
			return;

		if(!loadVariables(false))
			return;

		std::vector<string> errs;
		ArrayXXd	dummyData	= *dataArray,
					dummyX		= *xDataArray;
		ArrayXd		resX;

		ArrayXd  dummyCoefs = *coefficients;
		std::string expr = clrToString(tb->Text);
		ParserSolver parse(errs, expr, dummyX, dummyData, dummyCoefs);
		ArrayXd res = parse.EvaluateExpression(resX);

		if(!errs.empty()) {
			String ^erS = gcnew String(L"");
			for(int i = 0; i < (int)errs.size(); i++)
				erS += "\n" + stringToClr(errs[i]);

			System::Windows::Forms::MessageBox::Show(erS, 
					"Invalid expression", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
		
			return;
		}

		fitNowButton->Enabled = true;

	}

	void TWindow::fitExpressionTextBox_KeyPress(System::Object ^sender, System::Windows::Forms::KeyPressEventArgs ^e) {
#ifdef _DEBUG
		wchar_t deb = e->KeyChar;
#endif
		if((!(
			Char::IsLetterOrDigit(e->KeyChar) ||
			Char::IsWhiteSpace(e->KeyChar) ||
			(e->KeyChar == '/' || e->KeyChar == '*' || e->KeyChar == '+' || e->KeyChar == '-') || 
			(e->KeyChar == '(' || e->KeyChar == ')' || e->KeyChar == '.' || e->KeyChar == '/') ||
			(e->KeyChar == Convert::ToChar(Keys::Back))
			))				&&
			// Exceptions
			// copy and paste
			!(int(e->KeyChar) == 3 || int(e->KeyChar) == 22)
			)
			e->Handled = true;

	}

	void TWindow::fitNowButton_Click(System::Object ^sender, System::EventArgs ^e) {
		fitNowButton->Enabled = false;
		// TODO: Make a different thread

		// Read expression
		// Add columns based on number of variables
		// Loop
		//	Read file
		//	Send file data to fit
		//	Write final parameters to datagridview
		// End loop

		int iters = (int)clrToDouble(fitIterationsToolStripTextBox->Text);
		SetFitIterations(iters);
		fitIterationsToolStripTextBox->Text = iters.ToString();

		String ^ex = fitExpressionTextBox->Text;
		// Check to see that the chosen panels can work
		if(ex->Length < 1)
			return;

		// organize relevant curves and variables
		if(!loadVariables(true))
			return;

		std::vector<string> errs;
		ArrayXXd	dummyData	= *dataArray,
					dummyX		= *xDataArray;
		ArrayXd  dummyCoefs		= *coefficients, y, x, wts, paramErrs, modelErrs, pMut;

		seriesDataGridView->ColumnCount = 2;	// filename and index

		seriesDataGridView->Columns->Add(L"nameColumn", L"Name");
		seriesDataGridView->Columns["nameColumn"]->ReadOnly = false;

		String ^str = gcnew String(" ");
		for(char ch = 'a'; ch <= 'z'; ch++) {
			str = str->Replace(str[0], ch);
			if(ex->Contains(str)) {
				seriesDataGridView->Columns->Add(str + L"Column", str);
				seriesDataGridView->Columns[str + L"Column"]->AutoSizeMode = DataGridViewAutoSizeColumnMode::DisplayedCells;
			}
		}
		seriesDataGridView->Columns->Add(L"errColumn", L"\u03c7\u00b2"); // Chi^2
		seriesDataGridView->Columns->Add(L"err2Column", L"R\u00b2"); // R^2

		if(seriesDataGridView->ColumnCount < 3)	// Nothing to fit
			/* Column count:
			 *	1 - filename
			 *	2 - fileIndColumn	*/
			return;

		int ma = 26;
		ArrayXd a  = ArrayXd::Zero(ma); // Parameter vector
		cons pMin(ma), pMax(ma);
		ArrayXi pmut = ArrayXi::Zero(ma); // Mutability vector
		for(int k = 0; k < ma; k++)
			if(variableDataGridView["MutColumn", k]->Value->Equals("Y"))
				pmut[k]++;

		for(int i = 0; i < seriesDataGridView->Rows->Count; i++) {
			int fileInd = Convert::ToInt32(seriesDataGridView[L"fileIndColumn", i]->Value);
			std::wstring file = clrToWstring(files[fileInd]);

			std::vector<double> xV, yV;

			// The program cannot tolerate files with 1 point or less
			ReadDataFile(file.c_str(), xV, yV);
			if(xV.size() <= 1) {
				seriesDataGridView[L"nameColumn", i]->Value = gcnew String("Bad file");
				continue;
			} else {
				x.resize(xV.size());
				y.resize(yV.size());
				for(int j = 0; j < (int)xV.size(); j++) {
					x(j) = xV[j];
					y(j) = yV[j];
				}
			} // else

			//TODO: Take expression, determine which coefficients and curves should be used (delete others?) (as PreCalculate)
			//		Move parser and calculator to backend.
			//		collect mutables and ranges
			//		

			bool sub = FitCoeffs(y, x, clrToString(ex), dummyX, dummyData, dummyCoefs,
				 pmut, &pMin, &pMax, paramErrs, modelErrs);


			// Fill in the coefficients with the new ones
			string ch = "a";
			ch[0]--;
			for(int k = 0; k < ma; k++) {
				ch[0]++;
				if(pmut[k] < 1 || !(fitExpressionTextBox->Text->Contains(stringToClr(ch))))
					continue;
				char g[64] = {0};
				sprintf(g, "%.*f", 6, dummyCoefs(k));

				str = str->Replace(str[0], char(int('a' + k)));
				seriesDataGridView[str + L"Column", i]->Value = gcnew String(g);
			}
			char g[64] = {0};
			sprintf(g, "%.*f", 12, modelErrs(0));
			seriesDataGridView["errColumn", i]->Value = gcnew String(g);
			sprintf(g, "%.*f", 12, modelErrs(1));
			seriesDataGridView["err2Column", i]->Value = gcnew String(g);

		}	// for i < files->Length

		fitNowButton->Enabled = true;
	}


	void TWindow::rangeTB_KeyPress(System::Object ^sender, System::Windows::Forms::KeyPressEventArgs^  e) {
		if(!Char::IsDigit(e->KeyChar) && !(e->KeyChar == Convert::ToChar(Keys::Back)) && e->KeyChar != '.' &&
			!(e->KeyChar == Convert::ToChar(Keys::Back)) &&
			(!(int(e->KeyChar) == 3 || int(e->KeyChar) == 22)))
			e->Handled = true;

	}

	void TWindow::rangeTB_Leave(System::Object^  sender, System::EventArgs^  e) {
		TextBox^ tb = (TextBox^)sender;

		if(tb->Text)
			tb->Text = clrToDouble(tb->Text).ToString("0.000000");
		else
			tb->Text = L"0.000000";
	}

	void TWindow::exportSeriesToolStripMenuItem_Click(System::Object ^sender, System::EventArgs ^e) {
		if(sender == exportSeriesTableToolStripMenuItem)
			if(seriesDataGridView->ColumnCount < 1)
				return;
		if(sender == exportSeriesCurvesToolStripMenuItem)
			if(seriesDataGridView->ColumnCount < 2)
				return;

		std::wstring file;
		String ^ commonName = gcnew String(L"");

		// See http://en.wikipedia.org/wiki/Longest_common_substring_problem
		//		for reason to do this later
		//commonName = (String^)seriesDataGridView[0, 0]->Value;
		//for(int i = 0; i < seriesDataGridView->RowCount; i++)
		//	commonName = 

		sfd->Filter = L"Tab Separated Values (*.tsv)|*.tsv|All files|*.*";
		sfd->FileName = commonName;

		sfd->Title = "Save output file as";
		if(sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		wstringstream str;
		std::string expr = clrToString(fitExpressionTextBox->Text);
		std::string ch = "A";

		// Header
		str << "# Fit to: " << expr.c_str() << "\n";
		for(; ch[0] <='Z'; ch[0]++) {
			if(fitExpressionTextBox->Text->Contains(stringToClr(ch))) {
				FileExpressionPanel^ ref = nullptr;
				
				for(int i = 0; i < FileExpressionList->Count; i++)
					if(FileExpressionList[i]->designationTextBox->Text->Contains(stringToClr(ch)))
						ref = FileExpressionList[i];
				if(ref == nullptr)
					continue;
				str << "#\t" << ch.c_str() << " = " << 
					clrToString(ref->fileRadioButton->Checked ? ref->filenameTextBox->Text : (ref->fileRadioButton->AutoCheck ? ref->expressionTextBox->Text : ref->filenameTextBox->Text)).c_str() << "\n";
			}
		}
		str << "# Range: " << clrToWstring(minRangeTextBox->Text).c_str() << " to: " << clrToWstring(maxRangeTextBox->Text).c_str() << "\n";

		if(sender == exportSeriesTableToolStripMenuItem) {
			// Column Headers
			for(int j = 0; j < seriesDataGridView->ColumnCount; j++)
				str << clrToWstring((String^)seriesDataGridView->Columns[j]->HeaderCell->Value) << "\t";
			str << "\n";
			// Table
			for(int i = 0; i < seriesDataGridView->RowCount; i++) {
				for(int j = 0; j < seriesDataGridView->ColumnCount; j++) {
					String^ tmpStr = (String^)seriesDataGridView[j, i]->Value;
					if(tmpStr)
						str << clrToWstring(tmpStr) << "\t";
					else
						str << " \t";
				}
				str << "\n";
			}
		} else if(sender == exportSeriesCurvesToolStripMenuItem) {
			if(!loadVariables(false))
				return;

			std::vector<string> errs;
			ArrayXXd	dummyData	= *dataArray,
						dummyX		= *xDataArray,
						resYarray;
			ArrayXd		resX;

			ArrayXd  dummyCoefs = *coefficients;
			ParserSolver parse(errs, expr, dummyX, dummyData, dummyCoefs);
			ArrayXd res = parse.EvaluateExpression(resX);

			if(!errs.empty()) {
				String ^erS = gcnew String(L"");
				for(int i = 0; i < (int)errs.size(); i++)
					erS += "\n" + stringToClr(errs[i]);

				System::Windows::Forms::MessageBox::Show(erS, 
						"Invalid expression", System::Windows::Forms::MessageBoxButtons::OK, 
						System::Windows::Forms::MessageBoxIcon::Warning);
				return;
			}

			resYarray.resize(resX.size(), seriesDataGridView->RowCount*2 + 1);
			resYarray.col(0) = resX;

			for(int i = 0; i < seriesDataGridView->RowCount; i++) {
				// Read file i
				// Interpolate curve to resX
				// Place in resYarray
				// Load modified coefficients
				// Replace 
				std::vector<double> x, y;
				ArrayXd		xA, yA;
				ReadDataFile(clrToWstring(files[i]).c_str(), x, y);
				if(x.size() < 3)
					continue;
				xA.resize(x.size());
				yA.resize(x.size());
				for(int j = 0;  j < (int)x.size(); j++) {
					xA(j) = x[j];
					yA(j) = y[j];
				}

				resYarray.col(2*i + 1) = parse.InterpolateCurve(xA, yA);

				for(string ch = "a"; ch[0] <= 'z'; ch[0]++) {
					if(fitExpressionTextBox->Text->Contains(stringToClr(ch)))
						dummyCoefs[ch[0] - 'a'] = clrToDouble((String^)seriesDataGridView[stringToClr(ch)+"Column", i]->Value);
				}
				parse.ReplaceCoefficients(dummyCoefs);
				resYarray.col(2*i + 2) = parse.CalculateExpression();
			}

			// Column Headers
			str << "q\t";
			for(int i = 0; i < seriesDataGridView->RowCount; i++)
				str << clrToWstring((String^)seriesDataGridView["fileColumn", i]->Value) << " Data\t"
					<< clrToWstring((String^)seriesDataGridView["fileColumn", i]->Value) << " Model\t";
			str << "\n";
			for(int i = 0; i < seriesDataGridView->RowCount; i++) {
				String^ tmpStr = (String^)seriesDataGridView["nameColumn", i]->Value;
				if(tmpStr)
					str << clrToWstring(tmpStr) << " Data\t" << clrToWstring(tmpStr) << " Model\t";
				else
					str << " \t";
			}
			str << "\n";

			// Data
			for(int j = 0; j < resYarray.rows(); j++) {
				for(int i = 0; i < resYarray.cols(); i++)
					str << resYarray(j,i) << "\t";
				str << "\n";
			}

		} else return;

		clrToString(sfd->FileName, file);
		try {
			// Open a file for writing
			FILE *fp;

			if ((fp = _wfopen(file.c_str(), L"w, ccs=UTF-8")) == NULL) {
				fprintf(stderr, "Error opening file %s for writing\n",
								file);
				
				MessageBox::Show("Please make sure that the file is not open.", "Error opening file for writing", MessageBoxButtons::OK,
										 MessageBoxIcon::Error);
				return;
			}
			fwprintf(fp, str.str().c_str());
			fclose(fp);

		} catch (...) {
			MessageBox::Show("#%&#%.", "Error writing file", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}
	}

	void TWindow::seriesDataGridView_ColumnAdded(System::Object^  sender, System::Windows::Forms::DataGridViewColumnEventArgs^  e) {
		e->Column->ReadOnly = true;
		// TEMPORARY: DISABLE SORT IN SERIESDATAGRIDVIEW UNTIL WE LINK THE FILES
		//e->Column->SortMode = DataGridViewColumnSortMode::Programmatic;
	}

	void TWindow::fitIterationsToolStripTextBox_Leave(System::Object ^sender, System::EventArgs ^e) {
		int iters = (int)clrToDouble(fitIterationsToolStripTextBox->Text);
		SetFitIterations(iters);
		fitIterationsToolStripTextBox->Text = iters.ToString();
	}

	void TWindow::addMultiplePanelsToolStripMenuItem_Click(System::Object ^sender, System::EventArgs ^e) {
		bool badFile = false;
		bool good = true;
		ofd->Multiselect = true;

		ofd->Title = L"Select files to be added";
		ofd->Filter = "Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";
		
		if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		for(int f = 0; f < ofd->FileNames->Length && good; f++) {
			std::wstring file = clrToWstring(ofd->FileNames[f]);
			std::vector<double> x, y;

			// The program cannot tolerate files with 1 point or less
			ReadDataFile(file.c_str(), x, y);
			if(x.size() <= 1) {
				badFile = true;
			} else {
				good = AddPanel(ofd->FileNames[f], x, y);
			}

			if(!good)
				continue;

			modifyGraph(FileExpressionList[FileExpressionList->Count - 1], false);
		} // for Filenames

		if(badFile)
			System::Windows::Forms::MessageBox::Show("One (or more) of the chosen files is invalid or empty and has been ignored", 
					"Invalid data file(s)", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);

		if(!good)
			System::Windows::Forms::MessageBox::Show("There appear to be 26 open files. This is the max allowed at this point in time.", 
					"No more files allowed", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);


	}

	void TWindow::wgt_MouseMove(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e) {
		if(!wgtGraph || !wgtGraph)
			return;
		 
		GraphToolkit::DoublePair loc (0.0, 0.0);
		loc = wgtGraph->PointToData(e->X - wgtGraph->xoff, e->Y - wgtGraph->yoff);

		if(wgtGraph->LogScaleX)
			loc.first = pow(10.0, loc.first);

		if(wgtGraph->LogScaleY)
			loc.second = pow(10.0, loc.second);

		LocOnGraph->Text = "("+ Double(loc.first).ToString((loc.first < 1.0e-3 || loc.first > 1.0e5) ? ("e5") : ("0.000000")) + ", "
							+ Double(loc.second).ToString((loc.second < 1.0e-3 || loc.second > 1.0e5) ? ("e5") : ("0.000000")) + ")";

	}
	
	bool TWindow::checkForChanges(bool &changeVec) {
		//int g = changeVec.
		return true;
	}
		
	void TWindow::exprFlowLayoutPanel_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		exprFlowLayoutPanel->BackColor = exprFlowLayoutPanel->DefaultBackColor;

		array<System::String ^> ^files = gcnew array<System::String ^>(1);
		ArrayList^ dropped = gcnew ArrayList();
		dropped->AddRange(dynamic_cast<array<System::String^>^> (e->Data->GetData(DataFormats::FileDrop, false)));
		
		files = GetDroppedFilenames(dropped);

		for(int i = 0; i < files->Length && LoadFile(files[i]); i++);

		if(files)
			delete files;
			
	}

	void TWindow::exprFlowLayoutPanel_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		if(e->Data->GetDataPresent(DataFormats::FileDrop)) {
			e->Effect = DragDropEffects::All;
			exprFlowLayoutPanel->BackColor = Color::Red;
		} else {
			exprFlowLayoutPanel->BackColor = exprFlowLayoutPanel->DefaultBackColor; //Color::Red; //
			e->Effect = DragDropEffects::None;
		}

	}
	
	void TWindow::exprFlowLayoutPanel_DragLeave(System::Object^  sender, System::EventArgs^  e) {
		exprFlowLayoutPanel->BackColor = exprFlowLayoutPanel->DefaultBackColor;
	}

				 
	void TWindow::panel_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		((FileExpressionPanel^)(sender))->BackColor = ((FileExpressionPanel^)(sender))->DefaultBackColor;

		array<System::String ^> ^files = gcnew array<System::String ^>(1);
		ArrayList^ dropped = gcnew ArrayList();
		dropped->AddRange(dynamic_cast<array<System::String^>^> (e->Data->GetData(DataFormats::FileDrop, false)));
		
		files = GetDroppedFilenames(dropped);

		if(files->Length > 0)
			LoadFile(files[0], ((FileExpressionPanel^)(sender)));

		if(files)
			delete files;
	}

	void TWindow::panel_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
		if(e->Data->GetDataPresent(DataFormats::FileDrop)) {
			e->Effect = DragDropEffects::All;
			((FileExpressionPanel^)(sender))->BackColor = Color::Red;
		} else {
			((FileExpressionPanel^)(sender))->BackColor = ((FileExpressionPanel^)(sender))->DefaultBackColor; //Color::Red; //
			e->Effect = DragDropEffects::None;
		}
	}

	void TWindow::panel_DragLeave(System::Object^  sender, System::EventArgs^  e) {
		((FileExpressionPanel^)(sender))->BackColor = ((FileExpressionPanel^)(sender))->DefaultBackColor;
	}

	array<System::String ^> ^ TWindow::GetDroppedFilenames(ArrayList^ droppedFiles) {
		ArrayList ^ more = gcnew ArrayList();
		for(int i = 0; i < droppedFiles->Count; i++) {
			System::String^ st = droppedFiles[i]->ToString();
			if(System::IO::Directory::Exists(st)) {
				more->AddRange(System::IO::Directory::GetFiles(st, "*", System::IO::SearchOption::AllDirectories));
				droppedFiles->RemoveAt(i--);
			}
		}

		droppedFiles->AddRange(more);
		return reinterpret_cast<array<System::String^>^>(droppedFiles->ToArray(System::String::typeid));
	}

} // namespace PopulationGUI
