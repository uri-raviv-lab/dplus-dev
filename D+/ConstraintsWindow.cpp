#include "ConstraintsWindow.h"
#include "clrfunctionality.h"


System::Void DPlus::ConstraintsWindow::ConstraintsWindow_Load(System::Object^ sender, System::EventArgs^ e) {
	paramStruct ps = extEnt->GetParameters();
	int layerr, paramm;

	System::Windows::Forms::DataGridViewTextBoxColumn^		valueColumn	 = gcnew System::Windows::Forms::DataGridViewTextBoxColumn();
	System::Windows::Forms::DataGridViewTextBoxColumn^		minColumn	 = gcnew System::Windows::Forms::DataGridViewTextBoxColumn();
	System::Windows::Forms::DataGridViewTextBoxColumn^		maxColumn	 = gcnew System::Windows::Forms::DataGridViewTextBoxColumn();
	System::Windows::Forms::DataGridViewComboBoxColumn ^	minIndColumn = gcnew System::Windows::Forms::DataGridViewComboBoxColumn();
	System::Windows::Forms::DataGridViewComboBoxColumn ^	maxIndColumn = gcnew System::Windows::Forms::DataGridViewComboBoxColumn();

	valueColumn->ReadOnly	= true;
	maxColumn->ReadOnly		= false;
	minColumn->ReadOnly		= false;
	maxIndColumn->ReadOnly	= false;
	minIndColumn->ReadOnly	= false;

	minIndColumn->HeaderText	= "Greater than";
	minIndColumn->Name			= "GreaterThan";
	minColumn->HeaderText		= "Absolute minimum";
	minColumn->Name				= "AbsoluteMinimum";
	valueColumn->HeaderText		= "Value";
	valueColumn->Name			= "Value";
	maxIndColumn->HeaderText	= "Less than";
	maxIndColumn->Name			= "LessThan";
	maxColumn->HeaderText		= "Absolute maximum";
	maxColumn->Name				= "AbsoluteMaximum";

	minIndColumn->SortMode	= DataGridViewColumnSortMode::Programmatic;
	minColumn->SortMode		= DataGridViewColumnSortMode::Programmatic;
	valueColumn->SortMode	= DataGridViewColumnSortMode::Programmatic;
	maxIndColumn->SortMode	= DataGridViewColumnSortMode::Programmatic;
	maxColumn->SortMode		= DataGridViewColumnSortMode::Programmatic;

	dataGridView1->SelectionMode = DataGridViewSelectionMode::CellSelect;

	dataGridView1->ColumnCount = 0;
	dataGridView1->Columns->AddRange(gcnew cli::array< System::Windows::Forms::DataGridViewColumn^  >(5) {minIndColumn,
								minColumn, valueColumn, maxColumn, maxIndColumn});

	if(Sender == CONS_EXTRAPARAMETERS || Sender == CONS_PARAMETERS) {
		dataGridView1->RowCount = int(extIndices->size());
	} else if(Sender == CONS_XYZABG) {
		dataGridView1->RowCount = 6;
	}
	dataGridView1->AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode::DisplayedCells;
	
	DataGridViewComboBoxCell ^comboBox;
	dict = gcnew System::Collections::Generic::Dictionary<String^, int>();

	for(int i = 0; i < dataGridView1->RowCount; i++) {
		Parameter par;

		if(Sender == CONS_EXTRAPARAMETERS)
			dataGridView1->Rows[i]->HeaderCell->Value = stringToClr(extEnt->modelUI->GetExtraParameter(extIndices->at(i).first).name);
		else if(Sender == CONS_PARAMETERS)
			dataGridView1->Rows[i]->HeaderCell->Value = stringToClr(extEnt->modelUI->GetLayerName(extIndices->at(i).second)) + " " + stringToClr(extEnt->modelUI->GetLayerParamName(extIndices->at(i).first));
		else if(Sender == CONS_XYZABG) {
			switch(i) {
			case 0:
				dataGridView1->Rows[i]->HeaderCell->Value = L"X";
				par = ps.x;
				break;
			case 1:
				dataGridView1->Rows[i]->HeaderCell->Value = L"Y";
				par = ps.y;
				break;
			case 2:
				dataGridView1->Rows[i]->HeaderCell->Value = L"Z";
				par = ps.z;
				break;
			case 3:
				dataGridView1->Rows[i]->HeaderCell->Value = L"Alpha";
				par = ps.alpha;
				break;
			case 4:
				dataGridView1->Rows[i]->HeaderCell->Value = L"Beta";
				par = ps.beta;
				break;
			case 5:
				dataGridView1->Rows[i]->HeaderCell->Value = L"Gamma";
				par = ps.gamma;
				break;
			default:
				break;
			}
		}

		comboBox = gcnew DataGridViewComboBoxCell();
		System::Collections::Specialized::StringCollection ^sc = gcnew System::Collections::Specialized::StringCollection();
		if(Sender == CONS_XYZABG) {
			dict["None"] = -1;
			sc->Add("None");
			// TODO: Maybe: consider comparing the indices
		} else if(Sender == CONS_PARAMETERS){
			dict["None"] = -1;
			sc->Add("None");
			for(int k = 0; k < ps.layers; k++) {
				if(k == extIndices->at(i).second) {
					continue;
				}
				String ^ sdfh = stringToClr(extEnt->modelUI->GetLayerName(k));
				sc->Add(stringToClr(extEnt->modelUI->GetLayerName(k)));
				dict[stringToClr(extEnt->modelUI->GetLayerName(k))] = k;
			}
		} else {
			// Valid for Extra Parameters
			dict["None"] = -1;
			sc->Add("None");
		} 

		// Populate fields
		if(Sender == CONS_EXTRAPARAMETERS || Sender == CONS_PARAMETERS) {
			layerr = extIndices->at(i).first;
			paramm = extIndices->at(i).second;
			if(Sender == CONS_EXTRAPARAMETERS)
				par = ps.extraParams[layerr];
			else
				par = ps.params[layerr][paramm];
		}

		((DataGridViewComboBoxCell^)dataGridView1["GreaterThan", i])->DataSource = sc;//dict->Keys;
		((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->DataSource = sc;//dict->Keys;


		dataGridView1["Value",i]->Value = gcnew String(Double(par.value).ToString());
		dataGridView1["AbsoluteMinimum",i]->Value = gcnew String(Double(par.consMin).ToString());
		dataGridView1["AbsoluteMaximum",i]->Value = gcnew String(Double(par.consMax).ToString());
		
		if(Sender == CONS_EXTRAPARAMETERS) {
			((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value = 
				((par.consMaxIndex == -1) ? "None" : stringToClr(extEnt->modelUI->GetExtraParameter(par.consMaxIndex).name));

			((DataGridViewComboBoxCell^)dataGridView1["GreaterThan", i])->Value =
				((par.consMinIndex == -1) ? "None" : stringToClr(extEnt->modelUI->GetExtraParameter(par.consMinIndex).name));
		} else if(Sender == CONS_PARAMETERS) {
 			((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value = 
				((par.consMaxIndex == -1) ? "None" : stringToClr(extEnt->modelUI->GetLayerName(par.consMaxIndex)));
 		
 			((DataGridViewComboBoxCell^)dataGridView1["GreaterThan", i])->Value =
 				((par.consMinIndex == -1) ? "None" : stringToClr(extEnt->modelUI->GetLayerName(par.consMinIndex)));
		}
	}
	
	dataGridView1->AutoResizeRowHeadersWidth(DataGridViewRowHeadersWidthSizeMode::AutoSizeToDisplayedHeaders);
}

System::Void DPlus::ConstraintsWindow::OKButton_Click(System::Object^ sender, System::EventArgs^ e) {
	paramStruct oldPs = extEnt->GetParameters();
	paramStruct newPs(oldPs);
	int layer = -1, paramInd = -1;
	double val;
	int ind = -1;
	int endLoop;

	if(Sender == CONS_EXTRAPARAMETERS || Sender == CONS_PARAMETERS)
		endLoop = int(extIndices->size());
	else if(Sender == CONS_XYZABG)
		endLoop = 6;

	for(int i = 0; i < endLoop; i++) {
		Parameter par;

		if(Sender == CONS_EXTRAPARAMETERS || Sender == CONS_PARAMETERS) {

		layer	 = (extIndices->at(i)).first;
		paramInd = (extIndices->at(i)).second;

		if(Sender == CONS_EXTRAPARAMETERS)
			par = oldPs.extraParams[layer];
		else if(Sender == CONS_PARAMETERS)
			par = oldPs.params[layer][paramInd];
		} else if(Sender == CONS_XYZABG) {
			switch(i) {
			case 0:
				par = oldPs.x;
				break;
			case 1:
				par = oldPs.y;
				break;
			case 2:
				par = oldPs.z;
				break;
			case 3:
				par = oldPs.alpha;
				break;
			case 4:
				par = oldPs.beta;
				break;
			case 5:
				par = oldPs.gamma;
				break;
			default:
				break;
			}
		}

		// Default in case of deleted string or invalid floating point string
		par.consMax = std::numeric_limits<double>::infinity();
		if(dataGridView1["AbsoluteMaximum",i]->Value) {
			if(Double::TryParse(dataGridView1["AbsoluteMaximum",i]->Value->ToString(), val))
				par.consMax = val;
		}

		// Default in case of deleted string or invalid floating point string
		par.consMin = -std::numeric_limits<double>::infinity();
		if(dataGridView1["AbsoluteMinimum",i]->Value) {
			if(Double::TryParse(dataGridView1["AbsoluteMinimum",i]->Value->ToString(), val))
				par.consMin = val;
		}

		ind = -1;
		if(Sender == CONS_PARAMETERS && ((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value) {
			String ^sdtsd = ((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value->ToString();
			Object ^srsd = ((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value;
			ind = dict[((DataGridViewComboBoxCell^)dataGridView1["LessThan", i])->Value->ToString()];
		}
		par.consMaxIndex = ind;

		ind = -1;
		if(Sender == CONS_PARAMETERS && ((DataGridViewComboBoxCell^)dataGridView1["GreaterThan", i])->Value) {
			ind = dict[((DataGridViewComboBoxCell^)dataGridView1["GreaterThan", i])->Value->ToString()];
		}
		par.consMinIndex = ind;

		if(Sender == CONS_EXTRAPARAMETERS)
			newPs.extraParams[layer] = par;
		else if(Sender == CONS_PARAMETERS)
			newPs.params[layer][paramInd] = par;
		else if(Sender == CONS_XYZABG) {
			switch(i) {
			case 0:
				newPs.x = par;
				break;
			case 1:
				newPs.y = par;
				break;
			case 2:
				newPs.z = par;
				break;
			case 3:
				newPs.alpha = par;
				break;
			case 4:
				newPs.beta = par;
				break;
			case 5:
				newPs.gamma = par;
				break;
			default:
				break;
			}
		}
	}
	
	extEnt->SetParameters(newPs, parentForm->GetLevelOfDetail());
}

System::Void DPlus::ConstraintsWindow::dataGridView1_CellValidating(System::Object^  sender, System::Windows::Forms::DataGridViewCellValidatingEventArgs^  e) {
	
	dataGridView1->Rows[e->RowIndex]->ErrorText = "";
	int newInteger;
	// Don't try to validate the 'new row' until finished 
	// editing since there
	// is not any point in validating its initial value.
	if (dataGridView1->Rows[e->RowIndex]->IsNewRow) { return; }

	String^ columnName = dataGridView1->Columns[e->ColumnIndex]->Name;
	int minColInd = dataGridView1->Columns["AbsoluteMinimum"]->Index;
	int maxColInd = dataGridView1->Columns["AbsoluteMaximum"]->Index;
	if (e->ColumnIndex == minColInd)
	{
		double newVal;
		double maxVal;
		if (!Double::TryParse(dataGridView1[maxColInd,e->RowIndex]->Value->ToString(), maxVal))
			maxVal = std::numeric_limits<double>::infinity();
		if (!Double::TryParse(e->FormattedValue->ToString(), newVal) || newVal > maxVal)
		{
				e->Cancel = true;
				dataGridView1->Rows[e->RowIndex]->ErrorText = "The value must be smaller than the value of Absolute Maximum";

		}
	}
	if (e->ColumnIndex == maxColInd)
	{
		double newVal;
		double minVal;
		if (!Double::TryParse(dataGridView1[minColInd, e->RowIndex]->Value->ToString(), minVal))
			minVal = std::numeric_limits<double>::infinity();
		if (!Double::TryParse(e->FormattedValue->ToString(), newVal) || newVal < minVal)
		{
			e->Cancel = true;
			dataGridView1->Rows[e->RowIndex]->ErrorText = "The value must be greater than the value of Absolute Maximum";
		}
	}
}