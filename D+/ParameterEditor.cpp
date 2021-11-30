#include <set>

#include "ParameterEditor.h"
#include "clrfunctionality.h"
#include "ConstraintsWindow.h"

void DPlus::ParameterEditor::consDummy() {
	allUIObjects = gcnew Generic::List<Control^>();
	allUIObjects->AddRange(GetAllSubControls(this));

	for(int i = 0; i < allUIObjects->Count; i++)
		allUIObjects[i]->Enabled = false;

	//MessageBox::Show("# of controls: " + Int32(allUIObjects->Count).ToString());
}

System::Void DPlus::ParameterEditor::handleComboBoxChangeInDataGridView(System::Object^ sender, int col, int row, ComboBox^ cb)
{
	DataGridView ^gv = dynamic_cast<DataGridView ^>(sender);
	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();

	if (!en)
		return;

	paramStruct ps = en->GetParameters();

	int p = col / 2;
	bool is_value = col % 2 == 0;

	System::Object^ val = cb->SelectedIndex;

	if (sender == parameterDataGridView)
	{
		if (val != nullptr)
		{
			if (is_value)
				ps.params[p][row].value = 0.01 + (Int32)val;
			else
				ps.params[p][row].isMutable = (Boolean)val;
		}
	}
	else if (sender == extraParamsDataGridView)
	{
		if (val != nullptr)
		{
			if (is_value)
				ps.extraParams[p].value = 0.01 + (Int32)val;
			else
				ps.extraParams[p].isMutable = (Boolean)val;
		}
	}
	// Commit parameter
	en->SetParameters(ps, parentForm->GetLevelOfDetail());

}

System::Void DPlus::ParameterEditor::ComboBoxSelectedIndexChange(System::Object^ sender, System::EventArgs^ e)
{
	ComboBox^ asCB = dynamic_cast<ComboBox^>(sender);

	if (asCB)
	{
		DataGridView^ gv = dynamic_cast<DataGridView^>(asCB->Parent->Parent);
		if (gv)
			handleComboBoxChangeInDataGridView(gv, gv->SelectedCells[0]->ColumnIndex, gv->SelectedCells[0]->RowIndex, asCB);
//		asCB->SelectedIndex
	}
}

void DPlus::ParameterEditor::dataGridView_EditingControlShowing(System::Object^ sender, System::Windows::Forms::DataGridViewEditingControlShowingEventArgs^ e) {
	DataGridViewTextBoxEditingControl^ cn = nullptr;

	cn = dynamic_cast<DataGridViewTextBoxEditingControl^>(e->Control);
	if(cn) {
		cn->KeyPress -= gcnew System::Windows::Forms::KeyPressEventHandler(this, &ParameterEditor::dataGridView_KeyPress);
		cn->KeyPress += gcnew System::Windows::Forms::KeyPressEventHandler(this, &ParameterEditor::dataGridView_KeyPress);
	}

	ComboBox^ asCB = dynamic_cast<ComboBox^>(e->Control);
	if (asCB)
	{
		asCB->SelectedIndexChanged -= gcnew System::EventHandler(this, &ParameterEditor::ComboBoxSelectedIndexChange);
		asCB->SelectedIndexChanged += gcnew System::EventHandler(this, &ParameterEditor::ComboBoxSelectedIndexChange);
	}

}

void DPlus::ParameterEditor::dataGridView_KeyPress(System::Object^ sender, System::Windows::Forms::KeyPressEventArgs^ e) {
#ifdef _DEBUG
	wchar_t deb = e->KeyChar;
#endif

	DataGridViewTextBoxEditingControl^ dgv = nullptr;
	dgv = dynamic_cast<DataGridViewTextBoxEditingControl^>(sender);

	if(!dgv)
		return;

	// If it's an expression, let the users type whatever they want
	if(dgv->Text->StartsWith("="))
		return;

	if((!(
		Char::IsDigit(e->KeyChar) ||
		(e->KeyChar == '-') || 
		(e->KeyChar == '.') ||
		(e->KeyChar == '=') ||
		(e->KeyChar == Convert::ToChar(Keys::Back) || e->KeyChar == Convert::ToChar(Keys::Delete))
		))				&&
		// Exceptions
		// copy and paste
		!(int(e->KeyChar) == 3 || int(e->KeyChar) == 22)
		)
		e->Handled = true;
}

System::Void DPlus::ParameterEditor::dataGridView_CellContentClick(System::Object^ sender, DataGridViewCellEventArgs^ e)
{
	int col = e->ColumnIndex, row = e->RowIndex;
	if (col < 0 || row < 0)
		return;

	DataGridView^ asDGV = dynamic_cast<DataGridView ^>(sender);

	DataGridViewCheckBoxCell^ ascheckbox = dynamic_cast<DataGridViewCheckBoxCell ^>(asDGV[col, row]);
	if (ascheckbox)
		asDGV->EndEdit();

	DataGridViewComboBoxCell^ asComboBox = dynamic_cast<DataGridViewComboBoxCell^>(asDGV);
	if (asComboBox)
		asDGV->EndEdit();
}

System::Void DPlus::ParameterEditor::handleCheckChangeInDataGridView(System::Object^ sender, int col, int row)
{
	DataGridView ^gv = dynamic_cast<DataGridView ^>(sender);
	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();

	if (!en)
		return;

	paramStruct ps = en->GetParameters();

	int p = col / 2;
	bool is_value = col % 2 == 0;

	if (sender == parameterDataGridView)
	{
		if (gv[col, row]->Value != nullptr)
		{
			if (is_value)
				ps.params[p][row].value = (Boolean)gv[col, row]->Value;
			else
				ps.params[p][row].isMutable = (Boolean)gv[col, row]->Value;
		}
	}
	else if (sender == extraParamsDataGridView)
	{
		if (gv[col, row]->Value != nullptr)
		{
			if (is_value)
				ps.extraParams[p].value = (Boolean)gv[col, row]->Value;
			else
				ps.extraParams[p].isMutable = (Boolean)gv[col, row]->Value;
		}
	}
	// Commit parameter
	en->SetParameters(ps, parentForm->GetLevelOfDetail());

	sv->tvInvalidate();
}

System::Void DPlus::ParameterEditor::DataGridView_OnCellValueChanged(System::Object^ sender, System::Windows::Forms::DataGridViewCellEventArgs^ e)
{
	int col = e->ColumnIndex, row = e->RowIndex;
	if (col < 0 || row < 0)
		return;

	DataGridView^ asDGV = dynamic_cast<DataGridView ^>(sender);
	
	DataGridViewCheckBoxCell^ ascheckbox = dynamic_cast<DataGridViewCheckBoxCell ^>(asDGV[col, row]);
	if (ascheckbox)
		handleCheckChangeInDataGridView(sender, col, row);

}

System::Void DPlus::ParameterEditor::DataGridView_OnCellMouseUp(System::Object^ sender, System::Windows::Forms::DataGridViewCellMouseEventArgs^ e)
{
	int col = e->ColumnIndex, row = e->RowIndex;
	if (col < 0 || row < 0)
		return;

	DataGridView^ asDGV = dynamic_cast<DataGridView ^>(sender);

	DataGridViewCheckBoxCell^ ascheckbox = dynamic_cast<DataGridViewCheckBoxCell ^>(asDGV[col, row]);
	if (ascheckbox)
		asDGV->EndEdit();

	// Determines whether a value was modified (or mutability otherwise)
	bool isValue = (col % 2 == 0);

	if (isValue)
		return;

	DataGridView^ gv = dynamic_cast<DataGridView ^>(sender);
	if (gv)
	{
		gv->EndEdit();
	}
}

System::Void DPlus::ParameterEditor::DataGridView_CellEndEdit(System::Object^ sender, System::Windows::Forms::DataGridViewCellEventArgs^ e) {	
	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();

	int col = e->ColumnIndex, row = e->RowIndex;
	if(!en || col < 0 || row < 0)
		return;

	// Determines whether a value was modified (or mutability otherwise)
	bool isValue = (col % 2 == 0); 


	// If an expression (begins with "="), parse as lua expression	
	if(isValue) {
		if(sender == parameterDataGridView && parameterDataGridView[col, row]->Value != nullptr) {
			String ^val = parameterDataGridView[col, row]->Value->ToString()->Trim();		
			if(val->StartsWith("=")) {
				Double dval = parentForm->LuaParseExpression(val->Substring(1));
				parameterDataGridView[col, row]->Value = dval;
			} else if(val->Contains("=")) {
				MessageBox::Show("Invalid expression");
				parameterDataGridView[col, row]->Value = 0.0;
			}
		} else if(sender == extraParamsDataGridView && extraParamsDataGridView[col, row]->Value != nullptr) {
			String ^val = extraParamsDataGridView[col, row]->Value->ToString()->Trim();
			EXTRA_PARAM_TYPE ept = en->modelUI->GetExtraParamType(col / 2);
			if(ept == EPT_DOUBLE) {
				if(val->StartsWith("=")) {
					Double dval = parentForm->LuaParseExpression(val->Substring(1));
					extraParamsDataGridView[col, row]->Value = dval;
				} else if(val->Contains("=")) {
					MessageBox::Show("Invalid expression");
					extraParamsDataGridView[col, row]->Value = 0.0;
				}
			}
		}
	}
	// END of Lua parsing	

	paramStruct ps = en->GetParameters();

	// Parameter modification
	if(sender == parameterDataGridView) { 
		int lp = col / 2;
		if(isValue) {
			Double val;
			if(parameterDataGridView[col, row]->Value == nullptr) // Revert to old value
				parameterDataGridView[col, row]->Value = ps.params[lp][row].value;

			if(Double::TryParse(parameterDataGridView[col, row]->Value->ToString(), val))
				ps.params[lp][row].value = val;
		} else {
			if(parameterDataGridView[col, row]->Value != nullptr)
				ps.params[lp][row].isMutable = (Boolean)parameterDataGridView[col, row]->Value;
		}
	} else if(sender == extraParamsDataGridView) { // Extra parameter modification
		int ep = col / 2;
		if(isValue) {		
			Double val;
			if(extraParamsDataGridView[col, row]->Value == nullptr) // Revert to old value
				extraParamsDataGridView[col, row]->Value = ps.extraParams[ep].value;
			EXTRA_PARAM_TYPE ept = en->modelUI->GetExtraParamType(ep);
			if(ept == EPT_DOUBLE) {
				if(Double::TryParse(extraParamsDataGridView[col, row]->Value->ToString(), val)) {
					ps.extraParams[ep].value = val;
				}
				if(extraParamsDataGridView[col, row]->Value != nullptr)
					ps.extraParams[ep].isMutable = (Boolean)extraParamsDataGridView[col + 1, row]->Value;

			} else if(ept == EPT_CHECKBOX) {
				ps.extraParams[ep].value = ((Boolean)(extraParamsDataGridView[col, row]->Value) ? 1.0 : 0.0);
				ps.extraParams[ep].isMutable = false;
			} else if(ept == EPT_MULTIPLE_CHOICE) {
				std::vector<std::string> ops = en->modelUI->GetExtraParamOptionStrings(ep);
				std::string chosen = clrToString(extraParamsDataGridView[col, row]->Value->ToString()->Trim());
				int ind;
				for(ind = 0; ind < ops.size(); ind++) {
					if(ops[ind].compare(chosen) == 0) {
						break;
					}
				}
				ps.extraParams[ep].value = double(ind);
				ps.extraParams[ep].isMutable = false;	
			}
		} else {
			if(extraParamsDataGridView[col, row]->Value != nullptr)
				ps.extraParams[ep].isMutable = (Boolean)extraParamsDataGridView[col, row]->Value;
		}
	}	
	// END of parameter modification

	// Commit parameter
	en->SetParameters(ps, parentForm->GetLevelOfDetail());

	sv->tvInvalidate();
}

void DPlus::ParameterEditor::FillParamGridView(Entity ^en) {	
	if(en->modelUI == NULL) {
		// If PDB or amplitude
		addLayerButton->Enabled = false;
		removeLayerButton->Enabled = false;

		parameterDataGridView->Rows->Clear();
		parameterDataGridView->Columns->Clear();
		extraParamsDataGridView->Rows->Clear();
		extraParamsDataGridView->Columns->Clear();

		return;
	}

	paramStruct ps = en->GetParameters();
	int nlp = ps.nlp;
	int minLayers = en->modelUI->GetMinLayers();

	parameterDataGridView->Rows->Clear();
	parameterDataGridView->Columns->Clear();

	// Prepare the columns
	for(int i = 0; i < nlp; i++) {
		System::Windows::Forms::DataGridViewTextBoxColumn^ valueColumn = gcnew System::Windows::Forms::DataGridViewTextBoxColumn();
		System::Windows::Forms::DataGridViewCheckBoxColumn^  mutColumn = gcnew System::Windows::Forms::DataGridViewCheckBoxColumn(false);
		
		valueColumn->HeaderText	= stringToClr(en->modelUI->GetLayerParamName(i));
		valueColumn->SortMode	= DataGridViewColumnSortMode::NotSortable;
		mutColumn->HeaderText	= "Mut";
		mutColumn->TrueValue	= true;
		mutColumn->SortMode		= DataGridViewColumnSortMode::NotSortable;
		parameterDataGridView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::DataGridViewColumn^  >(2) {valueColumn, 
			mutColumn});
	}

	// Add the rows
	for(int i = 0; i < ps.layers; i++) {
		array<Object ^> ^row = gcnew array<Object ^>(nlp * 2);
		std::vector<int> NAcols;

		for(int j = 0; j < nlp; j++) {
			if(en->modelUI->IsParamApplicable(i, j)) {
				row[j * 2 + 0] = ps.params[j][i].value; // Value
				row[j * 2 + 1] = ps.params[j][i].isMutable; // Mutability
			} else {
				row[j * 2 + 0] = "N/A";
				row[j * 2 + 1] = false;

				NAcols.push_back(j);
			}
		}

		if(nlp > 0) {
			parameterDataGridView->Rows->Add(row);	
			parameterDataGridView->Rows[i]->HeaderCell->Value = stringToClr(en->modelUI->GetLayerName(i));

			// Mark cells read-only
			if(NAcols.size() > 0) {
				for(std::vector<int>::iterator iter = NAcols.begin(); iter != NAcols.end(); ++iter) {
					parameterDataGridView->Rows[i]->Cells[(*iter)*2]->ReadOnly = true;     // Value
					parameterDataGridView->Rows[i]->Cells[(*iter) * 2 + 1]->ReadOnly = true; // Mutability
				}
			}
		}
	}	// for i

	// Enable/disable buttons
	addLayerButton->Enabled = (ps.layers < en->modelUI->GetMaxLayers() || en->modelUI->GetMaxLayers() < 0);
	removeLayerButton->Enabled = false;

	//////////////////////////////////////////////////////////////////////////

	// Extra parameters
	extraParamsDataGridView->Rows->Clear();
	extraParamsDataGridView->Columns->Clear();
	extraParamsDataGridView->AutoGenerateColumns = false;
	
	// Prepare the columns
	for(int i = 0; i < ps.nExtraParams; i++) {
		System::Windows::Forms::DataGridViewColumn^ valueColumn;
		System::Windows::Forms::DataGridViewCheckBoxColumn^  mutColumn = gcnew System::Windows::Forms::DataGridViewCheckBoxColumn(false);

		if(en->modelUI->GetExtraParamType(i) == EPT_CHECKBOX) {
			valueColumn = gcnew System::Windows::Forms::DataGridViewCheckBoxColumn();
			mutColumn->ReadOnly = true;
		} else if(en->modelUI->GetExtraParamType(i) == EPT_MULTIPLE_CHOICE) {
			valueColumn = gcnew System::Windows::Forms::DataGridViewComboBoxColumn();

			System::Collections::Specialized::StringCollection ^ds = gcnew System::Collections::Specialized::StringCollection();
			std::vector<std::string> options = en->modelUI->GetExtraParamOptionStrings(i);
			for(int k = 0; k < options.size(); k++)
				ds->Add(stringToClr(options[k]));
			
			((DataGridViewComboBoxColumn^)valueColumn)->DataSource = ds;
			mutColumn->ReadOnly = true;
		} else {
			valueColumn = gcnew System::Windows::Forms::DataGridViewTextBoxColumn();
		}

		valueColumn->HeaderText	= stringToClr(en->modelUI->GetExtraParameter(i).name);
		mutColumn->HeaderText	= "Mut";
		mutColumn->TrueValue	= true;
		extraParamsDataGridView->Columns->AddRange(gcnew cli::array< System::Windows::Forms::DataGridViewColumn^  >(2) {valueColumn, 
			mutColumn});
	}

	// Add the rows
	array<Object ^> ^eprow = gcnew array<Object ^>(ps.nExtraParams * 2);
	for(int i = 0; i < ps.nExtraParams; i++) {
		if(en->modelUI->GetExtraParamType(i) == EPT_CHECKBOX) {
			eprow[i * 2 + 0] = (ps.extraParams[i].value > 0.01);
			eprow[i * 2 + 1] = false;
		} else if(en->modelUI->GetExtraParamType(i) == EPT_MULTIPLE_CHOICE) {
			eprow[i * 2 + 0] = stringToClr((en->modelUI->GetExtraParamOptionStrings(i))[int(ps.extraParams[i].value + 0.1)]);
			eprow[i * 2 + 1] = false;
		} else {
			eprow[i * 2 + 0] = ps.extraParams[i].value;
			eprow[i * 2 + 1] = ps.extraParams[i].isMutable;
		}
	}
	if(ps.nExtraParams > 0)
		extraParamsDataGridView->Rows->Add(eprow);

	parameterDataGridView->AutoSizeColumnsMode		= DataGridViewAutoSizeColumnsMode::DisplayedCells;
	extraParamsDataGridView->AutoSizeColumnsMode	= DataGridViewAutoSizeColumnsMode::DisplayedCells;
	parameterDataGridView->AutoResizeRowHeadersWidth(DataGridViewRowHeadersWidthSizeMode::AutoSizeToDisplayedHeaders);	
}

System::Void DPlus::ParameterEditor::gridViewContextMenuStrip_Opening(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e) {
	// TODO::Check to see if eache element should be enabled
	linkToolStripMenuItem->Enabled = (sender != extraParamsDataGridView) && (parameterDataGridView->SelectedCells->Count > 1) ;

}

System::Void DPlus::ParameterEditor::polydispersityToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	MessageBox::Show("TODO: open polydispersity option.");
}

System::Void DPlus::ParameterEditor::linkToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	MessageBox::Show("TODO: Link all parameters of the same type.");
}

System::Void DPlus::ParameterEditor::editConstraintsToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	DataGridView^ dgv = nullptr;
	// Figure out which control opened the context menu
	ToolStripItem ^menuItem = dynamic_cast<ToolStripItem ^>(sender);
	if (menuItem != nullptr) {
		System::Windows::Forms::ContextMenuStrip ^strip = dynamic_cast<System::Windows::Forms::ContextMenuStrip ^>(menuItem->Owner);
		if (strip != nullptr)
			dgv = dynamic_cast<DataGridView ^>(strip->SourceControl);
	}
	if(dgv == nullptr)
		dgv = parameterDataGridView;

	std::vector<std::pair<unsigned int, unsigned int >> prms;
	for(int i = 0; i < dgv->SelectedCells->Count; i++) {
		if(dgv->SelectedCells[i]->ColumnIndex % 2 == 1)
			continue;
		if(!dgv->SelectedCells[i]->Value)
			continue;
		prms.push_back(std::pair<unsigned int, unsigned int>(dgv->SelectedCells[i]->ColumnIndex / 2, dgv->SelectedCells[i]->RowIndex));
	}

	ConstraintsWindow ^cw = gcnew ConstraintsWindow(parentForm, ((SymmetryView^)parentForm->PaneList[SYMMETRY_VIEWER])->GetSelectedEntity(), &prms, ((dgv == extraParamsDataGridView) ? CONS_EXTRAPARAMETERS : CONS_PARAMETERS));
	System::Windows::Forms::DialogResult dr = cw->ShowDialog();

	if(dr == System::Windows::Forms::DialogResult::OK) {
		FillParamGridView(((SymmetryView^)parentForm->PaneList[SYMMETRY_VIEWER])->GetSelectedEntity());
	}
}

System::Void DPlus::ParameterEditor::parameterDataGridView_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
	
	if(e->Button == ::MouseButtons::Right) {
		DataGridView^ dgv = dynamic_cast<DataGridView^>(sender);
		if(!dgv)
			return;
		DataGridView::HitTestInfo^ hti = dgv->HitTest(e->X, e->Y);
		if(!hti || hti->ColumnIndex == -1 || hti->RowIndex == -1) {
			return;
		}
		if( !(hti->RowIndex - 1 == dgv->RowCount) ) {
			if(! dgv->SelectedCells->Contains(dgv[hti->ColumnIndex, hti->RowIndex]) )
				dgv->CurrentCell = dgv[hti->ColumnIndex, hti->RowIndex];
		}
	}
}

System::Void DPlus::ParameterEditor::addLayerButton_Click(System::Object^ sender, System::EventArgs^ e) {
	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();

	paramStruct ps = en->GetParameters();
	if(ps.layers == en->modelUI->GetMaxLayers()) // Sanity check
		return;

	if(ps.layers == 0)
		ps.params.resize(ps.nlp);

	// Add the actual layer	
	for(int i = 0; i < ps.nlp; i++)
		ps.params[i].push_back(Parameter(en->modelUI->GetDefaultParamValue(ps.layers, i)));

	ps.layers++;

	// Commit the new parameters
	en->SetParameters(ps, parentForm->GetLevelOfDetail());

	// Update grid-view and buttons
	FillParamGridView(en);

	// Invalidate 3D viewport
	GraphPane3D ^g3 = (GraphPane3D ^)parentForm->PaneList[GRAPH3D];
	g3->glCanvas3D1->Invalidate();
	g3->Invalidate();

	// Scroll to bottom
	if(ps.nlp > 0)
		parameterDataGridView->CurrentCell = parameterDataGridView[0, ps.layers - 1];
}

System::Void DPlus::ParameterEditor::removeLayerButton_Click(System::Object^ sender, System::EventArgs^ e) {
	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();

	paramStruct ps = en->GetParameters();	

	// Rows to remove
	std::set<int> rowindices;
	for each(DataGridViewRow ^dgvr in parameterDataGridView->SelectedRows)
		rowindices.insert(dgvr->Index);


	// Loop over all rows to remove (backwards, to keep indices good)
	for(std::set<int>::reverse_iterator iter = rowindices.rbegin(); iter != rowindices.rend(); ++iter) {
		int rowind = *iter;

		if(ps.layers == en->modelUI->GetMinLayers()) // Sanity check
			return;

		// Remove the actual layer	
		for(int i = 0; i < ps.nlp; i++)
			ps.params[i].erase(ps.params[i].begin() + rowind);

		ps.layers--;
	}

	// Commit the new parameters
	en->SetParameters(ps, parentForm->GetLevelOfDetail());

	// Update grid-view and buttons
	FillParamGridView(en);

	// Invalidate 3D viewport
	GraphPane3D ^g3 = (GraphPane3D ^)parentForm->PaneList[GRAPH3D];
	g3->glCanvas3D1->Invalidate();
	g3->Invalidate();
}

System::Void DPlus::ParameterEditor::parameterDataGridView_SelectionChanged(System::Object^ sender, System::EventArgs^ e) {
	if(parameterDataGridView->SelectedRows->Count < 1) {
		removeLayerButton->Enabled = false;
		return;
	}

	SymmetryView^ sv = (SymmetryView^)(parentForm->PaneList[SYMMETRY_VIEWER]);
	Entity^ en = sv->GetSelectedEntity();
	int minLayers = en->modelUI->GetMinLayers();

	for each(DataGridViewRow ^dgvr in parameterDataGridView->SelectedRows) {
		if(dgvr->Index < minLayers) {
			removeLayerButton->Enabled = false;
			return;
		}
	}

	// If we haven't returned yet, we can delete the row(s)
	removeLayerButton->Enabled = true;
}


