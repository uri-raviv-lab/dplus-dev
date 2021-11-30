#include <vcclr.h>

#include <codecvt>
#include "SymmetryView.h"

#include "clrfunctionality.h"
#include <fstream>
#include <string>

#include "Common.h"
#include "CommProtocol.h"
#include "ModelUI.h"

#include "SymmetryEditor.h"
#include "ParameterEditor.h"
#include "ScriptPane.h"

using Aga::Controls::Tree::TreeNodeAdv;
using Aga::Controls::Tree::NodePosition;

using namespace System::Text::RegularExpressions;


namespace DPlus {

static paramStruct DOLToParamStruct(String ^filename, paramStruct ps) {
	if(filename == nullptr)
		return ps;

	std::wifstream inFile (clrToString(filename));

	if(inFile) {
		ps.params.resize(6);

		std::wstring line;
		line.resize(1024 * 8);

		System::String ^RX_FLOATING_POINT_STRING = L"([-\\+]?\\b\\d*\\.?\\d+(?:[eE][-\\+]?\\d+)?)";
		System::String ^delim = L"[;\\s]";
		System::String ^sixPattern = 
			RX_FLOATING_POINT_STRING + delim + RX_FLOATING_POINT_STRING + delim +
			RX_FLOATING_POINT_STRING + delim + RX_FLOATING_POINT_STRING + delim +
			RX_FLOATING_POINT_STRING + delim + RX_FLOATING_POINT_STRING + L"\\s*";
		System::String ^sevenPattern = L"\\s*\\d+[;\\s]" + sixPattern;

		// Check encoding
		int length_of_BOM = 0;
		if (!inFile.eof())
		{
			inFile.getline(&line[0], line.size());

			if (line.length() > 2)
			{
				if (line[0] == 0xEF && line[1] == 0xBB && line[2] == 0xBF)
				{
					printf("UTF8\n");
					length_of_BOM = 3;
					inFile.seekg(0);
					inFile.imbue(std::locale(inFile.getloc(),
						new std::codecvt_utf8<wchar_t, 0x10ffff, std::consume_header>));
				}
			}

			if (line.length() > 1)
			{
				if (line[0] == 0xFE && line[1] == 0xFF)
				{
					printf("UTF16_BE\n");
					inFile.imbue(std::locale(inFile.getloc(),
						new std::codecvt_utf16<wchar_t, 0x10ffff>));
					length_of_BOM = 2;
				}
				if (line[0] == 0xFF && line[1] == 0xFE)
				{
					printf("UTF16_LE\n");
					inFile.imbue(std::locale(inFile.getloc(),
						new std::codecvt_utf16<wchar_t, 0x10ffff, std::little_endian>));
					length_of_BOM = 2;
				}
			}

			inFile.seekg(length_of_BOM);
		}

		while (!inFile.eof()) {
			// TODO: Malformed files will probably crash D+
			//inFile.getline(line);
			inFile.getline(&line[0], line.size());
			System::String ^lineString = stringToClr(line);
			Regex ^justFloats = gcnew Regex(sixPattern, RegexOptions::ECMAScript);
			Regex ^indexAndFloats = gcnew Regex(sevenPattern, RegexOptions::ECMAScript);
			if (!indexAndFloats->IsMatch(lineString) /*&& !justFloats->IsMatch(lineString)*/)
				continue;

			double x, y, z, alpha, beta, gamma;
			
			int offset = 0;
			Match ^m;
			if (indexAndFloats->IsMatch(lineString))
			{
				offset = 1;
				m = indexAndFloats->Match(lineString);
			}
			else
				m = justFloats->Match(lineString);

			// For some reason, justFloats doesn't give the correct string as Value, rather, the entire input is spit out as m->Groups[0]->Value...
			x = Double::Parse(m->Groups[offset++]->Value);
			y = Double::Parse(m->Groups[offset++]->Value);
			z = Double::Parse(m->Groups[offset++]->Value);
			alpha = Double::Parse(m->Groups[offset++]->Value);
			beta = Double::Parse(m->Groups[offset++]->Value);
			gamma = Double::Parse(m->Groups[offset++]->Value);

			ps.params[0].push_back(x); ps.params[1].push_back(y); ps.params[2].push_back(z);
			ps.params[3].push_back(alpha); 
			ps.params[4].push_back(beta); 
			ps.params[5].push_back(gamma);

			ps.layers++;
		}
	} else {
		inFile.close();
		return ps;
	}
	inFile.close();

	return ps;
}

System::Void SymmetryView::buttonAdd_Click(System::Object^ sender, System::EventArgs^ e) {
	GraphPane3D^ g3 = nullptr;
	g3  = dynamic_cast<GraphPane3D^>(parentForm->PaneList[GRAPH3D]);

	ModelInfo ^selmodel = (ModelInfo ^)entityCombo->SelectedItem;
	Entity ^ent = nullptr;

	if(selmodel->GetID() == 999) {
		// Hardcoded code for PDBs
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Title = "Select a PDB file...";
		ofd->Filter = "PDB Files (*.pdb)|*.pdb|All Files (*.*)|*.*";
		ofd->FileName = "";

		if(ofd->ShowDialog() != System::Windows::Forms::DialogResult::OK)
			return;
		String ^pdbfilename = ofd->FileName;
		String ^anomfilename = "";
		// Puts the basename without extension in the treeview
		// Example: C:\1SVA.pdb --> "1SVA (PDB)"
		ent = g3->RegisterPDB(pdbfilename, anomfilename, parentForm->GetLevelOfDetail(), CenterChecked());
	} 
	else if(selmodel->GetID() == 1000) 
	{
		// Hardcoded code for AMPs
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Title = "Select an amplitude grid file...";
		ofd->Filter = "Amplitude Grid Files (*.ampj)|*.ampj|All Files (*.*)|*.*";
		ofd->FileName = "";

		if(ofd->ShowDialog() != System::Windows::Forms::DialogResult::OK)
			return;

		// Puts the basename without extension in the treeview
		// Example: C:\1SVA.amp --> "1SVA (AMP)"		
		ent = g3->RegisterAMPGrid(ofd->FileName, parentForm->GetLevelOfDetail(), CenterChecked());
	} 
	else if(selmodel->GetID() >= 1001 && selmodel->GetID() <= 1003)
	{
		// Hardcoded code for scripted geometries, models and symmetries (in this order)
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Title = "Select a script...";
		ofd->Filter = "D+ Script Files (*.lua)|*.lua|All Files (*.*)|*.*";
		ofd->FileName = "";

		if(ofd->ShowDialog() != System::Windows::Forms::DialogResult::OK)
			return;

		ent = parentForm->RegisterLuaEntity(ofd->FileName, selmodel->GetID());	
	} 
	else if(selmodel->GetID() >= 0)
	{
		FrontendComm *frontend = parentForm->frontend;
		std::wstring contstr;
		const wchar_t *container = NULL;

		// Copy container from String^ to wchar_t*
		if(selmodel->GetContainer() != nullptr) {
			contstr = clrToWstring(selmodel->GetContainer());
			container = contstr.c_str();
		}
		
		ent = parentForm->CreateEntityFromID(container, selmodel->GetID(), selmodel->ToString());

		if(ent != nullptr && selmodel->GetContainer() == nullptr && selmodel->ToString() == "Manual Symmetry")
		{
			// Hardcoded code for manual symmetries (DOLs)
			String ^fname = nullptr;
			if(MessageBox::Show("Would you like to import locations from file?", "Import", MessageBoxButtons::YesNo) == System::Windows::Forms::DialogResult::Yes) {
				OpenFileDialog ^ofd = gcnew OpenFileDialog();
				ofd->Title = "Select a file...";
				ofd->Filter = "D+ Location/Rotation Files (*.dol)|*.dol|All Files (*.*)|*.*";
				ofd->FileName = "";

				if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
					fname = ofd->FileName;
				}
			}
			else
			{ // creating manual symmetry with at least one layer (like the function ParameterEditor::addLayerButton_Click)
				AddLayerManualSymmetry(sender, e, ent);
					
			}
			ent->SetParameters(DOLToParamStruct(fname, ent->GetParameters()), parentForm->GetLevelOfDetail());
			ent->modelName += " (" + fname + ")";// As per Roi's request
		}
	}

	if(ent != nullptr)
		parentForm->entityTree->Nodes->Add(ent);

	tvInvalidate();
}

System::Void SymmetryView::buttonRemove_Click(System::Object^ sender, System::EventArgs^ e) {


	// Add entities to "remove list"
	System::Collections::Generic::List<Entity ^> ^ents = gcnew System::Collections::Generic::List<Entity ^>();
	for (int i = 0; i < treeViewAdv1->SelectedNodes->Count; i++)
	{
		Entity^ en = (Entity ^)treeViewAdv1->SelectedNodes[i]->Tag;
		if (en->anomfilename && en->anomfilename != "")
			this->anomalousCheckBox->Checked = false;
		ents->Add(en);
	}

	// Remove entities from list
	for(int i = ents->Count - 1; i >= 0; i--) {
		ents[i]->Parent->Nodes->Remove(ents[i]);
		delete ents[i];

	}

	// Refresh UI
	GraphPane3D^ g3 = nullptr;
	g3  = dynamic_cast<GraphPane3D^>(parentForm->PaneList[GRAPH3D]);
	treeViewAdv1->Invalidate();
	g3->glCanvas3D1->Invalidate();
	g3->Invalidate();
}

System::Void SymmetryView::buttonGroup_Click(System::Object^ sender, System::EventArgs^ e) {
	GraphPane3D^ g3 = nullptr;
	g3  = dynamic_cast<GraphPane3D^>(parentForm->PaneList[GRAPH3D]);

	ModelInfo ^selmodel = (ModelInfo ^)entityCombo->SelectedItem;
	FrontendComm *frontend = parentForm->frontend;
	Entity ^ent = nullptr;

	if(selmodel->GetID() == 1003) { // Scripted symmetry
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Title = "Select a script...";
		ofd->Filter = "D+ Script Files (*.lua)|*.lua|All Files (*.*)|*.*";
		ofd->FileName = "";

		if(ofd->ShowDialog() != System::Windows::Forms::DialogResult::OK)
			return;

		ent = parentForm->RegisterLuaEntity(ofd->FileName, selmodel->GetID());
	} else {
		std::wstring contstr;
		const wchar_t *container = NULL;
		// Copy container from String^ to wchar_t*
		if(selmodel->GetContainer() != nullptr) {
			contstr = clrToWstring(selmodel->GetContainer());
			container = contstr.c_str();
		}

		ModelInformation mi = frontend->QueryModel(container, selmodel->GetID());
		ModelCategory cat = frontend->QueryCategory(container, mi.category);
		if (cat.type != MT_SYMMETRY) {
			MessageBox::Show("A symmetry must be selected in the list to the left of the Group Selected button.");
			return;
		}

		ent = parentForm->CreateEntityFromID(container, selmodel->GetID(), selmodel->ToString());

		if(ent != nullptr && selmodel->GetContainer() == nullptr && selmodel->ToString() == "Manual Symmetry") {
			// Hardcoded code for manual symmetries (DOLs)
			String ^fname = nullptr;
			if(MessageBox::Show("Would you like to import locations from file?", "Import", MessageBoxButtons::YesNo) == System::Windows::Forms::DialogResult::Yes) {
				OpenFileDialog ^ofd = gcnew OpenFileDialog();
				ofd->Title = "Select a file...";
				ofd->Filter = "D+ Location/Rotation Files (*.dol)|*.dol|All Files (*.*)|*.*";
				ofd->FileName = "";

				if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
					fname = ofd->FileName;
				}
			}
			else
			{ // creating manual symmetry with at least one layer (like the function ParameterEditor::addLayerButton_Click)
				AddLayerManualSymmetry(sender, e, ent);
			}

			ent->SetParameters(DOLToParamStruct(fname, ent->GetParameters()), parentForm->GetLevelOfDetail());
		}
	}
	
	// Move the selected entities into ent
	System::Collections::Generic::List<Entity ^> ^ents = gcnew System::Collections::Generic::List<Entity ^>();
	for(int i = 0; i < treeViewAdv1->SelectedNodes->Count; i++)
		ents->Add((Entity ^)treeViewAdv1->SelectedNodes[i]->Tag);
	Node ^point = ((Entity ^)treeViewAdv1->SelectedNodes[0]->Tag)->Parent;
	point->Nodes->Add(ent);
	for(int i = 0; i < ents->Count; i++)
		ent->Nodes->Add(ents[i]);
	
	tvInvalidate();

}

static void ClearCEntitySelection(Entity ^ent) {
	ent->selected = false;

	for(int i = 0; i < ent->Nodes->Count; i++)
		ClearCEntitySelection((Entity ^)ent->Nodes[i]);
}

static void ClearEntitySelection(MainWindow ^parentForm) {	
	for(int i = 0; i < parentForm->entityTree->Nodes->Count; i++)
		ClearCEntitySelection((Entity ^)parentForm->entityTree->Nodes[i]);
}

System::Void SymmetryView::treeViewAdv1_SelectionChanged(System::Object^ sender, System::EventArgs^ e) {
	this->parentForm->InSelectionChange = true;
	this->anomalousCheckBox->Visible = false;
	if(treeViewAdv1->SelectedNodes->Count > 1) {
		this->contextMenuModelName->Enabled = false;
		buttonRemove->Enabled = true;
		buttonGroup->Enabled = true;
		ChangeParamEditorEnabled(false);
		ChangeEditorEnabled(false);

		// Deselect all and reselect the selected items
		ClearEntitySelection(parentForm);
		for each (TreeNodeAdv ^tna in treeViewAdv1->SelectedNodes) {
			Entity^ en = (Entity ^)tna->Tag;
			en->selected = true;
		}
		
		tvInvalidate();

	} else if(treeViewAdv1->SelectedNodes->Count == 1) {
		this->contextMenuModelName->Enabled = true;
		buttonRemove->Enabled = true; 
		buttonGroup->Enabled = true;
		ChangeParamEditorEnabled(true);
		ChangeEditorEnabled(true);
		Entity^ en = (Entity ^)treeViewAdv1->SelectedNodes[0]->Tag;
		
		// Deselect all and reselect the selected item
		ClearEntitySelection(parentForm);
		en->selected = true;
		this->anomalousCheckBox->Visible = (en->type == EntityType::TYPE_PDB);
		if (en->anomfilename && en->anomfilename!="")
			this->anomalousCheckBox->Checked = true;
		else
			this->anomalousCheckBox->Checked = false;


		SymmetryEditor^ se		= (SymmetryEditor^)(parentForm->PaneList[SYMMETRY_EDITOR]);
		paramStruct ps = en->GetParameters();
		se->xTextBox->Text		= (gcnew Double(ps.x.value))->ToString();
		se->yTextBox->Text		= (gcnew Double(ps.y.value))->ToString();
		se->zTextBox->Text		= (gcnew Double(ps.z.value))->ToString();
		se->alphaTextBox->Text	= (gcnew Double(Degree(Radian(ps.alpha.value)).deg))->ToString();
		se->betaTextBox->Text	= (gcnew Double(Degree(Radian(ps.beta.value )).deg))->ToString();
		se->gammaTextBox->Text	= (gcnew Double(Degree(Radian(ps.gamma.value)).deg))->ToString();

		se->xMutCheckBox->Checked = ps.x.isMutable;
		se->yMutCheckBox->Checked = ps.y.isMutable;
		se->zMutCheckBox->Checked = ps.z.isMutable;
		se->aMutCheckBox->Checked = ps.alpha.isMutable;
		se->bMutCheckBox->Checked = ps.beta.isMutable;
		se->gMutCheckBox->Checked = ps.gamma.isMutable;
		se->useGridAtLevelCheckBox->Checked = ps.bSpecificUseGrid;
		
		ParameterEditor^ pe = (ParameterEditor^)(parentForm->PaneList[PARAMETER_EDITOR]);
		pe->FillParamGridView(en);

		tvInvalidate();
	} else {
		this->contextMenuModelName->Enabled = false;
		buttonRemove->Enabled = false;
		buttonGroup->Enabled = false;
		ChangeParamEditorEnabled(false);
		ChangeEditorEnabled(false);

		// Deselect all
		ClearEntitySelection(parentForm);
		tvInvalidate();
	}
	this->parentForm->InSelectionChange = false;
}

System::Void SymmetryView::entityCombo_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
	if(entityCombo->SelectedIndex < 0) {
		buttonAdd->Enabled = false;
		return;
	}

	ModelInfo ^mi = (ModelInfo ^)entityCombo->SelectedItem;
	buttonAdd->Enabled = (mi->GetID() >= 0);

	this->centerPDBCheckBox->Visible = (mi->GetID() == 999 || mi->GetID() == 1000);
	this->buttonGroup->Visible = !this->centerPDBCheckBox->Visible;
	
}

System::Void SymmetryView::treeViewAdv1_ItemDrag(System::Object^ sender, System::Windows::Forms::ItemDragEventArgs^ e) {
	treeViewAdv1->DoDragDropSelectedNodes(DragDropEffects::Move);
}

System::Void SymmetryView::treeViewAdv1_DragDrop(System::Object^ sender, System::Windows::Forms::DragEventArgs^ e) {
	treeViewAdv1->BeginUpdate();

	array<TreeNodeAdv ^> ^nodes = (array<TreeNodeAdv ^> ^)e->Data->GetData(array<TreeNodeAdv ^>::typeid);
	Entity ^dropNode = (Entity ^)treeViewAdv1->DropPosition.Node->Tag;
	if (treeViewAdv1->DropPosition.Position == NodePosition::Inside) {
		if(dropNode->type == EntityType::TYPE_SYMMETRY) {
			for(int i = 0; i < nodes->Length; i++)
			{
				TreeNodeAdv ^n = nodes[i];
				((Node ^)(n->Tag))->Parent = dropNode;
			}
			treeViewAdv1->DropPosition.Node->IsExpanded = true;
		}
	}
	else
	{
		Node ^parent = dropNode->Parent;
		Node ^nextItem = dropNode;
		if (treeViewAdv1->DropPosition.Position == NodePosition::After)
			nextItem = dropNode->NextNode;

		for(int i = 0; i < nodes->Length; i++)
		{
			TreeNodeAdv ^n = nodes[i];
			((Node ^)(n->Tag))->Parent = nullptr;
		}
		
		int index = -1;
		index = parent->Nodes->IndexOf(nextItem);
		for(int i = 0; i < nodes->Length; i++)
		{
			Node ^item = (Node ^)(nodes[i]->Tag);
			if (index == -1)
				parent->Nodes->Add(item);
			else
			{
				parent->Nodes->Insert(index, item);
				index++;
			}			
		}		
	}
	
	if (dropNode->GetParameters().bSpecificUseGrid)
		dropNode->SetUseGrid(true);

	treeViewAdv1->EndUpdate();
	tvInvalidate();
}

bool CheckNodeParent(TreeNodeAdv ^parent, TreeNodeAdv ^node)
{
	while (parent != nullptr)
	{
		if (node == parent)
			return false;
		else
			parent = parent->Parent;
	}
	return true;
}

System::Void SymmetryView::treeViewAdv1_DragOver(System::Object^ sender, System::Windows::Forms::DragEventArgs^ e) {
	if (e->Data->GetDataPresent(array<TreeNodeAdv ^>::typeid) && treeViewAdv1->DropPosition.Node != nullptr)
	{
		array<TreeNodeAdv ^> ^nodes = (array<TreeNodeAdv ^> ^)e->Data->GetData(array<TreeNodeAdv ^>::typeid);
		TreeNodeAdv ^parent = treeViewAdv1->DropPosition.Node;
		if (treeViewAdv1->DropPosition.Position != NodePosition::Inside)
			parent = parent->Parent;

		// If this list item is not a symmetry, we cannot add objects into it
		Entity ^ent = (Entity ^)(treeViewAdv1->DropPosition.Node->Tag);
		if(treeViewAdv1->DropPosition.Position == NodePosition::Inside && ent->type != EntityType::TYPE_SYMMETRY) {
			e->Effect = DragDropEffects::None;
			return;
		}

		for(int i = 0; i < nodes->Length; i++)
		{
			TreeNodeAdv ^node = nodes[i];
			if (!CheckNodeParent(parent, node))
			{
				e->Effect = DragDropEffects::None;
				return;
			}
		}

		e->Effect = e->AllowedEffect;
	}
}

void SymmetryView::ChangeEditorEnabled(bool en) {
	SymmetryEditor^ se = (SymmetryEditor^)(parentForm->PaneList[SYMMETRY_EDITOR]);
	se->xTextBox->Enabled = en;		se->xTrackBar->Enabled = en; se->xMutCheckBox->Enabled = en;
	se->yTextBox->Enabled = en;		se->yTrackBar->Enabled = en; se->yMutCheckBox->Enabled = en;
	se->zTextBox->Enabled = en;		se->zTrackBar->Enabled = en; se->zMutCheckBox->Enabled = en;
	se->alphaTextBox->Enabled = en;	se->alphaTrackBar->Enabled = en; se->aMutCheckBox->Enabled = en;
	se->betaTextBox->Enabled = en;	se->betaTrackBar->Enabled = en; se->bMutCheckBox->Enabled = en;
	se->gammaTextBox->Enabled = en;	se->gammaTrackBar->Enabled = en; se->gMutCheckBox->Enabled = en;
	se->constraintsButton->Enabled = en; se->useGridAtLevelCheckBox->Enabled = en;
}

void SymmetryView::ChangeParamEditorEnabled(bool en) {
	ParameterEditor^ pe = (ParameterEditor^)(parentForm->PaneList[PARAMETER_EDITOR]);
	for(int i = 0; i < pe->allUIObjects->Count; i++)
		pe->allUIObjects[i]->Enabled = en;
	DataGridView^ pgv = pe->parameterDataGridView;
	DataGridView^ epgv = pe->extraParamsDataGridView;
	pgv->Enabled = en;
	epgv->Enabled = en;
	if(!en) {	// Load relevant parameters and extra parameters
		pgv->ColumnCount = 0;
		epgv->ColumnCount = 0;
	}
}

Entity^ SymmetryView::GetSelectedEntity() {
	if(treeViewAdv1->SelectedNodes->Count == 1)
		return ((Entity ^)treeViewAdv1->SelectedNodes[0]->Tag);
	else return nullptr;
}

void SymmetryView::tvInvalidate() {
	GraphPane3D^ g3 = nullptr;
	g3  = dynamic_cast<GraphPane3D^>(parentForm->PaneList[GRAPH3D]);

	avgpopsizeText->Text = "" + parentForm->populationSizes[populationTabs->SelectedIndex];
	avgpopsizeMut->Checked = parentForm->populationSizeMutable[populationTabs->SelectedIndex];

	treeViewAdv1->Invalidate();
	g3->glCanvas3D1->Invalidate();
	g3->Invalidate();
}

System::Void SymmetryView::treeViewAdv1_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	//If the user selected indices and pressed delete/backspace, remove indicies
	if(e->KeyCode == Keys::Delete || e->KeyCode == Keys::Back) {
		buttonRemove_Click(sender, e);
		e->Handled = true;
	}
}

void SymmetryView::RemoveSelectedNodes() {
	buttonRemove_Click(nullptr, nullptr);
}

System::Void SymmetryView::treeViewAdv1_DoubleClick(System::Object^ sender, System::EventArgs^ e) {
	if(treeViewAdv1->SelectedNodes->Count == 1) {
		Entity^ en = (Entity ^)treeViewAdv1->SelectedNodes[0]->Tag;

		// Double-clicking a script causes it to open in the editor pane
		if(en->modelContext != nullptr) {
			ScriptPane ^sp = (ScriptPane ^)(parentForm->PaneList[SCRIPT_EDITOR]);
			sp->OpenFile(en->filename);
		} else {
			// Double-clicking on a model causes the 3D window to focus
			GraphPane3D ^g3 = (GraphPane3D ^)(parentForm->PaneList[GRAPH3D]);
			g3->Show();
		}
	}
}


void SymmetryView::AddPopulation()
{
	if(populationTabs->TabCount >= 1024)
	{
		MessageBox::Show("Cannot add new population. Maximal population number reached.");
		return;
	}

	TabPage ^newpage = gcnew TabPage("Population " + populationTabs->TabCount);

	// Create entity tree in parent form
	parentForm->populationTrees->Add(gcnew Aga::Controls::Tree::TreeModel());
	parentForm->populationSizes->Add(1.0);
	parentForm->populationSizeMutable->Add(false);

	// Resize number of domain models
	parentForm->ResizeNumDomainModels(parentForm->populationTrees->Count);

	// (this must be the last action)
	populationTabs->TabPages->Insert(populationTabs->TabCount - 1, newpage);
}

void SymmetryView::RemovePopulation(int index)
{
	if(populationTabs->TabCount <= 2)
		return;

	TabPage ^page = populationTabs->TabPages[index];

	// Remove population info from parent form
	parentForm->populationTrees->RemoveAt(index);
	parentForm->populationSizes->RemoveAt(index);
	parentForm->populationSizeMutable->RemoveAt(index);

	// Resize number of domain models
	parentForm->ResizeNumDomainModels(parentForm->populationTrees->Count);

	// Remove tab page (this must be done after removing population info, this triggers the "Selecting" event)
	populationTabs->TabPages->Remove(page);

	// Rename rest of tabs (excluding last)
	for(int i = 0; i < populationTabs->TabCount - 1; i++)
		if(populationTabs->TabPages[i]->Tag != "Renamed")
			populationTabs->TabPages[i]->Text = "Population " + (i + 1);
}


System::Void SymmetryView::populationTabs_Selecting(System::Object^ sender, System::Windows::Forms::TabControlCancelEventArgs^ e)
{
	if(e->TabPage == addPopulationFakeTab)
	{
		AddPopulation();

		e->Cancel = true;
		return;
	}

	// else, save the current tab
	tabIndex = e->TabPageIndex;

	// When changing tabs, clear treeview selection
	treeViewAdv1->ClearSelection();

	// Change 3D visualization, tree model and text box
	avgpopsizeText->Text = "" + parentForm->populationSizes[tabIndex];
	avgpopsizeMut->Checked = parentForm->populationSizeMutable[tabIndex];
	parentForm->entityTree = parentForm->populationTrees[tabIndex];
	treeViewAdv1->Model = parentForm->entityTree;
	tvInvalidate();
}

System::Void SymmetryView::closeToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	if(contextTab >= 0)
		RemovePopulation(contextTab);
}

System::Void SymmetryView::contextMenuStrip1_Opening(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
{
	if(contextTab < 0 || contextTab == populationTabs->TabCount - 1)
	{
		contextMenuStrip1->Enabled = false;
		return;
	}

	contextMenuStrip1->Enabled = true;

	if(populationTabs->TabCount <= 2)
		closeToolStripMenuItem->Enabled = false;
	else
		closeToolStripMenuItem->Enabled = true;
}



System::Void ConfirmationClickHandler(System::Object^ sender, System::EventArgs^ e) { ((System::Windows::Forms::Form ^)((Button ^)sender)->Parent)->Close(); }
static System::String ^ShowInputDialog(System::String ^ text, System::String ^ caption)
{
	System::Windows::Forms::Form ^prompt = gcnew System::Windows::Forms::Form();
	prompt->Width = 500;
	prompt->Height = 150;
	prompt->FormBorderStyle = FormBorderStyle::FixedDialog;
	prompt->Text = caption;
	prompt->StartPosition = FormStartPosition::CenterScreen;
	prompt->MaximizeBox = false;
	prompt->MinimizeBox = false;
	Label ^textLabel = gcnew Label(); textLabel->Left = 50; textLabel->Top=20; textLabel->AutoSize = true; textLabel->Text=text;
	TextBox ^textBox = gcnew TextBox(); textBox->Left = 50; textBox->Top=50; textBox->Width=400;
	Button ^confirmation = gcnew Button(); confirmation->Text = "OK";confirmation->Left=350;confirmation->Width=100; confirmation->Top=70;
	confirmation->Click += gcnew System::EventHandler(&ConfirmationClickHandler);	
	prompt->Controls->Add(textBox);
	prompt->Controls->Add(confirmation);
	prompt->Controls->Add(textLabel);
	prompt->AcceptButton = confirmation;
	prompt->ShowDialog();
	return textBox->Text;

	
}


System::Void SymmetryView::modelRenameToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	Entity^ en;
	try
	{
		en = (Entity ^)treeViewAdv1->SelectedNodes[0]->Tag;
	}
	catch (...)
	{
		return;
	}
	String^ uniquetest;
	bool unique;
	Aga::Controls::Tree::TreeViewAdv ^ population_comparer = (gcnew Aga::Controls::Tree::TreeViewAdv());
	do 
	{
		unique = true;
		System::String ^result = ShowInputDialog("Rename " + en->displayName + " to:", "Rename " + en->Text);
		if (result->Length > 0)
		{
			for each (Aga::Controls::Tree::TreeModel ^ pop in parentForm->populationTrees)
			{
				population_comparer->Model = pop;
				for each (Aga::Controls::Tree::TreeNodeAdv ^ node in population_comparer->AllNodes)
				{
					uniquetest = ((Entity ^)node->Tag)->displayName;
					if (result == uniquetest)
					{
						unique = false;
						break;
					}
				}
			}

			if (unique)
				en->displayName = result;
			else
				MessageBox::Show("Model name is not unique", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	} while (!unique);

}
System::Void SymmetryView::modelDeleteNameToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e)
{
	Entity^ en;
	try
	{
		en = (Entity ^)treeViewAdv1->SelectedNodes[0]->Tag;
	}
	catch (...)
	{
		return;
	}
	if (en->displayName != "")
	{
		en->displayName = "";
		//MessageBox::Show("Model name deleted");
	}

}

System::Void SymmetryView::treeViewAdv1_MouseClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {


	//nothing, for now
}




System::Void SymmetryView::renameToolStripMenuItem_ShortcutClick(System::Object^ sender, System::EventArgs^ e)
{
	System::String ^result = ShowInputDialog("Rename " + populationTabs->SelectedTab->Text + " to:", 
											 "Rename " + populationTabs->SelectedTab->Text);
	if(result->Length > 0)
	{
		populationTabs->SelectedTab->Text = result;
		populationTabs->SelectedTab->Tag = "Renamed";
	}
}

System::Void SymmetryView::renameToolStripMenuItem_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{	
	if(contextTab < 0)
		return;

	System::String ^result = ShowInputDialog("Rename " + populationTabs->TabPages[contextTab]->Text + " to:", 
											 "Rename " + populationTabs->TabPages[contextTab]->Text);
	if(result->Length > 0)
	{
		populationTabs->TabPages[contextTab]->Text = result;
		populationTabs->TabPages[contextTab]->Tag = "Renamed";
	}
}

int SymmetryView::GetHoveredTab(System::Windows::Forms::MouseEventArgs^ e)
{
	int result = -1;

	// iterate through all the tab pages
	for(int i = 0; i < populationTabs->TabCount; i++)
	{
		// get their rectangle area and check if it contains the mouse cursor
		Rectangle r = populationTabs->GetTabRect(i);
		if (r.Contains(e->Location))
		{
			result = i;
		}
	}

	return result;
}

System::Void SymmetryView::populationTabs_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
{
	// check if the right mouse button was pressed
	if(e->Button == System::Windows::Forms::MouseButtons::Right)
	{
		contextTab = GetHoveredTab(e);
	}
}
	
System::Void SymmetryView::populationTabs_MouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e)
{
	if(e->Button == System::Windows::Forms::MouseButtons::Middle)
	{
		contextTab = GetHoveredTab(e);

		// If this tab can be closed, close it
		if(contextTab >= 0 && contextTab < populationTabs->TabCount - 1)
			RemovePopulation(contextTab);
	}	
}

System::Void SymmetryView::avgpopsizeText_Leave(System::Object^ sender, System::EventArgs^ e)
{
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;

	if(Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if(source->Text->StartsWith("=")) {
			res = parentForm->LuaParseExpression(source->Text->Substring(1));
			source->Text = res.ToString();
		}

		// Set the population size parameter in the parent form
		parentForm->populationSizes[tabIndex] = res;
	}
}

System::Void SymmetryView::avgpopsizeText_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return
		|| e->KeyCode == Keys::Escape) {
			parentForm->takeFocus(sender, e);
			e->Handled = true;
	}
}

System::Void SymmetryView::avgpopsizeMut_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
{
	parentForm->populationSizeMutable[populationTabs->SelectedIndex] = avgpopsizeMut->Checked;
}

System::Void SymmetryView::anomalous_CheckedClick(System::Object^ sender, System::EventArgs^ e)
{
	Entity ^ ent = GetSelectedEntity();

	if (AnomalousChecked())
	{
		String ^anomfilename = "";
		OpenFileDialog ^ofd = gcnew OpenFileDialog();
		ofd->Title = "Select the anomalous file...";
		ofd->Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*";
		ofd->FileName = "";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK)
			anomfilename = ofd->FileName;

		if (anomfilename && anomfilename!="")
		{

			UpdateModelPtr(ent, anomfilename);
		}

		else
		{
			this->anomalousCheckBox->Checked = false;
		}
	}
	else
	{
		if (ent->anomfilename != "")
		{
			String ^anomfilename = "";
			UpdateModelPtr(ent, anomfilename);
		}
	}
}
System::Void SymmetryView::UpdateModelPtr(Entity ^ ent, String ^anomfilename)
	{
		std::string fnStr(clrToString(ent->filename));
		std::string afnStr;
		std::vector<const char*> filenames{ fnStr.c_str() };

		std::vector<unsigned int> fnLens{ unsigned int(ent->filename->Length) };

		if (anomfilename->Length)
		{
			afnStr = clrToString(anomfilename);
			filenames.push_back(afnStr.c_str());
			fnLens.push_back(anomfilename->Length);
		}
		ent->anomfilename = anomfilename;
		ent->BackendModel = ent->frontend->CreateFileAmplitude(parentForm->job, AF_PDB, nullptr,
			nullptr, filenames.data(), fnLens.data(), int(fnLens.size()), ent->bCentered);
	}

System::Void SymmetryView::AddLayerManualSymmetry(System::Object ^ sender, System::EventArgs^ e, Entity^ ent)
{
	
	 // creating manual symmetry with at least one layer (like the function ParameterEditor::addLayerButton_Click)
		ParameterEditor^ pe = (ParameterEditor^)(parentForm->PaneList[PARAMETER_EDITOR]);
		paramStruct ps = ent->GetParameters();
		if (ps.layers == 0)
		{
			ps.params.resize(ps.nlp);

			// Add the actual layer	
			for (int i = 0; i < ps.nlp; i++)
				ps.params[i].push_back(Parameter(ent->modelUI->GetDefaultParamValue(ps.layers, i)));

			ps.layers++;

			// Commit the new parameters
			ent->SetParameters(ps, parentForm->GetLevelOfDetail());

			// Update grid-view and buttons
			pe->FillParamGridView(ent);

			// Invalidate 3D viewport
			GraphPane3D ^g3 = (GraphPane3D ^)parentForm->PaneList[GRAPH3D];
			g3->glCanvas3D1->Invalidate();
			g3->Invalidate();

			// Scroll to bottom
			if (ps.nlp > 0)
				pe->parameterDataGridView->CurrentCell = pe->parameterDataGridView[0, ps.layers - 1];
		}

}

System::Void DPlus::SymmetryView::textBox_Leave(System::Object^ sender, System::EventArgs^ e)
{
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;

	if (Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if (source->Text->StartsWith("=")) {
			res = parentForm->LuaParseExpression(source->Text->Substring(1));
			source->Text = res.ToString();
		}

		// Set the scale parameter in the parent form
		if (source == scaleBox)
			parentForm->domainScale = res;
		else if (source == constantBox)
			parentForm->domainConstant = res;
	}
}

System::Void DPlus::SymmetryView::textBox_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e)
{
	if (e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return
		|| e->KeyCode == Keys::Escape) {
		parentForm->takeFocus(sender, e);
		e->Handled = true;
	}
}

System::Void DPlus::SymmetryView::SetDefaultParams()
{
	this->scaleMut->Checked = false;
	this->scaleMut->CheckState = System::Windows::Forms::CheckState::Unchecked;

	this->constantMut->Checked = false;
	this->constantMut->CheckState = System::Windows::Forms::CheckState::Unchecked;

	this->scaleBox->Text = L"1";
	this->constantBox->Text = L"0";
}

};