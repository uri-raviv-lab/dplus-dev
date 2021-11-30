#include "SymmetryEditor.h"
#include "ConstraintsWindow.h"

System::Void DPlus::SymmetryEditor::scrollTimer_Tick(System::Object^ sender, System::EventArgs^ e) {

	// For x, y, and z trackbars
	for(int i = 0; i < 3; i++) {
		TrackBar^ tb = TrackBarList[i];
		int mid = int((tb->Maximum - tb->Minimum) / 2.0);
		if(tb->Value == mid)
			continue;

		double delta, oldVal;
		if(!Double::TryParse(TextBoxList[i]->Text, oldVal))
			continue;
		delta = double(tb->Value - mid) * 0.025 * fabs(oldVal) / mid;
		TextBoxList[i]->Text = Double(oldVal + delta).ToString("0.000");
		transTextBox_Leave(TextBoxList[i], nullptr);
	}
}

void DPlus::SymmetryEditor::consDummy() {
	TextBoxList = gcnew System::Collections::Generic::List<TextBox^>();
	TrackBarList = gcnew System::Collections::Generic::List<TrackBar^>();

	TextBoxList->Add(xTextBox);		TrackBarList->Add(xTrackBar);
	TextBoxList->Add(yTextBox);		TrackBarList->Add(yTrackBar);
	TextBoxList->Add(zTextBox);		TrackBarList->Add(zTrackBar);
	TextBoxList->Add(alphaTextBox);	TrackBarList->Add(alphaTrackBar);
	TextBoxList->Add(betaTextBox);	TrackBarList->Add(betaTrackBar);
	TextBoxList->Add(gammaTextBox);	TrackBarList->Add(gammaTrackBar);

	for(int i = 0; i < TrackBarList->Count; i++) {
		DPlus::centerTrackBar(TrackBarList[i]);
		TrackBarList[i]->TickFrequency = TrackBarList[i]->Value;
	}

	scrollTimer->Enabled = true;


}

System::Void DPlus::SymmetryEditor::angleTrackBar_Scroll(System::Object^ sender, System::EventArgs^ e) {
	TrackBar^ sen = dynamic_cast<TrackBar^>(sender);
	if(!sen) {
		TextBox^ tb = dynamic_cast<TextBox^>(sender);
		if(!tb)
			return;
		return;
	}
	int ind = TrackBarList->IndexOf(sen);
	if(sen->Value == sen->Maximum) {
		sen->Value = sen->Minimum + 1;
		if(/*mouse pressed*/true/*for now*/) {
			this->Cursor = gcnew System::Windows::Forms::Cursor( ::Cursor::Current->Handle );
			System::Drawing::Point po = ::Cursor::Position;
			po.X -= sen->Width - 21;
			::Cursor::Position = po;
		}
	}
	if(sen->Value == sen->Minimum) {
		sen->Value = sen->Maximum - 1;
		if(/*mouse pressed*/true/*for now*/) {
			this->Cursor = gcnew System::Windows::Forms::Cursor( ::Cursor::Current->Handle );
			System::Drawing::Point po = ::Cursor::Position;
			po.X += sen->Width - 21;
			::Cursor::Position = po;
		}
	}
	TextBoxList[ind]->Text = Double(0.25 * (double)(sen->Value - 1)).ToString();
	angleTextBox_Leave(TextBoxList[ind], nullptr);
}

// Always returns a positive result, DO NOT TOUCH IT JUST WORKS
static double Mod(double x, double y)
{
	if (0 == y)
		return x;

	return x - y * floor(x/y);
}

System::Void DPlus::SymmetryEditor::angleTextBox_Leave(System::Object^ sender, System::EventArgs^ e) {

	TrackBar^ tb;
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;
	if(!source) {
		if(tb = dynamic_cast<TrackBar^>(sender))
			return;
	}


	if(Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if(controlledForm->treeViewAdv1->SelectedNodes->Count == 0)
			return;
		Entity^ ent = dynamic_cast<Entity^>(controlledForm->treeViewAdv1->SelectedNodes[0]->Tag);
		if(!ent) {
			MessageBox::Show("Bad Entity","Bad Entity! This code needs fixing...");
			return;
		}

		if(source->Text->StartsWith("="))			
			res = parentForm->LuaParseExpression(source->Text->Substring(1));

		// Result is modulo 360
		source->Text = Mod(res, 360.0).ToString();
		angleTextBox_TextChanged(sender, e);

 		if(source == alphaTextBox)
 			ent->SetAlpha(Radian(Degree((float)res)));
 		else if(source == betaTextBox)
			ent->SetBeta(Radian(Degree((float)res)));
 		else if(source == gammaTextBox)
			ent->SetGamma(Radian(Degree((float)res)));
		
		controlledForm->treeViewAdv1->Invalidate();
		((GraphPane3D^)(parentForm->PaneList[GRAPH3D]))->glCanvas3D1->Invalidate();
		((GraphPane3D^)(parentForm->PaneList[GRAPH3D]))->Invalidate();

	}

}

System::Void DPlus::SymmetryEditor::angleTextBox_TextChanged(System::Object^ sender, System::EventArgs^ e) {
	TextBox^ tex;
	if(!(tex = dynamic_cast<TextBox^>(sender)))
		return;
	if(dynamic_cast<TextBox^>(this->ActiveControl) == tex)
		return;
	double res;
	TrackBar^ tb;
	if(!(Double::TryParse(tex->Text, res)))
		return;
	tb = TrackBarList[TextBoxList->IndexOf(tex)];
	tb->Value = std::max(tb->Minimum + 1, std::min(tb->Maximum - 1, (int)(res / 0.25 + 1.0)));

}

System::Void DPlus::SymmetryEditor::transTrackBar_MouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
	TrackBar^ tb = dynamic_cast<TrackBar^>(sender);
	if(!tb)
		return;
	DPlus::centerTrackBar(tb);
}

System::Void DPlus::SymmetryEditor::transTextBox_Leave(System::Object^ sender, System::EventArgs^ e) {
	TrackBar^ tb;
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;
	if(!source) {
		if(tb = dynamic_cast<TrackBar^>(sender))
			return;
	}

	if(Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if(controlledForm->treeViewAdv1->SelectedNodes->Count == 0)
			return;
		Entity^ ent = dynamic_cast<Entity^>(controlledForm->treeViewAdv1->SelectedNodes[0]->Tag);
		if(!ent) {
			MessageBox::Show("Bad Entity","Bad Entity! This code needs fixing...");
			return;
		}
		if(source->Text->StartsWith("=")) {
			res = parentForm->LuaParseExpression(source->Text->Substring(1));
			source->Text = res.ToString();
		}

		if(source == xTextBox)
			ent->SetX(res);
		else if(source == yTextBox)
			ent->SetY(res);
		else if(source == zTextBox)
			ent->SetZ(res);

		controlledForm->treeViewAdv1->Invalidate();
		((GraphPane3D^)(parentForm->PaneList[GRAPH3D]))->glCanvas3D1->Invalidate();
		((GraphPane3D^)(parentForm->PaneList[GRAPH3D]))->Invalidate();

	}
}

System::Void DPlus::SymmetryEditor::TextBox_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return
		|| e->KeyCode == Keys::Escape) {
			parentForm->takeFocus(sender, e);
			e->Handled = true;
	}
}

System::Void DPlus::SymmetryEditor::MutCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	CheckBox ^cb = dynamic_cast<CheckBox^>(sender);

	if(!cb)
		return;

	SymmetryView ^sv = (SymmetryView ^)parentForm->PaneList[SYMMETRY_VIEWER];
	Entity ^ent = sv->GetSelectedEntity();
	if(ent == nullptr)
		return;

	if(cb == xMutCheckBox) {
		ent->SetXMut(cb->Checked);
	} else if(cb == yMutCheckBox) {
		ent->SetYMut(cb->Checked);
	} else if(cb == zMutCheckBox) {
		ent->SetZMut(cb->Checked);
	} else if(cb == aMutCheckBox) {
		ent->SetAlphaMut(cb->Checked);
	} else if(cb == bMutCheckBox) {
		ent->SetBetaMut(cb->Checked);
	} else if(cb == gMutCheckBox) {
		ent->SetGammaMut(cb->Checked);
	} else {
		MessageBox::Show("Blah");
	}
}

System::Void DPlus::SymmetryEditor::constraintsButton_Click(System::Object^ sender, System::EventArgs^ e) {
	ConstraintsWindow ^cw = gcnew ConstraintsWindow(parentForm, ((SymmetryView^)parentForm->PaneList[SYMMETRY_VIEWER])->GetSelectedEntity(), NULL, CONS_XYZABG);
	System::Windows::Forms::DialogResult dr = cw->ShowDialog();

	if(dr == System::Windows::Forms::DialogResult::OK) {
		// TODO: Change the text boxes and invalidate
//		FillParamGridView(((SymmetryView^)parentForm->PaneList[SYMMETRY_VIEWER])->GetSelectedEntity());
		;
	}
}

System::Void DPlus::SymmetryEditor::useGridAtLevelCheckBox_CheckedChanged( System::Object^ sender, System::EventArgs^ e ) {
	CheckBox ^cb = dynamic_cast<CheckBox^>(sender);

	if(!cb)
		return;

	SymmetryView ^sv = (SymmetryView ^)parentForm->PaneList[SYMMETRY_VIEWER];
	Entity ^ent = sv->GetSelectedEntity();
	if(ent == nullptr)
		return;
	if (parentForm->InSelectionChange)
	{
		ent->SetUseGrid(cb->Checked);
		return;
	}
	if (!cb->Checked && ent->IsParentUseGrid() )
	{
		System::Windows::Forms::DialogResult result;
		result = MessageBox::Show("Unselecting 'use grid from here' will cause all parent nodes to have this be unselected as well. Do you wish to proceed?", "Question", MessageBoxButtons::OKCancel, MessageBoxIcon::Question);
		if (result == ::DialogResult::OK)
			ent->SetUseGrid(cb->Checked);
		else
			useGridAtLevelCheckBox->Checked = true;
	}
	else
		ent->SetUseGrid(cb->Checked);
}


