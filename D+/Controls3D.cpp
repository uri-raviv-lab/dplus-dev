#include "Controls3D.h"
#include <math.h>

System::Void DPlus::Controls3D::showAxesCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	controlledForm->glCanvas3D1->ShowAxis = showAxesCheckBox->Checked;
	controlledForm->glCanvas3D1->ShowConstantSizedAxis = fixedSize_checkBox->Checked;
	controlledForm->glCanvas3D1->Invalidate();
	controlledForm->Invalidate();
}

System::Void DPlus::Controls3D::showCornerAxesCheckBox_CheckedChanged( System::Object^ sender, System::EventArgs^ e ) {
	controlledForm->glCanvas3D1->ShowCornerAxis = showCornerAxesCheckBox->Checked;
	controlledForm->glCanvas3D1->Invalidate();
	controlledForm->Invalidate();
}


System::Void DPlus::Controls3D::zoomTextBox_Leave(System::Object^ sender, System::EventArgs^ e) {
	double val;
	if(!(Double::TryParse(zoomTextBox->Text, val))) {
		// deal; or not...
		return;
	}
	if(fabs(val) < 1.0e-6)
		val = 1.0e-6;
	controlledForm->glCanvas3D1->Distance = (float)val;
	controlledForm->glCanvas3D1->Invalidate();
}

System::Void DPlus::Controls3D::angleTextBox_Leave(System::Object^ sender, System::EventArgs^ e) {
/* // GRRRRRRR Doesn't work!
	float *target = NULL;
	if(sender == angle1TextBox)
		target = &(controlledForm->glCanvas3D1->Yaw);
	if(sender == angle2TextBox)
		target = &(controlledForm->glCanvas3D1->Yaw);
	if(target == NULL) {
		return;
	}
*/

	TrackBar^ tb;
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;
	if(!source) {
		if(tb = dynamic_cast<TrackBar^>(sender))
			return;
	}


	if(Double::TryParse(source->Text, res)) {
		//*target = (float)res;
		if(source == pitchTextBox)
			controlledForm->glCanvas3D1->Pitch = (float)res;
		else if(source == yawTextBox)
			controlledForm->glCanvas3D1->Yaw = (float)res;
		else if(source == rollTextBox)
			controlledForm->glCanvas3D1->Roll = (float)res;

		controlledForm->glCanvas3D1->Invalidate();
	}

}

void DPlus::Controls3D::ConsDummy() {
	TextBoxList = gcnew System::Collections::Generic::List<TextBox^>();
	TrackBarList = gcnew System::Collections::Generic::List<TrackBar^>();
	
	TextBoxList->Add(pitchTextBox);	TrackBarList->Add(angle1TrackBar);
	TextBoxList->Add(yawTextBox);	TrackBarList->Add(angle2TrackBar);
	TextBoxList->Add(rollTextBox);	TrackBarList->Add(angle3TrackBar);
	TextBoxList->Add(zoomTextBox);		TrackBarList->Add(zoomTrackBar);

	// This works!
	// TrackBarList->ForEach(gcnew Action<TrackBar^>(DPlus::centerTrackBar));
	for(int i = 0; i < TrackBarList->Count; i++) {
		DPlus::centerTrackBar(TrackBarList[i]);
		TrackBarList[i]->TickFrequency = TrackBarList[i]->Value;
	}

	scrollTimer->Enabled = true;
}

System::Void DPlus::Controls3D::angleTrackBar_Scroll(System::Object^ sender, System::EventArgs^ e) {
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

System::Void DPlus::Controls3D::scrollTimer_Tick(System::Object^ sender, System::EventArgs^ e) {
	int mid = int((zoomTrackBar->Maximum - zoomTrackBar->Minimum) / 2.0);
	if(zoomTrackBar->Value == mid)
		return;

	double delta, oldVal;
	if(!Double::TryParse(zoomTextBox->Text, oldVal))
		return;
	delta = double(zoomTrackBar->Value - mid) * 0.03 * oldVal / mid;
	zoomTextBox->Text = Double(oldVal + delta).ToString("0.000");
	zoomTextBox_Leave(zoomTextBox, nullptr);
}

System::Void DPlus::Controls3D::zoomTrackBar_MouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
	TrackBar^ tb = dynamic_cast<TrackBar^>(sender);
	if(!tb)
		return;
	DPlus::centerTrackBar(tb);
}

void DPlus::Controls3D::angleTextBox_TextChanged(System::Object^ sender, System::EventArgs^ e) {
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

void DPlus::Controls3D::TextBox_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e) {
	if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return
								 || e->KeyCode == Keys::Escape) {
		parentForm->takeFocus(sender, e);
		e->Handled = true;
	}
}

System::Void DPlus::Controls3D::generateButton_Click( System::Object^ sender, System::EventArgs^ e ) {
	parentForm->Generate(); //1
}

System::Void DPlus::Controls3D::fitButton_Click(System::Object^ sender, System::EventArgs^ e) {
	parentForm->Fit();
}

System::Void DPlus::Controls3D::stopButton_Click(System::Object^ sender, System::EventArgs^ e) {
	parentForm->Stop();
}

System::String ^DPlus::Controls3D::SerializePreferences() {
	String ^contents = "";

	contents += "Viewport = {\n";
	GLView::Point3D camPos = this->controlledForm->glCanvas3D1->camPosition;
	GLView::Point3D camTar = this->controlledForm->glCanvas3D1->camTarget;

	contents += "\tctx = " + Double(camTar.X) + ",\n";
	contents += "\tcty = " + Double(camTar.Y)  + ",\n";
	contents += "\tctz = " + Double(camTar.Z)  + ",\n";

	contents += "\tcpx = " + Double(camPos.X) + ",\n";
	contents += "\tcpy = " + Double(camPos.Y)  + ",\n";
	contents += "\tcpz = " + Double(camPos.Z)  + ",\n";

	contents += "\tcPitch = " + Double(this->controlledForm->glCanvas3D1->camPitch)  + ",\n";
	contents += "\tcRoll = " + Double(this->controlledForm->glCanvas3D1->camRoll)  + ",\n";
		
	contents += "\tPitch = " + Double(this->controlledForm->glCanvas3D1->Pitch   ) + ",\n";
	contents += "\tYaw = "   + Double(this->controlledForm->glCanvas3D1->Yaw     ) + ",\n";
	contents += "\tRoll = "  + Double(this->controlledForm->glCanvas3D1->Roll    ) + ",\n";
	contents += "\tZoom = "  + Double(this->controlledForm->glCanvas3D1->Distance) + ",\n";

	contents += "\tAxes_at_origin = " + (this->showAxesCheckBox->Checked ? "true" : "false") + ",\n";
	contents += "\tAxes_in_corner = " + (this->showCornerAxesCheckBox->Checked ? "true" : "false") + ",\n";


	contents += "};\n";

	return contents;
}

void DPlus::Controls3D::DeserializePreferences(LuaTable ^viewPrefs) {

	if(viewPrefs == nullptr) // Load defaults
		return;

	if(viewPrefs["cpx"] != nullptr && viewPrefs["cpy"] != nullptr && viewPrefs["cpz"] != nullptr) {
		GLView::Point3D pnt;
		pnt.X = (float)LuaItemToDouble(viewPrefs["cpx"]);
		pnt.Y = (float)LuaItemToDouble(viewPrefs["cpy"]);
		pnt.Z = (float)LuaItemToDouble(viewPrefs["cpz"]);
		controlledForm->glCanvas3D1->camPosition = (pnt);
	}

	if(viewPrefs["ctx"] != nullptr && viewPrefs["cty"] != nullptr && viewPrefs["ctz"] != nullptr) {
		GLView::Point3D pnt;
		pnt.X = (float)LuaItemToDouble(viewPrefs["ctx"]);
		pnt.Y = (float)LuaItemToDouble(viewPrefs["cty"]);
		pnt.Z = (float)LuaItemToDouble(viewPrefs["ctz"]);
		controlledForm->glCanvas3D1->camTarget = (pnt);
	}

	if(viewPrefs["cPitch"] != nullptr) {
		controlledForm->glCanvas3D1->camPitch = (float)LuaItemToDouble(viewPrefs["cPitch"]);
	}
	if(viewPrefs["cRoll"] != nullptr) {
		controlledForm->glCanvas3D1->camPitch = (float)LuaItemToDouble(viewPrefs["cRoll"]);
	}

	if(viewPrefs["Pitch"] != nullptr) {
		this->pitchTextBox->Text = LuaItemToDouble(viewPrefs["Pitch"]).ToString();
		angleTextBox_Leave(pitchTextBox, nullptr);
	}
	if(viewPrefs["Yaw"] != nullptr) {
		this->yawTextBox->Text = LuaItemToDouble(viewPrefs["Yaw"]).ToString();
		angleTextBox_Leave(yawTextBox, nullptr);
	}
	if(viewPrefs["Roll"] != nullptr) {
		this->rollTextBox->Text = LuaItemToDouble(viewPrefs["Roll"]).ToString();
		angleTextBox_Leave(rollTextBox, nullptr);
	}


	if(viewPrefs["Zoom"] != nullptr) {
		this->zoomTextBox->Text = LuaItemToDouble(viewPrefs["Zoom"]).ToString();
		zoomTextBox_Leave(zoomTextBox, nullptr);
	}

	if(viewPrefs["Axes_at_origin"] != nullptr)
		this->showAxesCheckBox->Checked = LuaItemToBoolean(viewPrefs["Axes_at_origin"]);
	if(viewPrefs["Axes_in_corner"] != nullptr)
		this->showCornerAxesCheckBox->Checked = LuaItemToBoolean(viewPrefs["Axes_in_corner"]);
}

System::Void DPlus::Controls3D::textBox_Leave(System::Object^ sender, System::EventArgs^ e)
{
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;

	if(Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if(source->Text->StartsWith("=")) {
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

System::Void DPlus::Controls3D::textBox_KeyDown(System::Object^ sender, System::Windows::Forms::KeyEventArgs^ e)
{
	if(e->KeyCode == Keys::Enter || e->KeyCode == Keys::Return
		|| e->KeyCode == Keys::Escape) {
			parentForm->takeFocus(sender, e);
			e->Handled = true;
	}
}


System::Void DPlus::Controls3D::SetDefaultParams()
{

	this->pitchTextBox->Text = L"0";
	angleTextBox_Leave(pitchTextBox, nullptr);
	this->yawTextBox->Text = L"0";
	angleTextBox_Leave(yawTextBox, nullptr);
	this->rollTextBox->Text = L"0";
	angleTextBox_Leave(rollTextBox, nullptr);

	this->zoomTextBox->Text = L"4";
	zoomTextBox_Leave(NULL, nullptr);
	this->showAxesCheckBox->Checked = true;
	this->showAxesCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;

	this->showCornerAxesCheckBox->Checked = true;
	this->showCornerAxesCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;

	this->fixedSize_checkBox->Checked = false;
	this->fixedSize_checkBox->CheckState = System::Windows::Forms::CheckState::Unchecked;
	
	this->scaleMut->Checked = false;
	this->scaleMut->CheckState = System::Windows::Forms::CheckState::Unchecked;

	this->constantMut->Checked = false;
	this->constantMut->CheckState = System::Windows::Forms::CheckState::Unchecked;

	this->scaleBox->Text = L"1";
	this->constantBox->Text = L"0";
}