#include "PreferencesPane.h"
#include "SymmetryView.h"
#include "GraphPane3D.h"
#include "ControlButtonsPane.h"

System::Void DPlus::PreferencesPane::usGridCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	bool ch = useGridCheckBox->Checked;
	gridSizeLabel->Enabled = ch;
	gridSizeTextBox->Enabled = ch;
}



paramStruct DPlus::PreferencesPane::GetDomainPreferences() {
	paramStruct ps;
	//return ps; //TODO: this function appears to be useless (as tested by returning here), investigate further (eg Lua) and remove


	ps.nlp = 7; // THE NUMBER OF PREFERENCES
	ps.layers = 1;
	ps.params.resize(ps.nlp);

	double dbl;
	if (!Double::TryParse(this->integIterTextBox->Text, dbl))
	{
		throw gcnew UserInputException("The text in the \"Integration Iterations\" field in the Preferences Pane must be a valid number.");
	}
	if (dbl < 0)
		throw gcnew UserInputException("The text in the \"Integration Iterations\" field in the Preferences Pane must be a positive .");
	ps.params[0].push_back(Parameter(dbl));

	int gridSize;
	if (!Int32::TryParse(this->gridSizeTextBox->Text, gridSize))
	{
		throw gcnew UserInputException("The text in the \"Grid Size\" field in the Preferences Pane must be a valid number.");
	}
	if (gridSize % 2 != 0)
		throw gcnew UserInputException("grid size must be even");
	if (gridSize < 20)
		throw gcnew UserInputException("Minimum grid size is 20");

	ps.params[1].push_back(Parameter(dbl));

	ps.params[2].push_back(Parameter(this->useGridCheckBox->Checked ? 1.0 : 0.0));


	if(!Double::TryParse(this->convTextBox->Text, dbl))
	{
		throw gcnew UserInputException("The text in the \"Convergence\" field in the Preferences Pane must be a valid number.");
	}
	if (dbl< 0)
		throw gcnew UserInputException("The text in the \"Convergence\" field in the Preferences Pane must be positive.");

	ps.params[3].push_back(Parameter(dbl));

	if(!Double::TryParse(this->qMaxTextBox->Text, dbl) && parentForm->loadedSignal != nullptr)
	{
		throw gcnew UserInputException("The text in the \"q Max\" field in the Preferences Pane must be a valid number.");
	}
	if (parentForm->loadedSignal != nullptr)
	{
		dbl = parentForm->qvec[parentForm->qvec->Length - 1];
	}
	ps.params[4].push_back(Parameter(dbl));

	int ind  = this->integrationMethodComboBox->SelectedIndex;
	ps.params[5].push_back(Parameter(double(ind)));
	
	if (!Double::TryParse(this->qMinTextBox->Text, dbl) && parentForm->loadedSignal != nullptr)
	{
		throw gcnew UserInputException("The text in the \"q Min\" field in the Preferences Pane must be a valid number.");
	}
	ps.params[6].push_back(Parameter(dbl));

	return ps;
}

String ^DPlus::PreferencesPane::SerializePreferences() {
	if(this->InvokeRequired) {
		return ((String ^)(this->Invoke(gcnew FuncNoParamsReturnString(this, &PreferencesPane::SerializePreferences))));
	}	

	 String ^ applyResolutionCheckBox = ((ControlButtonsPane ^)(this->parentForm->PaneList[CALC_BUTTONS]))->applyResolutionCheckBox->Checked ? "true" : "false";
	 String ^ resolutionSigmaTextBox = ((ControlButtonsPane ^)(this->parentForm->PaneList[CALC_BUTTONS]))->resolutionSigmaTextBox->Text;

	String ^contents = "";

	contents += "DomainPreferences = {\n";

	//contents += "\tLive_Generation = " + (this->parentForm->liveGenerationToolStripMenuItem->Checked ? "true" : "false") + ",\n";
	contents += "\tFitting_UpdateGraph = " + (this->parentForm->updateFitGraphToolStripMenuItem->Checked ? "true" : "false") + ",\n";
	contents += "\tFitting_UpdateDomain = " + (this->parentForm->updateFitDomainToolStripMenuItem->Checked ? "true" : "false") + ",\n";

	contents += "\tOrientationMethod = [[" + this->integrationMethodComboBox->Text + "]],\n";
	contents += "\tOrientationIterations = " + this->integIterTextBox->Text + ",\n";
	contents += "\tGridSize = " + this->gridSizeTextBox->Text + ",\n";
	contents += "\tUseGrid = " + (this->useGridCheckBox->Checked ? "true" : "false") + ",\n";

	contents += "\tConvergence = " + this->convTextBox->Text + ",\n";
	contents += "\tqMin = " + this->qMinTextBox->Text + ",\n";
	contents += "\tqMax = " + this->qMaxTextBox->Text + ",\n";
	contents += "\tUpdateInterval = " + this->updateIntervalMSTextBox->Text + ",\n";
	contents += "\tGeneratedPoints = " + this->genResTextBox->Text + ",\n";

	contents += "\tApplyResolution = " + applyResolutionCheckBox + ",\n";
	contents += "\tResolutionSigma = " + resolutionSigmaTextBox + ",\n";

	if(parentForm->loadedSignal != nullptr)
		contents += "\tSignalFile = [[" + parentForm->signalFilename + "]],\n";

	

	contents += "};\n";

	return contents;
} 

static int ClampInt(double value, int low, int high)  {
	if(value < low)
		return low;
	if(value > high)
		return high;

	return (int)(value + 0.1);
}

void DPlus::PreferencesPane::DeserializePreferences(LuaTable ^domainPrefs) {
	if(domainPrefs == nullptr) // Load defaults
		return;

	//if(domainPrefs["Live_Generation"] != nullptr) 
		//this->parentForm->liveGenerationToolStripMenuItem->Checked = LuaItemToBoolean(domainPrefs["Live_Generation"]);

	if(domainPrefs["Fitting_UpdateGraph"] != nullptr) 
		this->parentForm->updateFitGraphToolStripMenuItem->Checked = LuaItemToBoolean(domainPrefs["Fitting_UpdateGraph"]);

	if(domainPrefs["Fitting_UpdateDomain"] != nullptr) 
		this->parentForm->updateFitDomainToolStripMenuItem->Checked = LuaItemToBoolean(domainPrefs["Fitting_UpdateDomain"]);
		
	if(domainPrefs["OrientationIterations"] != nullptr)
		this->integIterTextBox->Text = Int32(LuaItemToDouble(domainPrefs["OrientationIterations"])).ToString();

	if(domainPrefs["OrientationMethod"] != nullptr) {
		bool bFound = false;
		System::String ^im = dynamic_cast<String ^>(domainPrefs["OrientationMethod"]);
		for(int i = 0;  i < integrationMethodComboBox->Items->Count && im != nullptr; i++) {
			if(integrationMethodComboBox->Items[i]->Equals(im)) {
				this->integrationMethodComboBox->SelectedIndex = i;
				bFound = true; break;
			}
		}
		if(!bFound)
			this->integrationMethodComboBox->SelectedIndex = 0;
	} else {
		this->integrationMethodComboBox->SelectedIndex = 0;
	}

	if(domainPrefs["GridSize"] != nullptr)
		this->gridSizeTextBox->Text = Int32(LuaItemToDouble(domainPrefs["GridSize"])).ToString();

	if(domainPrefs["UseGrid"] != nullptr)
		this->useGridCheckBox->Checked = LuaItemToBoolean(domainPrefs["UseGrid"]);

	if (domainPrefs["ApplyResolution"] != nullptr)
		((ControlButtonsPane ^)(this->parentForm->PaneList[CALC_BUTTONS]))->applyResolutionCheckBox->Checked = 
														 LuaItemToBoolean(domainPrefs["ApplyResolution"]);
	
	if (domainPrefs["ResolutionSigma"] != nullptr)
		((ControlButtonsPane ^)(this->parentForm->PaneList[CALC_BUTTONS]))->resolutionSigmaTextBox->Text =
														LuaItemToDouble(domainPrefs["ResolutionSigma"]).ToString(); 

	if(domainPrefs["Convergence"] != nullptr)
		this->convTextBox->Text = LuaItemToDouble(domainPrefs["Convergence"]).ToString();

	if (domainPrefs["qMin"] != nullptr)
		this->qMinTextBox->Text = LuaItemToDouble(domainPrefs["qMin"]).ToString();

	if(domainPrefs["qMax"] != nullptr)
		this->qMaxTextBox->Text = LuaItemToDouble(domainPrefs["qMax"]).ToString();

	if (domainPrefs["GeneratedPoints"] != nullptr)
		this->genResTextBox->Text = LuaItemToDouble(domainPrefs["GeneratedPoints"]).ToString();

	if(domainPrefs["UpdateInterval"] != nullptr)
		this->updateIntervalMSTextBox->Text = Int32(LuaItemToDouble(domainPrefs["UpdateInterval"])).ToString();

	if(domainPrefs["SignalFile"] != nullptr && dynamic_cast<String ^>(domainPrefs["SignalFile"]))
		parentForm->LoadSignal((String ^)domainPrefs["SignalFile"]);

}


System::Void DPlus::PreferencesPane::gridSizeTextBox_Validating(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
{
	TextBox ^ tb = (TextBox ^)(sender);

	Int32 asInt;
	
	if (Int32::TryParse(tb->Text, asInt))
	{
		if (asInt % 2 == 1)
		{
			MessageBox::Show("The grid size must be even. The size will be increased by one upon calculation.",
									 "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
			gridSizeTextBox->Text = L"80";
		}
	}
	else
	{
		MessageBox::Show("The grid size must be an integer.", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
		gridSizeTextBox->Text = L"80";
	}
}

System::Void DPlus::PreferencesPane::qMinTextBox_Validating(System::Object ^ sender, System::ComponentModel::CancelEventArgs ^ e)
{
	TextBox ^ tb = (TextBox ^)(sender);

	double qmin;
	double qmax;
	if (Double::TryParse(tb->Text, qmin)) {
		if (Double::TryParse(qMaxTextBox->Text, qmax)) {
			if (qmin > qmax) {
				MessageBox::Show("qmin value must be smaller than qmax value", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
				qMinTextBox->Text = prev_qmin.ToString();
				return;
			}
			prev_qmin = qmin;
		}
	}
	else
	{ 
		MessageBox::Show("The qmin value must be a valid number.", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
		qMinTextBox->Text = prev_qmin.ToString();
	}
	return;
}

System::Void DPlus::PreferencesPane::qMaxTextBox_Validating(System::Object ^ sender, System::ComponentModel::CancelEventArgs ^ e)
{
	TextBox ^ tb = (TextBox ^)(sender);

	double qmin;
	double qmax;
	if (Double::TryParse(tb->Text, qmax)) {
		if (Double::TryParse(qMinTextBox->Text, qmin)) {
			if (qmax < qmin) {
				MessageBox::Show("qmax value must be bigger than qmin value", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
				qMaxTextBox->Text = prev_qmax.ToString();
				return;
			}
			prev_qmax = qmax;
		}
	}
	else 
	{
		MessageBox::Show("The qmax value must be a valid number.", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
		qMaxTextBox->Text = prev_qmax.ToString();
	}
	return;
}

System::Void DPlus::PreferencesPane::qMinTextBox_TextChanged(System::Object ^ sender, System::EventArgs ^ e)
{
	// if validate_qs is false, the system now loading data from signal file, therfore there is no need to run validations on qmin and qmax
	if (!validate_qs)
		return;
	TextBox ^ tb = (TextBox ^)(sender);
	double qmin;
	double qmax;
	if (Double::TryParse(tb->Text, qmin)) {
		if (Double::TryParse(qMaxTextBox->Text, qmax)) {
			if (qmin < 0) {
				MessageBox::Show("qmin value must be bigger than 0", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
				qMinTextBox->Text = prev_qmin.ToString();
				return;
			}
		}
	}
}
System::Void DPlus::PreferencesPane::qMaxTextBox_TextChanged(System::Object ^ sender, System::EventArgs ^ e)
{
	//// if validate_qs is false, the system now loading data from signal file, therfore there is no need to run validations on qmin and qmax
	//if (!validate_qs)
	//	return;
	//TextBox ^ tb = (TextBox ^)(sender);
	//double qmin;
	//double qmax;
	//if (Double::TryParse(tb->Text, qmax)) {
	//	if (Double::TryParse(qMinTextBox->Text, qmin)) {
	//		if (qmax < qmin) {
	//			MessageBox::Show("qmax value must be bigger than qmin value", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
	//			qMaxTextBox->Text = prev_qmax.ToString();
	//			return;
	//		}
	//		prev_qmax = qmax;
	//	}
	//}
}

System::Void DPlus::PreferencesPane::SetDefaultParams()
{
	this->integIterTextBox->Text = L"1000000";
	this->gridSizeTextBox->Text = L"80";
	this->useGridCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
	this->qMaxTextBox->Text = L"7.5";
	this->qMinTextBox->Text = L"0";
	this->genResTextBox->Text = L"800";
	this->convTextBox->Text = L"0.001";
	this->updateIntervalMSTextBox->Text = L"100";
	this->integrationMethodComboBox->SelectedIndex = 0;
}