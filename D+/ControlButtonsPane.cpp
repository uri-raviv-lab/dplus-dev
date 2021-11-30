#include "ControlButtonsPane.h"
#include <math.h>

System::Void DPlus::ControlButtonsPane::generateButton_Click(System::Object^ sender, System::EventArgs^ e) {
	parentForm->Generate(); //1
}

System::Void DPlus::ControlButtonsPane::fitButton_Click(System::Object^ sender, System::EventArgs^ e) {
	parentForm->Fit();
}

System::Void DPlus::ControlButtonsPane::stopButton_Click(System::Object^ sender, System::EventArgs^ e) {
	parentForm->Stop();
}

System::Void DPlus::ControlButtonsPane::applyResolutionCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	bool ch = applyResolutionCheckBox->Checked;
	resolutionSigmaLabel->Enabled = ch;
	resolutionSigmaTextBox->Enabled = ch;
}

System::Void DPlus::ControlButtonsPane::resolutionSigmaTextBox_TextChanged(System::Object^ sender, System::EventArgs ^ e)
{
	// TO DO!!
	// if validate_qs is false, the system now loading data from signal file, therfore there is no need to run validations on sigma.
	if (!validate_qs)
		return;
	TextBox ^ tb = (TextBox ^)(sender);

	Double sigma;
	if (Double::TryParse(tb->Text, sigma))
	{
		if (sigma < 0) {
			MessageBox::Show("The resolution sigma value must be a positive number.", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
			resolutionSigmaTextBox->Text = prev_sigma.ToString();
			return;
		}
		prev_sigma = sigma;
	}
	else
	{
		/*MessageBox::Show("The resolution sigma value must be a valid number");
		resolutionSigmaTextBox->Text = prev_sigma.ToString();
		return;*/
	}
}

System::Void DPlus::ControlButtonsPane::resolutionSigmaTextBox_Validating(System::Object ^ sender, System::ComponentModel::CancelEventArgs ^ e)
{
	TextBox ^ tb = (TextBox ^)(sender);

	Double sigma;
	if (!Double::TryParse(tb->Text, sigma)) {
		MessageBox::Show("The resolution sigma value must be a valid number", "Invalid input", MessageBoxButtons::OK, MessageBoxIcon::Error);
		resolutionSigmaTextBox->Text = prev_sigma.ToString();
	}
	return;
}



System::Void DPlus::ControlButtonsPane::SetDefaultParams()
{
	this->applyResolutionCheckBox->CheckState = System::Windows::Forms::CheckState::Unchecked;
	this->resolutionSigmaTextBox->Text = L"0.02";
}