#include "ServerConfigForm.h"

namespace DPlus
{

	System::Void ServerConfigForm::ok_button_Click(System::Object^  sender, System::EventArgs^  e)
	{
		if (!this->serverAddressTextbox->Text->EndsWith("/"))
			this->serverAddressTextbox->Text += "/";

		String ^ address = this->serverAddressTextbox->Text;
		String ^ code = this->codeTextbox->Text;



		try
		{
			bool valid_input = checkServerAddress(address) && checkCode(code);

			if (valid_input)
			{
				this->serverAddress = address;
				this->validationCode = code;
				this->Close();
			}
		}

			catch (Exception ^ e)
			{
				setErrorMessage(e->Message);
			}

	}

	System::Void ServerConfigForm::serverAddressTextbox_TextChanged(System::Object^  sender, System::EventArgs^  e)
	{
		if (!this->serverAddressTextbox->Text->Contains("http://"))
			setErrorMessage("Warning: Server address should begin with http://");
		else
			setErrorMessage("");
	}

	System::Void ServerConfigForm::EnableEditCheck_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if (this->EnableEditCheck->Checked)
		{
			EnableEdit();
		}

	}

	void ServerConfigForm::EnableEdit()
	{
		this->serverAddressTextbox->Enabled = true;
		this->codeTextbox->Enabled = true;
		this->EnableEditCheck->Enabled = false;
	}

	bool ServerConfigForm::checkServerAddress(String ^ address)
	{
		if (address == "")
		{
			setErrorMessage("Please enter a server address");
			return false;
		}
		
		return true;
	}

	bool ServerConfigForm::checkCode(String ^ code)
	{
		if (code == "")
		{
			setErrorMessage("You must enter an activation code");
			return false;
		}
		return true;
	}


	System::Void ServerConfigForm::setErrorMessage(String ^ msg)
	{
		this->errorMessage->Visible = true;
		this->errorMessage->Text = msg;
	}

}