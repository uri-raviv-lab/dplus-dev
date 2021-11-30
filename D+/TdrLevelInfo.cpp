#include "TdrLevelInfo.h"


System::Void DPlus::TdrLevelInfo::buttonUseCPU_Click(System::Object^  sender, System::EventArgs^  e)
{
	useCPU = true;
	this->Close();

}

System::Void DPlus::TdrLevelInfo::buttonRetry_Click(System::Object^  sender, System::EventArgs^  e)
{

	retry = true;
	this->Close();

}
System::Void DPlus::TdrLevelInfo::TdrLevelInfo_VisibleChanged(System::Object^  sender, System::EventArgs^  e)
{
	// the initialize should happen jusst when the window is opened
	if (this->Visible)
		retry = false;
}
System::Void DPlus::TdrLevelInfo::linkLabel_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e)
{
	try
	{
		VisitLink();
	}
	catch (Exception^ ex)
	{
		MessageBox::Show("Unable to open link that was clicked.");
	}
}

void  DPlus::TdrLevelInfo::VisitLink()
{
	// Change the color of the link text by setting LinkVisited   
	// to true.  
	this->linkLabel->LinkVisited = true;
	//Call the Process.Start method to open the default browser   
	//with a URL:  
	System::Diagnostics::Process::Start("https://scholars.huji.ac.il/sites/default/files/uriraviv/files/tdrlevelbugfix.pdf");
}