#include "UIsettings.h"
#include "AboutDialog.h"

namespace xraytools {
	void AboutDialog::AboutDialog_Load(System::Object^  sender, System::EventArgs^  e) {
		label2->Text += XRVERSION;
	}

};
