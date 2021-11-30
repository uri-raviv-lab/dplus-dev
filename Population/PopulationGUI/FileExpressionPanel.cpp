#include "FileExpressionPanel.h"
#include "clrfunctionality.h"


namespace PopulationGUI {

	FileExpressionPanel::FileExpressionPanel(char des) {
		InitializeComponent();
		init();

		std::string gr = " ";
		gr[0] = des;
		this->designationTextBox->Text = gcnew System::String(gr.c_str());;
	}
	FileExpressionPanel::FileExpressionPanel() {
		InitializeComponent();
		init();
        }

	FileExpressionPanel::FileExpressionPanel(String ^fileName, std::vector<double> xI, std::vector<double> yI, char desI) {
		InitializeComponent();
		init();

		std::string gr = " ";
		gr[0] = desI;
		this->designationTextBox->Text = gcnew System::String(gr.c_str());;

		*xF = xI;
		*yF = yI;
		x   = xF;
		y   = yF;

		filenameTextBox->Text = CLRBasename(fileName)->Remove(0, 1);
		ttip->SetToolTip(filenameTextBox, fileName);
	}

	FileExpressionPanel::FileExpressionPanel(String ^expression, char des) {
		InitializeComponent();
		init();
	}

	FileExpressionPanel::~FileExpressionPanel() {
            if (components) {
                delete components;
            }
			if(xE)
				delete xE;
			if(yE)
				delete yE;
			if(xF)
				delete xF;
			if(yF)
				delete yF;
        }

	void FileExpressionPanel::expressionTextBox_KeyPress(System::Object^  sender, System::Windows::Forms::KeyPressEventArgs^  e) {
#ifdef _DEBUG
		wchar_t deb = e->KeyChar;
#endif
		if((!(
			Char::IsLetterOrDigit(e->KeyChar) ||
			Char::IsWhiteSpace(e->KeyChar) ||
			(e->KeyChar == '/' || e->KeyChar == '*' || e->KeyChar == '+' || e->KeyChar == '-') || 
			(e->KeyChar == '(' || e->KeyChar == ')' || e->KeyChar == '.' || e->KeyChar == '/') ||
			(e->KeyChar == Convert::ToChar(Keys::Back))
			))				&&
			// Exceptions
			// copy and paste
			!(int(e->KeyChar) == 3 || int(e->KeyChar) == 22)
			)
			e->Handled = true;
	}
    void FileExpressionPanel::expressionTextBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
    }

	void FileExpressionPanel::expressionTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e) {
	}
	
	void FileExpressionPanel::init() {
		srand((unsigned)time(NULL));
		this->colorLabel->BackColor =  System::Drawing::Color::FromArgb(
			(rand() % 256),
			(rand() % 256), 
			(rand() % 256));

		this->ttip = (gcnew ToolTip(gcnew System::ComponentModel::Container()));
		this->ttip->SetToolTip(this->filenameTextBox, "this should be the path");

		this->xE = new std::vector<double>();
		this->yE = new std::vector<double>();
		this->xF = new std::vector<double>();
		this->yF = new std::vector<double>();

		this->visCheckBox->Checked = true;
		this->fileRadioButton->Checked = true;

		this->graphIndex = -1;
	}

	// Required in order to show the tool tip after it pops	
	void FileExpressionPanel::ResetTT(System::Object^ sender, System::EventArgs^ e) {
		this->ttip->Active = false;
		this->ttip->Active = true;
	}

	void FileExpressionPanel::expressionTextBox_Leave(System::Object^  sender, System::EventArgs^  e) {
		/* TODO: (Parent?)
			1) Check to make sure that any pasted information was legal
			2) Parse the expression
			3) Calculate expression and set to y
			4) Draw the graph
		*/
	}

	void FileExpressionPanel::colorLabel_MouseClick(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e) {
		/* TODO:
			1) The parent object should changethe color of the corresponding curve
		*/
		int r, g, b;
		r = rand() % 256;
		g = rand() % 256;
		b = rand() % 256;

		// Right click causes manual color selection
		if(e->Button == Windows::Forms::MouseButtons::Right) {
			ColorDialog cd;
			if(cd.ShowDialog() == Windows::Forms::DialogResult::OK)	{
				r = cd.Color.R;
				g = cd.Color.G;
				b = cd.Color.B;
			} else
				return;
		}
		this->colorLabel->BackColor =  System::Drawing::Color::FromArgb(r,g,b);
	}

	void FileExpressionPanel::fileRadioButton_Clicked(System::Object ^sender, System::EventArgs ^e) {
		return;
		// Should be a function in the enclosing class that opens an OFD to get a file
		// Also, the file and path Strings should be set

		// Additional TODO:
		// radioButton_CheckedChanged should draw a different graph
	}

	int FileExpressionPanel::GetColorRef() {
		return System::Drawing::ColorTranslator::ToWin32(colorLabel->BackColor);
	}

	void FileExpressionPanel::toggleFilenameToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if(fileRadioButton->Checked)
			return;
		fileRadioButton->AutoCheck ^= true;	// Toggle
		filenameTextBox->ReadOnly = fileRadioButton->AutoCheck;

		if(!filenameTextBox->ReadOnly) {
			file = filenameTextBox->Text;
			filenameTextBox->Text = L"";
		} else {
			filenameTextBox->Text = file;
		}
	}

}
