#include "ExtractBackground.h"

//#include "calculation_external.h"
//#include "clrfunctionality.h"


namespace GUICLR {

	void ExtractBackground::logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph->graph) {
			wgtGraph->graph->SetScale(0, (logScale->Checked) ? 
				SCALE_LOG : SCALE_LIN);
		}
		wgtGraph->Invalidate();
	}

	void ExtractBackground::OpenInitialGraph() {
		std::vector<double> dx, dy, bgx, bgy;
		RECT area;
		
		ReadCLRFile(_dataFile, dx, dy);
		ReadCLRFile(_bgFile, bgx, bgy);

		area.top = 0;
		area.left = 0;
		area.right = wgtGraph->Size.Width + area.left;
		area.bottom = wgtGraph->Size.Height + area.top;
		wgtGraph->graph = gcnew Graph(
						area, 
						RGB(255, 0, 0), 
						DRAW_LINES, dx, dy, 
						false,	//TODO::LogLog make logX checkbox
						logScale->Checked);
		wgtGraph->graph->Add(RGB(0, 0, 255), DRAW_LINES,
						 	 bgx, bgy);
	}

	void ExtractBackground::extractOneFile_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring savename, bgFile, dataFile;
		struct graphLine graphs[2];

		saveFileDialog1->Title = "Choose an output extracted file:";
		saveFileDialog1->ShowDialog();

		bool bFactor = !(MessageBox::Show("Would you like to scale the background to the signal?", "Extracting", 
			MessageBoxButtons::YesNo, MessageBoxIcon::Question) == System::Windows::Forms::DialogResult::No);

		
		clrToString(saveFileDialog1->FileName, savename);
		clrToString(_dataFile, dataFile);
		clrToString(_bgFile, bgFile);

		ImportBackground(bgFile.c_str(), dataFile.c_str(), savename.c_str(), bFactor);

		
		graphs[0].color = RGB(255, 0, 0);
		graphs[1].color = RGB(0, 0, 255);

		graphs[0].legendKey = "Original Data";
		graphs[1].legendKey = "Subtracted Data";

		ReadDataFile(dataFile.c_str(), graphs[0].x, graphs[0].y);
		ReadDataFile(savename.c_str(), graphs[1].x, graphs[1].y);
		
		ResultsWindow res(graphs, 2);

		res.ShowDialog();
	}

	void ExtractBackground::batchButton_Click(System::Object^  sender, System::EventArgs^  e) {
		System::String ^src, ^dest;
		bool multFiles = false;
		if( (System::Windows::Forms::Control::ModifierKeys & Keys::Shift) == Keys::Shift)
			multFiles = true;

		if(!multFiles) {
			// Choose source folder
			folderBrowserDialog1->Description = "Choose the folder to subtract background from:";
			if(folderBrowserDialog1->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
				return;

			src = gcnew System::String(folderBrowserDialog1->SelectedPath);

			if(MessageBox::Show("Are you sure you want to extract ALL files in this directory?", "Extracting", 
				MessageBoxButtons::YesNo, MessageBoxIcon::Question) == System::Windows::Forms::DialogResult::No)
				return;

		} else {
			openFileDialog1->Title = "Choose data files to subtract background...";
			openFileDialog1->Filter = "Data files (*.dat, *.chi, *.out)|*.dat;*.chi;*.out|All files|*.*";
			openFileDialog1->Multiselect = true;

			if(openFileDialog1->ShowDialog() ==
				System::Windows::Forms::DialogResult::Cancel)
				return;
		}

		folderBrowserDialog1->Description = "Choose the folder to save the files to:";
		if(folderBrowserDialog1->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;

		dest = gcnew System::String(folderBrowserDialog1->SelectedPath);

		bool bFactor = !(MessageBox::Show("Would you like to scale the background to the signal?", "Extracting", 
			MessageBoxButtons::YesNo, MessageBoxIcon::Question) == System::Windows::Forms::DialogResult::No);


		// For all files in source: ImportBackground to target
		array<System::String ^> ^srcFiles = (multFiles ? openFileDialog1->FileNames : 
									System::IO::Directory::GetFiles(src));
		
		std::wstring srcName, destName, fname, bgFile;
		clrToString(_bgFile, bgFile);
		double origMinBG = GetMinimumSig();
		SetMinimumSig(0.0);
		for(int i = 0; i < srcFiles->Length; i++) {
			destName.clear();
			srcName.clear();
			fname.clear();

			IO::FileInfo a(srcFiles[i]);
			
			clrToString(srcFiles[i], srcName);
			clrToString(dest, destName);
			clrToString(a.Name, fname);

			destName += L"\\" + fname;
			
			ImportBackground(bgFile.c_str(), srcName.c_str(), destName.c_str(), bFactor);
		}
		SetMinimumSig(origMinBG);
	}

	void ExtractBackground::chooseFile(int num) {
		std::wstring name;
		openFileDialog1->Multiselect = false;
		if(!openDataFile(openFileDialog1, "Choose a " + ((num == 1) ? "background" : "data") + " file to manipulate", name, false))
			return;
		vector<double> x, y;
		ReadDataFile(name.c_str(), x, y);

		if(num == 1) {
			delete _bgFile;
			_bgFile = gcnew System::String(name.c_str());
		}
		else {
			delete _dataFile;
			_dataFile = gcnew System::String(name.c_str());
		}

		wgtGraph->graph->Modify(num, x, y);
		wgtGraph->Invalidate();
	}

	void ExtractBackground::backgroundButton_Click(System::Object^  sender, System::EventArgs^  e) {
		chooseFile(1);
	}

	void ExtractBackground::dataButton_Click(System::Object^  sender, System::EventArgs^  e) {
		chooseFile(0);
	}
}
