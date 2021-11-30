#include "DockingWindow.h"
#include "clrFunc.h"
//#include "pdbData.h"
#include <ctime>
#include "Windows.h"

namespace DockingTest {
	void WriteDataFileWHeader(const wchar_t *filename, vector<FACC>& x,
		vector<FACC>& y, std::stringstream& header)
	{
		FILE *fp;

		if ((fp = _wfopen(filename, L"w")) == NULL) {
			fprintf(stderr, "Error opening file %s for writing\n",
				filename);
			exit(1);
		}

		fprintf(fp, "%s\n", header.str().c_str());

		int size = (x.size() < y.size()) ? x.size() : y.size();
		for(int i = 0; i < size; i++)
			fprintf(fp,"%.8g\t%.8g\n", x.at(i), y.at(i));

		fclose(fp);
	}
	
	void DockingWindow::L_trackBar_Scroll(System::Object^ sender, System::EventArgs^ e) {
		if(sender == L_trackBar) {
			L_label->Text = L_label->Text->Substring(0, L_label->Text->LastIndexOf(" ") + 1) + L_trackBar->Value.ToString();
		}
		if(sender == iterationsTrackBar) {
			iterationsLabel->Text = iterationsLabel->Text->Substring(0, iterationsLabel->Text->LastIndexOf("^") + 1) + iterationsTrackBar->Value.ToString();
		}
	}


	void DockingWindow::loadPDB_Button_Click(System::Object^ sender, System::EventArgs^ e) {
		if(((Button^)(sender)) == loadPDB_Button) {
			PDB_openFileDialog->Title = "Choose a PDB file";
			PDB_openFileDialog->Filter = "PDB File (*.pdb)|*.pdb|All files|*.*";
			if(PDB_openFileDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
				return;
			if(pdb) {
				delete pdb;
				pdb = nullptr;
			}
			pdb = new PDBModel(clrToString(PDB_openFileDialog->FileName));
			bLoadedGrid = false;
		} else {
			PDB_openFileDialog->Title = "Choose a grid file";
			PDB_openFileDialog->Filter = "AMP File (*.amp)|*.amp|All files|*.*";
			if(PDB_openFileDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
				return;
			if(pdb) {
				delete pdb;
				pdb = nullptr;
			}
			pdb = new PDBModel();
			pdb->ReadAmplitudeFromFile(clrToString(PDB_openFileDialog->FileName));
			bLoadedGrid = true;
		}
	}

	void DockingWindow::DummyCons() {
		bLoadedGrid = false;
		Q = new std::vector<FACC>;
		res = new std::vector<FACC>;

		progressrep = gcnew CLRProgressFunc(this, &DockingWindow::ProgressReport);
		notifycomp  = gcnew CLRNotifyCompletionFunc(this, &DockingWindow::NotifyCompletion);

	}

	System::Void DockingWindow::calculateButton_Click(System::Object^ sender, System::EventArgs^ e) {
		int sec = L_trackBar->Value;
		unsigned long long iters = unsigned long long(pow(10.0, double(iterationsTrackBar->Value)));
		clock_t begin, end, c_time;
		String^ timeStr = L"";
		FACC qmax = FACC(Double::Parse(qMaxTextBox->Text)), qSmin = 0.001;
		int size = 1000;
		if(bLoadedGrid) {
			qmax = pdb->GetGridStepSize() * double(pdb->GetGridSize()) / 2.0;
			sec = pdb->GetGridSize();
		}
		Q->resize(size);
		res->resize(size);
		for(int i = 0; i < size; i++)
			Q->at(i) = FACC((qmax - qSmin) / FACC(size - 1) * FACC(i));
#ifdef NO_GRID
		sec = 2;	// TODO::DEBUG
		pdb->calculateGrid(qmax, sec);
#else
		if(!bLoadedGrid) {
			begin = clock();
			pdb->calculateGrid(qmax, sec);
			end = clock();
			Beep(435,250);
			c_time = end - begin;
			if(c_time > 10000)
				timeStr = Int32(c_time / CLOCKS_PER_SEC) + L" seconds";
			else
				timeStr = Int32(c_time * double(1000.0 / double(CLOCKS_PER_SEC))) + L" ms";
			matrixLabel->Text = matrixLabel->Text->Substring(0, matrixLabel->Text->LastIndexOf(":") + 2)
				+ timeStr;
#ifdef WRITE_READ_FILE
			std::stringstream ss;
			ss << "# PDB file: " << clrToString(PDB_openFileDialog->FileName) << "\n";
			ss << "# Program revision: "<< SVN_REV_STR << "\n";
			ss << "# N^3; N = " << pdb->GetGridSize() << "\n";

			pdb->WriteAmplitudeToFile(clrToString(PDB_openFileDialog->FileName->Replace(".pdb","_" + Int32(sec).ToString() + ".amp")), ss);
#endif	// WRITE_READ_FILE
		}
#endif	// NO_GRID

		begin = clock();
		//pdb->calculateIVec(*Q, *res, 1.0e-4, iters);	// No longer valid
		IntensityCalculator ic;
		std::vector<AmplitudeModel*> aps;
		aps.push_back((AmplitudeModel*)(pdb));
		ic.SetAmplitudes(aps);
		ic.CalculateIntensityVector(*Q, *res, 1.0e-4, iters);	
		end = clock();
		Beep(435,500);
		c_time = end - begin;
		if(c_time > 10000)
			timeStr = Int32(c_time / CLOCKS_PER_SEC) + L" seconds";
		else
			timeStr = Int32(c_time * double(1000.0 / double(CLOCKS_PER_SEC))) + L" ms";
 		orientationLabel->Text = orientationLabel->Text->Substring(0, orientationLabel->Text->LastIndexOf(":") + 2)
 								+ timeStr;

	}

	System::Void DockingWindow::save_Button_Click(System::Object^ sender, System::EventArgs^ e) {
		std::wstring file;
		sfd->FileName = PDB_openFileDialog->FileName->Substring(0, PDB_openFileDialog->FileName->LastIndexOf("."));
		sfd->Filter = "OUT Files (*.out)|*.out|All Files (*.*)|*.*";
		sfd->Title = "Choose a filename";
		if(sfd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
			return;
		std::stringstream blank;
		blank << "# PDB file: " << clrToString(PDB_openFileDialog->FileName) << "\n";
		blank << "# Program revision: "<< SVN_REV_STR << "\n";
#ifndef NO_GRID
		blank << "# Matrix calculation time: " << clrToString(matrixLabel->Text->Substring(matrixLabel->Text->LastIndexOf(":")+2)) << "\n" ;
#endif
		blank << "# Orientation calculation time: " << clrToString(orientationLabel->Text->Substring(orientationLabel->Text->LastIndexOf(":")+2)) << "\n" ;
#ifdef NO_GRID
	#ifdef GAUSS_LEGENDRE_INTEGRATION
		blank << "# 10^" << iterationsTrackBar->Value << " steps of Gauss-Legendre integration (no pre-calculated grid)\n";
	#else
		blank << "# 10^" << iterationsTrackBar->Value << " Iterations of MC (no pre-calculated grid)\n";
	#endif
#else
		blank << "# N^3; N = " << pdb->GetGridSize() << "\n";
#endif
		blank << "# Notes: With 3D linear interpolation.\n";
#ifdef CF4_CHEAT
		blank << "# Notes: Calculated using an analytical expression of CF4.\n";
#endif
#ifdef CF4_QVEC
		blank << "# Notes: Calculated using an analytical expression of CF4 where \\vec{Q} is parallel to a C-F bond.\n";
#endif
		WriteDataFileWHeader(clrToWstring(sfd->FileName).c_str(), *Q, *res, blank);

	}

	void DockingWindow::qMaxTextBox_KeyPress(System::Object^ sender, System::Windows::Forms::KeyPressEventArgs^ e) {
#ifdef _DEBUG
		wchar_t deb = e->KeyChar;
#endif

		if((!(
			Char::IsDigit(e->KeyChar) ||
			(e->KeyChar == '-') || 
			(e->KeyChar == '.') ||
			(e->KeyChar == Convert::ToChar(Keys::Back) || e->KeyChar == Convert::ToChar(Keys::Delete))
			))				&&
			// Exceptions
			// copy and paste
			!(int(e->KeyChar) == 3 || int(e->KeyChar) == 22)
			)
			e->Handled = true;
	}

	void DockingWindow::ChangeEnabled(bool en) {
		loadPDB_Button->Enabled		= en;
		L_trackBar->Enabled			= en;
		qMaxTextBox->Enabled		= en;
		iterationsTrackBar->Enabled	= en;
		loadButton->Enabled			= en;
		calculateButton->Enabled	= en;
		save_Button->Enabled		= en;

		gridProgressBar->Visible = !en;
		iterationsProgressBar->Visible = !en;
	}

	void DockingWindow::ProgressReport(void *args, double progress) {
		// Invoke from inside UI thread, if required
		if(this->InvokeRequired) {
			array<Object^> ^fparams = { IntPtr(args), progress };
			this->Invoke(progressrep, fparams);
			return;
		}

		// Update progress bar value
		//progressBar1->Value = int(progress * progressBar1->Maximum);
	}

	void DockingWindow::NotifyCompletion(void *args, int error) {
		// Invoke from inside UI thread, if required
		if(this->InvokeRequired) {
			array<Object^> ^fparams = { IntPtr(args), error };
			this->Invoke(notifycomp, fparams);
			return;
		}

		if(error != PDB_OK)
			MessageBox::Show("Error! Error code: " + error, "ERROR",
								MessageBoxButtons::OK, MessageBoxIcon::Error);

		ChangeEnabled(true);

	}


}
