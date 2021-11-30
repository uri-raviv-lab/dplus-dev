#include "ErrorTableWindow.h"
#include "clrfunctionality.h"
#include "UnicodeChars.h"

using namespace System::Windows::Forms;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/*
#ifndef _UNICODE
#define _UNICODE
#endif
*/
namespace GUICLR {


	void ErrorTableWindow::ErrorTableWindow_Load(System::Object^  sender, System::EventArgs^  e) {
		this->CloseButton->Text = L"&Close";
		this->ExportButton->Text = L"&Export...";

		System::Collections::Generic::List<ListView^>^ listViews = gcnew System::Collections::Generic::List<ListView^>();
		listViews->Add(FFLV);
		listViews->Add(SFLV);
		listViews->Add(PhLV);
		listViews->Add(BGLV);
		
		for (int i = 0; i < listViews->Count; i++) {
			for(int j = 0; j < LVs[i]->Items->Count ; j++) {
				for(int k = 1 + i / 3; k < LVs[i]->Columns->Count; k +=2) {
					// for the BG LV, start from the name and not the number
					if( ((i == 1 && LVs[i]->Columns->Count == 4) || (i == 2)  ) && k > 1)
						// if it's the phases (i==2)
						// if it's caille (i == 1 && LVs[i]->Columns->Count == 4)
						// then we only have 1 column of data
						continue;
					ListViewItem ^lvi = gcnew ListViewItem(LVs[i]->Items[j]->SubItems[0 + i / 3]->Text + " - " + LVs[i]->Columns[k]->Text);
					lvi->SubItems->Add(LVs[i]->Items[j]->SubItems[k]->Text);
					System::String^ str = gcnew String(L"?");
					if( (errorsVectors->at(i).size() <= (unsigned)(j + (k/2) * LVs[i]->Items->Count) ) || (!errorsVectors[i].empty() &&
						(errorsVectors->at(i).at(j + (k/2) * LVs[i]->Items->Count) < 0.0) ))
						//if there is no error or it is set as -1.0 (not mutable)
						str = L"-";
					else
						str = errorsVectors->at(i).at(j + (k/2) * LVs[i]->Items->Count).ToString();
					if(clrToDouble(str) < 0.0)
						str = L"-";
					lvi->SubItems->Add(str);
					listViews[i]->Items->Add(lvi);
				}
			}
		}

		// Extra parameters
		for(int i = 0; i < LVs[4]->Items->Count; i++) {
			ListViewItem ^lvi = gcnew ListViewItem(LVs[4]->Items[i]->SubItems[0]->Text);;
			lvi->SubItems->Add(LVs[4]->Items[i]->SubItems[1]->Text);
			System::String^ str = gcnew String(L"?");
			if(errorsVectors->at(0).size() <= (unsigned)LVs[0]->Items->Count || (!errorsVectors[0].empty() &&
				errorsVectors->at(0).at(listViews[0]->Items->Count) < 0.0))
				str = L"-";
			else
				str = errorsVectors->at(0).at(listViews[0]->Items->Count).ToString();
			lvi->SubItems->Add(str);
			listViews[0]->Items->Add(lvi);
		}

		// Caille
		//if(LVs[1]->Columns->Count == 4) { //it's Caille!! YUPI
		//	listViews[1]->Items->Clear();
		//	for( int i = 0; i < LVs[1]->Items->Count; i++) {
		//		ListViewItem ^lvi;
		//		lvi->SubItems->Add(LVs[1]->Items[i]->SubItems[0]->Text);
		//		lvi->SubItems->Add(LVs[1]->Items[i]->SubItems[1]->Text);
		//		listViews[1]->Items->Add(lvi);
		//	}
		//}
		
		// Phases - remove " - Value"
		for(int i = 0; i < listViews[2]->Items->Count; i++)
			//listViews[2]->Items[i]->SubItems[0]->Text->Remove(1, 8);
			listViews[2]->Items[i]->SubItems[0]->Text = listViews[2]->Items[i]->SubItems[0]->Text->Replace(L" - Value", L"");
		
		//Re-arrange the window
		int longest = 0, shortest = 0, b1 = 0, b2 = 0, right;

		// Resize the listViews
		for(int i = 0; i  < listViews->Count; i++) {
			if(listViews[i]->Items->Count < 1) {
				shortest = i;
				listViews[i]->Visible = false;
				listViews[i]->Width  = 1;
				listViews[i]->Height = 1;
				continue;
			}
			listViews[i]->AutoResizeColumns(System::Windows::Forms::ColumnHeaderAutoResizeStyle::ColumnContent);
			if(i == 2)
				listViews[i]->Columns[0]->AutoResize(System::Windows::Forms::ColumnHeaderAutoResizeStyle::HeaderSize);
			listViews[i]->Columns[listViews[i]->Columns->Count-1]->Width = max(listViews[i]->Columns[listViews[i]->Columns->Count-1]->Width, 37);
			listViews[i]->Width = listViews[i]->Items[listViews[i]->Items->Count - 1]->Bounds.Right + 21;
			listViews[i]->Height = min(300, listViews[i]->Items[listViews[i]->Items->Count - 1]->Bounds.Bottom + 10);
			
			shortest	= (listViews[i]->Height < listViews[shortest]->Height)	? i : shortest;
			longest		= (listViews[i]->Height > listViews[longest]->Height)	? i : longest;
			
			//Highlight every other row
			for(int j = 0; j < listViews[i]->Items->Count; j++) {
				if(j % 2 == 1)
					listViews[i]->Items[j]->BackColor = System::Drawing::Color::Gainsboro;
			}
		}

		for(int i = 0; i < listViews->Count; i++) {
			if(i == longest || i == shortest)
				continue;
			b1 = i;
			break;
		}
		for(int i = b1 + 1; i < listViews->Count; i++) {
			if(i == longest || i == shortest)
				continue;
			b2 = i;
			break;
		}

		listViews[2]->Columns[0]->Width += 6;
		listViews[longest]->Location = System::Drawing::Point(12, 12);
		listViews[shortest]->Location = System::Drawing::Point(12, 30 + listViews[longest]->Height);
		right = max(listViews[longest]->Width, listViews[shortest]->Width) + 30;
		listViews[b1]->Location = System::Drawing::Point(right, 12);
		listViews[b2]->Location = System::Drawing::Point(right, 30 + listViews[b1]->Height);

		int pos = 45 + listViews[b1]->Height + listViews[b2]->Height;
		ExportButton->Location = System::Drawing::Point(right, pos);
		CloseButton->Location  = System::Drawing::Point(right + ExportButton->Width + 30, pos);

		this->Height = 50 + max(pos +CloseButton->Height, listViews[longest]->Height + listViews[shortest]->Height + 20);
		this->Width = right + max(listViews[b1]->Width, listViews[b2]->Width) + 20;

	}

	void ErrorTableWindow::ExportButton_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring file;
		sfd->FileName = CLRBasename(_dataFile)->Remove(0, 1);
		sfd->FileName += "_report";

		sfd->Filter = "TSV Files (*.tsv)|*.tsv|All Files (*.*)|*.*";
		sfd->Title = "Choose a filename";
		if(sfd->ShowDialog() == 
			System::Windows::Forms::DialogResult::Cancel)
			return;
		
		clrToString(sfd->FileName, file);


		// Open a file for writing
		FILE *fp;

		if ((fp = _wfopen(file.c_str(), L"w, ccs=UTF-8")) == NULL) {
			fprintf(stderr, "Error opening file %s for writing\n",
							file);
			
			MessageBox::Show("Please make sure that the file is not open.", "Error opening file for writing", MessageBoxButtons::OK,
									 MessageBoxIcon::Error);
			return;
		}

		// Write each tab with its parameters and titles (+chisqr/Rsqr)
		// Collect titles (header field names)
		System::Collections::Generic::List<ListView^>^ LV = gcnew System::Collections::Generic::List<ListView^>();
		LV->Add(FFLV);
		LV->Add(SFLV);
		LV->Add(PhLV);
		LV->Add(BGLV);

		for(int cnt = 0; cnt < LV->Count; cnt++) {
			if(LV[cnt]->Items->Count > 0) {
				for(int i = 0; i < LV[cnt]->Columns->Count; i++) {
					// fwprintf doesn't help write the \sigma correctly, it just screws up the whole file
					fwprintf(fp, L"%s\t ", clrToWstring(LV[cnt]->Columns[i]->Text).c_str());
				}
				fwprintf(fp, L"\t ");
			}
		}

		fwprintf(fp, L"\n");
		
		// Fill in the data for the tables
		int maxRows = 0;
		for(int j = 0; j < LV->Count; j++)
			maxRows = max(maxRows, LV[j]->Items->Count);

		for(int row = 0; row < maxRows; row++) {
			for(int cnt = 0; cnt < LV->Count; cnt++) {
				if(LV[cnt]->Items->Count > 0) {
					for(int itm = 0; itm < LV[cnt]->Columns->Count; itm++) {
						if(row < LV[cnt]->Items->Count) {
							fwprintf(fp, L"%s", clrToWstring(LV[cnt]->Items[row]->SubItems[itm]->Text).c_str());
						}
						fwprintf(fp, L" \t ");
					}
					if(!(cnt == LV->Count - 1))
						fwprintf(fp, L"\t ");
				}
			}
			fwprintf(fp, L"\n");
		}

		// Close file
		fclose(fp);
	}
}