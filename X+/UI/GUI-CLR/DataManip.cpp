#include "clrfunctionality.h"
#include "DataManip.h"
#include <string>
#include <cstring>
#include <fstream>
#include <cmath>

using std::ifstream;

namespace GUICLR {

	void DataManip::doneButton_Click(System::Object ^sender, System::EventArgs ^e) {
		this->Close();
	}

	void DataManip::chooseFilesButton_Click(System::Object ^sender, System::EventArgs ^e) {
		bool angst = angstromCheckbox->Checked, fact = factorCheckbox->Checked, rmv = rmvHeadersCheckbox->Checked;

		openFileDialog1->Title = "Choose data files to manipulate...";
		openFileDialog1->Filter = "Data files (*.dat, *.chi, *.out)|*.dat;*.chi;*.out|All files|*.*";

		if(openFileDialog1->ShowDialog() ==
			System::Windows::Forms::DialogResult::Cancel)
			return;

		double exponent = clrToDouble(factorTextBox->Text);
		double factor = 1.0 * pow(10, exponent);
		FILE *fp;
		
		for(int i = 0; i < openFileDialog1->FileNames->Length; i++) {
			bool threeCols = true, err = true, useOrig = false;
			std::string filename = clrToString(openFileDialog1->FileNames[i]),
				dir = clrToString(CLRDirectory(openFileDialog1->FileNames[i])),
				base = clrToString(CLRBasename(openFileDialog1->FileNames[i])),
				newDir, str;

			ifstream in (filename.c_str());

			if(!in) {
				fprintf(stderr, "Error opening file %s for reading\n",
						filename);
				continue;
			}

			// Create a new subdirectory to save the new files in
			newDir = dir + "\\manip";
			String ^grr = gcnew String(newDir.c_str());
			System::IO::Directory::CreateDirectory(grr);
			clrToString(openFileDialog1->FileNames[i], str);
			
			if ((fp = fopen((newDir.append(base).append(".dat").c_str()), "w")) == NULL) {
				fprintf(stderr, "Error opening file %s for writing\n",
					(newDir.append(base).append(".dat").c_str()));
				continue;
			}

			

			while(!in.eof()) {
				threeCols = false; err = true; useOrig = false;
				std::string line, tmpline, origLine;
				size_t pos1, pos2, end;
				getline(in, line);
				origLine = line;  // In case we don't want to manipulate text
				
				while(line[0]  == ' ' || line[0] == '\t' || line[0] == '\f')
					line.erase(0, 1);

				//Remove initial whitespace
				while(line[0]  == ' ' || line[0] == '\t' || line[0] == '\f')
					line.erase(0, 1);

				//Replaces whitespace with one tab
				for(int cnt = 1; cnt < (int)line.length(); cnt++) {
					if(line[cnt] == ' ' || line[cnt + 1] == ',' || line[cnt] == '\t' || line[cnt] == '\f') {
									while(((int)line.length() > cnt + 1) && (line[cnt + 1] == ' ' || line[cnt + 1] == ',' || line[cnt + 1] == '\t' || line[cnt + 1] == '\f'))
							line.erase(cnt + 1, 1);
						
						line[cnt] = '\t';
					}
				}
					
				pos1 = line.find("\t");
				// Less than 2 words/columns
				if(pos1 == std::string::npos) 
					useOrig = true;

				end = line.find("\t", pos1 + 1);

				if(end == std::string::npos)
					end = line.length() - pos1;
				else
					end = end - pos1;

				if(end == 0) 	// There is no second word
					useOrig = true;

				//Find third word
				pos2 = pos1 + end;
				end = line.find("\t", pos2 + 1);
				if(end == std::string::npos)
					end = line.length() - pos2;
				else
					end = end - pos2;

				if(!useOrig && end > 0) 	// There is a third word
					threeCols = true;

				// Check to make sure the two/three words are doubles
				std::string strC;
				char *ptr;
				strC = line.substr(0, pos1);
				strtod(strC.c_str(), &ptr);
				if(ptr == strC)
					useOrig = true;

				strC = line.substr(pos1 + 1, pos2);
				strtod(strC.c_str(), &ptr);
				if(ptr == strC)
					useOrig = true;

				if(!useOrig && threeCols) {
					strC = line.substr(pos2 + 1, end);
					strtod(strC.c_str(), &ptr);
					if(ptr == strC)
						err = false;
				}

				double x, y, er = 0.0;

				// Write the line to the new file depending on whether or not the data
				// was manipulated.
				if(!useOrig) {
					x = strtod(line.substr(0, pos1).c_str(), NULL);
					y = strtod(line.substr(pos1 + 1, pos2).c_str(), NULL);
					if(threeCols)
						er = strtod(line.substr(pos2 + 1, end).c_str(), NULL);
					if(fact) {
						y *= factor;
						if(threeCols && err)
							er *= factor;
					}
					if(angst)
						x *= 10.0;

					fprintf(fp, "%*f\t%*e", pos1, x, pos2 - pos1 + 1, y);
					if(threeCols) {
						if(err)
							fprintf(fp, "\t%*e", end - pos2 + 1, er);
						else {
							fprintf(fp, "%s", line.substr(pos2, end).c_str());
						}
					}
					fprintf(fp, "\n");
				}
				else if(!rmv)
					fprintf(fp, "%s\n", origLine.c_str());
			}
			
			fclose(fp);
		}

	}

	void DataManip::Checkbox_CheckedChanged(System::Object ^sender, System::EventArgs ^e) {
		if(angstromCheckbox->Checked || factorCheckbox->Checked || rmvHeadersCheckbox->Checked)
			chooseFilesButton->Enabled = true;
		else
			chooseFilesButton->Enabled = false;
	}

	void DataManip::factorTextBox_Leave(System::Object^  sender, System::EventArgs^  e) {
		double res;
		std::string str;
		
		clrToString(((TextBox ^)(sender))->Text, str);

		res = strtod(str.c_str(), NULL);
		((TextBox ^)(sender))->Text = res.ToString();
	}
}

