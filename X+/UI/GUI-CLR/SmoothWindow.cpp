#include "SmoothWindow.h"
#include "clrfunctionality.h"

namespace GUICLR {

	void SmoothWindow::OpenInitialGraph() {
		std::vector<double> dx, dy;
		RECT area;
		
		ReadCLRFile(_dataFile, dx, dy);

		area.top = 0;
		area.left = 0;
		area.right = wgtGraph->Size.Width + area.left;
		area.bottom = wgtGraph->Size.Height + area.top;
		wgtGraph->graph = gcnew Graph(
						area, 
						RGB(255, 0, 0), 
						DRAW_LINES, dx, dy, 
						logscaleX->Checked,	logScale->Checked);


		origY->y = dy;
	}

	void SmoothWindow::logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph->graph) {
			wgtGraph->graph->SetScale(0, (logScale->Checked) ? 
				SCALE_LOG : SCALE_LIN);
		}
		wgtGraph->Invalidate();
	}

	void SmoothWindow::logscaleX_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph->graph) {
			wgtGraph->graph->SetScale(1, (logscaleX->Checked) ? 
				SCALE_LOG : SCALE_LIN);
		}
		wgtGraph->Invalidate();
	}

	void SmoothWindow::saveAs_Click(System::Object^  sender, System::EventArgs^  e) {
		// Save As...
		std::wstring savefile;

		if(_bOverwrite)
			savefile = clrToWstring(_dataFile);
		else {
			if(saveFileDialog1->ShowDialog() != Windows::Forms::DialogResult::OK)
				return;

			clrToString(saveFileDialog1->FileName, savefile);
		}

		WriteDataFile(savefile.c_str(), wgtGraph->graph->x[0], 
					  wgtGraph->graph->y[0]);
		this->Close();
	}

	void SmoothWindow::trackBar_Scroll(System::Object^  sender, System::EventArgs^  e) {
			vector<double> cury = origY->y,
				           x = wgtGraph->graph->x[0];

			int pos = trackBar1->Value;
			int pos2 = trackBar2->Value;
			
			if(pos > 0) {
				double rPos = double(pos) / double(trackBar1->Maximum);
				double rPos2 = double(pos2) / double(trackBar2->Maximum);
				// rPos is 0.0->0.99
				if(gaussianBlurToolStripMenuItem->Checked)
					smoothVector(int(rPos * 10.0), cury);
				else
					cury = bilateralFilter(cury, x, (rPos * 10.0), (exp(rPos2 * 10.0) - 1.0));
			}

			wgtGraph->graph->Modify(0, x, cury);
			wgtGraph->Invalidate();
	}

	void SmoothWindow::optionToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if(sender == gaussianBlurToolStripMenuItem)
			if(!(gaussianBlurToolStripMenuItem->Checked)) {
				bilateralFilterToolStripMenuItem->Checked = false;
				gaussianBlurToolStripMenuItem->Checked = true;
				tableLayoutPanel2->RowStyles[0]->Height = 0.0;
				tableLayoutPanel1->RowStyles[2]->Height /= 2;
				trackBar_Scroll(sender, e);
			}
		if(sender == bilateralFilterToolStripMenuItem)
			if(!(bilateralFilterToolStripMenuItem->Checked)) {
				bilateralFilterToolStripMenuItem->Checked = true;
				gaussianBlurToolStripMenuItem->Checked = false;
				tableLayoutPanel2->RowStyles[0]->Height = tableLayoutPanel2->RowStyles[1]->Height;
				tableLayoutPanel1->RowStyles[2]->Height *= 2;
				trackBar_Scroll(sender, e);
			}
	}

};
