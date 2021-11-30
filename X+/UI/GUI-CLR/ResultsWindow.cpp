#include "ResultsWindow.h"
#include "ExportGraph.h"


namespace GUICLR {
	void ResultsWindow::OpenInitialGraph() {
			RECT area;
			std::vector<std::string> keys;

			area.top = 0;
			area.left = 0;
			area.right = wgtGraph->Size.Width;
			area.bottom = wgtGraph->Size.Height;
			wgtGraph->graph = gcnew Graph(
							area, 
							_graphs[0].color, 
							DRAW_LINES, _graphs[0].x, _graphs[0].y,
							false,
							false);

			keys.push_back(_graphs[0].legendKey);

			for(int i = 1; i < _cnt; i++) {
				keys.push_back(_graphs[i].legendKey);
				wgtGraph->graph->Add(_graphs[i].color, DRAW_LINES,
							 		 _graphs[i].x, _graphs[i].y);
			}

			wgtGraph->graph->Legend(keys);
		}

	void ResultsWindow::logScale_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if(wgtGraph->graph) {
			wgtGraph->graph->SetScale(0, (logScale->Checked) ? 
				SCALE_LOG : SCALE_LIN);
		}
		wgtGraph->Invalidate();
	}

	void ResultsWindow::exportGraph_Click(System::Object^  sender, System::EventArgs^  e) {
		ExportGraph eg(_graphs, _cnt);
		eg.ShowDialog();
	}
};
