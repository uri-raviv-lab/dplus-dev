#include "ExportGraph.h"

#include <vector>

namespace GUICLR {

	void ExportGraph::OpenInitialGraph() {
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
							false, false);

			keys.push_back(_graphs[0].legendKey);

			for(int i = 1; i < _cnt; i++) {
				keys.push_back(_graphs[i].legendKey);
				wgtGraph->graph->Add(_graphs[i].color, DRAW_LINES,
							 		 _graphs[i].x, _graphs[i].y);
			}

			wgtGraph->graph->Legend(keys);	
	}

}
