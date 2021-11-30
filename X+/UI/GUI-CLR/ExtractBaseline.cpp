
#include "clrfunctionality.h"

#include "ExtractBaseline.h"
//#include "calculation_external.h"
//#include "FrontendExported.h"

#include "genbackground.h"	// includes "FrontendExported.h"

using std::map;

#ifndef max
#define max(a,b)	(((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)	(((a) < (b)) ? (a) : (b))
#endif

/**
* Returns the slope and intercept (line equation) from given two points
* (x1,y1) and (x2,y2).
*/
inline void LineFunction(double x1, double y1, double x2, double y2, 
						 double *slope, double *intercept) {
	 double x11 = x1, x22 = x2, y11 = y1, y22 = y2;

	 *slope = ((y22 - y11) / (x22 - x11));
	 *intercept = y11 - ((*slope) * x11);
}


namespace GUICLR {
	double ExtractBaseline::InterpolatePoint(double x0, const std::vector<double>& x, const std::vector<double>& y) {
		for(int i = 0; i < (int)x.size(); i++) {
			if(x0 <= x[i] && i != 0)
				return y[i - 1] + (x0 - x[i - 1]) * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]));
		}
		return 0.0;
	}

	void ExtractBaseline::mapToVectors( std::map<double, double>& m, 
										std::vector<double>& sbx, 
										std::vector<double>& sby) {
		sbx.clear();
		sby.clear();
		// Rewriting the vectors
		for(std::map<double, double>::const_iterator iter = m.begin(); 
			iter != m.end(); iter++) {
				sbx.push_back((*iter).first);
				sby.push_back((*iter).second);
		}
	}

	void ExtractBaseline::VectorsToLogscale(const std::vector<double>& datax,
						   const std::vector<double>& datay,
						   std::vector<double>& sx,
						   std::vector<double>& sy) {
		std::vector<int> a;

		if(sx.size() < 2)
			return;

		int ctr = 0;
		for (int i = 1; i < (int)datax.size(); i++){
			if ((datax.at(i) - sx.at(ctr)) > 0.0){
				a.push_back(i - ((fabs(datax.at(i) - sx.at(ctr)) > fabs(datax.at(i - 1) - sx.at(ctr)) ) ? 1 : 0)); 
				ctr++;
				if(ctr == (int)sx.size())
					break;
			}
		}

		vector<double> bx (a.back(), 0.0), by (a.back(), 0.0);
		double slope, intercept;

		double sya = InterpolatePoint(sx.at(0), datax, datay), 
			   syb = InterpolatePoint(sx.at(1), datax, datay);
		for (int j = 0; j < (int)sx.size() - 1; j++) {
			slope = ( log10(syb) - log10(sya) ) / ( log10(sx[j+1]) - log10(sx[j]) );

			intercept = log10(sya) - slope*log10(sx[j]);

			if(j + 1 < (int)a.size())
				for (int i = a[j]; i < a[j+1]; i++) {
					bx[i] = datax[i];
					by[i] = pow(10.0, intercept) * pow(datax[i],slope);
				}

			sya = syb;
			if(j + 2 < (int)sx.size())
				syb = InterpolatePoint(sx.at(j + 2), datax, datay);
		}

		double lastx = sx.back(), lasty = sy.back();
		sx.clear();
		sy.clear();
		for (int i = a[0]; i < a[a.size() - 1]; i++) {
			sx.push_back(bx[i]);
			sy.push_back(by[i]);
		}
	
		sx.push_back(lastx);
		sy.push_back(lasty);

	}

	void ExtractBaseline::__findIntersections() {
		bool bIntersection = false;
		int interPt = -1, endPt = -1;
		
		std::vector<double> &signaly = wgtGraph->graph->y[0];
		std::vector<double> &baseliney = wgtGraph->graph->y[1];
		std::vector<double> &signalx = wgtGraph->graph->x[0];
		std::vector<double> &baselinex = wgtGraph->graph->x[1];

		if(baselinex.size() > 1) {
			int i, left;

			for(left = 0; signalx[left] < baselinex[0]; left++);

			// Search for the first intersection from the left
			for(i = 1; (i < (int)baseliney.size() - 1) && signaly[i + left] >= baseliney[i]; i++);

			if(i < (int)baseliney.size() - 1) {
				bIntersection = true;
				interPt = i + left;
				// Search for the first intersection from the right
				for(i = baseliney.size() - 1; i > -1 && signaly[i + left] >= baseliney[i]; i--);
				endPt = i + left;

			}
		}

		label1->Visible = bIntersection;

		wgtGraph->graph->HighlightSingle(0, interPt, endPt, RGB(0, 0, 255));
	}

	void ExtractBaseline::OpenInitialGraph() {
		std::vector<double> dx, dy, sbx, sby;
		RECT area;
		
		ReadCLRFile(_dataFile, dx, dy);
		if(_angstrom)
			for(int i = 0; i < (int)dx.size(); i++)
				dx[i] *= 10.0;

		while(!dx.empty() && dx[0] < 1.0e-9) {
			dx.erase(dx.begin());
			dy.erase(dy.begin());
		}

		_curleft=0;
		_curright=dx.size();

		area.top = 0;
		area.left = 0;
		area.right = wgtGraph->Size.Width + area.left;
		area.bottom = wgtGraph->Size.Height + area.top;
		wgtGraph->graph = gcnew Graph(
						area, 
						RGB(145, 100, 80), 
						DRAW_LINES, dx, dy, 
						false,	//TODO::LogLog make logX checkbox
						logScale->Checked);
					
		if(bUsingOld) {
			ReadCLRFile(_targetFile, sbx);
		
			_map->m.clear();

			for(int i = 0; i < int(sbx.size()); i++)
				_map->m[sbx[i]] = InterpolatePoint(sbx[i], dx, dy);

		}

		mapToVectors(_map->m, sbx, sby);
		VectorsToLogscale(dx, dy, sbx, sby);

		wgtGraph->graph->Add(RGB(130, 205, 50), DRAW_LINES, sbx, sby);

		__findIntersections();
	}

	void ExtractBaseline::saveAs_Click(System::Object^  sender, System::EventArgs^  e) {
		std::wstring dataFile, wPath, target;
		clrToString(_dataFile, dataFile);
		clrToString(_targetFile, target);
		clrToString(_workspace, wPath);

		if(!_bAuto) {
			std::vector<double> sbx, sby;
			mapToVectors(_map->m, sbx, sby);
			wgtGraph->graph->Modify(1, sbx, sby);
		}

		_bHasBaseline = (wgtGraph->graph->x[1].size() >1);
		if(_bHasBaseline && !_bAuto)
			Write1DDataFile(target.c_str(), wgtGraph->graph->x[1]);
		
		this->DialogResult = Windows::Forms::DialogResult::OK;	
		this->Close();
	}

	void ExtractBaseline::removeAll_Click(System::Object^  sender, System::EventArgs^  e) {
		std::vector<double> sb;
		_map->m.clear();
		if(_bAuto)
			AutoBaseline_Click(sender, e);
		wgtGraph->graph->Modify(1, sb, sb);
		label1->Visible = false;
		wgtGraph->graph->HighlightSingle(0, 0, 0, RGB(0, 0, 255));
		wgtGraph->Invalidate();
	}

	/**
	 * takes data from coordinate and returns its position in the data vector
	 **/
	int	ExtractBaseline::PosToDataIndex(double x, std::vector <double> datax)  {
		for(unsigned int i = 1; i < datax.size(); i++) {
				if(datax.at(i) >= x) {
					double mean = (datax.at(i) + datax.at(i - 1)) / 2.0 ;
					if(x <= mean)
						return i - 1;
					else
						return i;
				}
		}
		
		return 0;
	}


	void ExtractBaseline::wgtGraph_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
		
		if(e->Button == System::Windows::Forms::MouseButtons::Left) {
			if(_bCrop || _bAuto) return;
			if(!wgtGraph || !wgtGraph->graph)
				return;
			//the point to which we are pointing at in the q & I coordinates.
			std::pair <double, double> point = wgtGraph->graph->PointToData(e->X-wgtGraph->graph->xoff,e->Y-wgtGraph->graph->yoff);
			int index = PosToDataIndex(point.first,wgtGraph->graph->x[0]);

			
			//maps the q into I in our current baseline
			if(_map->m.find(wgtGraph->graph->x[0].at(index)) == _map->m.end())
				_map->m[wgtGraph->graph->x[0].at(index)] = wgtGraph->graph->y[0].at(index);
			else
				_map->m.erase(_map->m.find(wgtGraph->graph->x[0].at(index)));
					
			//sbx and sby are vectors that contains the baseline
			std::vector<double> sbx, sby;
			mapToVectors(_map->m, sbx, sby);

			VectorsToLogscale(wgtGraph->graph->x[0], wgtGraph->graph->y[0], sbx, sby);

			wgtGraph->graph->Modify(1, sbx, sby);

			//find intersections and colors the problems in blueish
			__findIntersections();

			wgtGraph->Invalidate();
		}
	}
	
	
	void ExtractBaseline::Crop_Click(System::Object^  sender, System::EventArgs^  e) {				 
		wgtGraph->graph->Deselect();
		if(!_bCrop) {
			cancelcrop->Enabled = true;
			AutoBaseline->Enabled = false;
			_bCrop=true;
			_oldleft = -1; _oldright = -1;
			if(!_bAuto)
				removeAll_Click(sender, e);
			removeAll->Enabled = false;
			saveAs->Enabled = false;
			Crop->Enabled = false;
		} else { 
			_bCrop=false;
			saveAs->Enabled = true;
			cancelcrop->Enabled = false;
			AutoBaseline->Enabled = true;

			//for iterative cropping
			_curleft+= _oldleft;
			_curright= _oldright;
			std::vector <double> newx=wgtGraph->graph->x[0], newy=wgtGraph->graph->y[0];
			
			newx.erase(newx.begin()+_oldright, newx.end());
			newy.erase(newy.begin()+_oldright, newy.end());
			newx.erase(newx.begin(), newx.begin() + _oldleft);
			newy.erase(newy.begin(), newy.begin() + _oldleft);
			
			// Crop the AutoBaseline
			if(_bAuto) {
				wgtGraph->graph->x[1].erase(wgtGraph->graph->x[1].begin()+_oldright, wgtGraph->graph->x[1].end());
				wgtGraph->graph->y[1].erase(wgtGraph->graph->y[1].begin()+_oldright, wgtGraph->graph->y[1].end());
				wgtGraph->graph->x[1].erase(wgtGraph->graph->x[1].begin(), wgtGraph->graph->x[1].begin() + _oldleft);
				wgtGraph->graph->y[1].erase(wgtGraph->graph->y[1].begin(), wgtGraph->graph->y[1].begin() + _oldleft);
			}
			wgtGraph->graph->Modify(0,newx,newy);
			_oldleft = -1; _oldright = -1;
			removeAll->Enabled = true;

			wgtGraph->graph->FitToAllGraphs();
		}
		wgtGraph->Invalidate();

	}

	void ExtractBaseline::wgtGraph_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e){
		if(!_bCrop) return; 
			
		if(e->Button==System::Windows::Forms::MouseButtons::Left) {
			if (_firstCropX == -1) return;
			//the point to which we are pointing at in the q & I coordinates.
			std::pair <double, double> point = wgtGraph->graph->PointToData(e->X-wgtGraph->graph->xoff,e->Y-wgtGraph->graph->yoff);
			int secondCropX = PosToDataIndex(point.first,wgtGraph->graph->x[0]);
			// we find the cropping margins and zone
			int left = min(_firstCropX, secondCropX), right = max(_firstCropX, secondCropX);
			_firstCropX = -1;   
			wgtGraph->graph->SetCropping(false, false);
			_Xdown =0 ;
			if (right - left == 0)
				return;
			if (_oldleft != -1 && _oldright != -1) {
				if (!((System::Windows::Forms::Control::ModifierKeys & Keys::Shift) == Keys::Shift)) {
					 right = max(_oldright,right);
					 left  = min(_oldleft,left);
				}
				else {
					if ( _oldleft < left && _oldright > right) {
						right = _oldright;
						left = _oldleft;
					}
					else if (_oldleft > left && _oldright > right) {
						left = right;
						right = _oldright;
					}
					else if (_oldleft < left && _oldright < right) {
						right = left;
						left = _oldleft;
					}
					else { 
						wgtGraph->graph->Deselect();
						_oldleft = _oldright = -1;
						Crop->Enabled = false;
						wgtGraph->Invalidate();
						return;
						
					} 
				}
			}
		
			wgtGraph->graph->RemoveMask();
			wgtGraph->graph->Mask(0,left,right,RGB(100,150,175));

			Crop->Enabled = true;

			wgtGraph->Invalidate();
				
						//paint in black the areas we don't want

			
			_oldleft = left; _oldright = right;
		}
	}

	void ExtractBaseline::wgtGraph_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e){
		if(!_bCrop) return;
		if(!wgtGraph || !wgtGraph->graph)
				 return;
		if(e->Button==System::Windows::Forms::MouseButtons::Left) {
			// The point to which we are pointing at in the q & I coordinates.
			std::pair <double, double> point = wgtGraph->graph->PointToData(e->X-wgtGraph->graph->xoff,e->Y-wgtGraph->graph->yoff);
			_firstCropX = PosToDataIndex(point.first,wgtGraph->graph->x[0]);
			_Xdown = e->X;
			


		}
	}
	
	void ExtractBaseline::wgtGraph_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e){
		if (_bCrop && _Xdown > 0 ) {
			wgtGraph->graph->SetCropping(true, false);
		}
		else {
			wgtGraph->graph->SetCropping(false, false); 
			return;
		}
		wgtGraph->graph->DrawPeakOutline(_Xdown, 2, e->X, 3);
		
		std::pair<double, double> loc;
		loc = wgtGraph->graph->PointToData(e->X - wgtGraph->graph->xoff, e->Y - wgtGraph->graph->yoff);
		std::vector<double> x = wgtGraph->graph->x[0];
		std::vector<double> y = wgtGraph->graph->y[0];

		if(y.empty() || x.empty()) return;

		int pos = x.size() - 1;
		for (unsigned int i = 1; i < x.size(); i++)
			if (x[i] > loc.first) { pos = i - 1; break; }

		//LocOnGraph->Text = "("+ Double(x[pos]).ToString() + ","+ Double(y[pos]).ToString()+ ")";
		wgtGraph->Invalidate();
		

	}
	void ExtractBaseline::AutoBaseline_Click(System::Object^  sender, System::EventArgs^  e) {
		if(!_bAuto) {
			std::vector <double> x, bly, signaly;

			x		= wgtGraph->graph->x[0];
			signaly	= wgtGraph->graph->y[0];

			AutoBaselineGen(x, signaly, bly);

			wgtGraph->graph->HighlightSingle(0, 0, 0, RGB(0, 0, 255));

			wgtGraph->graph->Modify(1, x, bly);
			wgtGraph->Invalidate();
			_bAuto = true;
			AutoBaseline->Text = "Manual Baseline";
			label1->Visible = false;

		}
		else {
			//find intersections and colors the problems in blueish
			__findIntersections();
			
			//sbx and sby are vectors that contains the baseline
			std::vector<double> sbx, sby;
			mapToVectors(_map->m, sbx, sby);

			wgtGraph->graph->Modify(1, sbx, sby);
			wgtGraph->Invalidate();

			_bAuto = false;
			AutoBaseline->Text = "Automatic Baseline";
		}
		
	}
	void ExtractBaseline::getAutoBaseline(std::vector<double> &bl) {
		bl = wgtGraph->graph->y[1];
	}

	void ExtractBaseline::getCroppedSignal(std::vector<double> &x, std::vector<double> &y) {
		x = wgtGraph->graph->x[0];
		y = wgtGraph->graph->y[0];
	}

};