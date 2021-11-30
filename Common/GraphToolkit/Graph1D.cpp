#include "Graph1D.h"
#include "clrfunctionality.h"

#include <limits>

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::Drawing::Drawing2D;

using std::vector;

namespace GraphToolkit {

template <typename T>
inline bool isNotReal(T num) {
	return ((num != num) || (num == std::numeric_limits<T>::infinity()) ||
		(num == -std::numeric_limits<T>::infinity()));
}

// Positive/Negative infinity definitions
#define POSINF (std::numeric_limits<double>::infinity())
#define NEGINF (-std::numeric_limits<double>::infinity())

/**
 * Graph Drawing Toolkit for Windows (WGDT) - Tal Ben-Nun
 */
void Graph1D::InitializeGraph() {

	double ymin, ymax;
	
	graphColors = new vector<COLORREF>();
	drawtypes = new vector<GraphDrawTypeC>();
	legendKeys = new vector<std::string>();

	highlightIndexes = new vector<std::pair<int, int>>();

	highlightMaskIndexes = new vector<std::pair<int, int>>();

	//highlightStart = highlightEnd = -1;
	
	zoomX = zoomY = zoomendx = -1;
	
	bRedraw = true;
	bRedrawBackground = true;
	bRedrawZoom = false;
	bRedrawPeak = false;
	bRedrawCrop = false;
	bPeaks = false;
	bCrop = false;

	bGrid = true;
	bXTicks = bYTicks = true;
	tickres = 5;


	xoff = this->Location.X;
	yoff = this->Location.Y;
	height = this->Height;
	width = this->Width;

	title = "";
	xLabel = "";
	yLabel = "";

	SetGraphDimensions();

	defscaleX = defoffX = defscaleY = defoffY = ymax = ymin = 0;

	curoffX = defoffX; curoffY = defoffY;
	curscaleX = defscaleX; curscaleY = defscaleY; 

	numGraphs = 0;

	GeneratePixelGraphs(-1);


	vis = new std::vector<bool>;
	vis->push_back(true);
}

// Generates the cx/cy vectors, the graph points in pixels 
// (id = -1 means all graphs are to be regenerated)
void Graph1D::GeneratePixelGraphs(int id) {
	for(int k = 0; k < numGraphs; k++) {
		if(id != -1 && k != id)
			continue;

		vector<double> ccx, ccy;
		for(unsigned int i = 0; i < x[k].size(); i++) {
			double xval = x[k][i], yval = y[k][i];

			if(bLogScaleX)
				xval = log10(xval);
			if(bLogScaleY)
				yval = log10(yval);

			if(isNotReal(yval) || isNotReal(xval))
				continue;

			ccx.push_back((xval - curoffX) * 
				 		  curscaleX);
			ccy.push_back(graphHeight - ((yval - curoffY) * 
					      curscaleY));
		}
		cx[k] = ccx;
		cy[k] = ccy;
	}
}

bool isOutOfGraph(double x, double y, int w, int h) {
	return !( x>=0 && x<=w && y>=0 && y<=h);
}

std::pair <double, double> GetLine(double x1,double y1, double x2,double y2) {
					double a, b;
					//returns a & b;
					a = y2 - y1;
					if (fabs(x2-x1) < 1e-9) a = 1000000;
					else
						a/= x2 - x1;
					b = -a*x1 + y1;
					return std::pair <double, double> (a,b);
					
}

bool Graph1D::ChangeColor(int i, COLORREF col) {
	if(i >= this->numGraphs)
		return false;
	graphColors->at(i) = col;
	return true;
}

bool Graph1D::ChangeColor(int i, int r, int g, int b) {
	if(i >= this->numGraphs)
		return false;
	graphColors->at(i) = RGB(r, g, b);
	return true;
}

void Graph1D::Add(COLORREF color, 
				GraphDrawType drawtype,
			    array<double> ^tx, 
				array<double> ^ty) {
	mutex.WaitOne();

	numGraphs++;
	vector<double> *tempx = new vector<double>[numGraphs];
	vector<double> *tempy = new vector<double>[numGraphs];
	vector<double> *tempcx = new vector<double>[numGraphs];
	vector<double> *tempcy = new vector<double>[numGraphs];
	for(int i = 0; i < numGraphs - 1; i++) {
		tempx[i] = x[i];
		tempy[i] = y[i];
		tempcx[i] = cx[i];
		tempcy[i] = cy[i];
	}
	delete[] x;
	delete[] y;
	delete[] cx;
	delete[] cy;

	x = tempx;
	y = tempy;
	cx = tempcx;
	cy = tempcy;

	x[numGraphs - 1] = arraytovector(tx);
	y[numGraphs - 1] = arraytovector(ty);
	graphColors->push_back(color);
	drawtypes->push_back(static_cast<GraphDrawTypeC>(drawtype));

	GeneratePixelGraphs(numGraphs - 1);

	vis->push_back(true);
	bRedraw = true;

	mutex.ReleaseMutex();
}

void Graph1D::Remove(int num) {
	if(num > numGraphs - 1 || num < 0) return;
	
	mutex.WaitOne();

	/*
	graphColors = new vector<COLORREF>(1, defColor);
	drawtypes = new vector<GraphDrawType>(1, defDrawtype);
	legendKeys = new vector<std::string>();

	zoomHistX = new vector< std::pair<double, double> >();
	zoomHistY = new vector< std::pair<double, double> >();
	*/
	
	numGraphs--;
	graphColors->erase(graphColors->begin() + num);
	drawtypes->erase(drawtypes->begin() + num);

	vector<double> *tempx = new vector<double>[numGraphs];
	vector<double> *tempy = new vector<double>[numGraphs];
	vector<double> *tempcx = new vector<double>[numGraphs];
	vector<double> *tempcy = new vector<double>[numGraphs];

	for(int i = 0, j = 0; i < numGraphs; i++, j++) {
		if (i == j && i == num) { i--; continue;}
		tempx[i] = x[j];
		tempy[i] = y[j];
		tempcx[i] = cx[j];
		tempcy[i] = cy[j];
	}
	delete[] x;
	delete[] y;
	delete[] cx;
	delete[] cy;

	x = tempx;
	y = tempy;
	cx = tempcx;
	cy = tempcy;
	
	vis->erase(vis->begin() + num);
	bRedraw = true;

	mutex.ReleaseMutex();
}

void Graph1D::SetDefaultScale(array<double> ^atx, array<double> ^aty) {
	if(atx == nullptr || aty == nullptr)
		return;

	std::vector<double> tx = arraytovector(atx), ty = arraytovector(aty);

	if(ty.size() == 0 || numGraphs < 1) {
		mutex.WaitOne();
		defscaleX = defscaleY = 1.0;
		curscaleX = defscaleX;
		curscaleY = defscaleY; 
		mutex.ReleaseMutex();
		return;
	}

	double ymin = std::numeric_limits<double>::infinity();
	double ymax = -std::numeric_limits<double>::infinity();

	mutex.WaitOne();

	if(tx.back() == tx[0])
		defscaleX = 1.0;
	else {
		if(bLogScaleX) {
			const double minPosi = std::numeric_limits<double>::epsilon();
			int mpos = 0;
			while(tx[mpos++] < minPosi);
			mpos--;

			defscaleX = graphWidth / (log10(tx.back()) - log10(tx[mpos]));
		}
		else
			defscaleX = graphWidth / (tx.back() - tx[0]);
	}

	defoffX = tx[0];
	if(bLogScaleX) {
		for(int ind = 0; ind < tx.size(); ind++) {
			defoffX = log10(tx[ind]);
			if(defoffX == defoffX && fabs(defoffX) != std::numeric_limits<double>::infinity()) {
				break;
			}
		}

		if(defoffX != defoffX && fabs(defoffX) != std::numeric_limits<double>::infinity()) {
			defoffX = 0.;
		}
	}

	for(unsigned int i = 0; i < tx.size(); i++) {
		if(bLogScaleY && isNotReal(log10(ty[i])))
			continue;
		if(bLogScaleX && isNotReal(log10(tx[i])))
			continue;

		if(ty[i] > ymax)
			ymax = ty[i];
		if(ymin > ty[i])
			ymin = ty[i];
	}

	// 20% whitespace
	if(ymax == ymin)
		defscaleY = 1.0;
	else {
		if(bLogScaleY)
			defscaleY = graphHeight / ((log10(ymax) - log10(ymin)) * 1.2);
		else
			defscaleY = graphHeight / ((ymax - ymin) * 1.2);
	}

	// 5% whitespace
	defoffY = (bLogScaleY ? log10(ymin) : ymin) - (graphHeight*0.05 / defscaleY);

	curoffX = defoffX; curoffY = defoffY;
	curscaleX = defscaleX; curscaleY = defscaleY; 

	mutex.ReleaseMutex();
}

void Graph1D::GetGraph(int num, array<double> ^%tx, array<double> ^%ty)
{
	mutex.WaitOne();
	tx = vectortoarray(x[num]);
	ty = vectortoarray(y[num]);
	mutex.ReleaseMutex();
}

void Graph1D::Modify(int num, array<double> ^tx, array<double> ^ty) {
	mutex.WaitOne();
	x[num] = arraytovector(tx);
	y[num] = arraytovector(ty);

	mutex.ReleaseMutex();
	// Set default scale and offset
	// Avi: this makes the first Graph1D be redrawn on it's own scale.
	//	if there are other graphs that have already been drawn, they
	//	remain on their own scale...
	//if(num == 0)
	//	SetDefaultScale(x[num], y[num]);
	// If the comment remains, modifying the first Graph1D will not resize the drawn area

	mutex.WaitOne();

	GeneratePixelGraphs(num);

	bRedraw = true;

	mutex.ReleaseMutex();
}

void Graph1D::Legend(array<String ^> ^keys) {
	std::vector<std::string> keyvec (keys->Length);
	for(int i = 0; i < keys->Length; i++)
		keyvec[i] = clrToString(keys[i]);

	mutex.WaitOne();
	*legendKeys = keyvec;
	bLegend = true;
	mutex.ReleaseMutex();
}

void Graph1D::Highlight(int num, int startX, int endX, COLORREF color) {
	mutex.WaitOne();
	highlightInd = num;
	highlightIndexes->push_back(std::pair<int, int>(startX, endX));
	//highlightStart = startX;
	//highlightEnd = endX;
	highlightColor = color;
	bRedraw = true;
	mutex.ReleaseMutex();
}

void Graph1D::HighlightSingle(int num, int startX, int endX, COLORREF color) {
	mutex.WaitOne();
	highlightInd = num;
	highlightIndexes->clear();
	highlightIndexes->push_back(std::pair<int, int>(startX, endX));
	if(startX == -1 && endX == -1)
		highlightIndexes->clear();
	highlightColor = color;
	bRedraw = true;
	mutex.ReleaseMutex();
}

void Graph1D::Deselect() {
	mutex.WaitOne();
	highlightIndexes->clear();
	//highlightStart = highlightEnd = -1;
	bRedraw = true;
	mutex.ReleaseMutex();
}

void Graph1D::Mask(int num, int startX, int endX, COLORREF color) {
	mutex.WaitOne();
	highlightMaskInd = num;
	highlightMaskIndexes->push_back(std::pair<int, int>(startX, endX));
	highlightMaskColor = color;
	bRedraw = true;
	mutex.ReleaseMutex();
}

void Graph1D::RemoveMask() {
	mutex.WaitOne();
	highlightMaskIndexes->clear();
	bRedraw = true;
	mutex.ReleaseMutex();
}

int Graph1D::GetHighlightStart(std::vector< std::pair<int, int> > &ind) {
	if(ind.size() == 0)
		return -1;

	int res = ind[0].first;
	for(int i = 1; i < (int)ind.size(); i++)
		res = min(res, ind[i].first);
	return res;
}

int Graph1D::GetHighlightEnd(std::vector< std::pair<int, int> > &ind) {
	if(ind.size() == 0)
		return -1;

	int res = ind[0].second;
	for(int i = 1; i < (int)ind.size(); i++)
		res = max(res, ind[i].second);
	return res;
}

void Graph1D::SetScale(int axis, GraphScale gs) {
	if(axis == 0)	// y-axis
		SetScale(gs,  (bLogScaleX ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN)/*, 0*/);
	else
		SetScale((bLogScaleY ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN), gs/*, 0*/);
}
void Graph1D::SetScale(GraphScale gsY, GraphScale gsX) {
	bLogScaleX = (gsX == GraphScale::SCALE_LOG);

	bLogScaleY = (gsY == GraphScale::SCALE_LOG);

	ResizeGraph();
}

void Graph1D::ResizeGraph() {
	xoff = this->Location.X;
	yoff = this->Location.Y;
	height = this->Height;
	width = this->Width;

	SetGraphDimensions();

	mutex.WaitOne();

	mutex.ReleaseMutex();
	
	if(numGraphs > 0) {
		int ind = FindFirstGraph();
		if(ind >= 0)
			SetDefaultScale(vectortoarray(x[ind]), vectortoarray(y[ind]));
	}

	ResetZoom();

	bRedrawBackground = true;
	bRedraw = true;
}

void Graph1D::SZoom(int startX, int startY) {
	zoomX = startX;
	zoomY = startY;
	bRedrawZoom = true;
}

void Graph1D::EZoom(int endX, int endY) {
	endX  -= graphOffsetX;
	endY  -= graphOffsetY;
	zoomX -= graphOffsetX;
	zoomY -= graphOffsetY;
	double xmin = min(zoomX, endX), 
		   ymin = min(zoomY, endY),
		   xmax = max(zoomX, endX),
		   ymax = max(zoomY, endY);
		
	double tmpscaleX, tmpscaleY, tmpoffX, tmpoffY;

	mutex.WaitOne();

	if( (xmax - xmin) == 0 || 
		(ymax - ymin) == 0 ) {
			bRedrawZoom = false; bRedraw = true;
			return;
	}
		

	tmpscaleX = (double)graphWidth / (double)(xmax - xmin);
	tmpoffX = xmin;
	tmpscaleY = (double)graphHeight / (double)(ymax - ymin);
	tmpoffY = graphHeight - ymax;

	curoffX += (tmpoffX / curscaleX);
	curscaleX *= tmpscaleX;

	curoffY += (tmpoffY / curscaleY);
	curscaleY *= tmpscaleY;

	GeneratePixelGraphs(-1);
	
	bRedrawBackground = true;
	bRedrawZoom = false;
	bRedraw = true;
	
	ShowGrid = true;

	mutex.ReleaseMutex();
}

void Graph1D::ResetZoom() {
	mutex.WaitOne();
	curoffX = defoffX; curoffY = defoffY;
	curscaleX = defscaleX; curscaleY = defscaleY; 

	GeneratePixelGraphs(-1);

	zoomendx = -1;
	bRedrawBackground = true;
	bRedraw = true;
	mutex.ReleaseMutex();
}

void Graph1D::FitToAllGraphs() {
	if(numGraphs < 1)
		return;

	double ymin = POSINF;
	double ymax = NEGINF;
	double xmin = POSINF;
	double xmax = NEGINF;
	
	mutex.WaitOne();
	for(int i = 0; i < numGraphs; i++) {
		if(!vis->at(i)) continue;
		for(unsigned int j = 0; j < x[i].size(); j++) {
			if(bLogScaleY && isNotReal(log10(y[i].at(j))))
				continue;
			if(bLogScaleX && isNotReal(log10(x[i].at(j))))
				continue;

			if(x[i].at(j) > xmax)
				xmax = x[i].at(j);
			if(xmin > x[i].at(j))
				xmin = x[i].at(j);

			if(y[i].at(j) > ymax)
				ymax = y[i].at(j);
			if(ymin > y[i].at(j))
				ymin = y[i].at(j);
		}
	}

	if(bLogScaleX) {
		xmin = log10(xmin);
		xmax = log10(xmax);
	}

	if(xmin == xmax)
		defscaleX = 1.0;
	else
		defscaleX = graphWidth / (xmax - xmin);
	defoffX = xmin;

	if(bLogScaleY) {
		ymin = log10(ymin);
		ymax = log10(ymax);
	}
 
	// 20% whitespace
	if(ymin == ymax)
		defscaleY = 1.0;
	else
		defscaleY = graphHeight / ((ymax - ymin) * 1.2);

	// 5% whitespace
	defoffY = ymin - (graphHeight*0.05 / defscaleY);

	curoffX = defoffX; curoffY = defoffY;
	curscaleX = defscaleX; curscaleY = defscaleY; 

	mutex.ReleaseMutex();

	ResetZoom();
}

DoublePair Graph1D::PointToData(int x, int y) {
	double tempx = x;
	double tempy = y;
	
	tempx = tempx - graphOffsetX;
	tempy = tempy - graphOffsetY;

	mutex.WaitOne();

	tempx = (tempx / curscaleX) + curoffX;
	tempy = ((graphHeight - tempy) / curscaleY) + curoffY;

	mutex.ReleaseMutex();
	return DoublePair(tempx, tempy);
}

void Graph1D::PaintTicks(Graphics ^g) {
	int res = tickres;
	double ytics = double(graphHeight) / double(res) / 2.0;
	double xtics = double(graphWidth) / double(res) / 2.0;
	
	Pen ^pen = gcnew Pen(Color::Blue, 2.0);
	StringFormat ^sf = gcnew StringFormat();
	sf->Alignment = StringAlignment::Far;

	System::Drawing::Font ^textFont = gcnew System::Drawing::Font(gcnew FontFamily("Arial"), 7, 
						   FontStyle::Regular, GraphicsUnit::Point, 
						   DEFAULT_CHARSET);

	// Draw log(I) ticks and values
	if(bLogScaleY && bYTicks && numGraphs > 0) {
		double ymin = double::MaxValue, ymax = double::MinValue, orders;
		for(int n = 0; n < numGraphs; n++) {
			for(int u = 0; (unsigned)u < y[n].size(); u++) {
				if(y[n].at(u) > 1.0e-100) {	// Ignore negative and zero values
					ymin = min(ymin, y[n].at(u));
					ymax = max(ymax, y[n].at(u));
				}
			}
		}

		orders = (log10(ymax) - log10(ymin));

		for(int i = (int)log10(ymin) - 2; i < int(log10(ymax) + 3.0); i++) {
			int pos = 0;
			for(int j = 2; j < 11; j++) {
				pos = int(graphHeight - ((log10(j * pow(10.0, i)) - curoffY) * curscaleY));
				if(pos < 1 || pos > graphHeight || pos != pos)
					continue;
				g->DrawLine(pen, graphOffsetX, graphOffsetY + pos, 
							graphOffsetX + j, graphOffsetY + pos);
				if((orders / curscaleY * defscaleY * 750 / graphHeight) < 1.0 || 
						j == 10 || 
						(11 - j) % int(orders / curscaleY * defscaleY * 750 / graphHeight) == 0) {
					std::pair<double, int> inten = toSci(pow(10, PointToData(0, (pos + graphOffsetY)).second));
					g->DrawString(inten.first.ToString("0.") + "E " + inten.second.ToString("0."),
								textFont, gcnew SolidBrush(Color::Black),
								Drawing::Rectangle(graphOffsetX - 45, graphOffsetY + pos - 4, 42, 10), sf);
				}
			}
		} // end for i
	} // end if blogscaleY

	// Draw the log(q) ticks and values
	if(bLogScaleX && bXTicks && numGraphs > 0) {
		int ind = FindFirstGraph();
		if(ind < 0)
			return;

		double xmax = x[ind].back(), xmin = x[ind].front(), xscale = defscaleX;
		int pos, j;
		double orders;
		int cropped = int(x->size() - cx->size());

		//Check to make sure that the q values are shown in the Graph1D
		for(j = 0; (unsigned)j < cx[ind].size(); j++) {
			if(x[ind].at(cropped + j) > 0.0) {
				xmin = x[ind].at(cropped + j);
				break;
			}
		}
		for(int end = (int)cx[ind].size() - 1; end > j; end--) {
			if(cx[ind].at(end) / 1.00001 <= graphWidth ) {
				xmax = x[ind].at(cropped + end);
				break;
			}
		}		

		orders = (log10(xmax) - log10(xmin));

		for(int i = (int)floor(log10(xmin)); i < int(floor(log10(xmax)) + 1.0); i++) {
			for(j = 2; j < 11; j++) {
				pos = int((log10(j * pow(10.0, i)) - curoffX) * curscaleX);

				//Don't draw lines that are out of the Graph1D area
				if(pos > graphWidth || pos < 1 || pos != pos)
					continue;

				g->DrawLine(pen, graphOffsetX + pos, graphOffsetY + graphHeight, 
							graphOffsetX + pos, graphOffsetY + graphHeight - j);
				if(orders < 0.41 || j == 10 || (11 - j) % int(orders * 3.0) == 0) {
					std::pair<double, int> inten = toSci(pow(10.0, PointToData(pos + graphOffsetX, 0).first));
					g->DrawString(inten.first.ToString("0.") + "E " + inten.second.ToString("0."), textFont, gcnew SolidBrush(Color::Black),
							  Drawing::Rectangle(i == 2*res ? graphOffsetX + pos - 22 : graphOffsetX + pos - 8, graphOffsetY + graphHeight + 2, 42, 10));
				}
			}
		}
	} // end if blogscaleX

	for(int i = 0; i < 2*res + 1; i++) {
		int pos;

		// Y Ticks
		if(bYTicks && !bLogScaleY) {
			pos = int(i * ytics);
			if(i == 2*res || i == 0); // Don't draw last/first line
			else if(i % 2 == 0)
				g->DrawLine(pen, graphOffsetX, graphOffsetY + pos, 
							graphOffsetX + 10, graphOffsetY + pos);
			else
				g->DrawLine(pen, graphOffsetX, graphOffsetY + pos, 
							graphOffsetX + 5, graphOffsetY + pos);

			if(i % 2 == 0) {
				std::pair<double, int> hi = toSci(PointToData(0, pos + graphOffsetY).second);

				g->DrawString(hi.first.ToString("0.00") + "E " + hi.second.ToString("0."), 
						  textFont,
						  gcnew SolidBrush(Color::Black),
						  Drawing::Rectangle(graphOffsetX - 45, graphOffsetY + pos - 4, 42, 10), sf);
			}
		}

		// X Ticks
		if(bXTicks && !LogScaleX) {
			pos = int(i * xtics);
			if(i == 2*res || i == 0); // Don't draw last/first line
			else if(i % 2 == 0)
				g->DrawLine(pen, graphOffsetX + pos, graphOffsetY + graphHeight, 
							graphOffsetX + pos, graphOffsetY + graphHeight - 10);
			else
				g->DrawLine(pen, graphOffsetX + pos, graphOffsetY + graphHeight, 
							graphOffsetX + pos, graphOffsetY + graphHeight - 5);
					
			if(i % 2 == 0) {
				g->DrawString(PointToData(pos + graphOffsetX, 0).first.ToString("0.000"), 
							  textFont,
		 					  gcnew SolidBrush(Color::Black),
							  Drawing::Rectangle(i == 2*res ? graphOffsetX + pos - 18 : graphOffsetX + pos - 4, graphOffsetY + graphHeight + 2, 30, 10));
			}

		}
	}
}

void Graph1D::PaintBackground(Graphics ^g) {
	g->Clear(Color::White);
	g->DrawRectangle(gcnew Pen(Color::Black), 0, 0, width - 1, height - 1);
}

void Graph1D::PaintZoom(Graphics ^g, int mouseX, int mouseY) {
	int tzoomX = zoomX, tzoomY = zoomY;

	//"Transparent" rectangle
	Pen ^p = gcnew Pen(Color::Black);
	p->DashStyle = Drawing2D::DashStyle::Dot;

	
	g->DrawLine(p, max(min(tzoomX, mouseX), graphOffsetX + 1), 
				max(min(tzoomY, mouseY), graphOffsetY + 1),
				min(max(mouseX, tzoomX), graphOffsetX + graphWidth - 2), 
				max(min(mouseY, tzoomY), graphOffsetY + 1));

	g->DrawLine(p, min(max(mouseX, tzoomX), graphOffsetX + graphWidth - 2), 
				max(min(mouseY, tzoomY), graphOffsetY + 1),
				min(max(mouseX, tzoomX), graphOffsetX + graphWidth - 2), 
				min(max(mouseY, tzoomY), graphOffsetY + graphHeight - 2));

	g->DrawLine(p, min(max(mouseX, tzoomX), graphOffsetX + graphWidth - 2), 
				min(max(mouseY, tzoomY), graphOffsetY + graphHeight - 2),
				max(min(mouseX, tzoomX), graphOffsetX + 1),
				min(max(mouseY, zoomY), graphOffsetY + graphHeight - 2));

	g->DrawLine(p, max(min(mouseX, tzoomX), graphOffsetX + 1),
				min(max(mouseY, zoomY), graphOffsetY + graphHeight - 2),
				max(min(mouseX, tzoomX), graphOffsetX + 1), 
				max(min(mouseY, tzoomY), graphOffsetY + 1));

	bRedrawZoom = false;
}

void Graph1D::PaintLabels(Graphics ^g) {
	// Title
	if(title->Length > 0) {
		StringFormat ^sf = gcnew StringFormat();
		sf->Alignment = StringAlignment::Center;
		g->DrawString(title, 
			gcnew System::Drawing::Font(gcnew FontFamily("Arial"), 8, 
			FontStyle::Regular, GraphicsUnit::Point, 
					   DEFAULT_CHARSET),
			gcnew SolidBrush(Color::Black),
			Drawing::Rectangle(0, 2, width, 20), sf);
	}
	//END Title

	// We need the border in order to paint the labels
	if(!(bXTicks || bYTicks))
		return;

	// X Label
	if(xLabel->Length > 0) {
		StringFormat ^sf = gcnew StringFormat();
		sf->Alignment = StringAlignment::Center;
		g->DrawString(xLabel, 
			gcnew System::Drawing::Font(gcnew FontFamily("Arial"), 8, 
			FontStyle::Regular, GraphicsUnit::Point, 
					   DEFAULT_CHARSET),
			gcnew SolidBrush(Color::Black),
			Drawing::Rectangle(graphOffsetX, height - 20, graphWidth, 20), sf);
	}
	//END X Label
	// Y Label
	if(yLabel->Length > 0) {
		StringFormat ^sf = gcnew StringFormat(StringFormatFlags::DirectionVertical);
		sf->Alignment = StringAlignment::Center;

		g->TranslateTransform(20.0f, (float)(graphHeight));
		g->RotateTransform(180.0f);
		g->DrawString(yLabel, 
			gcnew System::Drawing::Font(gcnew FontFamily("Arial"), 8, 
			FontStyle::Regular, GraphicsUnit::Point, 
					   DEFAULT_CHARSET, true),
			gcnew SolidBrush(Color::Black),
			Drawing::Rectangle(0, -20, 15, graphHeight), sf);
		g->ResetTransform();
	}
	//END Y Label
}

// Draws outline of peak if drawing a peak in SF mode
void Graph1D::PaintPeak(Graphics ^g, int mouseX, int mouseY) {
	Pen ^recPen = gcnew Pen(Color::FromArgb(200,0,55));

	//Draw a triangle representing a peak
	g->DrawLine(recPen, down_click_x, max(down_click_y, mouseY), 
				down_click_x, min(down_click_y, mouseY));
	g->DrawLine(recPen, down_click_x, min(down_click_y, mouseY),
				down_click_x + abs(mouseX - down_click_x), max(down_click_y, mouseY));
	g->DrawLine(recPen, down_click_x + abs(mouseX - down_click_x), max(down_click_y, mouseY),
				down_click_x - abs(mouseX - down_click_x), max(down_click_y, mouseY));
	g->DrawLine(recPen, down_click_x - abs(mouseX - down_click_x), max(down_click_y, mouseY),
				down_click_x, min(down_click_y, mouseY));

	bRedrawPeak = false;
}

void Graph1D::PaintCrop(Graphics ^g, int mouseX) {
	Pen ^recPen = gcnew Pen(Color::FromArgb(89,0,0));

	int h = graphOffsetY + graphHeight;

	//Draw a line 
	if (down_click_x < mouseX ){
		g->DrawLine(recPen, down_click_x, graphOffsetY + 7, down_click_x + 10, graphOffsetY + 7);
		g->DrawLine(recPen, down_click_x, graphOffsetY + 7, down_click_x, h - 7);
		g->DrawLine(recPen, down_click_x, h - 7, down_click_x + 10, h - 7);
	}
	else {
		g->DrawLine(recPen, down_click_x, graphOffsetY + 7, down_click_x - 10, graphOffsetY + 7);
		g->DrawLine(recPen, down_click_x, graphOffsetY + 7, down_click_x, h - 7);
		g->DrawLine(recPen, down_click_x, h - 7, down_click_x - 10, h - 7);
	}

	if(!bMask) {
		g->DrawLine(recPen, down_click_x - 20, graphOffsetY + 7 + 5, down_click_x - 10, graphOffsetY + 7 + 5);
		
		if(!((System::Windows::Forms::Control::ModifierKeys & System::Windows::Forms::Keys::Shift) == System::Windows::Forms::Keys::Shift)) {
			g->DrawLine(recPen, down_click_x - 15, graphOffsetY + 7 + 5, down_click_x - 15, graphOffsetY + 7);
			g->DrawLine(recPen, down_click_x - 15, graphOffsetY + 7, down_click_x - 15, graphOffsetY + 7 + 10);
		}
	}
	if (down_click_x < mouseX ) {
		g->DrawLine(recPen, mouseX, graphOffsetY + 7, mouseX - 10, graphOffsetY + 7);
		g->DrawLine(recPen, mouseX, graphOffsetY + 7, mouseX, h - 7);
		g->DrawLine(recPen, mouseX, h - 7, mouseX - 10, h - 7);
	}
	else {
		g->DrawLine(recPen, mouseX, graphOffsetY + 7, mouseX + 10, graphOffsetY + 7);
		g->DrawLine(recPen, mouseX, graphOffsetY + 7, mouseX, h - 7);
		g->DrawLine(recPen, mouseX, h - 7, mouseX + 10, h - 7);
	}

	bRedrawCrop = false;
}

bool Graph1D::IsInHighlight(int graphNum, int index) {
	if(graphNum == 0) {
		for(int i = 0; i < (int)highlightMaskIndexes->size(); i++) {
			std::pair<int, int> highlight = highlightMaskIndexes->at(i);
			if(index >= highlight.first && index <= highlight.second)
				return true;
		}
	} else if(graphNum == 1){
		for(int i = 0; i < (int)highlightIndexes->size(); i++) {
			std::pair<int, int> highlight = highlightIndexes->at(i);
			if(index >= highlight.first && index <= highlight.second)
				return true;
		}
	}
	return false;
}

void DrawLimits (double x1, double y1, double x2, double y2, int w, int h, Graphics ^g,Pen ^gcolor) {
	std::vector <std::pair <double,double> > finalPoints;
	finalPoints.clear();
	std::pair <double,double> line = GetLine(x1,y1,x2,y2);		
	double a = line.first, b = line.second;
	
	double left,right,top,bottom;
	left = b;
	right = a*w+b;
	if( !(fabs(a)< 1e-12)  ) {
		top = -b / a;
		bottom = (h - b) / a;
	}
	else 
		top = bottom = -1;
	
	bool bLeft   = (0 <= left && h >= left),
		 bRight  = (0 <= right && h >= right),
		 bTop    = (0 <= top && w >= top),
		 bBottom = (0 <= bottom && w >= bottom);

	if(bLeft  )
		finalPoints.push_back(std::pair<double, double>(0, left));	
	if(bRight  )
		finalPoints.push_back(std::pair<double, double>(w, right));	
	if(bTop )
		finalPoints.push_back(std::pair<double, double>(top, 0));	
	if(bBottom )
		finalPoints.push_back(std::pair<double, double>(bottom, h));
	

	if(finalPoints.size() == 2)
		g->DrawLine(gcolor, (float)finalPoints[0].first, (float)finalPoints[0].second, 
				(float)finalPoints[1].first, (float)finalPoints[1].second);

}

void Graph1D::DrawGraphFromVectors(int graphNum, Graphics ^g, Pen ^gcolor, Pen ^hcolor, const std::vector<double> &x, 
							     const std::vector<double> &y, int w, int h) {

	for(int i = 1; i < (int)x.size(); i++) {
		if(x[i] < 0.0 && i < (int)x.size() - 1) continue;
		if(isNotReal(y[i]) || isNotReal(y[i - 1])) continue;

		if (x[i-1] < 0.0) {
			if (isOutOfGraph(x[i],(float)h - 1.0,w,h)) {
				DrawLimits(x[i-1],y[i-1],x[i],y[i],w,h,g,IsInHighlight(graphNum, i) ? hcolor : gcolor);
			}
			else{ 
				std::pair <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
				if(y[i-1] < -1000.0*h || line.second < -1000.0*h || y[i-1] > 1000.0*h || line.second > 1000.0*h)
					continue;
				g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor,0.0,(float)line.second,(float)x[i],(float)y[i]);
			}
			continue;
		}
		if (x[i] > w) {
			if (isOutOfGraph(x[i-1],y[i-1],w,h)) {
				DrawLimits(x[i],y[i],x[i-1],y[i-1],w,h,g,IsInHighlight(graphNum, i) ? hcolor : gcolor);
			}
			else{ 
				std::pair  <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
				g->DrawLine(IsInHighlight(graphNum, i) ? hcolor :  gcolor,(float)w ,(float)line.first*float(w)+(float)line.second,(float)x[i-1],(float)y[i-1]);
			}
			return; 
		}


		if(y[i] < 0.0) {
			if(i > 0 && y[i-1] > 0.0) {
				if (isOutOfGraph(x[i-1],y[i-1],w,h)) {
					DrawLimits(x[i],y[i],x[i-1],y[i-1],w,h,g,IsInHighlight(graphNum, i) ? hcolor : gcolor);
				} else { 
					std::pair  <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
					g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor, (float)x[i-1], (float)y[i-1],float(-line.second/line.first) , 0.0);
				}
				while (i+1 < (int)x.size() && y[i+1] < 0.0) i++;
			}
			continue;
		}
			
		if(y[i] > float(h)) {
			if(i > 0 && y[i-1] < float(h)) {
				if (isOutOfGraph(x[i-1],y[i-1],w,h)) {
					DrawLimits(x[i],y[i],x[i-1],y[i-1],w,h,g,IsInHighlight(graphNum, i) ? hcolor : gcolor);
				} else { 
					std::pair <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
					g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor, (float)x[i-1],
							   (float)y[i-1], float((h-line.second)/line.first), float(h));
				}
				while (i+1 < (int)x.size() && y[i+1] > (float)h) i++;
			}
			continue;
		}
		//current point is in graph checking previous
		if (i == 0) continue;
		if (i>0 && isOutOfGraph(x[i-1],y[i-1],w,h)) {
			if(y[i-1] < 0.0){
				std::pair  <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
				g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor, (float)x[i], 
							(float)y[i], float(-line.second/line.first) , 0.0);

			}
			else {
				std::pair <double,double> line = GetLine(x[i],y[i],x[i-1],y[i-1]);
				g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor, (float)x[i], (float)y[i],
							float((h-line.second)/line.first), float(h));
			}
		}
		else
			g->DrawLine(IsInHighlight(graphNum, i) ? hcolor : gcolor, (float)x[i], (float)y[i], 
									(float)x[i-1], (float)y[i-1]);

		//drawline(between points)
		
			//TODO (???)
			//interpolate(x1,x2,y1,y2,y3);
			// y= a*x+b
			// if (graphBot(x) < || > edge(x)) graphBot = a*x +b => x = (graphBot - b) /a
			// same for graphTop.
			// draw line from graphTop to graphBot
			
		
	}
}

void Graph1D::PaintGraph(Graphics ^g, int w, int h) {
	if(w <= 0 || h <= 0)
		return;

	graphBmp = gcnew Bitmap(w, h);
	Graphics ^innerG = Graphics::FromImage(graphBmp);
	innerG->SmoothingMode = SmoothingMode::AntiAlias;

	innerG->DrawRectangle(gcnew Pen(Color::Black), 0, 0, w - 1, h - 1);

	for(int k = 0; k < numGraphs; k++) {
		//Don't waste time drawing invisible stuff
		if(!vis->at(k)) continue;
		
		bool bHighlight = (highlightInd == k && GetHighlightStart(*highlightIndexes) >= 0 && 
						   GetHighlightStart(*highlightIndexes) < GetHighlightEnd(*highlightIndexes) && 
						   GetHighlightEnd(*highlightIndexes) < (int)cx[k].size());
		
		bool bMaskHighlight = (highlightMaskInd == k && GetHighlightStart(*highlightMaskIndexes) >= 0 && 
						   GetHighlightStart(*highlightMaskIndexes) < GetHighlightEnd(*highlightMaskIndexes) && 
						   GetHighlightEnd(*highlightMaskIndexes) < (int)cx[k].size());
		
		int R = GetRValue(graphColors->at(k)), 
			G = GetGValue(graphColors->at(k)), 
			B = GetBValue(graphColors->at(k));
		Pen	^gcolor    = gcnew Pen(Color::FromArgb(R, G, B), 2.0f);

		Pen ^highlight;
		if(bHighlight) {
			R = GetRValue(highlightColor);
			G = GetGValue(highlightColor); 
			B = GetBValue(highlightColor);
			highlight = gcnew Pen(Color::FromArgb(R, G, B), 2.0f);
		}

		if(bMaskHighlight) {
			R = GetRValue(highlightMaskColor);
			G = GetGValue(highlightMaskColor); 
			B = GetBValue(highlightMaskColor);
			highlight = gcnew Pen(Color::FromArgb(R, G, B), 2.0f);
		}

		bHighlight = bHighlight || bMaskHighlight;

		switch(drawtypes->at(k)) {
			default:
			case DRAW_LINES:
				if(cx[k].size() <= 1)
					continue;

				DrawGraphFromVectors(k, innerG, gcolor, bHighlight ? highlight : gcolor, cx[k], cy[k], graphWidth, graphHeight); 
				break;

			case DRAW_SCATTER:
				gcolor->Width = 1.0;
				if(bHighlight)
					highlight->Width = 1.0;
				for(int i = 0; i < (int)cx[k].size(); i++) {
					if(cx[k][i] < 0.0 || cx[k][i] > graphWidth || cy[k][i] < 0.0 || cy[k][i] > graphHeight)
						continue;

					if(i % 3 == 0 && (!cy[k][i].IsInfinity(cy[k][i]) && cy[k][i] == cy[k][i]) ) {
						innerG->DrawEllipse(bHighlight ?  highlight : gcolor, (float)cx[k][i], 
											(float)cy[k][i], 3.0f, 3.0f);
					}
				}
				break;

			case DRAW_FULL_SCATTER:
				gcolor->Width = 1.0;
				Brush ^br, ^hbr;
				if(bHighlight)
					highlight->Width = 1.0;
				
				br = gcnew System::Drawing::SolidBrush(gcolor->Color);
				if(bHighlight)
					hbr = gcnew System::Drawing::SolidBrush(highlight->Color);
				
				for(int i = 0; i < (int)cx[k].size(); i++) {
					if(cx[k][i] < 0.0 || cx[k][i] > graphWidth || cy[k][i] < 0.0 || cy[k][i] > graphHeight)
						continue;

					if(i % 3 == 0) {
						//bool bInHighlight = bHighlight && IsInHighlight(i);
						innerG->FillEllipse(bHighlight ? (IsInHighlight(k, i) ? hbr : br) : br, (float)cx[k][i], (float)cy[k][i], 4.0f, 4.0f);
					}
				}
				break;
			case DRAW_VERTICAL_LINE:
					gcolor->Width = 1.0;
					if(bHighlight)
						highlight->Width = 1.0;
					for(int i = 0; i < (int)cx[k].size(); i++) {
						if(cx[k][i] < 0.0 || cx[k][i] > graphWidth)
							continue;
						bool bInHighlight = bHighlight && IsInHighlight(k, i);
						innerG->DrawLine(bInHighlight ? highlight : gcolor, (float)cx[k][i], 0.0f, 
										 (float)cx[k][i], (float)height);
					}
					break;
					
		}
	}
}

void Graph1D::PaintGrid(Graphics ^g) {
	int res = tickres;
	double xtics = double(graphWidth) / double(res);
	double ytics = double(graphHeight) / double(res);
	Pen ^p = gcnew Pen(Color::FromArgb(200, 200, 200));

	for(int i = 1; i < res; i++) {
		if(!LogScaleY)
			g->DrawLine(p, graphOffsetX, graphOffsetY + int(i * ytics), 
						graphOffsetX + graphWidth, graphOffsetY + int(i * ytics));
		if(!LogScaleX)
			g->DrawLine(p, graphOffsetX + int(i * xtics), graphOffsetY, 
						graphOffsetX + int(i * xtics), graphOffsetY + graphHeight);
	}

	if(LogScaleX && numGraphs > 0) {
		int ind = FindFirstGraph();
		if(ind < 0)
			return;

		double xmin = x[ind].front(), xmax = x[ind].back(), orders;
		if(xmin == 0.) {
			xmin = x[ind][1];
		}
		int pos, j;
		
		//Check to make sure that the q values are shown in the graph
		for(j = 0; (unsigned)j < cx[ind].size(); j++) {
			if((cx[ind].at(j) >= 0.0 || cx[ind].at(j) != cx[ind].at(j) ) && (!bLogScaleX || x[ind].at(j) > 0.0)) {
				xmin = x[ind].at(j);
				break;
			}
		}
		for(int end = (int)cx[ind].size() - 1; end > j; end--) {
			if(cx[ind].at(end) / 1.00001 <= graphWidth ) {
				xmax = x[ind].at(end);
				break;
			}
		}		

		orders = log10(xmax) - log10(xmin);

		for(int i = (int)floor(log10(xmin)); i < int(floor(log10(xmax)) + 1.0); i++) {
			for(int j = 2; j < 11; j++) {
				pos = int((log10(j * pow(10.0, i)) - curoffX) * curscaleX);
				if(pos > graphWidth || pos < 1 || pos != pos)
					continue;
				if(orders < 0.41 || j % 2 == 0 || (11 - j) % int(orders * 3.0) == 0) {
					//Don't draw lines that are out of the graph area
					g->DrawLine(p, graphOffsetX + pos, graphOffsetY, 
							graphOffsetX + pos, graphOffsetY + graphHeight);
				}
			}
		}
	} // end if(LogScaleX)

	if(LogScaleY && numGraphs > 0) {
		double ymin = double::MaxValue, ymax = double::MinValue, orders;
		int pos;

		for(int n = 0; n < numGraphs; n++) {
			for(int u = 0; (unsigned)u < y[n].size(); u++) {
				if(y[n].at(u) < 1.0e-100)
					continue;
				ymin = min(ymin, y[n].at(u));
				ymax = max(ymax, y[n].at(u));
			}
		}

		orders = (log10(ymax) - log10(ymin));

		for(int i = (int)floor(log10(ymin)); i < int(floor(log10(ymax)) + 2.0); i++) {
			for(int j = 2; j < 11; j++) { 
				pos = int(graphHeight - ((log10(j * pow(10.0, i)) - curoffY) * curscaleY));
				if(pos < 1 || pos > graphHeight || pos != pos)
					continue;
				if((orders / curscaleY * defscaleY * 750 / graphHeight) < 1.0 ||
						j == 10 ||
						(11 - j) % int(orders / curscaleY * defscaleY * 750 / graphHeight) == 0) {
					g->DrawLine(p, graphOffsetX, graphOffsetY + pos, 
								graphOffsetX + graphWidth, graphOffsetY + pos);
				}
			}
		}
	} // end if(LogScaleY)
}

void Graph1D::PaintLegend(Graphics ^g) {
	System::Drawing::Font ^textFont = gcnew System::Drawing::Font(gcnew FontFamily("Arial"), 8, 
																   FontStyle::Regular, GraphicsUnit::Point, 
																   DEFAULT_CHARSET);

	for(int i = 0; i < (int)legendKeys->size(); i++) {
		int R = GetRValue(graphColors->at(i)), 
			G = GetGValue(graphColors->at(i)), 
			B = GetBValue(graphColors->at(i));
		Pen ^p = gcnew Pen(Color::FromArgb(R, G, B), 1.0);
		
		if(drawtypes->at(i) == DRAW_LINES)
			g->DrawLine(p, graphOffsetX + 10, graphOffsetY + (graphHeight / 20 * (i + 1)), 
					graphOffsetX + 40, graphOffsetY + (graphHeight / 20 * (i + 1)));
		else if(drawtypes->at(i) == DRAW_SCATTER) {
			for(int j = graphOffsetX + 10; j < graphOffsetX + 40; j += 10)
				g->DrawEllipse(p, Drawing::Rectangle(j, graphOffsetY + int(graphHeight / 20 * (i + 1)) - 2, 4, 4));
		}
		
		g->DrawString(gcnew String(legendKeys->at(i).c_str()), textFont, gcnew SolidBrush(Color::Black),
					  (float)(graphOffsetX + 45), (float)(graphOffsetY + float(graphHeight / 20 * (i + 1)) - 6));
	}
}

void Graph1D::SetGraphDimensions() {
	graphOffsetX = graphOffsetY = 0;
	graphWidth = width;
	graphHeight = height;

	if(bXTicks || bYTicks) {
		graphOffsetX += 40;
		graphOffsetY += 20;
		graphWidth   -= 50;
		graphHeight  -= 40;

		if(xLabel->Length > 0)
			graphHeight -= 20;

		if(yLabel->Length > 0) {
			graphOffsetX += 20;
			graphWidth -= 20;
		}
	}
}

void Graph1D::Repaint(Graphics ^g, int mouseX, int mouseY) {
	if(width <= 0 || height <= 0)
		return;

	mutex.WaitOne();

	// Create/Use existing layers
	SetGraphDimensions();

	// Draw the layers
	if(bRedrawBackground) {
		bgBmp = gcnew Bitmap(width, height);
		Graphics ^innerG = Graphics::FromImage(bgBmp);

		PaintBackground(innerG);

		if(bGrid)
			PaintGrid(innerG);

		if(title->Length > 0 || xLabel->Length > 0 || yLabel->Length > 0)
			PaintLabels(innerG);

		if(bXTicks || bYTicks)
			PaintTicks(innerG);

		bRedrawBackground = false;
	}

	if(bgBmp)
		g->DrawImage(bgBmp, 0, 0);

	if(bRedraw) {
		PaintGraph(g, graphWidth, graphHeight);
		bRedraw = false;
	}

	if(graphBmp)
		g->DrawImage(graphBmp, graphOffsetX, graphOffsetY);

	if(bLegend)
		PaintLegend(g);

	if(bRedrawZoom)
		PaintZoom(g, mouseX, mouseY);

	if(bCrop && bRedrawCrop)
		PaintCrop(g, mouseX);
	
	if(bPeaks && bRedrawPeak)
		PaintPeak(g, mouseX, mouseY);

	mutex.ReleaseMutex();
}

void Graph1D::ExportImage(IntPtr hParent) {	
	SaveFileDialog ^sfd = gcnew SaveFileDialog();
	sfd->Title = "Save Graph As:";
	sfd->Filter = "JPEG Image (*.jpg, *.jpeg)|*.jpg;*.jpeg|PNG Image (*.png)|*.png|GIF Image (*.gif)|*.gif|Bitmap Image (*.bmp)|*.bmp|All files|*.*";
	sfd->FilterIndex = 1;

	
	
	if(sfd->ShowDialog() == DialogResult::Cancel)
		return;

	String ^filename = sfd->FileName;

	bool bAppendExt = !filename->Contains(".");

	mutex.WaitOne();

	Bitmap ^bitmap = gcnew Bitmap(width, height);
	Graphics ^g = Graphics::FromImage(bitmap);

	bRedraw = true;
	bRedrawBackground = true;
	bRedrawZoom = false;
	Repaint(g, 0, 0);

	Imaging::ImageFormat ^format;

	switch(sfd->FilterIndex) {
		default:
		case 1:
			format = Imaging::ImageFormat::Jpeg;
			if(bAppendExt)
				filename += ".jpg";
			break;
		case 2:
			format = Imaging::ImageFormat::Png;
			if(bAppendExt)
				filename += ".png";
			break;
		case 3:
			format = Imaging::ImageFormat::Gif;
			if(bAppendExt)
				filename += ".gif";
			break;
		case 4:
			format = Imaging::ImageFormat::Bmp;
			if(bAppendExt)
				filename += ".bmp";
			break;
    }

	bitmap->Save(filename, format);

	mutex.ReleaseMutex();
}

void Graph1D::DrawPeakOutline(int xi, int yi, int xf, int yf) {
	down_click_x = xi - xoff;
	down_click_y = yi - yoff;
	current_x = xf - xoff;
	current_y = yf - yoff;	
}

bool Graph1D::DrawPeak() {
	return bPeaks;
}

void Graph1D::SetDrawPeak(bool bDraw) {
	bRedrawPeak = bPeaks = bDraw;
}

bool Graph1D::Cropping() {
	return bCrop;
}

void Graph1D::SetCropping(bool _bCrop, bool _bMask) {
	bRedrawCrop = bCrop = _bCrop;
	bMask = _bMask;
}

bool Graph1D::GetGraphVisibility(int num) {
	return vis->at(num);
}
void Graph1D::SetGraphVisibility(int num, bool visible) {
	vis->at(num) = visible;
}

// Change to scientific form
std::pair<double, int> Graph1D::toSci(double x) {
	std::pair<double, int> res;
	res.second = (int)floor(log10(fabs(x)));
	res.first = x / pow(10.0, res.second);

	if(isNotReal(res.second) && res.second != 0) {
		res.second	= 0;
		res.first	= 0.0;
	}

	return res;
}


void Graph1D::CleanupGraph() {
	if(graphColors)
		delete graphColors;
	graphColors = NULL;

	if(highlightIndexes)
		delete highlightIndexes;
	highlightIndexes = NULL;

	if(highlightMaskIndexes)
		delete highlightMaskIndexes;
	highlightMaskIndexes = NULL;

	if(vis)
		delete vis;
	vis = NULL;

	if(drawtypes)
		delete drawtypes;
	drawtypes = NULL;

	if(legendKeys)
		delete legendKeys;
	legendKeys = NULL;

	if(x)
		delete[] x;
	x = NULL;

	if(y)
		delete[] y;
	y = NULL;

	if(cx)
		delete[] cx;
	cx = NULL;

	if(cy)
		delete[] cy;
	cy = NULL;
}

int Graph1D::FindFirstGraph()
{
	for(int i = 0; i < numGraphs; i++)
		if(x[i].size() > 0)
			return i;

	return -1;
}

};

