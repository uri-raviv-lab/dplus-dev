#ifndef __GRAPHTOOLKIT_H
#define __GRAPHTOOLKIT_H
#pragma once

#include "clrfunctionality.h"

#include <cmath>
#include <windows.h>
using namespace System;
using namespace System::Drawing;
using namespace System::Threading;

enum GraphDrawType {
	DRAW_LINES,
	DRAW_SCATTER,
	DRAW_FULL_SCATTER,
	DRAW_VERTICAL_LINE
};

enum GraphScale {
	SCALE_LIN,
	SCALE_LOG
};

struct graphLine {
	COLORREF color;
	GraphDrawType drawtype;
	std::vector<double> x, y;
	std::string legendKey;
};

typedef struct {
	std::vector< std::pair<double, double> > p;
} PairArray;


public ref class Graph {
private:
	// Graph colors vector
	std::vector<COLORREF> *graphColors;

	// Graph draw types vector
	std::vector<GraphDrawType> *drawtypes;

	// Graph legend keys vector
	std::vector<std::string> *legendKeys;

	// Highlight parameters
	int highlightInd;
	std::vector<std::pair<int, int>> *highlightIndexes; 
	COLORREF highlightColor;

	// Highlight (masking) parameters
	int highlightMaskInd;
	std::vector<std::pair<int, int>> *highlightMaskIndexes; 
	COLORREF highlightMaskColor;

	//Graph visibility vector
	std::vector<bool> *vis;

	// Multi-threading requires Mutual Exclusions
	Mutex mutex;

	// Tick/Grid resolution
	int tickres;

	// Zooming coordinates
	int zoomX, zoomY, zoomendx, zoomendy;

	// Cropping/Peak coordinates
	int down_click_x, down_click_y, current_x, current_y;

	// Actual graph (frame inside the graph containing the plot) coordinates
	int graphOffsetX, graphOffsetY, graphWidth, graphHeight;

	// Scale values for default zoom
	double defoffX, defoffY, defscaleX, defscaleY,
		   curoffX, curoffY, curscaleX, curscaleY;

	// Various labels
	String ^title, ^xLabel, ^yLabel;

	// Graphic layers of the graph
	Bitmap ^bgBmp, ^graphBmp;

	bool bPeaks;	// Flag to tell if the user is trying to draw a peak
	bool bCrop;		// if I am cropping in the remove baseline window
	bool bMask;		// Flag to tell if masking

	void SetDefaultScale(const std::vector<double> &tx, const std::vector<double> &ty);

	// Various graphics painting functions
	void PaintBackground(Graphics ^g);
	void PaintLegend(Graphics ^g);
	void PaintGrid(Graphics ^g);
	void PaintTicks(Graphics ^g);
	void PaintGraph(Graphics ^g, int w, int h);
	void PaintZoom(Graphics ^g, int mouseX, int mouseY);
	void PaintLabels(Graphics ^g);
	void PaintPeak(Graphics ^g, int mouseX, int mouseY);
	void PaintCrop(Graphics ^g, int mouseX);

	void DrawGraphFromVectors(int graphNum, Graphics ^g, Pen ^gcolor, Pen ^hcolor, const std::vector<double>& x, 
							  const std::vector<double>& y, int w, int h);

	void SetGraphDimensions();

	// Generates the cx/cy vectors
	void GeneratePixelGraphs(int id);

	bool bRedrawBackground, bRedraw, bRedrawZoom, bRedrawPeak, bRedrawCrop;

public:
	// Graph window coordinates
	int width, height, xoff, yoff;

	// Self explanatory boolean fields
	bool bLogScaleX, bLogScaleY, bGrid, bLegend, bXTicks, bYTicks;

	// Actual x/y values
	std::vector<double> *x, *y;

	// Actual x/y graph pixel values for quick drawing (with zoom and offsets)
	std::vector<double> *cx, *cy;

	// Number of graphs
	int numGraphs;

	Graph(RECT area, 
		  COLORREF defColor, 
          GraphDrawType defDrawtype, 
		  std::vector<double> tx, 
		  std::vector<double> ty,
		  bool logScaleX, bool logScaleY);

	~Graph();

	// Redraws graphs
	void Add(COLORREF color, 
			 GraphDrawType drawtype, 
		  	 std::vector<double> tx, 
			 std::vector<double> ty);

	// Redraws graphs
	void Modify(int num, std::vector<double> tx, 
		        std::vector<double> ty);

	// Redraws graphs
	void Remove(int num);

	// Redraws extras
	void Legend(const std::vector<std::string> &keys);

	// Redraw graph
	void Highlight(int num, int startX, int endX, COLORREF color);
	void HighlightSingle(int num, int startX, int endX, COLORREF color);
	void Deselect();

	// Change to scientific form
	std::pair<double, int> toSci(double x);

	// Masking highlighting
	void Mask(int num, int startX, int endX, COLORREF color);
	void RemoveMask();

	// Redraws extras
	void SZoom(int startX, int startY);

	// Redraws all
	void EZoom(int endX, int endY);

	// Redraw graph
	void ResetZoom();
	void FitToAllGraphs();

	// Does the redrawing
	void Repaint(Graphics ^g, int mouseX, int mouseY);

	bool LogScaleY();

	bool LogScaleX();

	// Redraws graph	Axis == 0 -> y-axis; Axis == 1 -> x-axis
	// Changes the indicated axis to gs (linear/log) and leaves the other unchanged
	void SetScale(int axis, GraphScale gs);
	
	// Sets the y-axis to gsY (linear/log) and the x-axis to gsX (linear/log)
	void SetScale(GraphScale gsY, GraphScale gsX);

	// Redraws background
	void ToggleGrid() { bGrid = !bGrid; bRedrawBackground = true; }

	// Redraws background, graph
	void ToggleXTicks() { bXTicks = !bXTicks; bRedrawBackground = true; bRedraw = true; }

	// Redraws background, graph
	void ToggleYTicks() { bYTicks = !bYTicks; bRedrawBackground = true; bRedraw = true; }

	// Redraws all
	void Resize(RECT area);

	void ExportImage(IntPtr hParent);

	// Redraw background, graph
	void SetTitle (const std::string &inTitle) { title  = gcnew String(inTitle.c_str()); bRedrawBackground = true; bRedraw = true; }
	void SetXLabel(const std::string &inLabel) { xLabel = gcnew String(inLabel.c_str()); bRedrawBackground = true; bRedraw = true; }
	void SetYLabel(const std::string &inLabel) { yLabel = gcnew String(inLabel.c_str()); bRedrawBackground = true; bRedraw = true; }

	void SetTitle (String ^inTitle) { title  = inTitle; bRedrawBackground = true; bRedraw = true; }
	void SetXLabel(String ^inLabel) { xLabel = inLabel; bRedrawBackground = true; bRedraw = true; }
	void SetYLabel(String ^inLabel) { yLabel = inLabel; bRedrawBackground = true; bRedraw = true; }

	// Redraws background
	void SetTickRes(int res) { tickres = res; bRedrawBackground = true; }

	std::pair<double, double> PointToData(int x, int y);

	int GetHighlightStart(std::vector<std::pair<int, int>> &ind);
	bool IsInHighlight(int graphNum, int index);
	int GetHighlightEnd(std::vector<std::pair<int, int>> &ind);

	bool GetGraphVisibility(int num);
	void SetGraphVisibility(int num, bool visible);

	// Change the color of the ith curve
	bool ChangeColor(int i, int r, int g, int b);
	bool ChangeColor(int i, COLORREF col);


	/**
	 * A function to draw two lines representing the size and position of a peak
	 * while the user draws one with the mouse (in SF tab)
	 * xi and yi are the initial xy co-ordinates, xf and yf are the final xy co-ordinates.
	**/
	void DrawPeakOutline(int xi, int yi, int xf, int yf);

	bool DrawPeak();

	// Redraws extras
	void SetDrawPeak(bool bDraw);

	bool Cropping();

	// Redraw extras
	void SetCropping(bool bCrop, bool bMask);

	void RedrawZoom(bool bDraw) { bRedrawZoom = bDraw; }
	void RedrawPeak(bool bDraw) { bRedrawPeak = bDraw; }
	void RedrawCrop(bool bDraw) { bRedrawCrop = bDraw; }

};
#endif