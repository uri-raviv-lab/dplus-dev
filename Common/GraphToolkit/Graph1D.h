#ifndef __GRAPH1D_H
#define __GRAPH1D_H
#pragma once

#include <vector>

#include <cmath>
#include <windows.h>


using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Threading;
using namespace System::Drawing;


namespace GraphToolkit {
	enum GraphDrawTypeC {
		DRAW_LINES = 0,
		DRAW_SCATTER,
		DRAW_FULL_SCATTER,
		DRAW_VERTICAL_LINE
	};

	public value class DoublePair {
	public:
		double first, second;
		DoublePair(double a, double b) { first = a; second = b; }
	};

	/// <summary>
	/// Summary for Graph1D
	/// </summary>
	public ref class Graph1D : public System::Windows::Forms::UserControl
	{
	public:	
		enum class GraphDrawType {
			DRAW_LINES = 0,
			DRAW_SCATTER,
			DRAW_FULL_SCATTER,
			DRAW_VERTICAL_LINE
		};

		enum class GraphScale {
			SCALE_LIN,
			SCALE_LOG
		};



	private:
	// Graph colors vector
	std::vector<COLORREF> *graphColors;

	// Graph draw types vector
	std::vector<GraphDrawTypeC> *drawtypes;

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
	private: System::Windows::Forms::Timer^  timer1;
			 bool bMask;		// Flag to tell if masking

	void SetDefaultScale(array<double> ^tx, array<double> ^ty);

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

	void DrawGraphFromVectors(int graphNum, Graphics ^g, Pen ^gcolor, Pen ^hcolor, const std::vector<double> &x, 
							  const std::vector<double> &y, int w, int h);

	void SetGraphDimensions();

	// Generates the cx/cy vectors
	void GeneratePixelGraphs(int id);

	bool bRedrawBackground, bRedraw, bRedrawZoom, bRedrawPeak, bRedrawCrop;

	bool bZooming;

public:
	// Mouse coordinates
	int curx, cury;

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

	// Redraws graphs
	void Add(COLORREF color, 
			 GraphDrawType drawtype, 
		  	 array<double> ^ tx, 
			 array<double> ^ ty);

	// Redraws graphs
	void Modify(int num, array<double> ^ tx, 
		        array<double> ^ ty);

	// Gets the values for the graph
	void GetGraph(int num, array<double> ^% tx, array<double> ^% ty);

	// Redraws graphs
	void Remove(int num);

	// Redraws extras
	void Legend(array<String ^> ^keys);

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

	//////////////////////////////////////////////////////////////////////////
	// Properties
	[Category("Appearance"), DefaultValue(false), Description("Determines whether the graph's X-axis will be logarithmically scaled by default.")]
	property bool LogScaleX {
		bool get() {
			return bLogScaleX;
		}

		void set(bool value) {
			bLogScaleX = value;
			SetScale(bLogScaleY ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN,
					 bLogScaleX ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN);
			this->Invalidate();
		}
	}

	[Category("Appearance"), DefaultValue(false), Description("Determines whether the graph's Y-axis will be logarithmically scaled by default.")]
	property bool LogScaleY {
		bool get() {
			return bLogScaleY;
		}

		void set(bool value) {
			bLogScaleY = value;
			SetScale(bLogScaleY ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN,
					 bLogScaleX ? GraphScale::SCALE_LOG : GraphScale::SCALE_LIN);
			this->Invalidate();
		}
	}

	[Category("Appearance"), DefaultValue(true), Description("If true, displays a grid in the background of the graph.")]
	property bool ShowGrid {
		bool get() {
			return bGrid;
		}

		void set(bool value) {
			bGrid = value;
			bRedrawBackground = true;
			this->Invalidate();
		}
	}

	[Category("Appearance"), DefaultValue(true), Description("If true, displays ticks on the bottom of the graph (the X-axis).")]
	property bool ShowXTicks {
		bool get() {
			return bXTicks;
		}

		void set(bool value) {
			bXTicks = value;
			bRedrawBackground = true;
			bRedraw = true;
			this->Invalidate();
		}
	}

	[Category("Appearance"), DefaultValue(true), Description("If true, displays ticks on the lefthand side of the graph (the Y-axis).")]
	property bool ShowYTicks {
		bool get() {
			return bYTicks;
		}

		void set(bool value) {
			bYTicks = value;
			bRedrawBackground = true;
			bRedraw = true;
			this->Invalidate();
		}
	}

	[Category("Appearance"), DefaultValue(5), Description("The number of ticks to show per-axis.")]
	property int TickResolution {
		int get() {
			return tickres;
		}

		void set(int value) {
			tickres = value;
			bRedrawBackground = true;
			this->Invalidate();
		}
	}

	[Category("Data"), Description("The graph's title.")]
	property String ^GraphTitle {
		String ^get() {
			return title;
		}

		void set(String ^value) {
			title = value;
			bRedrawBackground = true;
			bRedraw = true;
			this->Invalidate();
		}
	}

	[Category("Data"), Description("The graph's X-label title.")]
	property String ^XLabel {
		String ^get() {
			return xLabel;
		}

		void set(String ^value) {
			xLabel = value;
			bRedrawBackground = true;
			bRedraw = true;
			this->Invalidate();
		}
	}

	[Category("Data"), Description("The graph's Y-label title.")]
	property String ^YLabel {
		String ^get() {
			return yLabel;
		}

		void set(String ^value) {
			yLabel = value;
			bRedrawBackground = true;
			bRedraw = true;
			this->Invalidate();
		}
	}

	[Category("Layout"), DefaultValue(true), Description("Enables zooming in with the right mouse button and zooming out with the middle mouse button.")]
	property bool MouseZoom;

	[Category("Layout"), DefaultValue(false), Description("If enabled in conjunction with MouseZoom, zooming requires to drag the right mouse button.")]
	property bool MouseDragToZoom;

	protected:
	// Redraws graph	Axis == 0 -> y-axis; Axis == 1 -> x-axis
	// Changes the indicated axis to gs (linear/log) and leaves the other unchanged
	void SetScale(int axis, GraphScale gs);
	
	// Sets the y-axis to gsY (linear/log) and the x-axis to gsX (linear/log)
	void SetScale(GraphScale gsY, GraphScale gsX);


	void ExportImage(IntPtr hParent);

	int GetHighlightStart(std::vector< std::pair<int, int> > &ind);
	bool IsInHighlight(int graphNum, int index);
	int GetHighlightEnd(std::vector< std::pair<int, int> > &ind);

public:
	DoublePair PointToData(int x, int y);

	// Redraws all
	void ResizeGraph();

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

protected:
	void InitializeGraph();

	public:
		Graph1D(void)
		{
			// Some default values
			MouseZoom = true;
			MouseDragToZoom = false;

			InitializeComponent();

			// IMPORTANT: Makes graph panel double-buffered
			SetStyle(ControlStyles::DoubleBuffer |
				ControlStyles::UserPaint |
				ControlStyles::AllPaintingInWmPaint,
				true);
			UpdateStyles();
			
			InitializeGraph();

			// Initial painting
			curx = cury = 0;
			bZooming = false;
			Repaint(this->CreateGraphics(), 0, 0);
		}

	protected:
		void CleanupGraph();

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Graph1D()
		{
			if (components)
			{
				delete components;
			}

			CleanupGraph();
		}
private: System::ComponentModel::IContainer^  components;
protected: 


	protected: 

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->SuspendLayout();
			// 
			// timer1
			// 
			this->timer1->Interval = 10;
			this->timer1->Tick += gcnew System::EventHandler(this, &Graph1D::timer1_Tick);
			// 
			// Graph1D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->Name = L"Graph1D";
			this->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &Graph1D::Graph1D_Paint);
			this->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &Graph1D::Graph1D_MouseMove);
			this->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Graph1D::Graph1D_MouseClick);
			this->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &Graph1D::Graph1D_MouseDown);
			this->Resize += gcnew System::EventHandler(this, &Graph1D::Graph1D_Resize);
			this->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &Graph1D::Graph1D_MouseUp);
			this->ResumeLayout(false);

		}
#pragma endregion
		// Events
	private: System::Void Graph1D_Paint(System::Object^  sender, System::Windows::Forms::PaintEventArgs^  e) {
				 Repaint(e->Graphics, curx, cury);
			 }
private: System::Void Graph1D_Resize(System::Object^  sender, System::EventArgs^  e) {
				ResizeGraph();
				this->Invalidate();
		 }


		 // Mouse-zoom related events
private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e) {
			 if(!bZooming) {
				 timer1->Enabled = false;
				 this->Invalidate();
			 }
		 }
private: System::Void Graph1D_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 if(MouseZoom && !MouseDragToZoom) {
				 switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							if(!bZooming) {
								SZoom(curx, cury);
							} else
								EZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
						case Windows::Forms::MouseButtons::Middle:
							if(!bZooming) {
								FitToAllGraphs();
								ShowGrid = bGrid;
								timer1->Enabled = true;
							}
							break;
					}
				}
		 }
private: System::Void Graph1D_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 if(MouseZoom && MouseDragToZoom) {
				 switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							if(!bZooming)
								SZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
					}
				}
		 }
private: System::Void Graph1D_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 if(MouseZoom && MouseDragToZoom) {
				 switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							EZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
						case Windows::Forms::MouseButtons::Middle:
							if(!bZooming) {
								FitToAllGraphs();
								timer1->Enabled = true;
							}
							break;
					}
				}
		 }
private: System::Void Graph1D_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 if(MouseZoom) {
				 curx = e->X - xoff;
				 cury = e->Y - yoff;


				 if(bZooming || DrawPeak() || Cropping()) {
					 RedrawZoom(bZooming);
					 RedrawCrop(Cropping());
					 RedrawPeak(DrawPeak());

					 this->Invalidate();
				 }

			 }
		 }

		 // Find the first graph with content
		 int FindFirstGraph();
};
}
#endif
