#ifndef __WGTCONTROL_H
#define __WGTCONTROL_H
#pragma once
#include "graphtoolkit.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace GUICLR {

	/// <summary>
	/// Summary for WGTControl
	/// </summary>
	public ref class WGTControl : public System::Windows::Forms::UserControl
	{
	public:
		public: Graph ^graph;
				int curx, cury;
				bool bZooming;
				bool bDragToZoom;

		WGTControl(void) : curx(0), cury(0), bZooming(false)
		{
			InitializeComponent();

		 bDragToZoom = false;
		}

		void setDragToZoom(bool d) {
			bDragToZoom = d;
		}

		bool getDragToZoom() {
			return bDragToZoom;
		}

		Graph ^getGraph() {
			return graph;
		}

		void setGraph(Graph ^gr) { 
			graph = gr; 
		}

		virtual bool IsInputKey(System::Windows::Forms::Keys) override {
				return true;
			 }

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~WGTControl()
		{
			if(graph)
				delete graph;
			graph = nullptr;

			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Timer^  timer1;
	protected: 
	private: System::ComponentModel::IContainer^  components;

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
			this->timer1->Tick += gcnew System::EventHandler(this, &WGTControl::timer1_Tick);
			// 
			// WGTControl
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->Cursor = System::Windows::Forms::Cursors::Cross;
			this->Name = L"WGTControl";
			this->Load += gcnew System::EventHandler(this, &WGTControl::WGTControl_Load);
			this->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &WGTControl::WGTControl_Paint);
			this->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &WGTControl::WGTControl_MouseMove);
			this->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &WGTControl::WGTControl_MouseDown);
			this->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &WGTControl::WGTControl_MouseUp);
			this->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &WGTControl::WGTControl_MouseClick);
	
			this->Resize += gcnew System::EventHandler(this, &WGTControl::WGTControl_Resize);
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void WGTControl_Load(System::Object^  sender, System::EventArgs^  e) {
				 // IMPORTANT: Makes graph panel double-buffered
				 SetStyle(ControlStyles::DoubleBuffer |
						  ControlStyles::UserPaint |
						  ControlStyles::AllPaintingInWmPaint,
						  true);
				 UpdateStyles();
			 }
	private: System::Void WGTControl_Paint(System::Object^  sender, System::Windows::Forms::PaintEventArgs^  e) {
				if(graph) {
					//graph->bRedrawZoom = bZooming;
					graph->Repaint(e->Graphics, curx, cury);
				}
			 }
	private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e) {
				 if(!bZooming) {
					timer1->Enabled = false;
					if(graph)
						this->Invalidate();
				}
			 }
	private: System::Void WGTControl_MouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
				 if(graph && !bDragToZoom) {
					switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							if(!bZooming) {
								graph->SZoom(curx, cury);
							} else
								graph->EZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
						case Windows::Forms::MouseButtons::Middle:
							if(!bZooming) {
								//graph->ResetZoom();
								graph->FitToAllGraphs();
								graph->ToggleGrid();
								graph->ToggleGrid();
								//graph->bRedrawZoom = false;
								timer1->Enabled = true;
							}
							break;
					}
				}
			 }
	private: System::Void WGTControl_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
				 if(graph && bDragToZoom) {
					switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							if(!bZooming)
								graph->SZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
					}
				}
			 }
	private: System::Void WGTControl_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
				 if(graph && bDragToZoom) {
					switch(e->Button) {
						case Windows::Forms::MouseButtons::Right:
							if(curx <= 0 || cury <= 0)
								break;
							graph->EZoom(curx, cury);
							bZooming = !bZooming;
							timer1->Enabled = true;
							break;
						case Windows::Forms::MouseButtons::Middle:
							if(!bZooming) {
								//graph->ResetZoom();
								graph->FitToAllGraphs();
								timer1->Enabled = true;
							}
							break;
					}
				}
			 }
	private: System::Void WGTControl_Resize(System::Object^  sender, System::EventArgs^  e) {
				 if(graph) {
					RECT area;
					area.top = 0;
					area.left = 0;
					area.right = this->Size.Width;
					area.bottom = this->Size.Height;
					graph->Resize(area);
				 }
				 this->Invalidate();
			 }
private: System::Void WGTControl_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			 if(graph) {
					curx = e->X - graph->xoff;
					cury = e->Y - graph->yoff;
					
					
					if(bZooming || graph->DrawPeak() || graph->Cropping()) {
						graph->RedrawZoom(bZooming);
						graph->RedrawCrop(graph->Cropping());
						graph->RedrawPeak(graph->DrawPeak());

						this->Invalidate();
					}

				}
		 }
};
}
#endif