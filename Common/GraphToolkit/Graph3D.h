#pragma once

#include <windows.h>

#include <gl/gl.h>
#include <gl/glu.h>

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Windows;
using namespace System::Data;
using namespace System::Drawing;


namespace GraphToolkit {

	/// <summary>
	/// Summary for Graph3D
	/// </summary>
	public ref class Graph3D : public System::Windows::Forms::UserControl
	{
	protected:
		// OpenGL handles
		NativeWindow ^GLWnd;
		HDC m_hDC;
		HGLRC m_hglrc;

		// Fields
		unsigned int rInterval;

	public:
		[Description("The function called to draw the contents of this control.")]
		event EventHandler ^Render;

		[Category("Appearance")]
		[Description("The interval, in milliseconds, for caling the render event. If zero, renders continuously.")]		
		property unsigned int RenderInterval {
			unsigned int get() { return rInterval; }
			void set(unsigned int value) {
				rInterval = value;
				if(timer1)
					timer1->Interval = ((value == 0) ? 1 : value);
			}
		}

		Graph3D(void)
		{
			RenderInterval = 40; // 25 FPS max

			// IMPORTANT: Makes graph panel double-buffered
			SetStyle(ControlStyles::DoubleBuffer, true);
			UpdateStyles();

			InitializeComponent();			
		}

	protected:
		

		System::Void SwapOpenGLBuffers(System::Void)
		{
			SwapBuffers(m_hDC) ;
		}

		System::Void MakeCurrent(System::Void) {
			if((wglMakeCurrent(m_hDC, m_hglrc)) == NULL)
			{
				;//MessageBox::Show("wglMakeCurrent Failed");
			}
		}

		GLint MySetPixelFormat(HDC hdc)
		{
			static	PIXELFORMATDESCRIPTOR pfd=				// pfd Tells Windows How We Want Things To Be
			{
				sizeof(PIXELFORMATDESCRIPTOR),				// Size Of This Pixel Format Descriptor
				1,											// Version Number
				PFD_DRAW_TO_WINDOW |						// Format Must Support Window
				PFD_SUPPORT_OPENGL |						// Format Must Support OpenGL
				PFD_DOUBLEBUFFER,							// Must Support Double Buffering
				PFD_TYPE_RGBA,								// Request An RGBA Format
				16,										// Select Our Color Depth
				0, 0, 0, 0, 0, 0,							// Color Bits Ignored
				0,											// No Alpha Buffer
				0,											// Shift Bit Ignored
				0,											// No Accumulation Buffer
				0, 0, 0, 0,									// Accumulation Bits Ignored
				16,											// 16Bit Z-Buffer (Depth Buffer)  
				0,											// No Stencil Buffer
				0,											// No Auxiliary Buffer
				PFD_MAIN_PLANE,								// Main Drawing Layer
				0,											// Reserved
				0, 0, 0										// Layer Masks Ignored
			};

			GLint  iPixelFormat; 

			// get the device context's best, available pixel format match 
			if((iPixelFormat = ChoosePixelFormat(hdc, &pfd)) == 0)
			{
				MessageBox::Show("ChoosePixelFormat Failed");
				return 0;
			}

			// make that match the device context's current pixel format 
			if(SetPixelFormat(hdc, iPixelFormat, &pfd) == FALSE)
			{
				MessageBox::Show("SetPixelFormat Failed");
				return 0;
			}

			if((m_hglrc = wglCreateContext(m_hDC)) == NULL)
			{
				MessageBox::Show("wglCreateContext Failed: " + GetLastError());
				return 0;
			}

			if((wglMakeCurrent(m_hDC, m_hglrc)) == NULL)
			{
				MessageBox::Show("wglMakeCurrent Failed");
				return 0;
			}


			return 1;
		}

		bool InitGL(GLvoid)										// All setup for opengl goes here
		{
			glShadeModel(GL_SMOOTH);							// Enable smooth shading
			glClearDepth(1.0f);									// Depth buffer setup
			glEnable(GL_DEPTH_TEST);							// Enables depth testing
			glDepthFunc(GL_LEQUAL);								// The type of depth testing to do
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really nice perspective calculations
			return TRUE;										// Initialisation went ok
		}

		GLvoid ReSizeGLScene(GLsizei width, GLsizei height)		// Resize and initialise the gl window
		{
			if (height==0)										// Prevent A Divide By Zero By
			{
				height=1;										// Making Height Equal One
			}

			glViewport(0,0,width,height);						// Reset The Current Viewport

			glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
			glLoadIdentity();									// Reset The Projection Matrix

			// Calculate The Aspect Ratio Of The Window
			gluPerspective(45.0f,(float)width/(float)height,0.1f,100.0f);

			glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
			glLoadIdentity();									// Reset The Modelview Matrix
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Graph3D()
		{
			GLWnd->DestroyHandle();

			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Timer^  timer1;
	private: System::ComponentModel::IContainer^  components;
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
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->SuspendLayout();
			// 
			// label1
			// 
			this->label1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 32, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(177)));
			this->label1->ForeColor = System::Drawing::Color::White;
			this->label1->Location = System::Drawing::Point(0, 0);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(322, 279);
			this->label1->TabIndex = 0;
			this->label1->Text = L"3D";
			this->label1->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// timer1
			// 
			this->timer1->Interval = 40;
			this->timer1->Tick += gcnew System::EventHandler(this, &Graph3D::timer1_Tick);
			// 
			// Graph3D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::DimGray;
			this->Controls->Add(this->label1);
			this->Name = L"Graph3D";
			this->Size = System::Drawing::Size(322, 279);
			this->Load += gcnew System::EventHandler(this, &Graph3D::Graph3D_Load);
			this->Resize += gcnew System::EventHandler(this, &Graph3D::Graph3D_Resize);
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e) {
				 if(RenderInterval == 0) {
					timer1->Enabled = false;
					for(;;) {
						MakeCurrent();
						// TODO: Remove?
						glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear screen and depth buffer
						glLoadIdentity();									// Reset the current modelview matrix
						Render(this, gcnew EventArgs());
					}
				 }

				 MakeCurrent();
				 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear screen and depth buffer
				 glLoadIdentity();									// Reset the current modelview matrix
				 Render(this, gcnew EventArgs());
			 }
	private: System::Void Graph3D_Resize(System::Object^  sender, System::EventArgs^  e) {
				 if(GLWnd)
					 ReSizeGLScene(this->Width, this->Height);
			 }
private: System::Void Graph3D_Load(System::Object^  sender, System::EventArgs^  e) {
			 GLWnd = gcnew NativeWindow();
			 Forms::CreateParams^ cp = gcnew Forms::CreateParams;

			 // Set the position on the form
			 cp->X = 0;
			 cp->Y = 0;
			 cp->Height = this->Height;
			 cp->Width = this->Width;

			 // Specify the form as the parent.
			 cp->Parent = this->Handle;

			 // Create as a child of the specified parent and make OpenGL compliant (no clipping)
			 cp->Style = WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_DISABLED;

			 // Create the actual window
			 GLWnd->CreateHandle(cp);

			 m_hDC = GetDC((HWND)GLWnd->Handle.ToPointer());

			 if(m_hDC)
			 {
				 if(!MySetPixelFormat(m_hDC))
					 return;

				 ReSizeGLScene(this->Width, this->Height);
				 if(!InitGL())
					return;

				 // Set pseudo-transparent background
				 COLORREF bg = this->BackColor.ToArgb();
				 GLclampf r,g,b;

				 r = ((float)GetRValue(bg) / 255.0f);
				 g = ((float)GetGValue(bg) / 255.0f);
				 b = ((float)GetBValue(bg) / 255.0f);
				 glClearColor(b, g, r, 0.0f);

				 timer1->Enabled = true;
			 }
		 }
};
}
