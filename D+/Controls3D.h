#pragma once

#include "MainWindow.h"
#include "GraphPane3D.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace WeifenLuo::WinFormsUI::Docking;
using namespace LuaInterface;


namespace DPlus {

	/// <summary>
	/// Summary for Controls3D
	/// </summary>
	public ref class Controls3D : WeifenLuo::WinFormsUI::Docking::DockContent
	{
	protected:
		MainWindow ^parentForm;
	private: System::Windows::Forms::Timer^  scrollTimer;
	public: System::Windows::Forms::CheckBox^  showCornerAxesCheckBox;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	public: System::Windows::Forms::CheckBox^  fixedSize_checkBox;
 
	protected: 
		GraphPane3D ^controlledForm;
	public:
		Controls3D(MainWindow ^pform, GraphPane3D ^pane3d)
		{
			InitializeComponent();

			parentForm = pform;
			controlledForm = pane3d;
			ConsDummy();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Controls3D()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Collections::Generic::List<TrackBar^>^ TrackBarList;
	private: System::Collections::Generic::List<TextBox^>^ TextBoxList;
	public: System::Windows::Forms::TextBox^  pitchTextBox;
	public: System::Windows::Forms::TrackBar^  angle1TrackBar;
	public: System::Windows::Forms::Label^  angle1Label;
	private: System::Windows::Forms::Label^  zoomLabel;
	public: System::Windows::Forms::TrackBar^  zoomTrackBar;
	public: System::Windows::Forms::TextBox^  zoomTextBox;
	private: System::Windows::Forms::Label^  angle3Label;
	public: System::Windows::Forms::TrackBar^  angle3TrackBar;
	public: System::Windows::Forms::TextBox^  rollTextBox;
	private: System::Windows::Forms::Label^  angle2Label;
	public: System::Windows::Forms::TrackBar^  angle2TrackBar;
	public: System::Windows::Forms::TextBox^  yawTextBox;
	public: System::Windows::Forms::CheckBox^  showAxesCheckBox;
	private: System::ComponentModel::IContainer^  components;

	private: System::Windows::Forms::Label^  detailLabel;
	private: System::Windows::Forms::GroupBox^  graphicGroupBox;
	private: System::Windows::Forms::Label^  distLabel;
	public: System::Windows::Forms::TrackBar^  lodTrackbar;
	public: System::Windows::Forms::TrackBar^  drawDistTrackbar;

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
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(Controls3D::typeid));
			this->pitchTextBox = (gcnew System::Windows::Forms::TextBox());
			this->angle1TrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->angle1Label = (gcnew System::Windows::Forms::Label());
			this->zoomLabel = (gcnew System::Windows::Forms::Label());
			this->zoomTrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->zoomTextBox = (gcnew System::Windows::Forms::TextBox());
			this->angle3Label = (gcnew System::Windows::Forms::Label());
			this->angle3TrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->rollTextBox = (gcnew System::Windows::Forms::TextBox());
			this->angle2Label = (gcnew System::Windows::Forms::Label());
			this->angle2TrackBar = (gcnew System::Windows::Forms::TrackBar());
			this->yawTextBox = (gcnew System::Windows::Forms::TextBox());
			this->showAxesCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->scrollTimer = (gcnew System::Windows::Forms::Timer(this->components));
			this->showCornerAxesCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->fixedSize_checkBox = (gcnew System::Windows::Forms::CheckBox());
			this->lodTrackbar = (gcnew System::Windows::Forms::TrackBar());
			this->detailLabel = (gcnew System::Windows::Forms::Label());
			this->graphicGroupBox = (gcnew System::Windows::Forms::GroupBox());
			this->distLabel = (gcnew System::Windows::Forms::Label());
			this->drawDistTrackbar = (gcnew System::Windows::Forms::TrackBar());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle1TrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->zoomTrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle3TrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle2TrackBar))->BeginInit();
			this->groupBox1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->lodTrackbar))->BeginInit();
			this->graphicGroupBox->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->drawDistTrackbar))->BeginInit();
			this->SuspendLayout();
			// 
			// pitchTextBox
			// 
			this->pitchTextBox->Location = System::Drawing::Point(89, 48);
			this->pitchTextBox->Name = L"pitchTextBox";
			this->pitchTextBox->Size = System::Drawing::Size(100, 20);
			this->pitchTextBox->TabIndex = 0;
			this->pitchTextBox->TextChanged += gcnew System::EventHandler(this, &Controls3D::angleTextBox_TextChanged);
			this->pitchTextBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::TextBox_KeyDown);
			this->pitchTextBox->Leave += gcnew System::EventHandler(this, &Controls3D::angleTextBox_Leave);
			// 
			// angle1TrackBar
			// 
			this->angle1TrackBar->LargeChange = 20;
			this->angle1TrackBar->Location = System::Drawing::Point(196, 48);
			this->angle1TrackBar->Maximum = 361;
			this->angle1TrackBar->Minimum = -361;
			this->angle1TrackBar->Name = L"angle1TrackBar";
			this->angle1TrackBar->Size = System::Drawing::Size(74, 45);
			this->angle1TrackBar->TabIndex = 1;
			this->angle1TrackBar->Scroll += gcnew System::EventHandler(this, &Controls3D::angleTrackBar_Scroll);
			// 
			// angle1Label
			// 
			this->angle1Label->AutoSize = true;
			this->angle1Label->Location = System::Drawing::Point(10, 51);
			this->angle1Label->Name = L"angle1Label";
			this->angle1Label->Size = System::Drawing::Size(34, 13);
			this->angle1Label->TabIndex = 2;
			this->angle1Label->Text = L"Pitch:";
			// 
			// zoomLabel
			// 
			this->zoomLabel->AutoSize = true;
			this->zoomLabel->Location = System::Drawing::Point(10, 136);
			this->zoomLabel->Name = L"zoomLabel";
			this->zoomLabel->Size = System::Drawing::Size(37, 13);
			this->zoomLabel->TabIndex = 5;
			this->zoomLabel->Text = L"Zoom:";
			// 
			// zoomTrackBar
			// 
			this->zoomTrackBar->LargeChange = 20;
			this->zoomTrackBar->Location = System::Drawing::Point(196, 133);
			this->zoomTrackBar->Maximum = 1000;
			this->zoomTrackBar->Name = L"zoomTrackBar";
			this->zoomTrackBar->Size = System::Drawing::Size(74, 45);
			this->zoomTrackBar->TabIndex = 7;
			this->zoomTrackBar->Value = 500;
			this->zoomTrackBar->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &Controls3D::zoomTrackBar_MouseUp);
			// 
			// zoomTextBox
			// 
			this->zoomTextBox->Location = System::Drawing::Point(89, 133);
			this->zoomTextBox->Name = L"zoomTextBox";
			this->zoomTextBox->Size = System::Drawing::Size(100, 20);
			this->zoomTextBox->TabIndex = 6;
			this->zoomTextBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::TextBox_KeyDown);
			this->zoomTextBox->Leave += gcnew System::EventHandler(this, &Controls3D::zoomTextBox_Leave);
			// 
			// angle3Label
			// 
			this->angle3Label->AutoSize = true;
			this->angle3Label->Location = System::Drawing::Point(10, 103);
			this->angle3Label->Name = L"angle3Label";
			this->angle3Label->Size = System::Drawing::Size(28, 13);
			this->angle3Label->TabIndex = 8;
			this->angle3Label->Text = L"Roll:";
			// 
			// angle3TrackBar
			// 
			this->angle3TrackBar->LargeChange = 20;
			this->angle3TrackBar->Location = System::Drawing::Point(196, 103);
			this->angle3TrackBar->Maximum = 1440;
			this->angle3TrackBar->Name = L"angle3TrackBar";
			this->angle3TrackBar->Size = System::Drawing::Size(74, 45);
			this->angle3TrackBar->TabIndex = 5;
			this->angle3TrackBar->Scroll += gcnew System::EventHandler(this, &Controls3D::angleTrackBar_Scroll);
			// 
			// rollTextBox
			// 
			this->rollTextBox->Location = System::Drawing::Point(89, 100);
			this->rollTextBox->Name = L"rollTextBox";
			this->rollTextBox->Size = System::Drawing::Size(100, 20);
			this->rollTextBox->TabIndex = 4;
			this->rollTextBox->TextChanged += gcnew System::EventHandler(this, &Controls3D::angleTextBox_TextChanged);
			this->rollTextBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::TextBox_KeyDown);
			this->rollTextBox->Leave += gcnew System::EventHandler(this, &Controls3D::angleTextBox_Leave);
			// 
			// angle2Label
			// 
			this->angle2Label->AutoSize = true;
			this->angle2Label->Location = System::Drawing::Point(10, 77);
			this->angle2Label->Name = L"angle2Label";
			this->angle2Label->Size = System::Drawing::Size(31, 13);
			this->angle2Label->TabIndex = 11;
			this->angle2Label->Text = L"Yaw:";
			// 
			// angle2TrackBar
			// 
			this->angle2TrackBar->LargeChange = 20;
			this->angle2TrackBar->Location = System::Drawing::Point(195, 74);
			this->angle2TrackBar->Maximum = 1442;
			this->angle2TrackBar->Name = L"angle2TrackBar";
			this->angle2TrackBar->Size = System::Drawing::Size(74, 45);
			this->angle2TrackBar->TabIndex = 3;
			this->angle2TrackBar->Scroll += gcnew System::EventHandler(this, &Controls3D::angleTrackBar_Scroll);
			// 
			// yawTextBox
			// 
			this->yawTextBox->Location = System::Drawing::Point(89, 74);
			this->yawTextBox->Name = L"yawTextBox";
			this->yawTextBox->Size = System::Drawing::Size(100, 20);
			this->yawTextBox->TabIndex = 2;
			this->yawTextBox->TextChanged += gcnew System::EventHandler(this, &Controls3D::angleTextBox_TextChanged);
			this->yawTextBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::TextBox_KeyDown);
			this->yawTextBox->Leave += gcnew System::EventHandler(this, &Controls3D::angleTextBox_Leave);
			// 
			// showAxesCheckBox
			// 
			this->showAxesCheckBox->AutoSize = true;
			this->showAxesCheckBox->Checked = true;
			this->showAxesCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->showAxesCheckBox->Location = System::Drawing::Point(171, 11);
			this->showAxesCheckBox->Name = L"showAxesCheckBox";
			this->showAxesCheckBox->Size = System::Drawing::Size(91, 17);
			this->showAxesCheckBox->TabIndex = 8;
			this->showAxesCheckBox->Text = L"Axes at Origin";
			this->showAxesCheckBox->UseVisualStyleBackColor = true;
			this->showAxesCheckBox->CheckedChanged += gcnew System::EventHandler(this, &Controls3D::showAxesCheckBox_CheckedChanged);
			// 
			// scrollTimer
			// 
			this->scrollTimer->Tick += gcnew System::EventHandler(this, &Controls3D::scrollTimer_Tick);
			// 
			// showCornerAxesCheckBox
			// 
			this->showCornerAxesCheckBox->AutoSize = true;
			this->showCornerAxesCheckBox->Checked = true;
			this->showCornerAxesCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
			this->showCornerAxesCheckBox->Location = System::Drawing::Point(8, 29);
			this->showCornerAxesCheckBox->Name = L"showCornerAxesCheckBox";
			this->showCornerAxesCheckBox->Size = System::Drawing::Size(162, 17);
			this->showCornerAxesCheckBox->TabIndex = 9;
			this->showCornerAxesCheckBox->Text = L"Axes on Bottom-Right Corner";
			this->showCornerAxesCheckBox->UseVisualStyleBackColor = true;
			this->showCornerAxesCheckBox->CheckedChanged += gcnew System::EventHandler(this, &Controls3D::showCornerAxesCheckBox_CheckedChanged);
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->showCornerAxesCheckBox);
			this->groupBox1->Controls->Add(this->zoomTrackBar);
			this->groupBox1->Controls->Add(this->angle3TrackBar);
			this->groupBox1->Controls->Add(this->angle2TrackBar);
			this->groupBox1->Controls->Add(this->angle1Label);
			this->groupBox1->Controls->Add(this->pitchTextBox);
			this->groupBox1->Controls->Add(this->angle1TrackBar);
			this->groupBox1->Controls->Add(this->zoomTextBox);
			this->groupBox1->Controls->Add(this->fixedSize_checkBox);
			this->groupBox1->Controls->Add(this->showAxesCheckBox);
			this->groupBox1->Controls->Add(this->zoomLabel);
			this->groupBox1->Controls->Add(this->rollTextBox);
			this->groupBox1->Controls->Add(this->angle3Label);
			this->groupBox1->Controls->Add(this->angle2Label);
			this->groupBox1->Controls->Add(this->yawTextBox);
			this->groupBox1->Location = System::Drawing::Point(12, 5);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(276, 167);
			this->groupBox1->TabIndex = 16;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"3D View:";
			// 
			// fixedSize_checkBox
			// 
			this->fixedSize_checkBox->AutoSize = true;
			this->fixedSize_checkBox->Location = System::Drawing::Point(171, 29);
			this->fixedSize_checkBox->Name = L"fixedSize_checkBox";
			this->fixedSize_checkBox->Size = System::Drawing::Size(100, 17);
			this->fixedSize_checkBox->TabIndex = 8;
			this->fixedSize_checkBox->Text = L"Fixed Axes Size";
			this->fixedSize_checkBox->UseVisualStyleBackColor = true;
			this->fixedSize_checkBox->CheckedChanged += gcnew System::EventHandler(this, &Controls3D::showAxesCheckBox_CheckedChanged);
			// 
			// lodTrackbar
			// 
			this->lodTrackbar->Location = System::Drawing::Point(86, 19);
			this->lodTrackbar->Maximum = 5;
			this->lodTrackbar->Minimum = 1;
			this->lodTrackbar->Name = L"lodTrackbar";
			this->lodTrackbar->Size = System::Drawing::Size(104, 45);
			this->lodTrackbar->TabIndex = 9;
			this->lodTrackbar->Value = 1;
			this->lodTrackbar->Scroll += gcnew System::EventHandler(this, &Controls3D::lodTrackbar_Scroll);
			// 
			// detailLabel
			// 
			this->detailLabel->AutoSize = true;
			this->detailLabel->Location = System::Drawing::Point(5, 22);
			this->detailLabel->Name = L"detailLabel";
			this->detailLabel->Size = System::Drawing::Size(75, 13);
			this->detailLabel->TabIndex = 10;
			this->detailLabel->Text = L"Level of Detail";
			// 
			// graphicGroupBox
			// 
			this->graphicGroupBox->Controls->Add(this->distLabel);
			this->graphicGroupBox->Controls->Add(this->drawDistTrackbar);
			this->graphicGroupBox->Controls->Add(this->detailLabel);
			this->graphicGroupBox->Controls->Add(this->lodTrackbar);
			this->graphicGroupBox->Location = System::Drawing::Point(13, 178);
			this->graphicGroupBox->Name = L"graphicGroupBox";
			this->graphicGroupBox->Size = System::Drawing::Size(200, 75);
			this->graphicGroupBox->TabIndex = 11;
			this->graphicGroupBox->TabStop = false;
			this->graphicGroupBox->Text = L"Graphics:";
			// 
			// distLabel
			// 
			this->distLabel->AutoSize = true;
			this->distLabel->Location = System::Drawing::Point(6, 56);
			this->distLabel->Name = L"distLabel";
			this->distLabel->Size = System::Drawing::Size(77, 13);
			this->distLabel->TabIndex = 12;
			this->distLabel->Text = L"Draw Distance";
			// 
			// drawDistTrackbar
			// 
			this->drawDistTrackbar->LargeChange = 50;
			this->drawDistTrackbar->Location = System::Drawing::Point(87, 48);
			this->drawDistTrackbar->Maximum = 200;
			this->drawDistTrackbar->Minimum = 1;
			this->drawDistTrackbar->Name = L"drawDistTrackbar";
			this->drawDistTrackbar->Size = System::Drawing::Size(104, 45);
			this->drawDistTrackbar->TabIndex = 11;
			this->drawDistTrackbar->TickStyle = System::Windows::Forms::TickStyle::None;
			this->drawDistTrackbar->Value = 200;
			this->drawDistTrackbar->Scroll += gcnew System::EventHandler(this, &Controls3D::drawDistTrack_Scroll);
			// 
			// Controls3D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->AutoScroll = true;
			this->ClientSize = System::Drawing::Size(300, 293);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->graphicGroupBox);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"Controls3D";
			this->ShowIcon = false;
			this->Text = L"Viewport";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle1TrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->zoomTrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle3TrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle2TrackBar))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->lodTrackbar))->EndInit();
			this->graphicGroupBox->ResumeLayout(false);
			this->graphicGroupBox->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->drawDistTrackbar))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
		// Methods
		protected:
			/**
			 * Method that contains extra constructor steps
			 **/
			void ConsDummy();

		// Events
private: 
	/**
	 * Changes the visibility of the axes in controlledForm->glCanvas3D1
	 **/
	System::Void showAxesCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e);
	/**
	 * Changes the zoom of controlledForm->glCanvas3D1
	 **/
	System::Void zoomTextBox_Leave(System::Object^ sender, System::EventArgs^ e);
	/**
	 * Changes the pitch and yaw of controlledForm->glCanvas3D1
	 **/
	System::Void angleTextBox_Leave(System::Object^ sender, System::EventArgs^ e);
	/**
	 * Changes the values in the angle TextBoxes
	 **/
	System::Void angleTrackBar_Scroll(System::Object^  sender, System::EventArgs^  e);
	/**
	 * Allows the live changing of the zoom
	 **/
	System::Void scrollTimer_Tick(System::Object^  sender, System::EventArgs^  e);
	/**
	 * Calls centerTrackBar with the zoomTrackBar
	 **/
	System::Void zoomTrackBar_MouseUp(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e);
	/**
	 * When triggered by the textBox or a trackBar it should be ignored. Used
	 * when a value is assigned to the textBox->Text
	 **/
	System::Void angleTextBox_TextChanged(System::Object^  sender, System::EventArgs^  e);
	
private: System::Void TextBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
private: System::Void showCornerAxesCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		 public: String ^SerializePreferences(); 
				 void DeserializePreferences(LuaTable ^domainPrefs);
private: System::Void lodTrackbar_Scroll(System::Object^  sender, System::EventArgs^  e);
private: System::Void drawDistTrack_Scroll(System::Object^  sender, System::EventArgs^  e);
public: void SetDefaultParams();
};
}
