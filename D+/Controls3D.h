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
	public: System::Windows::Forms::Button^  generateButton;
	private:



	public: System::Windows::Forms::CheckBox^  showCornerAxesCheckBox;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	public: System::Windows::Forms::Button^  fitButton;
	private:
	public:


	public: System::Windows::Forms::Button^  stopButton;
	private: System::Windows::Forms::Label^  label1;
	public: System::Windows::Forms::TextBox^  scaleBox;
	private: 
	public: System::Windows::Forms::CheckBox^  scaleMut;
	public: System::Windows::Forms::CheckBox^  constantMut;

	public: System::Windows::Forms::TextBox^  constantBox;

	private: System::Windows::Forms::Label^  label2;
	public: System::Windows::Forms::CheckBox^  fixedSize_checkBox;

	public:
	public:
	public: 

	private: 
	private: 

	private: 
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
	public: 

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
			this->generateButton = (gcnew System::Windows::Forms::Button());
			this->showCornerAxesCheckBox = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->fixedSize_checkBox = (gcnew System::Windows::Forms::CheckBox());
			this->fitButton = (gcnew System::Windows::Forms::Button());
			this->stopButton = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->scaleBox = (gcnew System::Windows::Forms::TextBox());
			this->scaleMut = (gcnew System::Windows::Forms::CheckBox());
			this->constantMut = (gcnew System::Windows::Forms::CheckBox());
			this->constantBox = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle1TrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->zoomTrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle3TrackBar))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle2TrackBar))->BeginInit();
			this->groupBox1->SuspendLayout();
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
			// generateButton
			// 
			this->generateButton->Location = System::Drawing::Point(12, 58);
			this->generateButton->Name = L"generateButton";
			this->generateButton->Size = System::Drawing::Size(73, 23);
			this->generateButton->TabIndex = 30;
			this->generateButton->Text = L"Generate";
			this->generateButton->UseVisualStyleBackColor = true;
			this->generateButton->Click += gcnew System::EventHandler(this, &Controls3D::generateButton_Click);
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
			this->groupBox1->Location = System::Drawing::Point(12, 114);
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
			// fitButton
			// 
			this->fitButton->Location = System::Drawing::Point(12, 87);
			this->fitButton->Name = L"fitButton";
			this->fitButton->Size = System::Drawing::Size(73, 23);
			this->fitButton->TabIndex = 31;
			this->fitButton->Text = L"Fit";
			this->fitButton->UseVisualStyleBackColor = true;
			this->fitButton->Click += gcnew System::EventHandler(this, &Controls3D::fitButton_Click);
			// 
			// stopButton
			// 
			this->stopButton->Enabled = false;
			this->stopButton->Location = System::Drawing::Point(91, 87);
			this->stopButton->Name = L"stopButton";
			this->stopButton->Size = System::Drawing::Size(73, 23);
			this->stopButton->TabIndex = 32;
			this->stopButton->Text = L"Stop";
			this->stopButton->UseVisualStyleBackColor = true;
			this->stopButton->Click += gcnew System::EventHandler(this, &Controls3D::stopButton_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 10);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(76, 13);
			this->label1->TabIndex = 16;
			this->label1->Text = L"Domain Scale:";
			// 
			// scaleBox
			// 
			this->scaleBox->Location = System::Drawing::Point(105, 7);
			this->scaleBox->Name = L"scaleBox";
			this->scaleBox->Size = System::Drawing::Size(90, 20);
			this->scaleBox->TabIndex = 18;
			this->scaleBox->Text = L"1";
			this->scaleBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::textBox_KeyDown);
			this->scaleBox->Leave += gcnew System::EventHandler(this, &Controls3D::textBox_Leave);
			// 
			// scaleMut
			// 
			this->scaleMut->AutoSize = true;
			this->scaleMut->Location = System::Drawing::Point(201, 9);
			this->scaleMut->Name = L"scaleMut";
			this->scaleMut->Size = System::Drawing::Size(64, 17);
			this->scaleMut->TabIndex = 19;
			this->scaleMut->Text = L"Mutable";
			this->scaleMut->UseVisualStyleBackColor = true;
			// 
			// constantMut
			// 
			this->constantMut->AutoSize = true;
			this->constantMut->Location = System::Drawing::Point(201, 35);
			this->constantMut->Name = L"constantMut";
			this->constantMut->Size = System::Drawing::Size(64, 17);
			this->constantMut->TabIndex = 21;
			this->constantMut->Text = L"Mutable";
			this->constantMut->UseVisualStyleBackColor = true;
			// 
			// constantBox
			// 
			this->constantBox->Location = System::Drawing::Point(105, 33);
			this->constantBox->Name = L"constantBox";
			this->constantBox->Size = System::Drawing::Size(90, 20);
			this->constantBox->TabIndex = 20;
			this->constantBox->Text = L"0";
			this->constantBox->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &Controls3D::textBox_KeyDown);
			this->constantBox->Leave += gcnew System::EventHandler(this, &Controls3D::textBox_Leave);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(12, 36);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(91, 13);
			this->label2->TabIndex = 23;
			this->label2->Text = L"Domain Constant:";
			// 
			// Controls3D
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->AutoScroll = true;
			this->ClientSize = System::Drawing::Size(300, 287);
			this->Controls->Add(this->constantMut);
			this->Controls->Add(this->constantBox);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->scaleMut);
			this->Controls->Add(this->scaleBox);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->generateButton);
			this->Controls->Add(this->stopButton);
			this->Controls->Add(this->fitButton);
			this->Controls->Add(this->groupBox1);
			this->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->HideOnClose = true;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Name = L"Controls3D";
			this->ShowIcon = false;
			this->Text = L"Controls";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle1TrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->zoomTrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle3TrackBar))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->angle2TrackBar))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

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
private: System::Void generateButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void showCornerAxesCheckBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
private: System::Void fitButton_Click(System::Object^  sender, System::EventArgs^  e);
private: System::Void stopButton_Click(System::Object^  sender, System::EventArgs^  e);
		 public: String ^SerializePreferences(); 
				 void DeserializePreferences(LuaTable ^domainPrefs);
private: System::Void textBox_Leave(System::Object^  sender, System::EventArgs^  e);
private: System::Void textBox_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
public:
	void SetDefaultParams();
};
}
