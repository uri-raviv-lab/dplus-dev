#include <windows.h> // For COLORREF

#include "GraphPane2D.h"
#include "clrfunctionality.h"
#include "FrontendExported.h"

#include <vector>

using namespace GraphToolkit;

void DPlus::GraphPane2D::SetYAxisText(System::String^ yAxis) {
	this->graph1D1->YLabel = yAxis;
}

void DPlus::GraphPane2D::SetXAxisText(System::String^ xAxis) {
	this->graph1D1->XLabel = xAxis;
}

void DPlus::GraphPane2D::SetGraphTitleText(System::String^ title) {
	this->graph1D1->GraphTitle = title;
}

System::Void DPlus::GraphPane2D::logQcheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	this->graph1D1->LogScaleX = logQcheckBox->Checked;
}

System::Void DPlus::GraphPane2D::logIcheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	this->graph1D1->LogScaleY = logIcheckBox->Checked;
}

DPlus::GraphPane2D::GraphPane2D( MainWindow ^pform )
{
	InitializeComponent();

	bSignalSet = false;
	bModelSet = false;
	sigy = nullptr;
	mody = nullptr;
	
	parentForm = pform;

	// Create two graphs: One for the signal, one for the model
	array<double> ^empty = gcnew array<double>(0);

	graph1D1->Add(RGB(255,0,0), Graph1D::GraphDrawType::DRAW_LINES, empty, empty);
	graph1D1->Add(RGB(0,0,255), Graph1D::GraphDrawType::DRAW_LINES, empty, empty);
}

void DPlus::GraphPane2D::SetSignalGraph( array<double> ^x, array<double> ^y )
{
	graph1D1->Modify(0, x, y);

	if(!bSignalSet)
		graph1D1->FitToAllGraphs();

	graph1D1->Refresh();

	bSignalSet = true;
	sigy = y;

	if(bModelSet) {
		std::vector<double> vsigy = arraytovector(sigy), vmody = arraytovector(mody);
		chiSqrLabel->Text = "chi^2 = " + WSSR(vsigy, vmody).ToString("0.######");
		rSqrLabel->Text = "R^2 = " + RSquared(vsigy, vmody).ToString("0.######");
	}
}

void DPlus::GraphPane2D::ClearSignalGraph()
{
	array<double> ^empty = gcnew array<double>(0);
	graph1D1->Modify(0, empty, empty);
	
	if(bSignalSet)
		graph1D1->FitToAllGraphs();

	graph1D1->Refresh();

	bSignalSet = false;
	sigy = nullptr;

	chiSqrLabel->Text = "chi^2 = N/A";
	rSqrLabel->Text = "R^2 = N/A";
}

void DPlus::GraphPane2D::GetModelGraph(array<double> ^%x, array<double> ^%y)
{
	graph1D1->GetGraph(1, x, y);
}

void DPlus::GraphPane2D::SetModelGraph(array<double> ^x, array<double> ^y)
{
	graph1D1->Modify(1, x, y);

	if(!bModelSet)
		graph1D1->FitToAllGraphs();

	graph1D1->ResizeGraph();
	graph1D1->Refresh();

	bModelSet = true;
	mody = y;

	if(bSignalSet) {
		std::vector<double> vsigy = arraytovector(sigy), vmody = arraytovector(mody);
		chiSqrLabel->Text = "chi^2 = " + WSSR(vsigy, vmody).ToString("0.######");
		rSqrLabel->Text = "R^2 = " + RSquared(vsigy, vmody).ToString("0.######");
	}
}

void DPlus::GraphPane2D::ClearModelGraph()
{
	array<double> ^empty = gcnew array<double>(0);
	graph1D1->Modify(1, empty, empty);

	if(bModelSet)
		graph1D1->FitToAllGraphs();

	graph1D1->Refresh();

	bModelSet = false;
	mody = nullptr;

	chiSqrLabel->Text = "chi^2 = N/A";
	rSqrLabel->Text = "R^2 = N/A";
}

System::Void DPlus::GraphPane2D::graph1D1_MouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
	GraphToolkit::DoublePair dp = graph1D1->PointToData(e->X, e->Y);
	double tempx = dp.first;
	double tempy = dp.second;
	if (graph1D1->bLogScaleX)
		tempx = pow(10, tempx);
	if (graph1D1->bLogScaleY)
		tempy = pow(10, tempy);
	locationLabel->Text = "(" + Double(tempx).ToString("0.######") + ", " + Double(tempy).ToString("0.######") + ")";
}
