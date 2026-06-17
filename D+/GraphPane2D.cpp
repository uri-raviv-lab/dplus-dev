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

System::Void DPlus::GraphPane2D::nmCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	if (suppressUnitEvents) return;
	if (!nmCheckBox->Checked) {
		if (!angCheckBox->Checked) {
			suppressUnitEvents = true;
			nmCheckBox->Checked = true;
			suppressUnitEvents = false;
		}
		return;
	}
	suppressUnitEvents = true;
	angCheckBox->Checked = false;
	suppressUnitEvents = false;
	useAngstrom = false;
	ApplyUnits();
}

System::Void DPlus::GraphPane2D::angCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	if (suppressUnitEvents) return;
	if (!angCheckBox->Checked) {
		if (!nmCheckBox->Checked) {
			suppressUnitEvents = true;
			angCheckBox->Checked = true;
			suppressUnitEvents = false;
		}
		return;
	}
	suppressUnitEvents = true;
	nmCheckBox->Checked = false;
	suppressUnitEvents = false;
	useAngstrom = true;
	ApplyUnits();
}

void DPlus::GraphPane2D::ApplyUnits() {
	double scale = useAngstrom ? 0.1 : 1.0;
	if (sigx != nullptr && sigy != nullptr) {
		array<double>^ dx = gcnew array<double>(sigx->Length);
		for (int i = 0; i < sigx->Length; ++i) dx[i] = sigx[i] * scale;
		graph1D1->Modify(0, dx, sigy);
	}
	if (modx != nullptr && mody != nullptr) {
		array<double>^ dx = gcnew array<double>(modx->Length);
		for (int i = 0; i < modx->Length; ++i) dx[i] = modx[i] * scale;
		graph1D1->Modify(1, dx, mody);
	}
	graph1D1->XLabel = useAngstrom ? L"Reciprocal Space [Å⁻¹]" : L"Reciprocal Space [nm⁻¹]";
	graph1D1->FitToAllGraphs();
	graph1D1->Refresh();
}

DPlus::GraphPane2D::GraphPane2D( MainWindow ^pform )
{
	InitializeComponent();

	bSignalSet = false;
	bModelSet = false;
	sigy = nullptr;
	mody = nullptr;
	sigx = nullptr;
	modx = nullptr;
	useAngstrom = false;
	suppressUnitEvents = false;

	parentForm = pform;

	// Create two graphs: One for the signal, one for the model
	array<double> ^empty = gcnew array<double>(0);

	graph1D1->Add(RGB(255,0,0), Graph1D::GraphDrawType::DRAW_LINES, empty, empty);
	graph1D1->Add(RGB(0,0,255), Graph1D::GraphDrawType::DRAW_LINES, empty, empty);
}

void DPlus::GraphPane2D::SetSignalGraph( array<double> ^x, array<double> ^y )
{
	sigx = x;
	sigy = y;
	double scale = useAngstrom ? 0.1 : 1.0;
	array<double> ^dx = gcnew array<double>(x->Length);
	for (int i = 0; i < x->Length; ++i) dx[i] = x[i] * scale;
	graph1D1->Modify(0, dx, y);

	if(!bSignalSet)
		graph1D1->FitToAllGraphs();

	graph1D1->Refresh();

	bSignalSet = true;

	if(bModelSet) {
		std::vector<double> vsigy = arraytovector(sigy), vmody = arraytovector(mody);
		chiSqrLabel->Text = L"χ² = " + WSSR(vsigy, vmody).ToString("0.######");
		rSqrLabel->Text = L"R² = " + RSquared(vsigy, vmody).ToString("0.######");
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
	sigx = nullptr;

	chiSqrLabel->Text = L"χ² = N/A";
	rSqrLabel->Text = L"R² = N/A";
}

void DPlus::GraphPane2D::GetModelGraph(array<double> ^%x, array<double> ^%y)
{
	graph1D1->GetGraph(1, x, y);
}

void DPlus::GraphPane2D::SetModelGraph(array<double> ^x, array<double> ^y)
{
	modx = x;
	mody = y;
	double scale = useAngstrom ? 0.1 : 1.0;
	array<double> ^dx = gcnew array<double>(x->Length);
	for (int i = 0; i < x->Length; ++i) dx[i] = x[i] * scale;
	graph1D1->Modify(1, dx, y);

	if(!bModelSet)
		graph1D1->FitToAllGraphs();

	graph1D1->ResizeGraph();
	graph1D1->Refresh();

	bModelSet = true;

	if(bSignalSet) {
		std::vector<double> vsigy = arraytovector(sigy), vmody = arraytovector(mody);
		chiSqrLabel->Text = L"χ² = " + WSSR(vsigy, vmody).ToString("0.######");
		rSqrLabel->Text = L"R² = " + RSquared(vsigy, vmody).ToString("0.######");
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
	modx = nullptr;

	chiSqrLabel->Text = L"χ² = N/A";
	rSqrLabel->Text = L"R² = N/A";
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
