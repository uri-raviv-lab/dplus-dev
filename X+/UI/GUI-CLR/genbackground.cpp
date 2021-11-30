/*
//#include "genbackground.h"	// TODO::BG
//#include "calculation_external.h"
//#include "FrontendExported.h"
#include "ExtractBaseline.h"
#include "DUMMY_HEADER_FILE.h"

using GUICLR::ExtractBaseline;

bool GenerateBackground(const std::wstring datafile, std::vector <double>& bgx,
						std::vector <double>& bgy, std::vector <double>& ffy, bool angstrom) {

	int ndata;
	vector<double> datax, datay;
	vector<double> sx, sy;
	std::wstring outputFile;
	std::wstring workspace, basename;
	wchar_t temp[260] = {0};
	
	GetDirectory(datafile.c_str(), temp);
	workspace = temp;
	wcsnset(temp, 0, 260);
	GetBasename(datafile.c_str(), temp);
	basename = temp;
	

    outputFile = basename + L"-baseline.out";

    datax.clear(); datay.clear();
	sx.clear(); sy.clear();

	ReadDataFile(datafile.c_str(), datax, datay);
	ndata = datax.size();

	ExtractBaseline base(datafile.c_str(), (workspace + outputFile).c_str(), angstrom);

	if(base.ShowDialog() != System::Windows::Forms::DialogResult::OK)
		return false;
	
	if(base._bHasBaseline && !base.bAutoBaseline())
		GenerateBGLinesandFormFactor(datafile.c_str(), (workspace + outputFile).c_str(), bgx, bgy, ffy,angstrom);
	// Automated Baseline
    else if (base.bAutoBaseline()) {
		base.getCroppedSignal(bgx, ffy);
		base.getAutoBaseline(bgy);

		for (int i = 0; i < (int)bgx.size(); i++)
			ffy[i] = GetMinimumSig() + ffy[i] - bgy[i];
	}
	else {
		base.getCroppedSignal(bgx, ffy);
		for (int i = 0; i < (int)bgx.size(); i++)
			ffy[i] = GetMinimumSig() + ffy[i];
		bgy = ffy;
	}
	return true;
}


*/
