#include "FittingPrefsPane.h"
#include "fittingEntity.h"
#include "clrfunctionality.h"

System::Void DPlus::FittingPrefsPane::InitCeresItems()
{
	// Already initialized
	if(parentForm->fittingPrefsTree->Nodes->Count > 0)
		return;

	// Fitting method treeview
	fittingEntity ^lineSearchNode = gcnew fittingEntity(gcnew System::String(MinimizerTypeToCString(MinimizerType_Enum::LINE_SEARCH)), parentForm->fittingPrefsTree);
	
	lineSearchNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(LineSearchTypeToCString(LineSearchType_Enum::ARMIJO)), parentForm->fittingPrefsTree));
	fittingEntity ^wolfeNode = gcnew fittingEntity(gcnew System::String(LineSearchTypeToCString(LineSearchType_Enum::WOLFE)), parentForm->fittingPrefsTree);
	lineSearchNode->Nodes->Add(wolfeNode);

	for each (fittingEntity ^nd in lineSearchNode->Nodes)
	{
		nd->Nodes->Add(gcnew fittingEntity(gcnew System::String(LineSearchDirectionTypeToCString(LineSearchDirectionType_Enum::STEEPEST_DESCENT)), parentForm->fittingPrefsTree));
		fittingEntity ^nlcgNode = gcnew fittingEntity(gcnew System::String(LineSearchDirectionTypeToCString(LineSearchDirectionType_Enum::NONLINEAR_CONJUGATE_GRADIENT)), parentForm->fittingPrefsTree);
		for(int i = 0; i < NonlinearConjugateGradientType_Enum::NonlinearConjugateGradientType_SIZE; i++) {
			nlcgNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(NonlinearConjugateGradientTypeToCString(NonlinearConjugateGradientType_Enum(i))), parentForm->fittingPrefsTree));
		}
		nd->Nodes->Add(nlcgNode);
	}
	// BFGS and L-BFGS are only valid with the Wolfe line search method (i.e. not Armijo)
	{
		wolfeNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(LineSearchDirectionTypeToCString(LineSearchDirectionType_Enum::LBFGS)), parentForm->fittingPrefsTree));
		wolfeNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(LineSearchDirectionTypeToCString(LineSearchDirectionType_Enum::BFGS)), parentForm->fittingPrefsTree));
	}
	parentForm->fittingPrefsTree->Nodes->Add(lineSearchNode);
	
	fittingEntity ^trustRegionNode = gcnew fittingEntity(gcnew System::String(MinimizerTypeToCString(MinimizerType_Enum::TRUST_REGION)), parentForm->fittingPrefsTree);
	trustRegionNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(TrustRegionStrategyTypeToCString(TrustRegionStrategyType_Enum::LEVENBERG_MARQUARDT)), parentForm->fittingPrefsTree));
	
	trustRegionNode->Nodes[0]->CheckState = System::Windows::Forms::CheckState::Checked; // We need a default to prevent users from trying to fit without a selected method.
	auto fe = dynamic_cast<fittingEntity ^>(trustRegionNode->Nodes[0]);
	if (fe)
		fe->checked = true;

	fittingEntity ^doglegNode = gcnew fittingEntity(gcnew System::String(TrustRegionStrategyTypeToCString(TrustRegionStrategyType_Enum::DOGLEG)), parentForm->fittingPrefsTree);
	doglegNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(DoglegTypeToCString(DoglegType_Enum::TRADITIONAL_DOGLEG)), parentForm->fittingPrefsTree));
	doglegNode->Nodes->Add(gcnew fittingEntity(gcnew System::String(DoglegTypeToCString(DoglegType_Enum::SUBSPACE_DOGLEG)), parentForm->fittingPrefsTree));
	trustRegionNode->Nodes->Add(doglegNode);
	parentForm->fittingPrefsTree->Nodes->Add(trustRegionNode);
	
	fittingTreeView->ExpandAll();
	fittingTreeView->Invalidate();


	// Cost function combo box
	for(int i = 0; i < LossFunction_Enum::LossFunction_SIZE; i++) {
		lossFunctionComboBox->Items->Add(gcnew System::String(LossFunctionToCString(LossFunction_Enum(i))));
	}
	lossFunctionComboBox->SelectedIndex = 0;

	// Residual combo box
	for(int i = 0; i < XRayResidualsType_Enum::XRayResidualsType_SIZE; i++) {
		residualsComboBox->Items->Add(gcnew System::String(XRayResidualsTypeToCString(XRayResidualsType_Enum(i))));
	}
	residualsComboBox->SelectedIndex = 0;

}
System::Void DPlus::FittingPrefsPane::FittingPrefsPane_Load( System::Object^ sender, System::EventArgs^ e ) {
	InitCeresItems();
}

CeresProperties DPlus::FittingPrefsPane::GetFittingMethod() {
	auto nodes = parentForm->fittingPrefsTree->Nodes;
	if(nodes->Count == 0)
		return CeresProperties();
	CeresProperties cp;
	fittingEntity^ curNode;
	fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nodes[0]);
	if(!fnd->Checked) {
		fnd = dynamic_cast<fittingEntity^>(nodes[1]);
	}
	curNode = fnd;
	try
	{
		cp.minimizerType = MinimizerTypefromCString(clrToString(curNode->name).c_str());
	}
	catch (Exception^ e){
		throw gcnew UserInputException("Please choose a valid minimizer type");
	}
	if(cp.minimizerType == LINE_SEARCH)
	{
		// Armijo
		fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[0]);
		if(fnd && fnd->Checked) {
			curNode = fnd;
		}
		else
		{
			// Wolfe
			fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[1]);
			if (fnd && fnd->Checked) {
				curNode = fnd;
			}
			else
			{
				MessageBox::Show("There seems to be a problem with the selected fitting method.", "ERROR", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
		try
		{
			cp.lineSearchType = LineSearchTypefromCString(clrToString(curNode->name).c_str());
		}
		catch (Exception^ e){
			throw gcnew UserInputException("Please choose a valid line search type");
		}

		for(int i = 0; i < curNode->Nodes->Count; i++) {
			fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[i]);
			if(fnd) {
				if(fnd->Checked) {
					curNode = fnd;
					break;
				}
			}
		}
		try
		{
			cp.lineSearchDirectionType = LineSearchDirectionTypefromCString(clrToString(curNode->name).c_str());
		}
		catch (Exception^ e){
			throw gcnew UserInputException("Please choose a valid line search direction type");
		}

		if(cp.lineSearchDirectionType == NONLINEAR_CONJUGATE_GRADIENT) {
			for(int i = 0; i < curNode->Nodes->Count; i++) {
				fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[i]);
				if(fnd) {
					if(fnd->Checked) {
						curNode = fnd;
						break;
					}
				}
			}
			try
			{
				cp.nonlinearConjugateGradientType = NonlinearConjugateGradientTypefromCString(clrToString(curNode->name).c_str());
			}
			catch (Exception^ e){
				throw gcnew UserInputException("Please choose a valid non linear conjugate gradient type");
			}
		}
	}
	else // TRUST_REGION
	{
		for(int i = 0; i < curNode->Nodes->Count; i++) {
			fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[i]);
			if(fnd) {
				if(fnd->Checked) {
					curNode = fnd;
					break;
				}
			}
		}
		try
		{
			cp.trustRegionStrategyType = TrustRegionStrategyTypefromCString(clrToString(curNode->name).c_str());
		}
		catch (Exception^ e){
			throw gcnew UserInputException("Please choose a valid trust region strategy type");
		}
		if(cp.trustRegionStrategyType == DOGLEG) {
			for(int i = 0; i < curNode->Nodes->Count; i++) {
				fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[i]);
				if(fnd) {
					if(fnd->Checked) {
						curNode = fnd;
						break;
					}
				}
			}
			try
			{
				cp.doglegType = DoglegTypefromCString(clrToString(curNode->name).c_str());
			}
			catch (Exception^ e){
				throw gcnew UserInputException("Please choose a valid dogleg type");
			}
		}


	}

	if (cp.trustRegionStrategyType >= TrustRegionStrategyType_SIZE ||
		cp.doglegType >= DoglegType_SIZE ||
		cp.lineSearchDirectionType >= LineSearchDirectionType_SIZE ||
		cp.minimizerType >= MinimizerType_SIZE ||
		cp.nonlinearConjugateGradientType >= NonlinearConjugateGradientType_SIZE)
	{
		throw gcnew UserInputException("Please choose a valid fitting method in the Fitting Preferences Pane.");
	}

	try
	{
		cp.lossFuncType = LossFunctionfromCString(clrToString(lossFunctionComboBox->SelectedItem->ToString()).c_str());
	}
	catch (Exception^ e){
		throw gcnew UserInputException("Please choose a valid loss function type");
	}

	try
	{
		cp.residualType = XRayResidualsTypefromCString(clrToString(residualsComboBox->SelectedItem->ToString()).c_str());
	}
	catch (Exception^ e){
		throw gcnew UserInputException("Please choose a valid residual type");
	}

	double res;
	if(!Double::TryParse(lossFunctionTextBox1->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Loss Function parameter 1\" field ("+ lossFunctionTextBox1->Text + ") in the Fitting Preferences Pane must be a valid number.");
	}

	cp.lossFunctionParameters[0] = res;
	if (!Double::TryParse(lossFunctionTextBox2->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Loss Function parameter 2\" field in the Fitting Preferences Pane must be a valid number.");
	}
	cp.lossFunctionParameters[1] = res;

	if (!Double::TryParse(convergenceTextBox->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Convergence\" field in the Fitting Preferences Pane must be a valid number.");
	}
	if (res < 0)
	{
		throw gcnew UserInputException("The text in the \"Convergence\" field in the Fitting Preferences Pane must be a possitive number.");
	}
	cp.fittingConvergence = res;

	if (!Double::TryParse(stepSizeTextBox->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Step Size\" field in the Fitting Preferences Pane must be a valid number.");
	}
	
	cp.derivativeStepSize = res;
	if (!Double::TryParse(derEpsTextBox->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Der eps\" field in the Fitting Preferences Pane must be a valid number.");
	}
	cp.derivativeEps = res;

	if (!Double::TryParse(iterationsTextBox->Text, res))
	{
		throw gcnew UserInputException("The text in the \"Iterations\" field in the Fitting Preferences Pane must be a valid number.");
	}

	return cp;

}

System::Void DPlus::FittingPrefsPane::fittingTreeView_MouseDown( System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e ) {
}

void DPlus::FittingPrefsPane::IsNodeLeaf( System::Object^ sender, NodeControls::NodeControlValueEventArgs^ e ) {
	e->Value = e->Node->Children->Count == 0;
}

System::Void DPlus::FittingPrefsPane::lossFunctionComboBox_SelectedIndexChanged( System::Object^ sender, System::EventArgs^ e )
{
	int numPars = LossFunctionNumberOfParameters(
		LossFunctionfromCString( clrToString(lossFunctionComboBox->SelectedItem->ToString()).c_str() ) );
	
	lossFunctionTextBox1->Enabled = 
		(numPars > 0) ? true : false;

	lossFunctionTextBox2->Enabled = 
		(numPars > 1) ? true : false;
}

System::Void DPlus::FittingPrefsPane::textBox_Leave( System::Object^ sender, System::EventArgs^ e ) {
	TextBox ^source = dynamic_cast<TextBox^>(sender);
	double res;

	if(Double::TryParse(source->Text, res) || source->Text->StartsWith("=")) {
		if(source->Text->StartsWith("="))		
			res = parentForm->LuaParseExpression(source->Text->Substring(1));

		source->Text = Double(res).ToString();

		if(!(Double::TryParse(source->Text, res)))
			return;

	}
}

void DPlus::FittingPrefsPane::SetFittingMethod( CeresProperties& dp ) {
	convergenceTextBox->Text	= Double(dp.fittingConvergence).ToString();
	stepSizeTextBox->Text		= Double(dp.derivativeStepSize).ToString();
	derEpsTextBox->Text			= Double(dp.derivativeEps).ToString();

	lossFunctionTextBox1->Text = Double(dp.lossFunctionParameters[0]).ToString();
	lossFunctionTextBox2->Text = Double(dp.lossFunctionParameters[1]).ToString();
	lossFunctionComboBox->Text = gcnew System::String(LossFunctionToCString(dp.lossFuncType));

	residualsComboBox->Text = gcnew System::String(XRayResidualsTypeToCString(dp.residualType));

	auto nodes = parentForm->fittingPrefsTree->Nodes;

	if(nodes->Count == 0)
		return;


	if (dp.minimizerType >= MinimizerType_SIZE || dp.minimizerType < 0) // The field was left blank
		return;

	for each (Node^ nd in nodes)
	{
		fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nd);
		if(fnd) {
			fnd->ClearChecked();
		}
	}

	auto curNode = (parentForm->fittingPrefsTree->Root);

#define ADD_TO_PATH(NAME)															\
	do {																			\
		for(int i = 0; i < curNode->Nodes->Count; i++) {							\
			fittingEntity^ fnd = dynamic_cast<fittingEntity^>(curNode->Nodes[i]);	\
			if(fnd)	{																\
				if(fnd->name->Equals(gcnew System::String(NAME)))					\
					curNode = curNode->Nodes[i];									\
			}																		\
		}																			\
	} while(false)


	ADD_TO_PATH(MinimizerTypeToCString(dp.minimizerType));

	if(dp.minimizerType == TRUST_REGION) {
		ADD_TO_PATH(TrustRegionStrategyTypeToCString(dp.trustRegionStrategyType));
		if(dp.trustRegionStrategyType == DOGLEG) {
			ADD_TO_PATH(DoglegTypeToCString(dp.doglegType));
		}
	}
	else // line search
	{
		// Armijo or Wolfe
		ADD_TO_PATH(LineSearchTypeToCString(dp.lineSearchType));

		ADD_TO_PATH(LineSearchDirectionTypeToCString(dp.lineSearchDirectionType));
			
		if(dp.lineSearchDirectionType == NONLINEAR_CONJUGATE_GRADIENT) {
			ADD_TO_PATH(NonlinearConjugateGradientTypeToCString(dp.nonlinearConjugateGradientType));
		}
	}

#undef ADD_TO_PATH

	fittingEntity^ ccurNode = dynamic_cast<fittingEntity^>(curNode);
	if(ccurNode)
	{
		ccurNode->IsChecked = true;
		ccurNode->CheckParents();
	}
}

System::String^ DPlus::FittingPrefsPane::SerializePreferences()
{
	if(this->InvokeRequired) {
		return ((String ^)(this->Invoke(gcnew FuncNoParamsReturnString(this, &FittingPrefsPane::SerializePreferences))));
	}

	CeresProperties cp = GetFittingMethod();

	String ^contents = "";

	contents += "FittingPreferences = {\n";

#define SET_TYPE_STRING(TAR, TYPE)	\
	contents += "\t" #TYPE " = [[" +  gcnew System::String(TYPE##ToCString(TAR)) + "]],\n"

	SET_TYPE_STRING(cp.minimizerType,					MinimizerType);
	SET_TYPE_STRING(cp.lineSearchDirectionType,			LineSearchDirectionType);
	SET_TYPE_STRING(cp.lineSearchType,					LineSearchType);
	SET_TYPE_STRING(cp.trustRegionStrategyType,			TrustRegionStrategyType);
	SET_TYPE_STRING(cp.doglegType,						DoglegType);
	SET_TYPE_STRING(cp.nonlinearConjugateGradientType,	NonlinearConjugateGradientType);
	SET_TYPE_STRING(cp.lossFuncType,					LossFunction);
	SET_TYPE_STRING(cp.residualType,					XRayResidualsType);

#undef SET_TYPE_STRING

	contents += "\tLossFuncPar1 = "			+ lossFunctionTextBox1->Text + ",\n";
	contents += "\tLossFuncPar2 = "			+ lossFunctionTextBox2->Text + ",\n";
	contents += "\tFittingIterations = "	+ iterationsTextBox->Text	 + ",\n";

	contents += "\tConvergence = "			+ convergenceTextBox->Text	 + ",\n";
	contents += "\tStepSize = "				+ stepSizeTextBox->Text		 + ",\n";
	contents += "\tDerEps = "				+ derEpsTextBox->Text		 + ",\n";
	contents += "};\n";

	return contents;

}

void DPlus::FittingPrefsPane::DeserializePreferences( LuaTable ^contents )
{
	if(contents == nullptr) // Load defaults
		return;

	// Needed when called from MainWindow_Load
	InitCeresItems();

	CeresProperties cp = GetFittingMethod();

#define GET_TYPE_STRING(TAR, TYPE)	\
	if(contents[#TYPE] != nullptr)	\
		TAR = TYPE##fromCString(clrToString(dynamic_cast<String ^>(contents[#TYPE])).c_str())

	GET_TYPE_STRING(cp.minimizerType,					MinimizerType);
	GET_TYPE_STRING(cp.lineSearchDirectionType,			LineSearchDirectionType);
	GET_TYPE_STRING(cp.lineSearchType,					LineSearchType);
	GET_TYPE_STRING(cp.trustRegionStrategyType,			TrustRegionStrategyType);
	GET_TYPE_STRING(cp.doglegType,						DoglegType);
	GET_TYPE_STRING(cp.nonlinearConjugateGradientType,	NonlinearConjugateGradientType);
	GET_TYPE_STRING(cp.lossFuncType,					LossFunction);
	GET_TYPE_STRING(cp.residualType,					XRayResidualsType);

#undef GET_TYPE_STRING

	if(contents["LossFuncPar1"] != nullptr)
		cp.lossFunctionParameters[0] = LuaItemToDouble(contents["LossFuncPar1"]);

	if(contents["LossFuncPar2"] != nullptr)
		cp.lossFunctionParameters[1] = LuaItemToDouble(contents["LossFuncPar2"]);


	SetFittingMethod(cp);

	if(contents["FittingIterations"] != nullptr)
		iterationsTextBox->Text = Int32(LuaItemToDouble(contents["FittingIterations"])).ToString();

	if(contents["Convergence"] != nullptr)
		convergenceTextBox->Text = Double(LuaItemToDouble(contents["Convergence"])).ToString();

	if(contents["StepSize"] != nullptr)
		stepSizeTextBox->Text = Double(LuaItemToDouble(contents["StepSize"])).ToString();

	if(contents["DerEps"] != nullptr)
		derEpsTextBox->Text = Double(LuaItemToDouble(contents["DerEps"])).ToString();

}

void DPlus::FittingPrefsPane::SetDefaultParams(){

	this->nodeCheckBox1->EditEnabled = true;
	this->lossFunctionTextBox1->Text = L"0.5";
	this->lossFunctionTextBox2->Text = L"0.5";
	this->iterationsTextBox->Text = L"20";
	this->stepSizeTextBox->Text = L"0.01";
	this->convergenceTextBox->Text = L"0.1";
	this->derEpsTextBox->Text = L"0.1";
	this->lossFunctionComboBox->SelectedIndex = 0;
	this->residualsComboBox->SelectedIndex = 0;

	auto nodes = parentForm->fittingPrefsTree->Nodes;
	for each (Node^ nd in nodes)
	{
		fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nd);
		if (fnd) {
			fnd->ClearChecked();
		}
	}

	if (nodes->Count > 0)
	{
		CeresProperties cp;
		fittingEntity^ curNode;
		fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nodes[1]);
		fnd->Nodes[0]->CheckState = System::Windows::Forms::CheckState::Checked;
		auto fe = dynamic_cast<fittingEntity ^>(fnd->Nodes[0]);
		if (fe)
			fe->checked = true;
	}
}