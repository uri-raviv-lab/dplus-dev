#include <windows.h> // For LoadLibrary etc.

#include "ExternalModelDialog.h"

namespace GUICLR {

	void ExternalModelDialog::FreeModelContainer() {
		if(hModule)
			FreeLibrary(hModule);

		hModule = NULL;

		// Trust the other windows to delete selectedmodel
		_selectedModel = NULL;
	}

	bool ExternalModelDialog::LoadModelContainer() {
		cats = lf->QueryCategoryCount(container);
		
		return (cats > 0);

// 		NumModels = NULL;
// 		ModelName = NULL;
// 		GetModelF = NULL;
// 
// 		// Free previous model container
// 		FreeModelContainer();
// 
// 		hModule = LoadLibrary(clrToWstring(filename).c_str());
// 		if(hModule == NULL) {
// 			MessageBox::Show("Invalid library");
// 			return false;
// 		}
// 
// 		NumModels = (numModelsFunc)GetProcAddress(hModule, "GetNumModels");
// 		ModelName = (modelNameFunc)GetProcAddress(hModule, "GetModelName");
// 		GetModelF = (getModelFunc) GetProcAddress(hModule, "GetModel");
// 		if(NumModels && ModelName && GetModelF)
// 			return true;
// 		
// 		// Handle failures
// 		if(!NumModels)
// 			MessageBox::Show("Invalid model container");
// 		else if(!ModelName)
// 			MessageBox::Show("Invalid model container (2)");
// 		else if(!GetModelF)
// 			MessageBox::Show("Invalid model container (3)");
// 		
// 		NumModels = NULL;
// 		ModelName = NULL;
// 		GetModelF = NULL;
// 		FreeModelContainer();
// 		
// 		return false;
	}

	void ExternalModelDialog::LoadDefaultModels() {
		GetAllModels();
	}
	
	bool ExternalModelDialog::GetAllModels() {
		models->Items->Clear();
		models->Enabled = false;

		// Try to load the model container
		if(!lf->IsValid())
			return false;

		// Populate the combobox
		cats = lf->QueryCategoryCount(container);

		for(int i = 0; i < cats; i++) {
			ModelCategory mc = lf->QueryCategory(container, i);
			if(mc.type == MT_FORMFACTOR) {
				for(int j = 0; j < 16; j++) {
					if(mc.models[j] == -1)
						break;

					(*modelInfos).push_back(lf->QueryModel(container, mc.models[j]));
					models->Items->Add(stringToClr((*modelInfos).at((*modelInfos).size() - 1).name));
				}
			}
		}

		models->Enabled = (models->Items->Count > 0);
		if(models->Enabled)
			models->SelectedIndex = 0;
		return models->Enabled;
	}
	
	void ExternalModelDialog::models_SelectedIndexChanged(System::Object^  sender, 
														  System::EventArgs^  e) {
		if(_selectedModel)
			delete _selectedModel;
		_selectedModel = NULL;

// 		// If it's a default model, get it from the default functions
// 		if(!hModule) {
// 			_selectedModel = dynamic_cast<FFModel *>(GetModel(models->SelectedIndex));
// 			ok->Enabled = (_selectedModel != NULL);
// 			return;
// 		}
// 
// 		// Try to get the model and downcast it to form factor
//		_selectedModel = dynamic_cast<FFModel *>(GetModelF(models->SelectedIndex));
		*_selectedModel = lf->QueryModel(container, models->SelectedIndex /*TODO::ChangeModel THIS IS NOT THE RIGHT INDEX*/);

		ok->Enabled = (_selectedModel != NULL);
	}

};
