#include <cstdio>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep((x) * 1000)
#else
#include <unistd.h>
#endif

#include "Common.h"

#include "Statistics.h" // For WSSR
#include "FrontendExported.h"    // For file management

// This UI does not use the frontend, instead, it uses the local communication
// to communicate directly with the backend.
#include "LocalComm.h"

#include "ModelUI.h"

#ifdef _DEBUG
	#include <conio.h>
#endif

#ifndef MAX_PATH
#define MAX_PATH 256
#endif

int const PBAR_SIZE = 40;

void __stdcall printProgress(void *args, double progress) {
    if(progress < 0.0)   progress = 0.0;
    if(progress > 1.0)   progress = 1.0;
	progress *= 100.0;
        
    printf("\r[");
    for(int i = 0; i < PBAR_SIZE; i++) {
        if(i < (int)((progress * PBAR_SIZE) / 100.0))
            printf("#");
        else
            printf(" ");
    }
    printf("] %.1f%%", progress);
    fflush(stdout);
}


/**
 * Prints the available model types and lets you choose one
 **/
ModelInformation InputModelType(FrontendComm *lf, const wchar_t *container) {
    char answer[16] = {0};
    printf("Available model types:\n");

	std::vector<ModelInformation> models;

	// Query all form factor models and put into a map
	int cats = lf->QueryCategoryCount(container);
	for(int i = 0; i < cats; i++) {
		ModelCategory mc = lf->QueryCategory(container, i);
		if(mc.type == MT_FORMFACTOR) {
			for(int j = 0; j < 16; j++) {
				if(mc.models[j] == -1)
					break;

				ModelInformation grr = lf->QueryModel(container, mc.models[j]);
				models.push_back(grr);
			}
		}
	}

	
	for(int i = 0; i < (int)models.size(); i++)
		printf("\t%d. %s\n", i + 1, models[i].name);

	int ind = -1;
	while(ind < 0 || ind > (int)models.size() - 1) {
		printf("Which model would you prefer? ");
		scanf("%s", answer);

		ind = (atoi(answer) - 1);
	}

	return models[ind];
}


/**
 * If there is a Gaussian ED model, allows the user to choose whether
 * or not the user would like to use it.
 **/
EDProfile InputEDType() {
	//if we don't have a Gaussian ED model
	//if(!(modelT == MODEL_SLAB)) 
		return EDProfile();

		/*
	char answer[16] = {0};
    printf("Three types of models exist:\n");
    printf("\t1. Discrete electron density profile\n"
           "\t2. Gaussian electron density profile\n"
           "Which model would you prefer? "
        );
    scanf("%s", answer);

    return (atoi(answer) == 2);
	*/
}
/**
 * Prints the available peak types and lets you choose one
 **/
PeakType InputPeakType() {
    char answer[16] = {0};
    printf("Available peak types:\n");
    printf("\t1. Gaussians\n"
           "\t2. Lorentzians\n"
           "\t3. Lorentzians Squared\n"
           //"\t4. Caille\n"
           "Which model would you prefer? "
        );
    scanf("%s", answer);

    switch(atoi(answer)) {
    default:
    case 1:
        return SHAPE_GAUSSIAN;
    case 2:
        return SHAPE_LORENTZIAN;
    case 3:
		return SHAPE_LORENTZIAN_SQUARED;
    //case 4:
    //    return SHAPE_CAILLE;
    }
}

/**
 * Simple helper function which empties the fit range vectors so that the fitter
 * won't crash
 **/
void setEmptyVectors(paramStruct *p, int layers) {
}

void requestData(const char *question, Parameter& param) {
    char answer[100] = {0};
    
    printf("Input %s ", question);
    scanf("%s", answer);
	param.value = strtod(answer, NULL);

    printf("Mutable? [y/n] ");
    scanf("%s", answer);
	param.isMutable = (tolower(answer[0]) == 'y');
}

paramStruct InputInitialGuess(FrontendComm *lf, wchar_t *container, ModelInformation type) {
    paramStruct result;
    int layers = 0;
    char question[100] = {0}, curLayer[100] = {0};
    
	do {
		char buf[32] = {0};
		itoa(type.maxLayers, buf, 10);

		printf("Input number of layers (%d - %s): ", (type.minLayers < 0 ? 0 : type.minLayers),
			   (type.maxLayers < 0 ? "inf" : buf));
		scanf("%s", question);
		layers = atoi(question);		
	} while((type.minLayers >= 0 && layers < type.minLayers) && 
		    (type.maxLayers >= 0 && layers > type.maxLayers));

	result.layers = layers;
	result.nlp = type.nlp;
	result.nExtraParams = type.nExtraParams;

	// Clear layer parameter vectors
	result.params.resize(type.nlp);
	for(int i = 0; i < type.nlp; i++)
		result.params[i].resize(layers);
	
	char **lpNames = new char*[type.nlp];
	for(int i = 0; i < type.nlp; i++)
		lpNames[i] = new char[256];
	int *app = new int[type.nlp];

	lf->GetLayerParamNames(container, type.modelIndex, lpNames, type.nlp);	

	// Get layer parameters
    for(int i = 0; i < layers; i++) {		
		char layerName[256] = {0};

		lf->GetLayerInfo(container, type.modelIndex, i, layerName, app, NULL, type.nlp);
		for(int j = 0; j < type.nlp; j++) {			
			if(app[j]) {
				sprintf(question, "%s %s:", layerName, lpNames[j]);
				requestData(question, result.params[j][i]);
			} else {
				// When the layer parameter is N/A
				result.params[j][i].value = -1;
				result.params[j][i].isMutable = false;
			}
		}
	}
    
	for(int i = 0; i < type.nlp; i++)
		delete[] lpNames[i];
	delete[] lpNames;
	delete[] app;

	ExtraParam *ep = new ExtraParam[type.nExtraParams];
	lf->GetExtraParamInfo(container, type.modelIndex, ep, type.nExtraParams);

	// Get extra parameters
	result.extraParams.resize(type.nExtraParams);
	for(int i = 0; i < type.nExtraParams; i++) {
		char expstr[300] = {0};
		sprintf(expstr, "%s:", ep[i].name);
		requestData(expstr, result.extraParams[i]);
	}

	delete[] ep;

    return result;
}

void printParameters(FrontendComm *lf, wchar_t *container, ModelInformation type, paramStruct *p) {
    printf("Parameters:\n\t");

	char **lpNames = new char*[type.nlp];
	for(int i = 0; i < type.nlp; i++)
		lpNames[i] = new char[256];
	int *app = new int[type.nlp];

	lf->GetLayerParamNames(container, type.modelIndex, lpNames, type.nlp);	

	// Print table header
	for(int i = 0; i < type.nlp; i++)
		printf("%s\t\t", lpNames[i]);
	printf("\n");

	// Print table contents
    for(int i = 0; i < p->layers; i++) {		
		char layerName[256] = {0};

		lf->GetLayerInfo(container, type.modelIndex, i, layerName, app, NULL, type.nlp);

		// Print layer number
		printf("%d\t", i);

		for(int j = 0; j < type.nlp; j++) {
			if(!app[j])
				printf("N/A\t\t");
			else
				printf("%.6f\t\t", p->params[j][i].value);
		}

		printf("\n");
    }

    printf("\n");


	for(int i = 0; i < type.nlp; i++)
		delete[] lpNames[i];
	delete[] lpNames;
	delete[] app;

	ExtraParam *ep = new ExtraParam[type.nExtraParams];
	lf->GetExtraParamInfo(container, type.modelIndex, ep, type.nExtraParams);

	// Print extra parameters
	for(int i = 0; i < (int)p->extraParams.size(); i++) {
		char printed[1024] = {0};

		// Print correct amount of decimal points
		sprintf(printed, "%%s: %%.%df\n",
				ep[i].decimalPoints);

        printf(printed, ep[i].name, p->extraParams[i].value);
	}

	delete[] ep;

}

int usage() {
    fprintf(stderr, "USAGE: xplus <DATA FILE> <OUTPUT MODEL> "
			"[-b/--baseline BASELINE FILE] [-i/--input INITIAL GUESS INI] [-o/--output RESULT INI]\n"
			"[-c/--container MODEL CONTAINER]\n"
            "Note that baseline and INI file are not required\n");
	Sleep(1500);
    return 1;
}

int main(int argc, char **argv) {
    // Filenames
    wchar_t data[MAX_PATH], baseline[MAX_PATH], params[MAX_PATH], outparams[MAX_PATH], modelFile[MAX_PATH],
			container[MAX_PATH];
    bool bBaseline = false, bIni = false, bOutIni = false, bSF = false, bFitSF = false, bModelContainer = false;	

    // Argument handling
    if(argc < 3)
        return usage();

    argc--; argv++; // Shifting the arguments forward
	mbstowcs(data, *argv, strlen(*argv) + 1);

    argc--; argv++; // Shifting the arguments forward
    mbstowcs(modelFile, *argv, strlen(*argv) + 1);

    argc--; argv++; // Shifting the arguments forward    
    while(argc > 0) {
		if(!strcmp(*argv, "-b") || !strcmp(*argv, "--baseline")) {
			argc--; argv++; // Shifting the arguments forward

			if(argc < 1)
				fprintf(stderr, "ERROR: No baseline file\n");
			else {
				bBaseline = true;
				mbstowcs(baseline, *argv, strlen(*argv) + 1);
				argc--; argv++; // Shifting the arguments forward
			}
		} else if(!strcmp(*argv, "-i") || !strcmp(*argv, "--input")) {
			argc--; argv++; // Shifting the arguments forward

			if(argc < 1)
				fprintf(stderr, "ERROR: No input file\n");
			else {
				bIni = true;
				mbstowcs(params, *argv, strlen(*argv) + 1);
				argc--; argv++; // Shifting the arguments forward
			}
		} else if(!strcmp(*argv, "-o") || !strcmp(*argv, "--output")) {
			argc--; argv++; // Shifting the arguments forward

			if(argc < 1)
				fprintf(stderr, "ERROR: No output file\n");
			else {
				bOutIni = true;
				mbstowcs(outparams, *argv, strlen(*argv) + 1);
				argc--; argv++; // Shifting the arguments forward
			}
		} else if(!strcmp(*argv, "-c") || !strcmp(*argv, "--container")) {
			argc--; argv++; // Shifting the arguments forward

			if(argc < 1)
				fprintf(stderr, "ERROR: No model container\n");
			else {
				bModelContainer = true;
				mbstowcs(container, *argv, strlen(*argv) + 1);
				argc--; argv++; // Shifting the arguments forward
			}
		} else {
		  printf("Error in argument %s\n", *argv);
		  argc--; argv++;
		}
    }

	FrontendComm *lf = new LocalFrontend();
	if(!lf->IsValid()) {
		printf("Cannot create frontend\n");
		delete lf;

		return 7;
	}
    printf("Input stage\n");

	// Get the model type
	ModelInformation type = InputModelType(lf, bModelContainer ? container : NULL);

	// Create the fitting job
	JobPtr job = lf->CreateJob(data, &printProgress);

	int res = lf->SetModel(job, MT_FORMFACTOR, bModelContainer ? container : NULL, type.modelIndex,
						   EDProfile());
	if(res) {
		printf("Error setting model: %d\n", res);
		lf->DestroyJob(job);
		delete lf;
		return 1;
	}

	paramStruct param;
	/*PeakType pType;
	peakStruct peaks;
	*/	

	void *iniFile = NewIniFile();

	ModelUI modelInfo;
	modelInfo.setModel(lf, bModelContainer ? container : NULL, type, 10);
	param.nExtraParams = modelInfo.GetNumExtraParams();
	param.nlp = modelInfo.GetNumLayerParams();

	if(bIni) {
		std::wstring prm(params);
		std::string nm(type.name);
		ReadParameters(prm, nm, &param, modelInfo, iniFile);
	}
	else // Input

//#define GRRRR
#ifdef GRRRR
	param.nlp = 2;
	param.layers = 3;
	param.params.resize(param.nlp);
	for(int i = 0; i < param.nlp; i++)
		param.params[i].resize(param.layers);
	param.bConstrain = false;
	param.nExtraParams = 2;
	param.params[0][0].value = -1.0;
	param.params[0][0].isMutable = false;
	param.params[0][1].value = 2.2;
	param.params[0][1].isMutable = true;
	param.params[0][2].value = 2.2;
	param.params[0][2].isMutable = true;
	param.params[1][0].value = 333.0;
	param.params[1][0].isMutable = false;
	param.params[1][1].value = 280.0;
	param.params[1][1].isMutable = false;
	param.params[1][2].value = 400.0;
	param.params[1][2].isMutable = false;
	param.extraParams.resize(param.nExtraParams);
	param.extraParams[0].value = 1.0;
	param.extraParams[0].isMutable = false;
	param.extraParams[1].value = 5.0;
	param.extraParams[1].isMutable = false;

#else
		param = InputInitialGuess(lf, bModelContainer ? container : NULL, type);
#endif
	/*
    printf("Does the signal include a structure factor? [y/n] ");
    char ans[256] = {0};
    scanf("%s", ans);

    
    if(tolower(ans[0]) == 'y') {
        bSF = true;
        
        printf("Would you like to fit the structure factor? [y/n] ");
        fflush(stdout);        

        scanf("%s", ans);
        if(tolower(ans[0]) == 'y')
            bFitSF = true;
    }

    
    if(bSF) {
		pType = InputPeakType();
		SetPeakType(pType);

		if(bIni)
			ReadPeaks(params, type->GetName(), &peaks, iniFile);
		else			
			peaks = InputInitialPeaks(pType);
	
    }

	CloseIniFile(iniFile);
	iniFile = NULL;
	*/

    // Work
    int pStop = 0;
    std::vector<double> ffx, ffy, bgly, resy, bg, errors, merrors;
	std::vector<bool> mask;
    bool bSuccess = true;
    /*
	// Baseline subtraction
    if(bBaseline) {
		// I assume the baseline file name should include "-baseline.out"
		// like in all other places...
		wcscat(baseline, L"-baseline.out");

        GenerateBGLinesandFormFactor(data, baseline, ffx, bgly, ffy, false);
    } else // File reading
	*/
    
	ReadDataFile(data, ffx, ffy);

	bg.resize(ffx.size(), 0.0);

    printf("\n");
	//if(!bOutIni)
	printParameters(lf, bModelContainer ? container : NULL, type, &param);
    
    /*if(!bFitSF) {
		if(bSF) {
			printf("\nGenerating Structure Factor...\n");
			bSuccess = GenerateStructureFactorU(ffx, resy, bg, &peaks, NULL, 
												&pStop, printProgress);
		}
        
		printf("\nFitting...\n");
		printProgress(0);

        bSuccess &= CreateModelU(ffx, ffy, resy, bg, mask, &param, errors, merrors, NULL, &pStop, 
                                 printProgress);
	
	} else {
	*/
	FittingProperties fp;
	fp.accurateDerivative	= false;
	fp.accurateFitting		= false;
	fp.bProgressReport		= true;
	fp.fitIterations		= 20;
	fp.logScaleFitting		= false;
	fp.method				= FIT_LM;
	fp.minSignal			= 0.0;
	fp.resolution			= 0.0;
	fp.usingGPUBackend		= false;
	fp.wssrFitting			= true;
	std::vector<int> masked;
	masked.resize(ffx.size(), 0);

	printf("\nFitting Form Factor...\n");
	printProgress(NULL, 0.0);		
	ErrorCode errfit = lf->Fit(job, MT_FORMFACTOR, &param, NULL, NULL, ffx, ffy, masked, fp);
	lf->WaitForFinish(job);
    printProgress(NULL, 1.0);
    printf("\n");
    if(errfit) {
        printf("Failed to fit model (%d), exiting.\n", errfit);

		// Destroy the job
		lf->DestroyJob(job);
		delete lf;

        return 2;
    }

	int resultSize = lf->GetGraphSize(job);
	resy.resize(resultSize);
	lf->GetGraph(job, &resy[0], resultSize);

    // Output
    WriteDataFile(modelFile, ffx, resy);	
	//lf->GetResults(job, ModelType::MT_FORMFACTOR, param);
	printf("\nFitted parameters:\n");
	paramStruct resp;
	lf->GetResults(job, MT_FORMFACTOR, resp);
	printParameters(lf, bModelContainer ? container : NULL, type, &resp);

#ifdef GENERATE_CODE_KUGYUK
 	GenerateProperties gp;
	gp.bProgressReport = true;
	gp.liveGenerate = false;

		printf("\nGenerating Form Factor...\n");
		printProgress(NULL, 0.0);		
		ErrorCode err = lf->Generate(job, MT_FORMFACTOR, &param, NULL, NULL, ffx, gp);
		
		lf->WaitForFinish(job);
 /*       
		printf("\nFitting...\n");
		printProgress(0);

		bSuccess &= FitStructureFactorU(ffx, ffy, resy, bg, mask, &peaks, errors, merrors, NULL, &pStop, 
										printProgress);
    }*/

    // Finish up
    printProgress(NULL, 1.0);
    printf("\n");

    if(err) {
        printf("Failed to fit model (%d), exiting.\n", err);

		// Destroy the job
		lf->DestroyJob(job);
		delete lf;

        return 2;
    }

	// Get results
	int resultSize2 = lf->GetGraphSize(job);
	resy.resize(resultSize2);
	lf->GetGraph(job, &resy[0], resultSize2);

    // Output
	modelFile[40]= 'X';
    WriteDataFile(modelFile, ffx, resy);	

	/*
	iniFile = NewIniFile();

	if(bOutIni) {
		WriteParameters(outparams, type->GetName(), &param, iniFile);
	} else {*/
		printf("Final ");
		printParameters(lf, job, type, &param);
	/*}

	SaveAndCloseIniFile(outparams, type->GetName(), iniFile);
	iniFile = NULL;
	*/
#endif

	// Destroy the job
	lf->DestroyJob(job);
	delete lf;


    printf("Done successfully! WSSR: %.12f\n", WSSR(ffy, resy, false));

#ifdef _DEBUG
	_getch();
#endif

	return 0;
}
