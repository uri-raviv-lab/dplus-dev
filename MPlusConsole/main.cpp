#include <cstdio>
#include <windows.h>


#include "../BackendCommunication/LocalCommunication/LocalComm.h"

void STDCALL MyProgress(void *args, double progress)
{
	printf("Progress: %f\n", progress * 100.0);
}

void STDCALL MyCompletion(void *args, int error)
{
	printf("Finished. Error: %s\n", g_errorStrings[error]);

	bool *pbIsComplete = (bool *)args;
	*pbIsComplete = true;
}

int main(int argc, char **argv)
{
	// Create frontend (connects to backend)
	LocalFrontend frontend;

	bool bIsComplete = false;
	JobPtr job = frontend.CreateJob(L"our job", MyProgress, MyCompletion, &bIsComplete);

	// Create the model tree
	ModelPtr domain = frontend.CreateDomainModel(job);

	ModelPtr sphereGeometry = frontend.CreateModel(job, NULL, 2, EDProfile());
	ModelPtr sphereAmplitude = frontend.CreateGeometricAmplitude(job, sphereGeometry);

	// Create q points for generation
	std::vector<double> x (500);
	for(int i = 0; i < 500; i++) x[i] = ((double)i / 100.0) + 0.001;
	
	// Set initial parameters
	paramStruct domparams;
	domparams.nlp = 6; // THE NUMBER OF PREFERENCES
	domparams.layers = 1;
	domparams.params.resize(domparams.nlp);

	domparams.params[0].push_back(10000.0);	// Orientation iterations
	domparams.params[1].push_back(300);            // Grid size
	domparams.params[2].push_back(Parameter(0.0)); // Use grid? NO
	domparams.params[3].push_back(Parameter(0.001)); // Epsilon
	domparams.params[4].push_back(Parameter(x[499])); // qmax
	domparams.params[5].push_back(Parameter(double(OA_MC))); // Integration method
	
	paramStruct sphereparams;
	sphereparams.nExtraParams = 2;
	sphereparams.nlp = 2;
	sphereparams.layers = 2;
	sphereparams.extraParams.push_back(1.0); // Scale
	sphereparams.extraParams.push_back(5.0); // Background
	
	sphereparams.params.resize(2);
	sphereparams.params[0].resize(2); // R
	sphereparams.params[1].resize(2); // ED
	sphereparams.params[0][0] = 8.0; // Rsolvent
	sphereparams.params[0][1] = 333.0; // EDsolvent
	sphereparams.params[1][0] = 4.0; // Router
	sphereparams.params[1][1] = 400.0; // EDouter


	// Model tree:
	// Domain
	// |
	// `-Sphere
	ParameterTree ptree;
	ptree.SetNodeModel(domain);
	ptree.SetNodeParameters(domparams);	
	ptree.AddSubModel(sphereAmplitude, sphereparams);
	
	// Set settings for generation
	FittingProperties props;
	props.bProgressReport = true;
	props.msUpdateInterval = 100;

	ErrorCode err = frontend.Generate(job, ptree, x, props);
	if(err)
	{
		frontend.DestroyModel(job, sphereGeometry);		
		frontend.DestroyModel(job, sphereAmplitude);		
		frontend.DestroyModel(job, domain);
		frontend.DestroyJob(job);
		printf("An error occurred! %s\n", g_errorStrings[err]);
		return 0;
	}

	// Now we wait
	frontend.WaitForFinish(job);

	// Request graph
	double *y = new double[500];
	if(!frontend.GetGraph(job, y, 500))
		printf("No graph returned\n");
	else
	{
		FILE *fp = fopen("bla.dat", "wb");
		if(fp)
		{
			for(int i = 0; i < 500; i++)
				fprintf(fp, "%f\t%f\n", x[i], y[i]);
			fclose(fp);
		}

		printf("OK!\n");
	}

	// Free up all the data
	delete[] y;

	frontend.DestroyModel(job, sphereGeometry);		
	frontend.DestroyModel(job, sphereAmplitude);		
	frontend.DestroyModel(job, domain);
	frontend.DestroyJob(job);

	return 0;
}
