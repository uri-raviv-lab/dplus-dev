
#include "Amplitude.h" // For ModelTypeToString
#include "Geometry.h"  // For ModelTypeToString

#include "JobManager.h"

#include <cstdarg>
#include <ctime>
#include <string>
#include <sstream>
#include <set>
#include <iostream>

#include "CommProtocol.h" // For back-transmitting messages to the frontend
#include "Model.h" // For displaying active jobs' model names

#include "modelfitting.h"
#include "LocalBackend.h"

using namespace std;

// Helper function to turn a ModelType into a string (for logging purposes)
static std::string ModelTypeToString(IModel *model)  {
	std::string result = "";

	// The function determines the type of model according to the following decision tree:
	// 1. Complex Domain Model
	// 2. Composite Model
	// 3. Simple Geometry Model

	if(dynamic_cast<DomainModel *>(model)) {
		result += "Complex Domain";
	} else if(dynamic_cast<CompositeModel *>(model)) {
		CompositeModel *cm = dynamic_cast<CompositeModel *>(model);

		std::stringstream ss;
		ss << "Composite Model (" << cm->GetSubModels().size() << " domains)";
		result += ss.str();
	} else if(dynamic_cast<Geometry *>(model)) {
		result += dynamic_cast<Geometry *>(model)->GetName();
	}

	// If still the result string is zero-lengthed, we have no models
	if(result.size() == 0)
		result = "None";

	return result;
}

JobPtr JobManager::CreateJob(const char *instigator, const wchar_t *desc) {
	lock_guard<mutex> lock(jobMutex);

	jobCtr++;
	jobs[jobCtr] = Job(jobCtr, (instigator ? instigator : "N/A"),
					   (desc ? desc : L"N/A"));

	return (JobPtr)jobCtr;
}

void JobManager::DestroyJob(JobPtr job) {
	lock_guard<mutex> lock(jobMutex);

	if(jobs.find(job) != jobs.end()) {
		jobs[job].Destroy();

		jobs.erase(job);
	}
}

static void FitJobThread(void *args) {

}

ErrorCode JobManager::StartFitJob(LocalBackend *backend, JobPtr job, const ParameterTree& tree,
								  const std::vector<double>& x, 
								  const std::vector<double>& y, 
								  const std::vector<int>& mask, 
							 	  const FittingProperties& fp) {
	return ERROR_JOBNOTFOUND;
}

static void GenerateJobThread(void *args) {
	if(!args)
		return;
	fitJobArgs *gja = (fitJobArgs *)args;
	fitJobArgs localArgs = *gja;

	// Delete the just-now created argument structure (because we have a local copy)
	delete gja; gja = NULL;

	ErrorCode err = OK;
	std::string errMsg = "";
	
	try
	{
		err = PerformModelGeneration(&localArgs);
	}
	catch (backend_exception &be)
	{
		err = (ErrorCode)be.GetErrorCode();
		errMsg = be.GetErrorMessage();
		//throw be;
	}

	// Update the job in the manager
	JobManager::GetInstance().CompleteJob(localArgs.backend, localArgs.jobID, localArgs.fp.bProgressReport, err, errMsg);
}

static void Generate2DJobThread(void* args) {

	if (!args)
		return;
	fitJobArgs* gja = (fitJobArgs*)args;
	fitJobArgs localArgs = *gja;

	// Delete the just-now created argument structure (because we have a local copy)
	delete gja; gja = NULL;

	ErrorCode err = OK;
	std::string errMsg = "";

	try
	{
		err = PerformModelGeneration2D(&localArgs);
	}
	catch (backend_exception& be)
	{
		err = (ErrorCode)be.GetErrorCode();
		errMsg = be.GetErrorMessage();
		//throw be;
	}

	// Update the job in the manager
	JobManager::GetInstance().CompleteJob(localArgs.backend, localArgs.jobID, localArgs.fp.bProgressReport, err, errMsg);
}

ErrorCode JobManager::StartGenerateJob(LocalBackend *backend, JobPtr job, const ParameterTree& tree,
									   const std::vector<double>& x, 
								       const FittingProperties& fp) {
	Job j = GetJobInformation(job);
	if(!j.uid)
		return ERROR_JOBNOTFOUND;

	// Validate the param/model tree. If incorrect, return ERROR_INVALIDPARAMTREE
	if(!ValidateParamTree(job, tree))
		return ERROR_INVALIDPARAMTREE;	
		
	{
		lock_guard<mutex> joblock (*j.jobMutex);

		if(j.state != JS_IDLE)
			return ERROR_JOBRUNNING;

		// Reset parameters
		j.type = JT_GENERATE;
		j.progress = 0.0;
		j.resultGraph.clear();
		j.error = 0;
		j.errorMsg[0] = L'\0';
		if(j.pStop)
			*j.pStop = false;

		//reset job status
		j.jobStatus.code = -1;
		j.jobStatus.isRunning = true;
		j.jobStatus.progress = 0.0;

		// Set model(s) to generate
		if(j.tree)
			delete j.tree;
		j.tree = new ParameterTree(tree);

		// Get the root model from the tree
		IModel *topModel = NULL;
		ModelPtr mptr = j.tree->GetNodeModel();
		if(j.uidToModel.find(mptr) != j.uidToModel.end())
			topModel = j.uidToModel[mptr];

		// Print to log
		LogMessage(L"%llu: Started generating job %d (%ls) by %s.\n",
			(unsigned long long)time(NULL), job, j.description, j.instigator);
		LogMessage(L"%llu: Models in memory: %d; Amplitudes in memory: %d.\n",
			(unsigned long long)time(NULL), j.uidToModel.size(), j.uidToAmp.size());
		
		// Update job state prior to running it
		j.state = JS_RUNNING;
		j.lastAccess = j.beginning = time(NULL);
		UpdateJob(j);
	}

	// Create a thread for the job and run it
	{
		lock_guard<mutex> lock(jobMutex);

		// This structure is deleted immediately inside the thread (after copy)
		fitJobArgs *gja = new fitJobArgs;
		gja->backend = backend;
		gja->jobID = job;
		gja->x = x;
		gja->fp = fp;

		jobThreads[job] = new thread(GenerateJobThread, gja);
	}

	return OK;
}

ErrorCode JobManager::StartGenerate2DJob(LocalBackend* backend, JobPtr job, const ParameterTree& tree,
	const std::vector<double>& x,
	const FittingProperties& fp) {

	Job j = GetJobInformation(job);
	if (!j.uid)
		return ERROR_JOBNOTFOUND;

	// Validate the param/model tree. If incorrect, return ERROR_INVALIDPARAMTREE
	if (!ValidateParamTree(job, tree))
		return ERROR_INVALIDPARAMTREE;

	{
		lock_guard<mutex> joblock(*j.jobMutex);

		if (j.state != JS_IDLE)
			return ERROR_JOBRUNNING;

		// Reset parameters
		j.type = JT_GENERATE;
		j.progress = 0.0;
		j.resultGraph.clear();
		j.resultGraph2D = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>();
		j.error = 0;
		j.errorMsg[0] = L'\0';
		if (j.pStop)
			*j.pStop = false;

		//reset job status
		j.jobStatus.code = -1;
		j.jobStatus.isRunning = true;
		j.jobStatus.progress = 0.0;

		// Set model(s) to generate
		if (j.tree)
			delete j.tree;
		j.tree = new ParameterTree(tree);

		// Get the root model from the tree
		IModel* topModel = NULL;
		ModelPtr mptr = j.tree->GetNodeModel();
		if (j.uidToModel.find(mptr) != j.uidToModel.end())
			topModel = j.uidToModel[mptr];

		// Print to log
		LogMessage(L"%llu: Started generating job %d (%ls) by %s.\n",
			(unsigned long long)time(NULL), job, j.description, j.instigator);
		LogMessage(L"%llu: Models in memory: %d; Amplitudes in memory: %d.\n",
			(unsigned long long)time(NULL), j.uidToModel.size(), j.uidToAmp.size());

		// Update job state prior to running it
		j.state = JS_RUNNING;
		j.lastAccess = j.beginning = time(NULL);
		UpdateJob(j);
	}

	// Create a thread for the job and run it
	{
		lock_guard<mutex> lock(jobMutex);

		// This structure is deleted immediately inside the thread (after copy)
		fitJobArgs* gja = new fitJobArgs;
		gja->backend = backend;
		gja->jobID = job;
		gja->x = x;
		gja->fp = fp;

		jobThreads[job] = new thread(Generate2DJobThread, gja);
	}

	return OK;
}

void JobManager::StopJob(JobPtr job, bool bBlocking) {
	Job j = GetJobInformation(job);
	if(!j.uid)
		return;

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		if(j.state == JS_IDLE)
			return;

		j.lastAccess = time(NULL);
		UpdateJob(j);

		// Sends an asynchronous stop signal to the job
		j.Stop();
	}

	// Block on the thread, if blocking
	if(bBlocking)
		jobThreads[job]->join();
}

void JobManager::WaitForJob(JobPtr job) {
	Job j = GetJobInformation(job);
	if(j.uid == 0)
		return;

	if(j.state == JS_IDLE)
		return;

	// Block on the thread
	jobThreads[job]->join();
}

void JobManager::CompleteJob(LocalBackend *backend, JobPtr job, bool bNotifyFrontend, int error, const std::string &errMsg) {
	Job j = GetJobInformation(job);
	if(!j.uid)
		return;

	IModel *topModel = NULL;

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		if(j.state == JS_IDLE)
			return;

		j.progress = 1.0;
		j.error = error;
		// Assumes ASCII, dangerous?
		std::wstring w_last_message(errMsg.begin(), errMsg.end());
		wcsncpy(j.errorMsg, w_last_message.c_str(), 1024);
		j.state = JS_IDLE;

		//update job status
		j.jobStatus.code = error;
		j.jobStatus.isRunning = false;
		j.jobStatus.progress = 1.0;

		UpdateJob(j);

		// Get the root model from the tree
		ModelPtr mptr = j.tree->GetNodeModel();
		if(j.uidToModel.find(mptr) != j.uidToModel.end())
			topModel = j.uidToModel[mptr];
	}


	{
		lock_guard<mutex> lock(jobMutex);

		// Thread tries to delete itself.
		// TODO::Threads Think of a better design
		//thread *thr = jobThreads[job];
		//delete thr;

		jobThreads.erase(job);
	}

	// Print to log
	LogMessage(L"%llu: Completed %ls in job %d (%ls) by %s (errorcode %d %s). Models used: %s\n",
			   (unsigned long long)time(NULL), (j.type == JT_FIT) ? L"fitting" : L"generation", job, 
			   j.description, j.instigator, j.error, j.errorMsg,
			   ModelTypeToString(topModel).c_str());

	// Call backend completion handler (a field in Job)
	if (backend && bNotifyFrontend)
	{
		std::wstring wm(j.errorMsg);
		std::string m(wm.begin(), wm.end());
		backend->NotifyCompletion(job, (error != 0), error, m);
	}
}

Job JobManager::GetJobInformation(JobPtr job) {
	lock_guard<mutex> lock(jobMutex);

	if(jobs.find(job) != jobs.end())
		return jobs[job];

	return Job();
}

void JobManager::UpdateJob(Job job) {
	lock_guard<mutex> lock(jobMutex);

	// The following "if" disallows UID modification
	if(jobs.find(job.uid) != jobs.end())
		jobs[job.uid] = job;
}

std::vector<std::wstring> JobManager::GetActiveJobs(const char *instigator) 
{
	lock_guard<mutex> lock(jobMutex);

	std::vector<std::wstring> result;
	
	for(std::map<unsigned int, Job>::iterator iter = jobs.begin(); iter != jobs.end(); ++iter) {
		if(iter->second.state == JS_IDLE)
			continue;
		if(instigator && strncmp(iter->second.instigator, instigator, 256))
			continue;

		std::wstringstream wss;

		// Job data
		wss << iter->second.instigator << L"\t" << iter->second.description << L"\t"
			<< iter->second.type << L"\t";

		
		// Get the root model from the tree
		IModel *topModel = NULL;
		ModelPtr mptr = iter->second.tree->GetNodeModel();
		if(iter->second.uidToModel.find(mptr) != iter->second.uidToModel.end())
			topModel = iter->second.uidToModel[mptr];

		// Convert mbstring to unicode wide char string
		std::wstring wModelType;
		std::string cModelType = ModelTypeToString(topModel);
		wModelType.assign(cModelType.begin(), cModelType.end());

		// Models used in fitting
		wss << wModelType << L"\t";
		
		// Job progress
		wss << (int)(iter->second.progress * 100.0) << L"%\t";
		
		// Job times (beginning, last access)
		wss << iter->second.beginning << L"\t" << iter->second.lastAccess;

		result.push_back(wss.str());
	}

	return result;
}

void JobManager::LogMessage(const wchar_t *fmt, ...) 
{
#ifdef _WIN32
	FILE *fp = _wfopen(LOGFILE, L"a");
#else
    FILE *fp = fopen(wstringtoutf8(LOGFILE).c_str(), "a");
#endif
	
	va_list args;
	va_start (args, fmt);
	
	if(fp)
		vfwprintf(fp, fmt, args);
	else
		vwprintf(fmt, args);

	va_end (args);

	if(fp)
		fclose(fp);
}

#define MB(x)  do {} while(0)

bool JobManager::innerValidateParamTree(Job& j, const ParameterTree *tree, 
										bool bAmplitude,
										std::set<ModelPtr>& usedModels) {
	// Traverse tree
	ModelPtr model = tree->GetNodeModel();	

	// Empty model
	if(!model)
		return true;

	MB("INEM");

	// Verifies that the same model pointer isn't used twice in the same tree
	if(usedModels.find(model) != usedModels.end())
		return false; // Same model is used more than once

	MB("IFM");

	usedModels.insert(model);

	// First, validate the root model, then validate its children

	int num = tree->GetNumSubModels();
	bool bSubAmp = false;

	// Test that the model exists and contains the correct amount of children
	if(!bAmplitude) {
		MB("IAMPM");
		if(j.uidToModel.find(model) != j.uidToModel.end()) {
			IModel *mod = j.uidToModel[model];

			char buf[256]={0};
			sprintf(buf, "Model: %d, %p", model, mod);
			MB(buf);

			// Symmetry must contain amplitude children
			if(dynamic_cast<DomainModel *>(mod)) {
				MB("IAM2");
				bSubAmp = true;
				if(num <= 0)
					return false;
			} else if(dynamic_cast<CompositeModel *>(mod)) {
				MB("IAM3");
				if(num <= 0)
					return false;
			} else { // Other geometries cannot contain children			
				MB("IAM4");
				if(num > 0)
					return false;
			}
		} else
			return false;
	} else {
		MB("IAMPA");
		if(j.uidToAmp.find(model) != j.uidToAmp.end()) {
			Amplitude *amp = j.uidToAmp[model];
			MB("IAA1");

			// Symmetry must contain amplitude children
			if(dynamic_cast<ISymmetry *>(amp)) {
				MB("IAA2");
				bSubAmp = true;
				if(num <= 0)
					return false;
			} else {
				MB("IAA3");
				// Amplitude cannot contain children
				if(num > 0)
					return false;
			}
		
		} else // Model/Amplitude does not exist in current context
			return false;
	}

	// If it's a symmetry, it MUST have children. If it is not, it must have none	

	MB("IAC");

	// Validate this model's children (if should exist)
	for(int i = 0; i < num; i++) {
		if(!innerValidateParamTree(j, tree->GetSubModel(i), bSubAmp, usedModels))
			return false;
	}	

	return true;
}

bool JobManager::ValidateParamTree(JobPtr job, const ParameterTree& tree) {
	std::set<ModelPtr> usedModels;

	Job j = GetJobInformation(job);
	if(!j.uid)
		return false;

	MB("J");

	// Root model must not be empty
	if(!tree.GetNodeModel())
		return false;

	MB("NM");

	return innerValidateParamTree(j, &tree, false, usedModels);
}
