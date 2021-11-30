#ifndef __JOBMANAGER_H
#define __JOBMANAGER_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include "Common.h"
#include "../../BackendCommunication/LocalCommunication/LocalComm.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <thread>
#include <mutex>
using std::mutex;
using std::thread;

#include "Job.h" // For all Job-related classes

#define LOGFILE L"JobManager.log"

// Forward declarations
class BackendComm;
class LocalBackend;

// A singleton responsible for handling all the jobs, creating and destroying threads 
// for them to run concurrently and logging the job activity on this backend.
// @note This class MUST be thread-safe in order to function properly.
class JobManager {
protected:	
	// Job ID counter
	unsigned int jobCtr;

	std::map<unsigned int, Job> jobs;
	std::map<unsigned int, thread *> jobThreads;
	mutex jobMutex;
	

	JobManager() : jobCtr(0) {}
	JobManager(const JobManager& rhs) {}
	void operator=(const JobManager& rhs) {}
public:
	static JobManager& GetInstance() { 
		static JobManager jm;
		return jm;
	}

	JobPtr CreateJob(const char *instigator, const wchar_t *desc);

	void DestroyJob(JobPtr job);
	
	ErrorCode StartFitJob(LocalBackend *backend, JobPtr job, const ParameterTree& tree, 
						  const std::vector<double>& x, 
						  const std::vector<double>& y, const std::vector<int>& mask, 
						  const FittingProperties& fp);

	ErrorCode StartGenerateJob(LocalBackend *backend, JobPtr job, const ParameterTree& tree,
							   const std::vector<double>& x, 
							   const FittingProperties& fp);

	void StopJob(JobPtr job, bool bBlocking);

	void WaitForJob(JobPtr job);

	Job GetJobInformation(JobPtr job);

	void UpdateJob(Job job);

	// Called from the JOB THREAD
	void CompleteJob(LocalBackend *backend, JobPtr job, bool bNotifyFrontend, int error, const std::string &errMsg);

	// Columns:
	// Instigator | Description | Job Type (Fit/Generate) | FF Model | SF Model | 
	// BG Model | Progress (%) | Start Time | Last Access
	// If instigator is NULL, returns all active jobs
	std::vector<std::wstring> GetActiveJobs(const char *instigator = NULL);

	// Prints a message to the log
	static void LogMessage(const wchar_t *fmt, ...);

	// Validate an input parameter tree
	bool ValidateParamTree(JobPtr job, const ParameterTree& tree);

protected:
	bool innerValidateParamTree(Job& j, const ParameterTree *tree, bool bAmplitude, std::set<ModelPtr>& usedModels);
};


#endif // __JOBMANAGER_H
