#include <ctime>
#include <cstdarg>

#ifndef NOMINMAX
#define NOMINMAX
#endif // !NOMINMAX
#include <thread>
#include <mutex>
using std::mutex;
using std::lock_guard;

#include "Amplitude.h"
#include "Model.h"
#include "Geometry.h"
#include "fitting.h"

#include "LocalBackendParameterTree.h"

#include "Job.h"

/**
 * @file Job.cpp
 * @brief Implements the Job class, which encapsulates the state, resources, and lifecycle management
 *        for a single computational job (such as model fitting or generation) in the backend.
 *
 * The Job class is responsible for:
 *  - Managing all resources associated with a job, including models, amplitudes, fitters, and parameter trees.
 *  - Providing thread-safe operations via a per-job mutex.
 *  - Assigning unique identifiers to models and amplitudes within the job.
 *  - Handling job state, progress, error reporting, and stop signals.
 *  - Supporting resource cleanup and destruction of all associated objects.
 *  - Allowing interruption and reset of fitting operations.
 *
 * Key Concepts:
 *  - Each Job instance owns its models, amplitudes, and fitter, and is responsible for their lifetime.
 *  - Models and amplitudes are assigned unique IDs within the job for tracking and lookup.
 *  - The job maintains its own error state and supports formatted error reporting.
 *  - Thread safety is enforced for all resource and state modifications.
 *
 * Fields:
 *  - unsigned int uid:
 *      Unique identifier for the job.
 *  - char instigator[256]:
 *      String identifying the creator or owner of the job.
 *  - wchar_t description[256]:
 *      Human-readable description of the job.
 *  - JobType type:
 *      The type of job (e.g., JT_FIT, JT_GENERATE, JT_NONE).
 *  - JobState state:
 *      Current state of the job (e.g., JS_IDLE, JS_RUNNING).
 *  - double progress:
 *      Progress of the job (0.0–1.0).
 *  - int error:
 *      Last error code for the job.
 *  - wchar_t errorMsg[1024]:
 *      Last error message for the job.
 *  - time_t beginning:
 *      Timestamp when the job started.
 *  - time_t lastAccess:
 *      Timestamp of the last access to the job.
 *  - unsigned int lastUID:
 *      Counter for assigning unique model/amplitude IDs within the job.
 *  - ParameterTree* tree:
 *      Pointer to the parameter/model tree for the job.
 *  - Fitter* fitter:
 *      Pointer to the fitter object for fitting jobs (may be null).
 *  - std::map<unsigned int, IModel*> uidToModel:
 *      Maps model handles to model instances for this job.
 *  - std::map<unsigned int, Amplitude*> uidToAmp:
 *      Maps amplitude handles to amplitude instances for this job.
 *  - std::map<Amplitude*, unsigned int> ampToUid:
 *      Reverse map from amplitude instance to handle.
 *  - std::mutex* jobMutex:
 *      Pointer to the per-job mutex for thread safety.
 *  - int* pStop:
 *      Pointer to an integer flag used to signal job interruption.
 *  - std::vector<double> resultGraph:
 *      Stores the 1D result graph for the job.
 *  - Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> resultGraph2D:
 *      Stores the 2D result graph for the job.
 *  - JobStatus jobStatus:
 *      Tracks the current status, progress, and error code for the job.
 *
 * Main Methods:
 *  - Job (constructor): Initializes a new job with identifiers, state, and resources.
 *  - AssignModel / AssignAmplitude: Registers a new model or amplitude and assigns a unique ID.
 *  - Destroy: Cleans up all resources, including models, amplitudes, fitters, and synchronization primitives.
 *  - Stop: Signals the job (and its fitter, if present) to halt execution.
 *  - ResetFitter: Deletes and resets the job's fitter.
 *  - ReportError: Sets the job's error message using a formatted string.
 *
 * Threading and Safety:
 *  - All resource modifications are protected by a per-job mutex.
 *  - The stop signal is managed via an atomic integer pointer.
 *
 * Error Handling:
 *  - Errors are reported and stored in the job's errorMsg buffer.
 *  - The job tracks its own error code and message for reporting to the backend/frontend.
 *
 * See Job.h for class and method declarations.
 */


Job::Job(unsigned int id, const char *init, const wchar_t *desc) : uid(id), 
fitter(NULL), tree(NULL), state(JS_IDLE), type(JT_NONE), progress(0.0), error(0), beginning(0), lastUID(1) {

	lastAccess = time(NULL);
	errorMsg[0] = L'\0';

	if(init)
		strncpy(instigator, init, 256);
	if(desc)
		wcsncpy(description, desc, 256);

	jobMutex = new mutex();

	// Initialize the stop signal
	pStop = new int;
	*pStop = 0;

	//Initialize the status
	jobStatus.code = -1;
	jobStatus.isRunning = false;
	jobStatus.progress = 0.0;
}

unsigned int Job::AssignModel(IModel *model) {
	unsigned int tempRes;
	{
		// ASSUMING A LOCKED JOB
		//lock_guard<mutex> lock(*jobMutex);

		// Too many models
		if(lastUID > MAX_MODELS_PER_JOB)
			return 0;

		uidToModel[lastUID] = model;
		tempRes = lastUID;

		lastUID++;

	}
	return tempRes;
}

unsigned int Job::AssignAmplitude(Amplitude *amp) {
	unsigned int tempRes;
	{
		// ASSUMING A LOCKED JOB
		//lock_guard<mutex> lock(*jobMutex);

		// Too many models
		if(lastUID > MAX_MODELS_PER_JOB)
			return 0;

		// This ruined scripts and took me three days to debug and find!
		/*
		std::vector<unsigned int> fnd;
		// Collect all uids
		boost::copy(uidToModel | boost::adaptors::map_keys, std::back_inserter(fnd));
		boost::copy(uidToAmp   | boost::adaptors::map_keys, std::back_inserter(fnd));

		std::sort(fnd.begin(), fnd.end());
		
		lastUID = fnd.size();
		for(unsigned int fl = 1; fl < fnd.size(); fl++ ) {
			if(fnd[fl] != fl) {
				lastUID = fl;
				break;
			}

		}*/


		uidToAmp[lastUID] = amp;
		ampToUid[amp] = lastUID;
		tempRes = lastUID;

		lastUID++;

	}
	return tempRes;
}

void Job::Destroy() {
	{
		lock_guard<mutex> lock(*jobMutex);

		if(fitter) {
			delete fitter;
			fitter = NULL;
		}

		for(std::map<unsigned int, IModel *>::iterator iter = uidToModel.begin(); iter != uidToModel.end(); ++iter) {
			if(iter->second)
				delete iter->second;
		}

		for(std::map<unsigned int, Amplitude *>::iterator iter = uidToAmp.begin(); iter != uidToAmp.end(); ++iter) {
			if(iter->second)
				delete iter->second;
		}

		if(pStop) {
			*pStop = 1;
			delete pStop;
			pStop = NULL;
		}
	}
	
	if(jobMutex)
		delete jobMutex;
}

void Job::Stop() {	
	*pStop = 1;

	if(type == JT_FIT && fitter)
		fitter->Stop();
}

void Job::ResetFitter() {
	// Reset fitter
	if(fitter) {
		delete fitter;
		fitter = NULL;
	}
}

void Job::ReportError( const wchar_t *fmt, ... )
{
	va_list args;
	va_start (args, fmt);
	vswprintf(errorMsg, 1024, fmt, args);
	va_end (args);
}
