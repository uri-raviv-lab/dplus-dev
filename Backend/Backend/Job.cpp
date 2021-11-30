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
