#ifndef __JOB_H
#define __JOB_H

#include "Common.h"
#include "CommProtocol.h"
#include <map>

// The maximal amount of models per job
#define MAX_MODELS_PER_JOB 65536

// Forward declarations
class IModel;
class Amplitude;
class ModelFitter;
namespace tthread {
	class mutex;
};

typedef unsigned int JobPtr;

enum JobState {
	JS_IDLE,
	JS_RUNNING,
};

class LocalBackendParameterTreeConverter;
struct JobStatus;

struct Job {
	unsigned int uid;
	unsigned long long lastAccess;
	char instigator[256];
	wchar_t description[256];

	// Job-guarding mutex (for getGraph, startfit/gen etc.)
	mutex *jobMutex;

	// Data structures mapping between a model/amplitude UID and the corresponding objects
	std::map<unsigned int, IModel *> uidToModel;
	std::map<unsigned int, Amplitude *> uidToAmp;
	std::map<Amplitude *, unsigned int> ampToUid;
	unsigned int lastUID;

	ParameterTree *tree;
	ModelFitter *fitter;
	std::vector<double> resultGraph, ffGraph, sfGraph, bgGraph;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> resultGraph2D;

	// If there is a current job in progress
	JobState state;
	JobType type;
	unsigned long long beginning;
	double progress;
	int error;
	wchar_t errorMsg[1024];
	int *pStop;

	Job(unsigned int id = 0, const char *init = NULL, const wchar_t *desc = NULL);

	void ReportError(const wchar_t *fmt, ...);

	void Stop();

	void Destroy();

	void ResetFitter();

	// Create new map entries for model and amplitude
	unsigned int AssignModel(IModel *model);
	unsigned int AssignAmplitude(Amplitude *amp);

	JobStatus	jobStatus;
};

#endif
