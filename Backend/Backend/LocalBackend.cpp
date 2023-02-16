
#include "PDBAmplitude.h"
#include "LocalBackend.h"
#define RAPIDJSON_HAS_STDSTRING 1
#include "../../BackendCommunication/LocalCommunication/LocalComm.h"
#include "JobManager.h"
#include "Job.h"
#include "Model.h"
#include "Geometry.h"
#include "Symmetry.h"
#include "fitting.h"
#include "ModelContainer.h"

#include <string>
#include <sstream>

//#define RAPIDJSON_HAS_STDSTRING 1 // Needs to be defined before LocalComm.h which eventually include rapidjson
#include <rapidjson/document.h>
#include "../../Conversions/JsonWriter.h"

using namespace std;
using namespace rapidjson;

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif


#include <ctime>
#include <string>
#include <map>
#include <locale>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#define CONTAINER_PREFIX L""
#else
#define CONTAINER_PREFIX L"./Backend/lib"
#endif

#define DEFAULT_CONTAINER L"xplusmodels"

#ifdef _WIN32
#define CONTAINER_SUFFIX L".dll"
#else
#define CONTAINER_SUFFIX L".so"
#endif

// GLOBAL containing a temporary array of opened containers
static std::map<std::wstring, HMODULE> s_containers;

typedef int (*qCountFunc)();
typedef ModelCategory (*qCatFunc)(int catInd);
typedef ModelInformation (*qModelFunc)(int ind);
typedef IModel* (*getModelFunc)(int ind);
typedef void* (*getModelInfoProcFunc)(int ind, InformationProcedure type);


//////////////////////////////////////////////////////////////////////////
// Container Queries
//////////////////////////////////////////////////////////////////////////

// Helper function
static HMODULE GetBackend(const wchar_t *container) {
	std::wstring scon;
	if(!container) 
	{
        scon = CONTAINER_PREFIX;
		scon += DEFAULT_CONTAINER;
	} 
	else
		scon = container;

	scon += CONTAINER_SUFFIX;

	HMODULE hMod = NULL;
	if(s_containers.find(scon) != s_containers.end()) 
	{
		// Module already loaded
		hMod = s_containers[scon];
	} 
	else 
	{
		// Module not loaded
#ifdef _WIN32
		hMod = LoadLibraryW(scon.c_str());
#else
        hMod = dlopen(wstringtoutf8(scon).c_str(), RTLD_LAZY);
        if(!hMod)
        	std::cout << "Can't open " << wstringtoutf8(scon) << ":\n" << dlerror() << "\n";
#endif	
		if(hMod)
			s_containers[scon] = hMod;
	}
	if (!hMod)
		throw backend_exception(ErrorCode::ERROR_BACKEND);
	
	return hMod;
}

LocalBackend::LocalBackend()
{
}

LocalBackend::~LocalBackend()
{
}

int LocalBackend::HandleQueryCategoryCount(const wchar_t *container) {
	return GetNumCategories();
}

int LocalBackend::HandleQueryModelCount(const wchar_t *container) {
	return GetNumModels();

}


ModelCategory LocalBackend::HandleQueryCategory(const wchar_t *container, int catInd) {
	return GetCategoryInformation(catInd);
}

ModelInformation LocalBackend::HandleQueryModel(const wchar_t *container, int index) {
	return GetModelInformation(index);
}

//////////////////////////////////////////////////////////////////////////

#define LOG(...) JobManager::GetInstance().LogMessage(__VA_ARGS__)

JobPtr LocalBackend::HandleCreateJob(const wchar_t *description) {
	LOG(L"Create job: %ws\n", description);
	JobPtr rc = JobManager::GetInstance().CreateJob("local", description);


	return rc;
}

void LocalBackend::HandleDestroyJob(JobPtr job) {
	LOG(L"Destroy job: %d\n", job);
	// Blocking-stop a currently running job
	JobManager::GetInstance().StopJob(job, true);

	JobManager::GetInstance().DestroyJob(job);
}

ModelPtr LocalBackend::HandleCreateModel(JobPtr job, const wchar_t *container, 
										 int modelIndex, EDProfile profile) {
	Job j = JobManager::GetInstance().GetJobInformation(job);

	LOG(L"Create model: %d %ws %d", job, container, modelIndex);

	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));
	{
		lock_guard<mutex> joblock (*j.jobMutex);

		// Try to create the model
		//HMODULE hMod = GetBackend(container);
		//if(!hMod)
		//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

		// Get model
		//getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");
		//if(!getModel)
		//	throw(backend_exception(ERROR_INVALIDCONTAINER, g_errorStrings[ERROR_INVALIDCONTAINER]));

		IModel *model = GetModel(modelIndex);
		if(!model)
			throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

		// If it is a symmetry
		if(dynamic_cast<Symmetry *>((ISymmetry *)model) != NULL)
			throw(backend_exception(ERROR_INVALIDMODEL, g_errorStrings[ERROR_INVALIDMODEL]));

		/* WHY ARE WE DICTATING A SYMMETRIC DISCRETE PROFILE? IT SHOULD BE IN THE MODEL INFORMATION */
		// Set its ED profile.
// 		if(dynamic_cast<Geometry *>(model))
// 			dynamic_cast<Geometry *>(model)->SetEDProfile(profile);

		res = (ModelPtr)j.AssignModel(model);

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L" --> %d\n", res);

	return res;
}

ModelPtr LocalBackend::HandleCreateCompositeModel(JobPtr job) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	LOG(L"Create combined model: %d --> ", job);

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		res = (ModelPtr)j.AssignModel(new CompositeModel());

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L"%d\n", res);

	return res;
}

ModelPtr LocalBackend::HandleCreateDomainModel(JobPtr job) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	LOG(L"Create domain model: %d --> ", job);

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		res = (ModelPtr)j.AssignModel(new DomainModel());

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L"%d\n", res);

	return res;
}

ModelPtr LocalBackend::HandleCreateScriptedModel(JobPtr job, const char *script, unsigned int len) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));
	{
		lock_guard<mutex> joblock (*j.jobMutex);

		std::string scrstr (script, len);

		res = (ModelPtr)j.AssignModel(new LuaModel(scrstr, NULL));

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L"Job %d: Created scripted model (uid %d --> %ws)\n", job, res, (j.uidToAmp.find(res) != j.uidToAmp.end()) ? L"exists." : L"doesn't exist!!");


	return res;
}

ModelPtr LocalBackend::HandleCreateScriptedSymmetry(JobPtr job, const char *script, unsigned int len) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		std::string scrstr (script, len);

		res = (ModelPtr)j.AssignAmplitude(new LuaSymmetry(scrstr, NULL));

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L"Job %d: Created scripted symmetry (uid %d --> %ws)\n", job, res, (j.uidToAmp.find(res) != j.uidToAmp.end()) ? L"exists." : L"doesn't exist!!");

	return res;
}


ModelPtr LocalBackend::HandleCreateFileAmplitude(JobPtr job, AmpFileType type, const wchar_t *filename, bool bCenter, const wchar_t *anomfilename, bool electronPDB) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND])); //ERROR_JOBNOTFOUND;

	LOG(L"Create file amplitude: %d %ws\n", job, filename);

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		switch(type) {
			default:
				throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

			case AF_PDB:
				{
					std::wstring wfname = filename;
					std::string fname;
					fname.assign(wfname.begin(), wfname.end());

					std::string anomfname = "";
					if (anomfilename)
					{
						std::wstring wanomfname = anomfilename;
						anomfname.assign(wanomfname.begin(), wanomfname.end());
					}
					if (electronPDB)
						res = (ModelPtr)j.AssignAmplitude(new ElectronPDBAmplitude(fname, bCenter, anomfname));
					else
						res = (ModelPtr)j.AssignAmplitude(new XRayPDBAmplitude(fname, bCenter, anomfname));
					break;
				}				

			case AF_AMPGRID:
				{
					std::wstring wfname = filename;
					std::string fname;
					fname.assign(wfname.begin(), wfname.end());

					res = (ModelPtr)j.AssignAmplitude(new AmpGridAmplitude(fname));
					break;
				}
				break;
		}
		

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	return res;
}

ModelPtr LocalBackend::HandleCreateFileAmplitude(JobPtr job, AmpFileType type, const char *buffer, unsigned int bufferSize, const char *fileNm, unsigned int fnSize, bool bCenter, const char *anomFilename, unsigned int anomfnSize, bool electronPDB) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND])); //ERROR_JOBNOTFOUND;

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		switch(type) {
			default:
				throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

			case AF_PDB:
				if (electronPDB)
					res = (ModelPtr)j.AssignAmplitude(new ElectronPDBAmplitude(buffer, bufferSize, fileNm, fnSize, bCenter, anomFilename, anomfnSize));
				else
					res = (ModelPtr)j.AssignAmplitude(new XRayPDBAmplitude(buffer, bufferSize, fileNm, fnSize, bCenter, anomFilename, anomfnSize));
				break;

			case AF_AMPGRID:
				res = (ModelPtr)j.AssignAmplitude(new AmpGridAmplitude(buffer, bufferSize));
				break;
		}


		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	return res;
}

ModelPtr LocalBackend::HandleCreateGeometricAmplitude(JobPtr job, ModelPtr model) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND])); //ERROR_JOBNOTFOUND;

	LOG(L"Create geometric amplitude: %d %d", job, model);

	if(model != 0) {
		if(j.uidToModel.find(model) == j.uidToModel.end())
			throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

		// Geometric amplitude must be of type FFModel
		if(!dynamic_cast<FFModel *>(j.uidToModel[model]))
			throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));
	}

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		if(model)
			res = (ModelPtr)j.AssignAmplitude(new GeometricAmplitude((FFModel *)j.uidToModel[model]));
		else
			res = (ModelPtr)j.AssignAmplitude(new GeometricAmplitude(NULL));
	
		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L" --> %d\n", res);

	return res;
}

ModelPtr LocalBackend::HandleCreateSymmetry(JobPtr job, const wchar_t *container, 
											int symmetryIndex) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		// Try to create the model
		//HMODULE hMod = GetBackend(container);
		//if(!hMod)
		//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

		// Get model
		//getModelFunc getModel = (getModelFunc)GetProcAddress(hMod, "GetModel");
		//if(!getModel)
		//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

		ISymmetry *isymm = (ISymmetry *)GetModel(symmetryIndex);
		if(!isymm)
			throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));
		Symmetry *symm = dynamic_cast<Symmetry *>(isymm);
		if(!symm)
			throw(backend_exception(ERROR_INVALIDMODELTYPE, g_errorStrings[ERROR_INVALIDMODELTYPE]));

		res = (ModelPtr)j.AssignAmplitude(symm);

		j.lastAccess = time(NULL);

		JobManager::GetInstance().UpdateJob(j);
	}

	LOG(L"Job %d: Created symmetry (uid %d --> %ws)\n", job, res, (j.uidToAmp.find(res) != j.uidToAmp.end()) ? L"exists." : L"doesn't exist!!");

	return res;
}

// Helper functions for HandleDestroyModel
static void DestroyGenericModel(IModel *mod, bool bDestroyChildren);
static void DestroyAmplitude(Amplitude *amp, bool bDestroyChildren);
static void DestroyDomain(DomainModel *domain, bool bDestroyChildren);
static void DestroyGeometricAmplitude(GeometricAmplitude *amp, bool bDestroyChildren);
static void DestroySymmetry(ISymmetry *amp, bool bDestroyChildren);
static void DestroyCompositeModel(CompositeModel *cm, bool bDestroyChildren);

static void DestroyGenericModel(IModel *mod, bool bDestroyChildren) {
	if (!mod)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);

	LOG(L"Destroy model: %p, %S children\n", mod, bDestroyChildren ? "With" : "Without");

	if(dynamic_cast<CompositeModel *>(mod))
		DestroyCompositeModel(dynamic_cast<CompositeModel *>(mod), bDestroyChildren);
	else if(dynamic_cast<DomainModel *>(mod))
		DestroyDomain(dynamic_cast<DomainModel *>(mod), bDestroyChildren);
	else // All the other cases
		delete mod;
}

static void DestroyAmplitude(Amplitude *amp, bool bDestroyChildren) {
	if(!amp)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);

	if(dynamic_cast<GeometricAmplitude *>(amp))
		DestroyGeometricAmplitude(dynamic_cast<GeometricAmplitude *>(amp), bDestroyChildren);
	else if(dynamic_cast<ISymmetry *>(amp))
		DestroySymmetry(dynamic_cast<ISymmetry *>(amp), bDestroyChildren);
	else // All the other cases
		delete amp;
	amp = NULL;
}

static void DestroyDomain(DomainModel *domain, bool bDestroyChildren) {
	LOG(L"Destroy domain model: %p, %S children\n", domain, bDestroyChildren ? "With" : "Without");

	if(!domain)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);

	if(bDestroyChildren) {
		int nc = domain->GetNumSubAmplitudes();

		for(int i = 0; i < nc; i++)
			DestroyAmplitude(domain->GetSubAmplitude(i), bDestroyChildren);

	}

	delete domain;
}

static void DestroyGeometricAmplitude(GeometricAmplitude *amp, bool bDestroyChildren) {
	if(!amp)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);
	
	if(bDestroyChildren)
		delete amp->GetGeometry();

	delete amp;
}

static void DestroySymmetry(ISymmetry *amp, bool bDestroyChildren) {
	if(!amp)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);

	if(bDestroyChildren) {
		int nc = amp->GetNumSubAmplitudes();

		for(int i = 0; i < nc; i++) {
			DestroyAmplitude(amp->GetSubAmplitude(i), bDestroyChildren);
//			j.uidToAmp.erase(model);	   // TODO: THIS NEEDS TO BE DONE SOMEWHERE!!
		}
	}

	delete amp;
}

static void DestroyCompositeModel(CompositeModel *cm, bool bDestroyChildren) {
	if(!cm)
		throw backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]);

	if(bDestroyChildren) {
		std::vector<IModel *> additives = cm->GetSubModels();
		std::vector<IModel *> multipliers = cm->GetMultipliers();

		for(IModel *mod : additives)
			DestroyGenericModel(mod, bDestroyChildren);
		for(IModel *mod : multipliers)
			DestroyGenericModel(mod, bDestroyChildren);		
	}
	
	delete cm;
}

ErrorCode LocalBackend::HandleDestroyModel(JobPtr job, ModelPtr model, bool bDestroyChildren /*= false*/) {

	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if (j.uid == 0)
	{
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));
	}

	LOG(L"Destroy model: %d %d %S children\n", job, model, bDestroyChildren ? "With" : "Without");

	{
		lock_guard<mutex> joblock (*j.jobMutex);

		JobManager::LogMessage(L"%llu: Deleting ModelPtr %d of job %d (%s) by %s.\n",
			(unsigned long long)time(NULL), model, job, j.description, j.instigator);
		if(j.uidToModel.find(model) != j.uidToModel.end()) {
			IModel *mod = j.uidToModel[model];

			DestroyGenericModel(mod, bDestroyChildren);

			j.uidToModel.erase(model);
		} else if(j.uidToAmp.find(model) != j.uidToAmp.end()) {
			Amplitude *amp = j.uidToAmp[model];

			std::vector<ModelPtr> subs, syms;
			ISymmetry *isa = dynamic_cast<ISymmetry *>(amp);
			if(isa != NULL) {
				syms.push_back(model);
			}
			while(bDestroyChildren && isa != NULL && syms.size() > 0) {
				ModelPtr ii = syms.back();
				syms.pop_back();
				isa = dynamic_cast<ISymmetry *>(j.uidToAmp[ii]);
				int nc = isa->GetNumSubAmplitudes();
				for(int i = 0; i < nc; i++) {
					Amplitude *ai = isa->GetSubAmplitude(i);
					// finds uid
					for (std::map<unsigned int, Amplitude *>::iterator it=j.uidToAmp.begin(); it!=j.uidToAmp.end(); ++it) {
						if(it->second == ai) {
							subs.push_back(it->first);
							if(dynamic_cast<ISymmetry *>(it->second) != NULL)
								syms.push_back((ModelPtr)it->first);
						} // if(it->second == ai)
					} // for map
				} // for i
			} // while
			
			DestroyAmplitude(amp, bDestroyChildren);
			j.ampToUid.erase(amp);
			// TODO: This does not remove the children in bDestroyChildren
			//TODO- do i need to destroy children in amptoUID?
			j.uidToAmp.erase(model);
			for (int i = 0; i < subs.size(); i++)
			{
				j.uidToAmp.erase(subs[i]);
			}


		}
		else 
		{
			throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));
		}
		JobManager::LogMessage(L"%llu: Remaining models of job %d:\n",
			(unsigned long long)time(NULL), job);
		for (std::map<unsigned int, Amplitude *>::iterator it=j.uidToAmp.begin(); it!=j.uidToAmp.end(); ++it) {
			JobManager::LogMessage(L"%llu:\t\t%u:\n",
				(unsigned long long)time(NULL), it->first);
		}
		JobManager::GetInstance().UpdateJob(j);
	}

	return OK;
}

ErrorCode LocalBackend::HandleFit(JobPtr job, const ParameterTree& tree, const std::vector<double>& x, 
								  const std::vector<double>& y, const std::vector<int>& mask, 
								  const FittingProperties& fp)
{
	return JobManager::GetInstance().StartFitJob(this, job, tree, x, y, mask, fp);
}

ErrorCode LocalBackend::HandleGenerate(JobPtr job, const ParameterTree& tree, const std::vector<double>& x,
										const FittingProperties& fp) {
	return JobManager::GetInstance().StartGenerateJob(this, job, tree, x, fp);
}

ErrorCode LocalBackend::HandleGenerate2D(JobPtr job, const ParameterTree& tree, const std::vector<double>& x,
	const FittingProperties& fp) {

	ErrorCode ret = JobManager::GetInstance().StartGenerate2DJob(this, job, tree, x, fp);
	return ret;
}

void LocalBackend::HandleStop(JobPtr job) {
	JobManager::GetInstance().StopJob(job, false);
}

bool LocalBackend::HandleGetLastErrorMessage(JobPtr job, wchar_t *message, size_t length)
{
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	if(length > 1024)
		length = 1024;

	wcsncpy(message, j.errorMsg, length);

	return true;
}

JobType LocalBackend::HandleGetJobType(JobPtr job) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	return j.type;
}

JobStatus LocalBackend::GetJobStatus(JobPtr job)
{
	Job j = JobManager::GetInstance().GetJobInformation(job);

	if (j.uid == 0)
		throw backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]);
	
	return j.jobStatus;
}

int LocalBackend::HandleGetGraphSize(JobPtr job) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]);

	return (int)j.resultGraph.size();
}

void LocalBackend::HandleGet2DGraphSize(JobPtr job, int& rows, int &cols) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if (j.uid == 0)
		throw backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]);

	cols = (int)j.resultGraph2D.cols();
	rows = (int)j.resultGraph2D.rows();
}

bool LocalBackend::HandleGetGraph(JobPtr job, OUT double *yPoints, int nPoints)
{
	if(!yPoints)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	memcpy(yPoints, &j.resultGraph[0], nPoints * sizeof(double));

	return true;
}

bool LocalBackend::HandleGet2DGraph(JobPtr job, OUT MatrixXd &yPoints, int rows, int cols)
{
	
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if (j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	yPoints = j.resultGraph2D;

	return true;
}

ErrorCode LocalBackend::HandleGetResults(JobPtr job, OUT ParameterTree& tree) {
	Job j = JobManager::GetInstance().GetJobInformation(job);
	ModelPtr res = (ModelPtr)NULL;
	if (j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	lock_guard<mutex> joblock (*j.jobMutex);
	tree = *j.tree;		

	if (j.error != 0) {
		std::wstring wm(j.errorMsg);
		std::string m(wm.begin(), wm.end());
		throw(backend_exception(j.error, m.c_str()));
	}
	return OK;
}

bool LocalBackend::HandleGetFittingErrors(JobPtr job, OUT double *paramErrors, 
										  OUT double *modelErrors,
										  int nMutParams, int nPoints) 
{
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	if(!j.fitter)
		throw(backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]));
	if(j.state != JS_IDLE)
		throw(backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]));
	
	// Get the results
	std::vector<double> pErrors, mErrors;
	j.fitter->GetFittingErrors(pErrors, mErrors);

	// Copy the results
	if(paramErrors)
		memcpy(paramErrors, &pErrors[0], nMutParams * sizeof(double));
	if(modelErrors)
		memcpy(modelErrors, &mErrors[0], nPoints * sizeof(double));

	return true;
}

ErrorCode LocalBackend::HandleGetLayerParamNames(const wchar_t *container, int index, OUT char **lpNames, int nlp)
{
	
	if(!lpNames)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	// Try to obtain the model
	//HMODULE hMod = GetBackend(container);
	//if(!hMod)
	//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

	// Get model information procedure
	//getModelInfoProcFunc getmipfunc = (getModelInfoProcFunc)GetProcAddress(hMod, "GetModelInformationProcedure");
	//if(!getmipfunc)
	//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

	fGetLayerParamName getfunc = (fGetLayerParamName)GetModelInformationProcedure(index, IP_LAYERPARAMNAME);
	if(!getfunc)
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	for(int i = 0; i < nlp; i++)
		strncpy(lpNames[i], getfunc(i, NULL /*TODO::EDP*/).c_str(), 256);

	return OK;
}

ErrorCode LocalBackend::HandleGetDisplayParamInfo(const wchar_t *container, int index, OUT char **disp, int nDisp) {
	
	if(!disp)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	// Try to obtain the model
	//HMODULE hMod = GetBackend(container);
	//if(!hMod)
	//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

	// Get model information procedure
	//getModelInfoProcFunc getmipfunc = (getModelInfoProcFunc)GetProcAddress(hMod, "GetModelInformationProcedure");
	//if(!getmipfunc)
	//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

	fGetDisplayParamName getfunc = (fGetDisplayParamName)GetModelInformationProcedure(index, IP_DISPLAYPARAMNAME);
	if(!getfunc)
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	for(int i = 0; i < nDisp; i++)
		strncpy(disp[i],getfunc(i).c_str(), 256);

	return OK;
}

ErrorCode LocalBackend::HandleGetExtraParamInfo(const wchar_t *container, int index, OUT ExtraParam *ep, int nEP) {
	
	if(!ep)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	// Try to obtain the model
	//HMODULE hMod = GetBackend(container);
	//if(!hMod)
	//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

	// Get model information procedure
	//getModelInfoProcFunc getmipfunc = (getModelInfoProcFunc)GetProcAddress(hMod, "GetModelInformationProcedure");
	//if(!getmipfunc)
	//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

	fGetExtraParameter getfunc = (fGetExtraParameter)GetModelInformationProcedure(index, IP_EXTRAPARAMETER);
	if(!getfunc)
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	for(int i = 0; i < nEP; i++)
		ep[i] = getfunc(i);

	return OK;
}

ErrorCode LocalBackend::HandleGetLayerInfo(const wchar_t *container, int index, int layerIndex, 
									  OUT char *layerName, OUT int *applicability, 
									  OUT double *defaultValues, int nlp) {
	
	if(layerIndex < 0)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	// Try to obtain the model
	//HMODULE hMod = GetBackend(container);
	//if(!hMod)
	//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

	// Get model information procedure
	//getModelInfoProcFunc getmipfunc = (getModelInfoProcFunc)GetProcAddress(hMod, "GetModelInformationProcedure");
	//if(!getmipfunc)
	//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

	fGetLayerName getlnfunc = (fGetLayerName)GetModelInformationProcedure(index, IP_LAYERNAME);
	fIsParamApplicable getappfunc = (fIsParamApplicable)GetModelInformationProcedure(index, IP_ISPARAMAPPLICABLE);
	fGetDefaultParamValue getdvfunc = (fGetDefaultParamValue)GetModelInformationProcedure(index, IP_DEFAULTPARAMVALUE);
	if(!getlnfunc || !getappfunc || !getdvfunc)
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));


	// Fill all the information
	if(layerName)
		strncpy(layerName, getlnfunc(layerIndex).c_str(), 256);
	for(int i = 0; i < nlp; i++) {
		if(applicability)
			applicability[i] = getappfunc(layerIndex, i) ? 1 : 0;
		if(defaultValues)
			defaultValues[i] = getdvfunc(i, layerIndex, NULL /*TODO::EDP*/);
	}

	return OK;
}

ErrorCode LocalBackend::HandleGetDisplayParams(const wchar_t *container, int index, const paramStruct *params, OUT double *disp, int nDisp) {
	
	if(!disp)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	// Try to obtain the model
	//HMODULE hMod = GetBackend(container);
	//if(!hMod)
	//	throw(backend_exception(ERROR_FILENOTFOUND, g_errorStrings[ERROR_FILENOTFOUND]));

	// Get model information procedure
	//getModelInfoProcFunc getmipfunc = (getModelInfoProcFunc)GetProcAddress(hMod, "GetModelInformationProcedure");
	//if(!getmipfunc)
	//	throw(backend_exception(ERROR_BACKEND, g_errorStrings[ERROR_BACKEND]));

	fGetDisplayParamValue getfunc = (fGetDisplayParamValue)GetModelInformationProcedure(index, IP_DISPLAYPARAMVALUE);
	if(!getfunc)
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	for(int i = 0; i < nDisp; i++)
		disp[i] = getfunc(i, params);
	
	return OK;
}


void LocalBackend::NotifyProgress(JobPtr job, double progress) {

	Job j = JobManager::GetInstance().GetJobInformation(job);
	j.jobStatus.code = -1;
	j.jobStatus.isRunning = true;
	j.jobStatus.progress = progress;
	JobManager::GetInstance().UpdateJob(j);
}

void LocalBackend::NotifyCompletion(JobPtr job, bool bSuccess, int code, const std::string & msg) {

	Job j = JobManager::GetInstance().GetJobInformation(job);
	j.jobStatus.code = code;
	j.jobStatus.isRunning = false;
	j.jobStatus.progress = 1.0;
	j.jobStatus.code_string = msg;
	JobManager::GetInstance().UpdateJob(j);
}


std::string LocalBackend::HandleGetAmplitude(JobPtr job, ModelPtr model, std::string filename){
	Job j = JobManager::GetInstance().GetJobInformation(job);
	if (j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	if (j.uidToAmp.find(model) == j.uidToAmp.end())
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	Amplitude * amp = j.uidToAmp[model];

	if (!amp->GridIsReadyToBeUsed()) {
		throw(backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]));
	}

	amp->WriteAmplitudeToFile(filename);
	return filename;
}

std::string LocalBackend::HandleGetAmplitude(JobPtr job, ModelPtr model) {
	
	stringstream ss;

	Job j = JobManager::GetInstance().GetJobInformation(job);
	if (j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	if (j.uidToAmp.find(model) == j.uidToAmp.end())
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));

	Amplitude * amp = j.uidToAmp[model];

	if(!amp->GridIsReadyToBeUsed()) {
		throw(backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]));
	}

	if (!amp->WriteAmplitudeToStream(ss) == PDB_OK)
	{
		throw(backend_exception(ERROR_GENERAL, g_errorStrings[ERROR_GENERAL]));
	}

	return ss.str();	
}

std::string LocalBackend::HandleGetPDB(JobPtr job, ModelPtr model, bool electron) {
	
	stringstream ss;

	

	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	Amplitude *amp = NULL;
	ISymmetry *sym = NULL;
	if (j.uidToAmp.find(model) == j.uidToAmp.end())
	{
		if (j.uidToModel.find(model) == j.uidToModel.end())
		{
			throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));
		}
		else
			sym = dynamic_cast<ISymmetry*>(j.uidToModel[model]);
	}
	else
	{
		amp = j.uidToAmp[model];
		sym = dynamic_cast<ISymmetry*>(amp);
	}

	if (sym && sym->SavePDBFile(ss)) {
		return ss.str();
	}

	PDBAmplitude* pa;
	if (electron)
	{
		pa = dynamic_cast<ElectronPDBAmplitude*>(amp);
	}
	else
	{
		pa = dynamic_cast<XRayPDBAmplitude*>(amp);
	}

	if (pa && pa->SavePDBFile(ss)) {
		return ss.str();
	}
	
	throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));
}

ErrorCode LocalBackend::HandleGetDomainHeader(JobPtr job, ModelPtr model, OUT char *hdr, int length) {
 	if(!hdr)
		throw(backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]));

	Job j = JobManager::GetInstance().GetJobInformation(job);
	if(j.uid == 0)
		throw(backend_exception(ERROR_JOBNOTFOUND, g_errorStrings[ERROR_JOBNOTFOUND]));

	if(j.uidToModel.find(model) == j.uidToModel.end())
		throw(backend_exception(ERROR_MODELNOTFOUND, g_errorStrings[ERROR_MODELNOTFOUND]));
	// catch errors that generate/fit throw before creating grid/ calculations
	  
	if (j.error != 0) {
		std::wstring wm(j.errorMsg);
		std::string m(wm.begin(), wm.end());
		throw(backend_exception(j.error, m.c_str()));
	}
	std::string hd;
		
	if(j.uidToAmp.find(model) != j.uidToAmp.end()) {	// It's amp
		Amplitude * amp = j.uidToAmp[model];
		amp->GetHeader(0, hd);
	} else {
		IModel * mdl = j.uidToModel[model];
		DomainModel *dmm = dynamic_cast<DomainModel *>(mdl);
		CompositeModel *cmm = dynamic_cast<CompositeModel *>(mdl);
		if(dmm) 
			dmm->GetHeader(0, hd);
		else if(cmm)
			cmm->GetHeader(0, hd);
		else
			throw(backend_exception(ERROR_INVALIDMODEL, g_errorStrings[ERROR_INVALIDMODEL]));

	}
	//	memcpy(container, &j.resultG/raph[0], nPoints * sizeof(double));
	strncpy(hdr, hd.c_str(), length);

	return OK;

}


