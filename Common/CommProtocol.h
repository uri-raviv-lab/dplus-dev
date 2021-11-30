#ifndef __COMM_PROTOCOL_H
#define __COMM_PROTOCOL_H

#include "Common.h"

#pragma warning (disable: 4251)

// TODO::dox Finish documentation

// The default interval in seconds between each backend->frontend progress report
#define UPDATE_INTERVAL_MS 1000

struct JobStatus
{
	bool	isRunning;
	double	progress;
	int		code;
	std::string code_string;
};



// An abstract class defining the frontend side of the communication
class EXPORTED FrontendComm
{
protected:
	FrontendComm() {}	
public:
	virtual ~FrontendComm() {}

	/**
	 * IsValid: Returns true iff the frontend is valid and can be used.
	 *
	 * @return True iff the frontend is valid.
	 **/
	virtual bool IsValid() = 0;

	virtual std::string GetAllModelMetadata() = 0; // No container name, get the data for all containers
	// Query containers for models
	//////////////////////////////////////////////////////////////////////////
	virtual int QueryCategoryCount(const wchar_t *container) = 0;
	virtual int QueryModelCount(const wchar_t *container) = 0;
	virtual ModelCategory QueryCategory(const wchar_t *container, int catInd) = 0;
	virtual ModelInformation QueryModel(const wchar_t *container, int index) = 0;

	// Job Management
	//////////////////////////////////////////////////////////////////////////
	/**
	 * Creates a new backend job.
	 * @param description A description of the job, up to 256 characters.
	 * @param progHandler A function pointer (can be NULL) for notifying progress
	 *                    back to the UI.
	 * @param completionHandler A function pointer (can be NULL) for notifying completion
	 *                          information back to the UI.
	 * @return NULL on error.
	 */
	virtual JobPtr CreateJob(const wchar_t *description, progressFunc progHandler = NULL,
							 notifyCompletionFunc completionHandler = NULL, void *args = NULL) = 0;
	
	/**
	 * Stops a job and destroys it.
	 * @param job The given job to destroy.
	 */
	virtual void DestroyJob(JobPtr job) = 0;

	// Model Management
	//////////////////////////////////////////////////////////////////////////

	/**
	 * Creates a new model for use in a geometric amplitude, combined model or by itself inside a backend job.
	 * @param job The job pointer.
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container.
	 * @param profile The Electron Density profile to use.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateModel(JobPtr job, const wchar_t *container, int modelIndex, EDProfile profile) = 0;
	
	/**
	 * Creates a new composite model (containing multiple domains, optionally background and outer SF) 
	 * for use in a backend job.
	 * @param job The job pointer.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateCompositeModel(JobPtr job) = 0;

	/**
	 * Creates a new script-based model for use inside a backend job.
	 * @param job The job pointer.
	 * @param script The script that defines the model.
	 * @param len The script length, in characters.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateScriptedModel(JobPtr job, const char *script, const char * filename, unsigned int len) = 0;

	/**
	 * Creates a new domain (containing the amplitude tree) for use in a backend job.
	 * @param job The job pointer.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateDomainModel(JobPtr job) = 0;

	/**
	 * Creates a new file amplitude (from filename) for use in a domain model.
	 * @param job The job pointer.
	 * @param type The file type (e.g. PDB, AmpGrid).
	 * @param filenames The filenames (in the backend) to use as input.
	 * @param numFiles The number of files to be transfered to the backend as input.
	 * @param bCenter Tells the backend if a PDB file is loaded, whether to center it or not.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateFileAmplitude(JobPtr job, AmpFileType type, const wchar_t **filenames, int numFiles, bool bCenter) = 0;

	/**
	 * Creates a new file amplitude (from a local buffer) for use in a domain model.
	 * @param job The job pointer.
	 * @param type The file type (e.g. PDB, AmpGrid).
	 * @param buffer The local buffer to use.
	 * @param bufferSize The buffer's size.
	 * @param fileNm The filename.
	 * @param fnSize The filename's size.
	 * @param bCenter Tells the backend if a PDB file is loaded, whether to center it or not.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateFileAmplitude(JobPtr job, AmpFileType type, const char **buffer, unsigned int *bufferSize, const char **fileNm, unsigned int *fnSize, int numFiles, bool bCenter) = 0;

	/**
	 * Creates a new geometric amplitude, optionally based on an existing model.
	 * @param job The job pointer.
	 * @param model The model to use inside the geometric amplitude (can be modified later by setting sub-model 0).
	 *              If model is NULL, creates an empty geometric amplitude.
	 * @return NULL on error (for instance, the model pointer is invalid), or the model UID if succeeded.
	 */
	virtual ModelPtr CreateGeometricAmplitude(JobPtr job, ModelPtr model) = 0;

	/**
	 * Creates a new symmetry amplitude based on an existing model.
	 * @param job The job pointer.
	 * @param symmetry Determines the symmetry type to use.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateSymmetry(JobPtr job, const wchar_t *container, int symmetryIndex) = 0;

	/**
	 * Creates a new scripted symmetry amplitude.
	 * @param job The job pointer.
	 * @param script Contains the script for the symmetry.
	 * @param len The script's length in characters.
	 * @return NULL on error, or the model UID if succeeded.
	 */
	virtual ModelPtr CreateScriptedSymmetry(JobPtr job, const char *script, const char * filename, unsigned int len) = 0;

	/**
	 * Destroys a model, if not in use. 
	 * NOTE: When a job is destroyed, its models are automatically destroyed, unless DestroyModel is called.
	 * NOTE: There are four different implementations to this:
	 *       1. Model (Where bDestroyChildren does not matter)
	 *		 2. Domain
	 *		 3. Symmetries and Amplitudes (Amplitude)
	 *		 4. CompositeModel
	 * @param job The job containing the model.
	 * @param model The given model.
	 * @param bDestroyChildren (Only apples to symmetries and geometric amplitudes) If true, destroys the
	 *                         underlying models as well.
	 */
	virtual ErrorCode DestroyModel(JobPtr job, ModelPtr model, bool bDestroyChildren = true) = 0;

	// Actions
	//////////////////////////////////////////////////////////////////////////
	/**
	 * Performs model fitting for the chosen model type. The rest of the models (if not NULL)
	 * are automatically added and multiplied to the fitting model.
	 * This function is non-blocking. This means that it will return immediately
	 * when the fitting process begins. Progress is reported using the HandleProgress
	 * and HandleCompletion functions.
	 *
	 * @param job The job to fit.
	 * @param model The model to fit.
	 * @param x The x points for the data to fit to.
	 * @param y The y points for the data to fit to.
	 * @param mask The points to mask the data in (0 - unmasked, 1 - masked)
	 * @param fp The fitting properties to use for this process.
	 * @return Returns the appropriate error codes.
	 */
	virtual ErrorCode Fit(JobPtr job, const wchar_t *luaScript, const std::vector<double>& x,
		const std::vector<double>& y, const std::vector<int>& mask, bool useGPU, std::string* message = nullptr) = 0;
	/**
	 * Performs model generation for the chosen model type. The rest of the models
	 * are NOT added to the final result, unless specified in the "types" parameter.
	 * This function is non-blocking. This means that it will return immediately
	 * when the generation process begins. Progress is reported using the HandleProgress
	 * and HandleCompletion functions.
	 *
	 * @param job The job to generate in.
	 * @param model The model to use.
	 * @param x The x points for the function to be evaluated in.	 
	 * @param fp The fitting properties to use for this process.
	 * @return Returns the appropriate error codes.
	 */
	virtual ErrorCode Generate(JobPtr job, const wchar_t *luaScript, const std::vector<double> &x, bool useGPU) = 0;

	/**
	Check the Capabilities of the backend computer
	@return Returns the appropriate error codes.
	*/
	virtual ErrorCode CheckCapabilities(bool checkTdr = true) = 0;

	// Non-blocking stop
	virtual void Stop(JobPtr job) = 0;


	/**
	 * WaitForFinish: Waits for the job to finish its task, or returns immediately
	 *                if job does not exist or is idle.
	 *
	 * @param job The job to wait for.
	 **/
	virtual void WaitForFinish(JobPtr job) = 0;

	// Retrieve data from backend
	//////////////////////////////////////////////////////////////////////////

	
	/**
	 * GetLastErrorMessage : Returns the job's last error message string (up to 1024 characters)
	 * @param job The given job.
	 * @return True iff succeeded
	 **/
	virtual bool GetLastErrorMessage(JobPtr job, wchar_t *message, size_t length) = 0;

	/**
	 * GetJobType : Returns the job's type (either generation or fitting).
	 * @param job The given job.
	 * @return The job's type
	 **/
	virtual JobType GetJobType(JobPtr job) = 0;

	/**
	 * GetGraphSize : Returns latest intermediate graph's size.
	 * NOTE: Graph updating is now governed by the UI, on handle progress.
	 *
	 * @param job The given job.
	 * @return The latest intermediate graph's size.
	 **/
	virtual int GetGraphSize(JobPtr job) = 0;

	/**
	 * GetGraph : Returns the latest intermediate graph.
	 * NOTE: Graph updating is now governed by the UI, on handle progress.
	 *
	 * @param job The given job.
	 * @param yPoints The y points of the resulting graph.
	 * @param nPoints The number of points in y.
	 * @return True iff succeeded.
	 **/
	virtual bool GetGraph(JobPtr job, OUT double *yPoints, int nPoints) = 0;

	/**
	 * ExportAmplitude: Returns model's amplitude through strm
	 * @param job The given job.
	 * @param model The ModelPtr whose amplitude we are requesting
	 * @param filename A file to which the amplitude is written
	 * @return bool true iff succeeded
	 **/
	virtual bool ExportAmplitude(JobPtr job, ModelPtr model, const wchar_t *filename) = 0;

	/**
	 * @name	SavePDB
	 * @brief	Saves a PDB file of the all the atoms up to \p model
	 * @param	JobPtr job The job id
	 * @param	ModelPtr model The model up to which the atoms will be save to a PDB file
	 * @param	const wchar_t * filename The save filename
	 * @ret
	*/
	virtual bool SavePDB(JobPtr job, ModelPtr model, const wchar_t *filename) = 0;

	virtual void CheckJobProgress(JobPtr job) = 0;
	virtual void MarkCheckJobProgressDone(JobPtr job) = 0;

	virtual bool CheckJobOvertime(JobPtr job, unsigned long long startTime, unsigned long long sec) = 0;
	/**
	 * GetResults : Returns the latest parameter results from the fitting process.
	 *
	 * @param job The given job.
	 * @param model The model to get the results from. 
	 * @param p The results
	 * @return The corresponding error code
	 **/
	virtual ErrorCode GetResults(JobPtr job, OUT ParameterTree& tree) = 0;

	/**
	 * GetFittingErrors:  Returns the fitting errors from the last fitting process.
	 *
	 * @param job The given job.
	 * @param paramErrors The parameter errors.
	 * @param modelErrors The model errors.
	 * @param nMutParams The size of "paramErrors". (The number of mutable parameters)
	 * @param nPoints The size of "modelErrors".
	 * @return True iff succeeded.
	 **/
	/*virtual bool GetFittingErrors(JobPtr job, OUT double *paramErrors, 
								  OUT double *modelErrors, int nMutParams, int nPoints) = 0; */

	
	// Retrieve model data
	//////////////////////////////////////////////////////////////////////////

	/**
	 * GetLayerParamNames: Returns the layer parameter names
	 *
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container.
	 * @param lpNames An array of strings, each consisting of 256 characters
	 * @return True iff succeeded.
	 **/
	virtual ErrorCode GetLayerParamNames(const wchar_t *container, int index, OUT char **lpNames, int nlp) = 0;

	/**
	 * GetDisplayParamInfo: Returns the names of the display parameters of the current model.
	 *
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container. 
	 * @param disp An array (to be filled) of "nDisp" 256-character strings
	 * @param nDisp The number of elements in "disp"
	 * @return True iff succeeded.
	 **/
	virtual ErrorCode GetDisplayParamInfo(const wchar_t *container, int index, OUT char **disp, int nDisp) = 0;

	/**
	 * GetExtraParamInfo: Returns information about all the extra parameters in
	 *                    the model.
	 *
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container.
	 * @param ep An array of "nEP" extra parameter information structures.
	 * @param nEP The number of extra parameters in this model.
	 * @return True iff succeeded.
	 **/
	virtual ErrorCode GetExtraParamInfo(const wchar_t *container, int index, OUT ExtraParam *ep, int nEP) = 0;

	/**
	 * GetLayerInfo: Returns the layer information regarding the specified layer.
	 *               This includes the name of the layer and the applicability of
	 *               its current layer parameters.
	 *
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container.
	 * @param layerIndex The layer index to obtain information for.
	 * @param layerName A string consisting of 256 characters, specifying the name
	 *                  of the current layer.
	 * @param applicability An array of size "nlp" containing parameter applicability
	 *                      (1 if parameter can be filled, 0 if "N/A" should be displayed).
	 * @param defaultValues An array of size "nlp" specifying the default values for the
	 *                      parameters in this layer.
	 * @param nlp The number of layer parameters in the selected model.
	 * @return True iff succeeded.
	 **/
	virtual ErrorCode GetLayerInfo(const wchar_t *container, int index, int layerIndex, OUT char *layerName,
								   OUT int *applicability, OUT double *defaultValues,
								   int nlp) = 0;	

	/**
	 * GetDisplayParams: Returns the display parameters of the current model.
	 *
	 * @param container The name of the model container to look in (or NULL for default model container).
	 * @param modelIndex The index of the model in the container.
	 * @param params The parameters to compute the display parameters by
	 * @param disp An array (to be filled) of display parameter values
	 * @param nDisp The number of elements in "disp"
	 * @return True iff succeeded.
	 **/
	virtual ErrorCode GetDisplayParams(const wchar_t *container, int index,  const paramStruct *params, OUT double *disp, int nDisp) = 0;

	virtual ErrorCode GetDomainHeader(JobPtr job, ModelPtr model, OUT char *container, OUT int &length) = 0;

	virtual void AddFileToExistingModel(int modelPtr, const wchar_t *fileNm) = 0;
	virtual void RemoveFileFromExistingModel(int modelPtr, const wchar_t *fileNm) = 0;
};

// An abstract class defining the backend side of the communication
class BackendComm {
protected:
	BackendComm() {}	
public:
	virtual ~BackendComm() {}	

	virtual std::string CallBackend(std::string json) = 0;
};




#endif
