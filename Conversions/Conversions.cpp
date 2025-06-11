#include "Conversions.h"
#include <rapidjson/document.h>

/**
 * @file Conversions.cpp
 * @brief Implements serialization and deserialization utilities for fitting and job status properties using JSON.
 *
 * The Conversions module is responsible for:
 *  - Translating between JSON representations (using rapidjson) and internal C++ data structures for fitting properties and job status.
 *  - Providing utility functions to serialize and deserialize fitting configuration and job status for use in the application's workflow.
 *
 * Key Concepts:
 *  - JSON Serialization/Deserialization: Converts between rapidjson::Value objects and C++ structs (FittingProperties, JobStatus).
 *  - Fitting Properties Mapping: Extracts fitting preferences and algorithm parameters from JSON to configure fitting routines.
 *  - Job Status Reporting: Serializes and deserializes job status for progress tracking and UI updates.
 *
 * Fields:
 *  - (via FittingProperties):
 *      - accurateDerivative, accurateFitting, fitIterations, logScaleFitting, method, minSignal, wssrFitting, ceresProps, bProgressReport, liveFitting, liveGenerate, msUpdateInterval
 *  - (via CeresProperties):
 *      - minimizerType, lineSearchDirectionType, lineSearchType, trustRegionStrategyType, doglegType, nonlinearConjugateGradientType, lossFuncType, residualType, lossFunctionParameters, fittingConvergence, derivativeStepSize, derivativeEps
 *  - (via JobStatus):
 *      - isRunning, progress, code, code_string
 *
 * Main Methods:
 *  - FittingProperties FittingPropertiesFromStateJSON(const rapidjson::Value& json):
 *      Parses fitting preferences from a JSON object and returns a configured FittingProperties struct.
 *  - void WriteJobStatusJSON(JsonWriter& writer, JobStatus jobStatus):
 *      Serializes a JobStatus struct to JSON using a provided JsonWriter.
 *  - JobStatus JobStatusFromJSON(const rapidjson::Value& status):
 *      Parses a JobStatus struct from a JSON object.
 *
 * Error Handling:
 *  - Assumes required JSON fields are present and valid; does not perform extensive error checking.
 *  - Relies on rapidjson for type safety and access.
 *
 * Dependencies:
 *  - Conversions.h for struct and function declarations.
 *  - rapidjson/document.h for JSON parsing and manipulation.
 *  - Definitions for FittingProperties, CeresProperties, JobStatus, and JsonWriter.
 *
 * See Conversions.h for struct and method declarations.
 */

FittingProperties FittingPropertiesFromStateJSON(const rapidjson::Value &json)
{
	FittingProperties fp;

	fp.accurateDerivative = false;
	fp.accurateFitting = true;
	fp.fitIterations = 20;
	fp.logScaleFitting = false;
	fp.method = FIT_LBFGS;
	fp.minSignal = 0.0;
	fp.wssrFitting = false;

	//Get the fitting method
	CeresProperties ceresProps;

	const rapidjson::Value &fit = json.FindMember("FittingPreferences")->value;

	ceresProps.minimizerType = MinimizerTypefromCString(fit["MinimizerType"].GetString());
	ceresProps.lineSearchDirectionType = LineSearchDirectionTypefromCString(fit["LineSearchDirectionType"].GetString());
	ceresProps.lineSearchType = LineSearchTypefromCString(fit["LineSearchType"].GetString());
	ceresProps.trustRegionStrategyType = TrustRegionStrategyTypefromCString(fit["TrustRegionStrategyType"].GetString());
	ceresProps.doglegType = DoglegTypefromCString(fit["DoglegType"].GetString());
	ceresProps.nonlinearConjugateGradientType = NonlinearConjugateGradientTypefromCString(fit["NonlinearConjugateGradientType"].GetString());
	ceresProps.lossFuncType = LossFunctionfromCString(fit["LossFunction"].GetString());
	ceresProps.residualType = XRayResidualsTypefromCString(fit["XRayResidualsType"].GetString());
	ceresProps.lossFunctionParameters[0] = fit["LossFuncPar1"].GetDouble();
	ceresProps.lossFunctionParameters[1] = fit["LossFuncPar2"].GetDouble();
	ceresProps.fittingConvergence = fit["Convergence"].GetDouble();
	ceresProps.derivativeStepSize = fit["StepSize"].GetDouble();
	ceresProps.derivativeEps = fit["DerEps"].GetDouble();

	fp.ceresProps = ceresProps;

	//Get the fitting iterations
	fp.fitIterations = fit["FittingIterations"].GetInt();

	fp.bProgressReport = true;
	fp.liveFitting = false;  // Always false
	fp.liveGenerate = false; // Always false

	fp.msUpdateInterval = 1000;  // Update interval can be hard coded, don't take it from the parameters	

	return fp;
}

void WriteJobStatusJSON(JsonWriter &writer, JobStatus jobStatus)
{
	writer.StartObject();

	writer.Key("isRunning");
	writer.Bool(jobStatus.isRunning);
	writer.Key("progress");
	writer.Double(jobStatus.progress);
	writer.Key("code");
	writer.Int(jobStatus.code);
	writer.Key("message");
	writer.String(jobStatus.code_string.c_str());

	writer.EndObject();

}


JobStatus JobStatusFromJSON(const rapidjson::Value &status)
{
	JobStatus jobStatus;

	jobStatus.isRunning = status["isRunning"].GetBool();
	jobStatus.progress = status["progress"].GetDouble();
	jobStatus.code = status["code"].GetInt();
	jobStatus.code_string = status["message"].GetString();
	return jobStatus;
}