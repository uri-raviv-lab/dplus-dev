#include "Conversions.h"
#include <rapidjson/document.h>

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