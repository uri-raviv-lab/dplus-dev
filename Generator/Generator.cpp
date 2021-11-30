// Generator.cpp : Defines the entry point for the console application.

#include <iostream>
#include <sstream>

#include <vector>
#include <limits>
#include <string>
#include <list>

#include <chrono>
#include <thread>

using std::vector;
#include <fstream>

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
};

#include "BackendInterface.h"
#include "LocalComm.h"

#include <rapidjson/document.h>
#include "JsonWriter.h"

#include "LUAConversions.h"
#include "ParamTreeConversions.h"
#include "LocalBackendParameterTree.h"
#include "BackendWrapper.h"
using namespace rapidjson;
using namespace std;

string ReadFile(const string filename);
//copied from filemgt
void ReadDataFile(const char *filename, vector<double>& x, vector<double>& y);

void WaitForFinish(void);
JobStatus GetJobStatus(JobPtr job);


void writeClientId(JsonWriter &writer);
void writeClientData(JsonWriter &writer);
void writeFunction(JsonWriter &writer, string functionName, string functionArgs);

string GetGenerateArgs(const char * fileName, const char *outputFileName);
string GetFitArgs(const char * fileName, const char *outputFileName);
string GetAmplitudeModelPtr(const char * fileName);
string GetPdbModelPtr(const char * fileName);

vector<double> GetQvec(std::string json);

BackendComm *backend;

JobPtr job;
std::ofstream outputFile;

using namespace rapidjson;
using namespace std;

bool jobDone = false;

/*
{
	client - id: <ID - of - client>
		client - data : <client specific data>


	function : <Name of function to call>
		   args : <arguments>
}

*/

int main(int argc, char *argv[])
{
	string functionArgs;
	backend = CreateBackendComm();

	cout << "Test bed for JSONized backend" << endl;

	if (argc != 3)
	{
		cerr << "Usage: Generator <state file> <output file>" << endl;
		return 0;
	}
	cout << "from state file:  " << argv[1] << endl;
	cout << "output file:  " << argv[2] << endl;
		
	while (true)
	{
		cout << "\n 1. GetAllModelMetadata \n 2. GetJobStatus \n 3. Stop \n 4. StartGenerate \n 5. GetGenerateResults \n 6. StartFit \n 7. GetFitResults \n 8. GetAmplitude \n 9. GetPDB \n 10. Unsupported function \n 11. Illegal JSON\n" << endl;
		cout << "Enter number of function to test or 0 to quit ";

		int functionNum;
		cin >> functionNum;

		//clear functionArgs
		functionArgs = "";

		if (!functionNum)
		{
			break;
		}

		JsonWriter writer;
		writer.StartObject();
		writeClientId(writer);
		writeClientData(writer);

		string result;

		switch (functionNum)
		{
		case 1:
			writeFunction(writer, "GetAllModelMetadata", "");
			break;
		case 2:
			// Call after starting a Generate or Fit job
			writeFunction(writer, "GetJobStatus", "");
			break;
		case 3:
			writeFunction(writer, "Stop", "");
			break;
		case 4:
			functionArgs = GetGenerateArgs(argv[1], argv[2]);
			writeFunction(writer, "StartGenerate", functionArgs);
			break;
		case 5:
			// Call only after Generate is done
			writeFunction(writer, "GetGenerateResults", "");
			break;
		case 6:
			functionArgs = GetFitArgs(argv[1], argv[2]);
			writeFunction(writer, "StartFit", functionArgs);
			break;
		case 7:
			// Call only after Fit is done
			writeFunction(writer, "GetFitResults", "");
			break;
		case 8:
			// TODO: Run Generate first, wait for results, call GetPDB on the ModelPtr that was generated.
			functionArgs = GetAmplitudeModelPtr(argv[1]);
			writeFunction(writer, "GetAmplitude", functionArgs);
			break;
		case 9:
			// TODO: Run Generate first, wait for results, call GetPDB on the ModelPtr that was generated.
			functionArgs = GetPdbModelPtr(argv[1]);
			writeFunction(writer, "GetPDB", functionArgs);
			break;
		case 10:
			writeFunction(writer, "blablatest", functionArgs);
			break;
		case 11:
			result = backend->CallBackend(string("this is not a json")); //should throw exception
			cout << "result of CallBackend " << result << endl;
			break;
		default: break;
		}

		if (functionNum == 11)
		{
			continue;
		}
		writer.EndObject();

		string json = writer.GetString();


		result = backend->CallBackend(json);
		//string result = backend->CallBackend(string("this is not a json"));
		cout << "result of CallBackend " << result << endl;

		if (functionNum == 4 || functionNum == 6)
		{
			WaitForFinish();
		}
	}
	
	

	//This is how to dump a string to file:
	//std::ofstream out2("pt2.json");
	//out2 << json2;
	//out2.close();
	return 0;

}



void WaitForFinish()
{
	JobStatus jobStatus = GetJobStatus(job);

	if (jobStatus.isRunning)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		WaitForFinish();
	}
	else
	{
		return;
	}
}





void writeClientId(JsonWriter &writer)
{

	writer.Key("client-id");	
	writer.String("clientIdString");

}


void writeClientData(JsonWriter &writer)
{
	

	writer.Key("client-data");
	writer.StartObject();

	writer.Key("list");
	writer.StartArray();
	writer.Int(1);
	writer.Int(2);
	writer.Int(3);
	writer.EndArray();

	writer.Key("object");
	writer.StartObject();
	writer.Key("a");
	writer.String("ayyyy");
	writer.Key("b");
	writer.String("beeee");
	writer.EndObject();

	writer.EndObject();

}


void writeFunction(JsonWriter &writer, string functionName, string functionArgs)
{
	writer.Key("function");
	writer.String(functionName.c_str());
	
	if (!functionArgs.empty())
	{
		writer.Key("args");
		writer.String(functionArgs.c_str());
	}
}




string ReadFile(const string filename)
{
	ifstream f;
	f.open(filename);
	if (f.fail())
		return "";

	// Adapter from here: http://stackoverflow.com/a/2602060/871910
	std::string s;

	f.seekg(0, ios::end);
	s.reserve(f.tellg());
	f.seekg(0, ios::beg);

	s.assign((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());

	return s;
}


string GetGenerateArgs(const char * fileName, const char *outputFileName)
{
	string script = ReadFile(fileName);
	if (script.empty())
	{
		cerr << "Can't read file " << fileName << endl;
		return "";
	}

	// Create a pretty writer
	JsonWriter writer;

	writer.StartObject();
	writer.Key("state");

	// Call the converter
	LuaToJSON luaToJsonConverter(script);
	luaToJsonConverter.WriteState(writer);

	std::vector<double> xv, yv;
	vector<double>  qvec;
	ReadDataFile(outputFileName, xv, yv);
	if (xv.size() == 0 || yv.size() == 0) {
		return "";
	}

	// Write the X vector with writer.StartArray,...
	writer.Key("x");
	writer.StartArray();
	for (auto itr = xv.begin(); itr != xv.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();

	writer.EndObject();

	string json = writer.GetString();

	return json;
}

string GetFitArgs(const char * fileName, const char *outputFileName)
{

	string script = ReadFile(fileName);
	if (script.empty())
	{
		cerr << "Can't read file " << fileName << endl;
		return "";
	}

	// Create a pretty writer
	JsonWriter writer;

	writer.StartObject();
	writer.Key("state");

	// Call the converter
	LuaToJSON luaToJsonConverter(script);
	luaToJsonConverter.WriteState(writer);

	std::vector<double> xv, yv;
	vector<double>  qvec;
	ReadDataFile(outputFileName, xv, yv);
	if (xv.size() == 0 || yv.size() == 0) {
		return "";
	}

	// Write the X vector with writer.StartArray,...
	writer.Key("x");
	writer.StartArray();
	for (auto itr = xv.begin(); itr != xv.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();

	writer.Key("y");
	writer.StartArray();
	for (auto itr = yv.begin(); itr != yv.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();

	writer.Key("mask");
	writer.StartArray();

	std::vector<int> mask(yv.size(), 0); //mask of zeroes (just like in MainWindow.cpp...)

	for (auto itr = mask.begin(); itr != mask.end(); ++itr)
	{
		writer.Int(*itr);
	}
	writer.EndArray();

	writer.EndObject();

	string json = writer.GetString();

	return json;
}

//TODO properly...
string GetAmplitudeModelPtr(const char * fileName)
{
	int model = 2;
	return (to_string(model));
}

string GetPdbModelPtr(const char * fileName)
{
	int model = 2;
	return (to_string(model));
}


JobStatus GetJobStatus(JobPtr job)
{
/*	std::string statusString = backend->GetJobStatus();

	rapidjson::Document doc;
	doc.Parse(statusString.c_str());

	//build the parameter tree from the json string
	const rapidjson::Value &status = doc;
*/
	JobStatus jobStatus;

/*	jobStatus.isRunning = status["isRunning"].GetBool();
	jobStatus.progress = status["progress"].GetDouble();
	jobStatus.code = status["code"].GetInt(); */
	return jobStatus;
}

//copied from filemgt
void ReadDataFile(const char *filename, vector<double>& x, vector<double>& y)
{
	int i = 0, n = -1;
	bool done = false, started = false;
	ifstream in(filename);

	if (!in) {
		char file[1024] = { 0 };
		//wcstombs(file, filename, 1024);
		fprintf(stderr, "Error opening file %s for reading\n",
			filename);
		exit(1);
	}

	while (!in.eof() && !done) {
		i++;
		if (n > 0 && i > n)
			break;

		std::string line;
		size_t pos, end;
		double curx = -1.0, cury = -1.0;
		getline(in, line);

		line = line.substr(0, line.find("#"));
		if (line.length() == 0)
			continue;

		//Remove initial whitespace
		while (line[0] == ' ' || line[0] == '\t' || line[0] == '\f')
			line.erase(0, 1);

		//Replaces whitespace with one tab
		for (int cnt = 1; cnt < (int)line.length(); cnt++) {
			if (line[cnt] == ' ' || line[cnt] == ',' || line[cnt] == '\t' || line[cnt] == '\f') {
				while (((int)line.length() > cnt + 1) && (line[cnt + 1] == ' ' || line[cnt + 1] == ',' || line[cnt + 1] == '\t' || line[cnt + 1] == '\f'))
					line.erase(cnt + 1, 1);
				line[cnt] = '\t';
			}
		}

		pos = line.find("\t");
		// Less than 2 words/columns
		if (pos == std::string::npos){
			if (started) done = true;
			continue;
		}
		end = line.find("\t", pos + 1);

		if (end == std::string::npos)
			end = line.length() - pos;
		else
			end = end - pos;

		if (end == 0) {	// There is no second word
			if (started) done = true;
			continue;
		}

		// Check to make sure the two words are doubles
		char *str = new char[line.length()];
		char *ptr;
		strcpy(str, line.substr(0, pos).c_str());
		strtod(str, &ptr);
		if (ptr == str) {
			if (started) done = true;
			delete[] str;
			continue;
		}
		strcpy(str, line.substr(pos + 1, end).c_str());
		strtod(str, &ptr);
		if (ptr == str) {
			if (started) done = true;
			delete[] str;
			continue;
		}
		delete[] str;

		curx = strtod(line.substr(0, pos).c_str(), NULL);
		cury = strtod(line.substr(pos + 1, end).c_str(), NULL);

		if (!started && !(fabs(cury) > 0.0)) continue;
		if (!started) started = true;
		x.push_back(curx);
		y.push_back(cury);
	}

	in.close();

	// Removes trailing zeroes
	if (y.empty()) return;
	while (y.back() == 0.0) {
		y.pop_back();
		x.pop_back();
	}
	//sort vectors
	for (unsigned int i = 0; i < x.size(); i++) {
		for (unsigned int j = i + 1; j < x.size(); j++) {
			if (x[j] < x[i]) {
				double a = x[j];
				x[j] = x[i];
				x[i] = a;
				a = y[j];
				y[j] = y[i];
				y[i] = a;
			}
		}
	}
}