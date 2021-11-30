#include "../../Backend/Amplitude.h"
#include "../../Backend/CommandLineBackendWrapper.h"
#include <iostream>
#include <sstream>
#include <rapidjson/document.h>
#include "../../../Conversions/JsonWriter.h"
#include <boost/filesystem.hpp>
#include "../../Backend/LocalBackend.h"
#include <thread>
#include <chrono>
using namespace rapidjson;
using namespace std;
namespace fs = boost::filesystem;


string slurp(ifstream& in) {
	stringstream sstr;
	sstr << in.rdbuf();
	return sstr.str();
}


void print_to_file(string directory, string filename, string message, string find = "")
{
	std::ofstream of;
	of.open(directory + filename);
	of << message;
	of.close();
}


void handle_errors(string directory, int errorcode, string errormessage) //backend_exception &be)
{
	std::ofstream of;
	of.open(directory + "/job.json");
	std::string status = "{\"isRunning\": false, \"progress\" : 1.0, \"code\" : " + std::to_string(errorcode) + ",\n \"message\": \"" + errormessage + "\"}";
	of << status;
	of.close();


	JsonWriter writer;
	writer.StartObject();
	writer.Key("error");
	writer.StartObject();

	writer.Key("code");
	writer.Int(errorcode);

	writer.Key("message");
	writer.String(errormessage.c_str());

	writer.EndObject();
	writer.EndObject();

	std::ofstream rf(directory + "/data.json");
	std::string str = writer.GetString();
	rf << str;
	rf.close();
	std::cout << errormessage;
}


void check_job(string directory, CommandLineBackendWrapper &wrapper)
{
	std::size_t found;
	while (true)
	{
		JsonWriter statuswriter;
		wrapper.GetJobStatus(statuswriter);
		string status = statuswriter.GetString();
		found = status.find("false");
		if (found != std::string::npos)
			break;
		print_to_file(directory, "/job.json", status);
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

void parse_args(fs::path directory, rapidjson::Document &doc)
{
	fs::path combined = directory / "args.json";
	string argsfilename = combined.string();
	ifstream argsf(argsfilename);
	string args = slurp(argsf);
	argsf.close();
	doc.Parse(args.c_str());
	if (doc.HasParseError())
	{
		throw runtime_error("args not found or corrupted");
	}
}

std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
}

void save_amp_pdb(string directory, string cache_dir, CommandLineBackendWrapper &wrapper)
{
	std::cout << "saving files...\n";
	for (ModelPtr ptr : wrapper.GetModelPtrs())
	{
		char _Dest[50];
		sprintf(_Dest, "%08d.ampj", ptr);
		std::string filename(_Dest);
		boost::filesystem::path filepath = boost::filesystem::path(cache_dir) / boost::filesystem::path(filename);
		if (!boost::filesystem::exists(filepath))
		{
			wrapper.SaveAmplitude(ptr, cache_dir);
		}
		wrapper.SavePDB(ptr, directory + "/pdb/");
	}
}

int main(int argc, char *argv[])
{
	string directory = argv[1];
	print_to_file(directory, "/notrunning.txt", "False");

	fs::path dir = directory;

	CommandLineBackendWrapper wrapper = CommandLineBackendWrapper();// directory);
	string cache_dir;
	if (argc > 2)
		cache_dir = argv[2];
	else
	{
		fs::path withcache = fs::path(directory) / fs::path("cache");
		cache_dir = withcache.string();
	}
	
	try
	{
		//parse arguments
		rapidjson::Document doc;
		parse_args(dir, doc);

		//initialize cache
		wrapper.initializeCache(cache_dir);

		//call function
		wrapper.StartFit(doc.FindMember("args")->value, doc.FindMember("options")->value);

		//check if function has finished
		check_job(directory, wrapper);

		//save function results to file
		JsonWriter writer;
		wrapper.GetFitResults(writer);
		string results = writer.GetString();
		//string fixed_results_1 = replaceAll(results, "-Infinity", "\"-inf\"");
		//string fixed_results_2 = replaceAll(fixed_results_1, "Infinity", "\"inf\"");
		print_to_file(directory, "/data.json", results);

		//write final job status to file
		string finishedJob = "{\n    \"isRunning\": false,\n    \"progress\": 1.0,\n    \"code\": 0, \"message\" : \"OK\"\n}";
		print_to_file(directory, "/job.json", finishedJob);
		save_amp_pdb(directory, cache_dir, wrapper);
		print_to_file(directory, "/notrunning.txt", "True");
		return 0;
	}
	catch (runtime_error)
	{
		handle_errors(directory, 9, "problem with input args");
	}

	catch (backend_exception &be)
	{
		handle_errors(directory, be.GetErrorCode(), be.GetErrorMessage());
	}

	catch (exception e)
	{
		handle_errors(directory, 19, "error: " + std::string(e.what()));
	}

	print_to_file(directory, "/notrunning.txt", "True");
	return -1;
}
