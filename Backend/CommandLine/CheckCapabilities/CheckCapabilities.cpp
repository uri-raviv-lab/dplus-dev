// GetAllMetadata.cpp : Defines the entry point for the console application.
//

#include "../../Backend/CommandLineBackendWrapper.h"
#include <iostream>
#include <rapidjson/document.h>
#include "../../../Conversions/JsonWriter.h"
#include "../../Backend/LocalBackend.h"
#include <boost/filesystem.hpp>

using namespace rapidjson;
using namespace std;

int main(int argc, char *argv[])
{
	std::cout << "entered executable"<< endl;
	//open args:
	string directory = argv[1];
	bool checkTdr = true;
	if (argc > 2)
		bool checkTdr = argv[2];
	JsonWriter writer;
	CommandLineBackendWrapper wrapper = CommandLineBackendWrapper();

	try {
		wrapper.CheckCapabilities(checkTdr);
		std::ofstream of(directory + "/check_capabilities.json");

		// checkCapabilities doesn't return anything if it ok, else it will throw an error
		JsonWriter writer;
		writer.StartObject();
		writer.Key("error");
		writer.StartObject();

		writer.Key("code");
		writer.Int(0);

		writer.Key("message");
		writer.String("OK");

		writer.EndObject();
		writer.EndObject();

		std::string str = writer.GetString();
		of << str;
		of.close();
		cout << "done";
		return 0;
	}

	catch (backend_exception &be)
	{
		JsonWriter writer;
		writer.StartObject();
		writer.Key("error");
		writer.StartObject();

		writer.Key("code");
		writer.Int(be.GetErrorCode());

		writer.Key("message");
		writer.String(be.GetErrorMessage().c_str());

		writer.EndObject();
		writer.EndObject();

		std::ofstream rf(directory + "/check_capabilities.json");
		std::string str = writer.GetString();
		rf << str;
		rf.close();
		std::cout << "done with errors";
	}
	return 1;
}
