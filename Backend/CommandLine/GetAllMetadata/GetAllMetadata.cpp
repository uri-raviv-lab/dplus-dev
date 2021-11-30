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
	std::cout<<"entered executable";
	//open args:
	string directory = argv[1];
	JsonWriter writer;
	CommandLineBackendWrapper wrapper = CommandLineBackendWrapper();
	wrapper.GetAllModelMetadata(writer);

	try{
		std::ofstream of(directory + "/metadata.json");
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

		std::ofstream rf(directory + "/metadata.json");
		std::string str = writer.GetString();
		rf << str;
		rf.close();
		std::cout << "done with errors";
	}
	return 1;
}
