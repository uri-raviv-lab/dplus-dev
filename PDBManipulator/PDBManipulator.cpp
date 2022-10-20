#include "../backend_version.h"

#include "PDBReaderLib.h"

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

#include<boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
namespace fs = boost::filesystem;
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/algorithm/string/predicate.hpp>

int PDBManipulate(std::string inFilename, std::string command, fs::path ipt, std::string saveFilename, bool electron)
{
	PDBReader::PDBReaderOb<float>* pdb;
	
	if (electron)
	{
		pdb = new PDBReader::ElectronPDBReaderOb<float>(inFilename, false);
	}
	else
	{
		pdb = new PDBReader::XRayPDBReaderOb<float>(inFilename, false);
	}
	
	std::stringstream header;
	if (boost::iequals(command, "gcenter"))
	{
		pdb->moveGeometricCenterToOrigin();
		header << "REMARK    This file was made by moving the file\n"
			"REMARK    " << ipt
			<< "\nREMARK    to its geometric center.\n";
	}
	if (boost::iequals(command, "mcenter"))
	{
		pdb->moveCOMToOrigin();
		header << "REMARK    This file was made by moving the file\n"
			"REMARK    " << ipt
			<< "\nREMARK    to its center of mass.\n";
	}
	if (boost::iequals(command, "align"))
	{
		pdb->AlignPDB();
		header << "REMARK    This file was made by moving the file\n"
			"REMARK    " << ipt
			<< "\nREMARK    to its center of mass and aligning its principal axes to Cartesian axes.\n";
	}

	if (PDB_OK != pdb->WritePDBToFile(saveFilename, header))
	{
		std::cerr << "Error occurred when attempting to write to " << saveFilename << ".\n";
		delete pdb;
		return -5;
	}

	delete pdb;
	return 0;
}


int main(int argc, char* argv[])
{

	//////////////////////////////////////////////////////////////////////////
	// Command line argument values
	std::string inFilename, saveFilename, helpModule, command;
	bool electron;

	//////////////////////////////////////////////////////////////////////////
	// Parse command line
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", po::value<std::string>(&helpModule)->implicit_value(""),
		"Print this help message")
		("command,c", po::value<std::string>(&command),
		"Command to execute. Valid options are: mcenter; gcenter; align.")
		("PDBFile,i", po::value<std::string>(&inFilename),
		"Input PDB file")
		("out,o", po::value<std::string>(&saveFilename),
		"The name of the file to write the results to")
		("electron",
			"To use ElectronPDBReader")
		;

	po::positional_options_description p;
	p.add("command", 1);
	p.add("PDBFile", 1);
	p.add("out", 1);
	p.add("electron", 1);

	po::variables_map vm;

	try
	{
		po::store(po::command_line_parser(argc, argv).
			options(desc).positional(p).run(), vm);
		po::notify(vm);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << "\n";
		return -11;
	}

	if (vm.count("help")) {
		if (helpModule != "")
		{
			std::cout << "The selected option for help does not have a detailed help.\n\n";
		}
		std::cout << "Usage: " << argv[0] << " <command> <input PDB filename> <output PDB filename> \n";
		std::cout << desc;
		return 0;
	}

	if (vm.count("command") == 1)
	{
		if (!boost::iequals(command, "gcenter") &&
			!boost::iequals(command, "mcenter") &&
			!boost::iequals(command, "align")
			)
		{
			std::cout << "Invalid command provided. You wrote \"" << command
				<< "\".\nValid options are: mcenter; gcenter; align.\n";

			return -32;
		}
		std::cout << "Command: " << command << "." << std::endl;
	}
	else
	{
		std::cout << "\nA (single) command is required.\n" << std::endl;
		std::cout << "Usage: " << argv[0] << " <command> <input PDB filename> <output PDB filename> \n";
		return -21;
	}

	if (vm.count("PDBFile") == 1)
	{
		std::cout << "Input PDB file: " << inFilename << "." << std::endl;
	}
	else
	{
		std::cout << "\nA (single) PDB file is required as input.\n" << std::endl;
		std::cout << "Usage: " << argv[0] << " <command> <input PDB filename> <output PDB filename> \n";
		return -21;
	}

	if (vm.count("out") == 1)
	{
		std::cout << "Output PDB file: " << saveFilename << "." << std::endl;
	}
	else
	{
		std::cout << "\nA (single) PDB filename is required as output.\n" << std::endl;
		std::cout << "Usage: " << argv[0] << " <command> <input PDB filename> <output PDB filename> \n";
		return -21;
	}

	if (vm.count("electron"))
	{
		electron = true;
	}
	else
	{
		electron = false;
	}

	fs::path ipt(inFilename);
	ipt = fs::system_complete(ipt);
	if (!fs::exists(ipt)) {
		std::cout << "No such file " << ipt << std::endl;
		return -3;
	}

	return PDBManipulate(inFilename, command, ipt, saveFilename, electron);
	

}

