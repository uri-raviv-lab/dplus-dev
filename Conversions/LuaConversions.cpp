#include "LUAConversions.h"

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
};
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

#include <algorithm>


#include <rapidjson/document.h>
#include "JsonWriter.h"

using namespace rapidjson;
using namespace std;

vector<string> LuaToJSON::_tableNames;

LuaToJSON::LuaToJSON(std::string lua)
{
	if (_tableNames.size() == 0)
	{
		_tableNames.push_back("DomainPreferences");
		_tableNames.push_back("FittingPreferences");
		_tableNames.push_back("Viewport");
		_tableNames.push_back("Domain");
	}

	_luaState = lua_open();
	luaL_openlibs(_luaState);

	std::string jsonLuaPath = EnsureJSONLua();
	std::string luaLoad = "JSON = (dofile \"" + jsonLuaPath + "\")";

	/*std::string loadJsonChunk = "JSON_Chunk = (loadfile \"JSON.Lua\")";
	if (luaL_dostring(_luaState, loadJsonChunk.c_str()))
	{
		const char *error = lua_tostring(_luaState, -1);
		throw std::runtime_error("Can't load JSON.lua");
	} */

	if (luaL_dostring(_luaState, luaLoad.c_str()))
	{
		const char *error = lua_tostring(_luaState, -1);
		throw std::runtime_error("Can't execute JSON.lua into the lua State");
	}

	if (luaL_loadstring(_luaState, lua.c_str()))
	{
		throw std::invalid_argument("Bad lua string passed to LuaToJSON");
	}
	
	if (lua_pcall(_luaState, 0, LUA_MULTRET, 0)) //TODO - Itay: review this added check
	{
		throw std::invalid_argument("Couldn't run lua string passed to LuaToJSON");
	}
}

LuaToJSON::~LuaToJSON()
{
	lua_close(_luaState);
}


void LuaToJSON::WriteState(JsonWriter &writer)
{
	writer.StartObject();
	//The tables
	int numTables = _tableNames.size();
	for (int i = 0; i < numTables; ++i)
	{
		string cmd = "return JSON:encode_pretty(" + _tableNames[i] + ")";
		if (luaL_dostring(_luaState, cmd.c_str()))
		{
			throw new std::invalid_argument("Can't extract table from Lua");
		}
		writer.Key(_tableNames[i].c_str());
		writer.RawString(lua_tostring(_luaState, -1));
	}
	writer.EndObject();	
}

std::wstring LuaToJSON::GetJSONLuaPathname()
{
#ifdef _WIN32
	wchar_t filename[5000];
	GetModuleFileName(nullptr, filename, sizeof(filename) / sizeof(wchar_t));

	wstring pathname(filename);
#else
    // Linux version
    char temp [ PATH_MAX ];
    std::string str;

    if (getcwd(temp, PATH_MAX) != 0) 
        str = std::string(temp);

    std::wstring pathname(str.begin(), str.end());
#endif
	std::replace(pathname.begin(), pathname.end(), L'\\', L'/');  // Lua gets confused by backslashes
    
	pathname = pathname.substr(0, pathname.find_last_of('/')) + L"/JSON.lua";

	return pathname;
}

bool LuaToJSON::FileExists(const std::wstring path)
{
#ifdef _WIN32
	// From here: http://stackoverflow.com/a/6218957/871910
	DWORD dwAttrib = GetFileAttributes(path.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		!(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    std::string nonunicode (path.begin(), path.end());
    struct stat buffer;
    return (stat (nonunicode.c_str(), &buffer) == 0); 
#endif
}

std::string LuaToJSON::EnsureJSONLua()
{
	std::wstring pathname = GetJSONLuaPathname();
	std::string pathnameStr(pathname.begin(), pathname.end());
	if (!FileExists(pathname.c_str()))
		SaveLuaResource(pathname);

	return pathnameStr;
}

std::string LuaToJSON::saveJsonFile()
{
	std::wstring pathname = GetJSONLuaPathname();
	std::string pathnameStr(pathname.begin(), pathname.end());
	if (!FileExists(pathname.c_str()))
		SaveLuaResource(pathname);

	return pathnameStr;
}

void LuaToJSON::SaveLuaResource(const std::wstring pathname)
{
    std::string nonunicode (pathname.begin(), pathname.end());
    
	std::ofstream output;
	output.open(nonunicode);
	output << _jsonLua;
	output.close();
}

