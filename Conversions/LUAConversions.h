#ifndef __LUA_CONVERSIONS_H
#define __LUA_CONVERSIONS_H

#include <string>
#include <vector>

#include <rapidjson/document.h>

class lua_State;
class JsonWriter;

class LuaToJSON
{
public:
	LuaToJSON(std::string lua);
	~LuaToJSON(); 
	static std::string saveJsonFile();
	
	void WriteState(JsonWriter  &writer);

private:
	lua_State *_luaState;
	static std::vector<std::string> _tableNames;

	static std::wstring GetJSONLuaPathname(); // Returns the pathname for JSON.Lua
	std::string EnsureJSONLua();// Make sure JSON.lua is stored in the same directory as the DLL
				
	static bool FileExists(const std::wstring path);
	static void SaveLuaResource(const std::wstring path);
	static const char *_jsonLua;
};

#endif