/*
 * A handy wrapper of rapidjson's PrettyWriter
 */

#ifndef JSON_WRITER_H
#define JSON_WRITER_H

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <string>

class JsonWriter : public rapidjson::PrettyWriter<rapidjson::StringBuffer>
{
protected:
	rapidjson::StringBuffer _buffer;

public:
	JsonWriter() : _buffer(), rapidjson::PrettyWriter<rapidjson::StringBuffer>(_buffer)
	{

	}

	// Writes a raw string with no quotes and no escapes.
	void RawString(const std::string &s)
	{
		PrettyPrefix(rapidjson::kStringType);
		for (int i = 0; i < s.size(); i++)
			_buffer.Put(s[i]);
	}

	rapidjson::StringBuffer &Buffer() { return _buffer; }

	const char *GetString() { return _buffer.GetString(); }
};

#endif