#ifndef __CONVERSIONS_H
#define __CONVERSIONS_H

#include <string>
#include "Common.h"
#include <rapidjson/document.h>
#include <rapidjson/document.h>
#include "JsonWriter.h"
#include "../Common/CommProtocol.h"

// Get a FittingProperties structure from its JSON representation
FittingProperties FittingPropertiesFromStateJSON(const rapidjson::Value &json);

void WriteJobStatusJSON(JsonWriter &writer, JobStatus jobStatus);
JobStatus JobStatusFromJSON(const rapidjson::Value &json);

#endif