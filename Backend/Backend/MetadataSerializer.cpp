
#include "MetadataSerializer.h"
#include "../Conversions/JsonWriter.h"
#include "LocalBackend.h"

#define DEFAULT_CONTAINER "xplusmodels"

MetadataSerializer::MetadataSerializer(LocalBackend *backend)
{
	_backend = backend;
}

void MetadataSerializer::Serialize(JsonWriter &writer)
{
	writer.StartArray();  // Array of containers
	writer.StartObject();
	writer.Key("containerName");
	writer.String(DEFAULT_CONTAINER);
	writeModels(nullptr, writer);
	writeModelCategories(nullptr, writer);
	writer.EndObject(); // Container
	writer.EndArray();
}


void MetadataSerializer::writeModels(wchar_t *container, JsonWriter &writer)
{
	writer.String("models");
	writer.StartArray();

	int numModels = _backend->HandleQueryModelCount(container);
	for (int i = 0; i < numModels; i++)
	{
		auto mi = _backend->HandleQueryModel(container, i);
		writer.StartObject();
		writer.String("index");
		writer.Int(mi.modelIndex);
		writer.Key("name");
		writer.String(std::string(mi.name).c_str()); // Copy to a non-fixed size array. Stupid Rapidjson bug.
		writer.Key("category");
		writer.Int(mi.category);
		writer.Key("gpuCompatible");
		writer.Bool(mi.isGPUCompatible);
		writer.Key("slow");
		writer.Bool(mi.isSlow);
		writer.Key("ffImplemented");
		writer.Bool(mi.ffImplemented);
		writer.Key("isLayerBased");
		writer.Bool(mi.isLayerBased);

		if (mi.minLayers >= 0)
			writeLayers(container, mi, writer);

		if (mi.nExtraParams)
			writeExtraParams(container, mi, writer);

		writer.EndObject();
	}

	writer.EndArray();
}

void MetadataSerializer::writeLayers(wchar_t *container, ModelInformation &mi, JsonWriter &writer)
{
	writer.Key("layers");
	writer.StartObject();
	writer.Key("min");
	writer.Int(mi.minLayers);
	writer.Key("max");
	writer.Int(mi.maxLayers);
	writer.Key("layerInfo");
	writer.StartArray();
	for (int i = 0; i < mi.minLayers; i++)
		writeLayer(container, mi, i, writer);
	if (mi.maxLayers == -1)
		writeLayer(container, mi, -1, writer);
	writer.EndArray();

	writer.Key("params");
	writeLayerParams(container, mi, writer);

	writer.EndObject();
}

void MetadataSerializer::writeLayer(wchar_t *container, ModelInformation &mi, int index, JsonWriter &writer)
{
	int layerIndex = index == -1 ? mi.minLayers : index;
	char layerName[500];
	int applicability[500];
	double defaultValues[500];

	_backend->HandleGetLayerInfo(container, mi.modelIndex, layerIndex, layerName, applicability, defaultValues, mi.nlp);
	writer.StartObject();
	writer.Key("index");
	writer.Int(index);
	writer.Key("name");
	writer.String(std::string(layerName).c_str());  // writer.String takes the length of the array (500) and checks for null at the end, not at the end of the actual string
	writer.Key("applicability");
	writer.StartArray();
	for (int i = 0; i < mi.nlp; i++)
		writer.Int(applicability[i]);
	writer.EndArray();
	writer.Key("defaultValues");
	writer.StartArray();
	for (int i = 0; i < mi.nlp; i++)
		writer.Double(defaultValues[i]);
	writer.EndArray();
	writer.EndObject();
}

void MetadataSerializer::writeLayerParams(wchar_t *container, ModelInformation &mi, JsonWriter &writer)
{
	char **names = new char*[mi.nlp];
	for (int i = 0; i < mi.nlp; i++)
		names[i] = new char[500];
	_backend->HandleGetLayerParamNames(container, mi.modelIndex, names, mi.nlp);

	writer.StartArray();
	for (int i = 0; i < mi.nlp; i++)
		writer.String(names[i]);
	writer.EndArray();

	for (int i = 0; i < mi.nlp; i++)
		delete names[i];
	delete names;
}


void MetadataSerializer::writeExtraParams(wchar_t *container, ModelInformation &mi, JsonWriter &writer)
{
	writer.Key("extraParams");
	writer.StartArray();

	ExtraParam eps[100];
	_backend->HandleGetExtraParamInfo(container, mi.modelIndex, eps, mi.nExtraParams);
	for (int i = 0; i < mi.nExtraParams; i++)
	{
		writer.StartObject();
		writer.Key("name");
		writer.String(std::string(eps[i].name).c_str());
		writer.Key("defaultValue");
		writer.Double(eps[i].defaultVal);
		if (eps[i].isRanged)
		{
			writer.Key("range");
			writer.StartObject();
			writer.Key("min");
			writer.Double(eps[i].rangeMin);
			writer.Key("max");
			writer.Double(eps[i].rangeMax);
			writer.EndObject();
		}
		writer.Key("isIntegral");
		writer.Bool(eps[i].isIntegral);
		writer.Key("decimalPoints");
		writer.Int(eps[i].decimalPoints);
		writer.Key("isAbsolute");
		writer.Bool(eps[i].isAbsolute);
		writer.Key("canBeInfinite");
		writer.Bool(eps[i].canBeInfinite);
		writer.EndObject();
	}
	writer.EndArray();
}

void MetadataSerializer::writeModelCategories(wchar_t *container, JsonWriter &writer)
{
	writer.String("modelCategories");
	writer.StartArray();

	int numCategories = _backend->HandleQueryCategoryCount(container);
	for (int i = 0; i < numCategories; i++)
	{
		ModelCategory cat = _backend->HandleQueryCategory(container, i);
		writer.StartObject();
		writer.Key("name");
		writer.String(std::string(cat.name).c_str());
		writer.Key("index");
		writer.Int(i);
		writer.Key("type");
		writer.Int(cat.type);
		writer.Key("models");
		writer.StartArray();
		for (int j = 0; cat.models[j] != -1; j++)
			writer.Int(cat.models[j]);
		writer.EndArray();

		writer.EndObject();
	}
	writer.EndArray();
}
