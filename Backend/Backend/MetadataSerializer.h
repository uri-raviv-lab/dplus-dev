#ifndef METADATA_SERIALIZER_H
#define METADATA_SERIALIZER_H

class LocalBackend;
class JsonWriter;
struct ModelInformation;

class MetadataSerializer
{
public:
	MetadataSerializer(LocalBackend *backend);

	void Serialize(JsonWriter &writer);

private:
	LocalBackend *_backend;

	void writeModels(wchar_t *container, JsonWriter &writer);
	void writeLayers(wchar_t *container, ModelInformation &mi, JsonWriter &writer);
	void writeLayer(wchar_t *container, ModelInformation &mi, int index, JsonWriter &writer);
	void writeLayerParams(wchar_t *container, ModelInformation &mi, JsonWriter &writer);
	void writeExtraParams(wchar_t *container, ModelInformation &mi, JsonWriter &writer);
	void writeModelCategories(wchar_t *container, JsonWriter &writer);
};

#endif