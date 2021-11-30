#ifndef __AMPCACHE
#define __AMPCACHE
#include "BackendInterface.h"
#include <string>
#include <vector>
#include "Common.h"
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "BackendWrapper.h"
#include "LocalBackendParameterTree.h"
namespace fs = boost::filesystem;

//class LocalBackendParameterTreeConverter;

class EXPORTED_BE AmplitudeCache
{
public: static std::string cache_directory;
		static void _setCacheDirectory(std::string directory);
		static std::wstring getFilepath(Amplitude * internalptr);
		static fs::path path_creator(ModelPtr modelPtr, bool forparam = false);

		static ModelPtr getModel(Amplitude *);
		static std::map<ModelPtr, VectorXd> previousParams;
		static std::map<ModelPtr, std::wstring> AmpCached;
		static void ampAddedtoCache(Amplitude *, VectorXd);
		static bool amp_is_cached(Amplitude *);
		static void initializeCache(std::string directory, LocalBackendParameterTreeConverter *conv);
		static void initializeCache(LocalBackendParameterTreeConverter *conv);
		static void _loadCache();

		static VectorXd previousParameters(Amplitude *);

		static LocalBackendParameterTreeConverter *Converter;
		static void _setConverter(LocalBackendParameterTreeConverter *conv);
};


#endif
