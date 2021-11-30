#include "Amplitude.h"
#include "AmplitudeCache.h"
#include <iomanip>

std::string AmplitudeCache::cache_directory;
LocalBackendParameterTreeConverter * AmplitudeCache::Converter;
std::map<ModelPtr, std::wstring> AmplitudeCache::AmpCached;
std::map<ModelPtr, VectorXd> AmplitudeCache::previousParams;


ModelPtr AmplitudeCache::getModel(Amplitude * amp)
{
	return Converter->StateFromAmp(amp);
}

VectorXd AmplitudeCache::previousParameters(Amplitude * amp)
{
	VectorXd res;
	ModelPtr modelPtr = Converter->StateFromAmp(amp);
	res = previousParams[modelPtr];
	if (res.size() == 0) // if there are no parameters in memory, try reading them from the file if it exists
	{
		fs::path p = path_creator(modelPtr, true);

		try
		{
			std::ifstream f(p.string());
			f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			if (f.is_open())
			{
				std::vector<double> tmpRes;
				double val;
				while (f >> val)
				{
					tmpRes.push_back(val);
				}
				res = Eigen::Map<Eigen::VectorXd>(tmpRes.data(), tmpRes.size());
			}
		}
		catch (std::ifstream::failure) // don't do anything
		{
		}


	}
	return res;
}

void  AmplitudeCache::ampAddedtoCache(Amplitude * amp, VectorXd param)
{
	ModelPtr modelPtr = Converter->StateFromAmp(amp);
	AmpCached[modelPtr] = amp->getrndPath();
	previousParams[modelPtr] = amp->GetPreviousParameters();
	fs::path p = path_creator(modelPtr, true);
	std::ofstream f(p.string());
	if (f.is_open())
	{
		f << std::setprecision(128);
		f << param;
	}/**/
}

bool  AmplitudeCache::amp_is_cached(Amplitude * amp)
{
	ModelPtr modelPtr = Converter->StateFromAmp(amp);
	std::wstring path = AmpCached[modelPtr];
	bool res = path.length() > 0;
	if (res)
		amp->setrndPath(path);
	return res;
}


void AmplitudeCache::initializeCache(std::string directory, LocalBackendParameterTreeConverter *conv)
{
	_setCacheDirectory(directory);
	_setConverter(conv);
	_loadCache();
}

void AmplitudeCache::initializeCache(LocalBackendParameterTreeConverter *conv)
{
	fs::path temp_dplus = fs::temp_directory_path() / fs::path("dplus");
	if (!fs::is_directory(temp_dplus))
		fs::create_directory(temp_dplus);
	fs::path session = fs::unique_path();
	fs::path dir = temp_dplus / session;
	boost::filesystem::create_directory(dir);
	cache_directory = dir.string();
	_setConverter(conv);
	//_loadCache();
}

void AmplitudeCache::_loadCache()
{
	fs::path p(cache_directory);
	//option one:
	boost::filesystem::directory_iterator itr(p);
	while (itr != boost::filesystem::directory_iterator())
	{

		string filename = itr->path().filename().string();
		string ext = filename.substr(9, 3);

		//create the model ptr
		string num = filename.substr(0, 8);
		double d = std::stod(num);
		ModelPtr mp = d;

		if (ext == "amp")
		{
			//add to ampCached
			AmpCached[mp] = true;
		}

		if (ext == "prm")
		{
			std::ifstream filein(itr->path().string().c_str());
			int count = 0;
			double buffer[int(1e2)];
			while (filein >> buffer[count])
			{
				count++;
			}
			VectorXd params(count);
			for (int i = 0; i < count; i++)
			{
				params(i) = buffer[i];
			}

			previousParams[mp] = params;
		}

		++itr;
	}



}

void AmplitudeCache::_setCacheDirectory(std::string directory)
{
	fs::path dir = directory;
	fs::create_directory(dir);
	cache_directory = dir.string();
}

void AmplitudeCache::_setConverter(LocalBackendParameterTreeConverter *conv)
{
	Converter = conv;
}

std::wstring AmplitudeCache::getFilepath(Amplitude * const amp)
{
	ModelPtr modelPtr = Converter->StateFromAmp(amp);
	fs::path p= path_creator(modelPtr);
	return p.wstring();
}

fs::path AmplitudeCache::path_creator(ModelPtr modelPtr, bool forparam)
{
	char _Dest[50];
	if (forparam)
		sprintf(_Dest, "%08d.prm", modelPtr);
	else
		sprintf(_Dest, "%08d.ampj", modelPtr);
	fs::path filename = std::string(_Dest);
	fs::path dir = cache_directory;
	fs::path full_path = dir / filename;
	return full_path;
}
