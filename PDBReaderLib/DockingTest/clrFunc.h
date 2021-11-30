#pragma once
#include <vector>
#include <limits>

using std::vector;

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

static inline void clrToString(System::String^ str, std::string& result) {
	array<wchar_t>^ arr = str->ToCharArray();

	result = "";
	for(int i = 0; i < arr->Length; i++)
		result += (char)arr[i];
}

static inline void clrToString(System::String^ str, std::wstring& result) {
	array<wchar_t>^ arr = str->ToCharArray();

	result.clear();
	for(int i = 0; i < arr->Length; i++)
		result += (wchar_t)arr[i];
}

static inline std::string clrToString(System::String^ str) {
	std::string res;
	clrToString(str, res);
	return res;
}

static inline std::wstring clrToWstring(System::String^ str) {
	std::wstring res;
	clrToString(str, res);
	return res;
}

static inline System::String ^stringToClr(const std::string& str) {
	return gcnew System::String(str.c_str());
}

static inline System::String ^stringToClr(const std::wstring& str) {
	return gcnew System::String(str.c_str());
}

static inline double clrToDouble(System::String^ str) {
	std::string ststr;
	clrToString(str, ststr);


	// Special values
	System::String ^mstr = str->ToLower();
	if(mstr->Equals("inf") || mstr->Equals("infinity") ||
		mstr->StartsWith("1.#INF")) {
			// Infinity
			return std::numeric_limits<double>::infinity();
	} else if(mstr->Equals("-inf") || mstr->Equals("-infinity") ||
		mstr->StartsWith("-1.#INF")) {
			// Negative infinity
			return -std::numeric_limits<double>::infinity();
	}

	return strtod(ststr.c_str(), NULL); 
}

template<typename T> static std::vector<T> arraytovector(array<T> ^arr) {
	std::vector<T> res (arr->Length);
	for(int i = 0; i < arr->Length; i++)
		res[i] = arr[i];

	return res;
}

template<typename T> static array<T> ^vectortoarray(const std::vector<T> &vec) {
	array<T> ^res = gcnew array<T>(vec.size());
	for(int i = 0; i < vec.size(); i++)
		res[i] = vec[i];
}
