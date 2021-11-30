#ifndef __CLRFUNCTIONALITY_H
#define __CLRFUNCTIONALITY_H
#pragma once
#include <vector>
#include <limits>
#include "FrontendExported.h"

using std::vector;

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

static inline void ReadCLRFile(System::String^ fname, vector<double>& x, vector<double>& y) {
		wchar_t fstr[MAX_PATH] = {0};
		array<wchar_t>^ arr = fname->ToCharArray();

		for(int i = 0; i < fname->Length; i++)
			fstr[i] = (wchar_t)arr[i];

		ReadDataFile(fstr, x, y);
}

static inline void ReadCLRFile(System::String^ fname, vector<double>& x) {
		wchar_t fstr[MAX_PATH] = {0};
		array<wchar_t>^ arr = fname->ToCharArray();

		for(int i = 0; i < fname->Length; i++)
			fstr[i] = (wchar_t)arr[i];

		Read1DDataFile(fstr, x);
}

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

static inline void clrToWchar_tp(System::String^ str, wchar_t* res) {
	for(int i = 0; i < str->Length; i++) {
		res[i] = str[i];
	}
	res[str->Length] = '\0';
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

static inline bool openDataFile(System::Windows::Forms::OpenFileDialog ^ofd,
								System::String ^title,
								std::wstring &res, std::vector<double> &x, 
								std::vector<double> &y, bool bOutput) {
	ofd->Title = title;
	if(bOutput)
		ofd->Filter = "Output Files (*.out)|*.out|Data Files (*.dat, *.chi)|*.dat;*.chi|All files|*.*";
	else
		ofd->Filter = "Data Files (*.dat, *.chi)|*.dat;*.chi|Output Files (*.out)|*.out|All files|*.*";

	if(ofd->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
		return false;

	clrToString(ofd->FileName, res);

	// The program cannot tolerate files with 1 point or less
	ReadDataFile(res.c_str(), x, y);
	if(x.size() <= 1) {
		System::Windows::Forms::MessageBox::Show("The chosen file is invalid or empty", 
					"Invalid data file", System::Windows::Forms::MessageBoxButtons::OK, 
					System::Windows::Forms::MessageBoxIcon::Warning);
		return false;
	}

	return true;
}

static inline bool openDataFile(System::Windows::Forms::OpenFileDialog ^ofd,
								System::String ^title,
								std::wstring &res, bool bOutput) {
	std::vector<double> x, y;
	return openDataFile(ofd, title, res, x, y, bOutput);

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

static System::String ^CLRBasename(System::String ^file) {
	wchar_t basename[260] = {0};

	GetBasename(clrToWstring(file).c_str(), basename);

	return gcnew System::String(basename);
}

static System::String ^CLRDirectory(System::String ^file) {
	wchar_t res[260] = {0};

	GetDirectory(clrToWstring(file).c_str(), res);

	return gcnew System::String(res);
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
#endif