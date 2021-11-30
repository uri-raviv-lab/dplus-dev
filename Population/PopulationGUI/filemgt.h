#ifndef __FILEMGT_H
#define __FILEMGT_H

#include <vector>
#include <sstream>

int  CheckSizeOfFile (const wchar_t *filename);
void ReadDataFile(const wchar_t *filename, std::vector<double>& x, std::vector<double>& y);
void Read1DDataFile(const wchar_t *filename, std::vector<double>& x);

void WriteDataFileWHeader(const wchar_t *filename, std::vector<double>& x, std::vector<double>& y, std::stringstream& header);
void WriteDataFile(const wchar_t *filename, std::vector<double>& x, std::vector<double>& y);
void Write3ColDataFile(const wchar_t *filename, std::vector<double>& x, std::vector<double>& y, std::vector<double>& err);
void Write1DDataFile(const wchar_t *filename, std::vector<double>& x);

void GetBasename(const wchar_t *file, wchar_t *result, int n);
void GetDirectory(const wchar_t *file, wchar_t *result, int n);

#endif
