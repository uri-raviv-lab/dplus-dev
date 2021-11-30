#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

using std::vector;
using std::ifstream;

int CheckSizeOfFile (const wchar_t *filename)
{
	int n = 0;
	std::string line;
    ifstream in (filename);

    if(!in)
		return 0;

    while(!in.eof()) {
        getline(in, line);
        n++;
    }

	return n;
}

void ReadDataFile(const wchar_t *filename, vector<double>& x,
                  vector<double>& y)
{
	int i = 0, n = -1;
	bool done = false, started = false;
	ifstream in (filename);

    if(!in) {
		char file[1024] = {0};
		wcstombs(file, filename, 1024);
        fprintf(stderr, "Error opening file %s for reading\n",
                file);
        exit(1);
	}

    while(!in.eof() && !done) {
        i++;
        if(n > 0 && i > n)
            break;

		std::string line;
        size_t pos, end;
        double curx = -1.0, cury = -1.0;
        getline(in, line);

        line = line.substr(0, line.find("#"));

        //Remove initial whitespace
        while(line[0]  == ' ' || line[0] == '\t' || line[0] == '\f')
            line.erase(0, 1);

		//Replaces whitespace with one tab
        for(int cnt = 1; cnt < (int)line.length(); cnt++) {
			if(line[cnt] == ' ' || line[cnt] == ',' || line[cnt] == '\t' || line[cnt] == '\f') {
				while(((int)line.length() > cnt + 1) && (line[cnt + 1] == ' ' || line[cnt + 1] == ',' || line[cnt + 1] == '\t' || line[cnt + 1] == '\f'))
					line.erase(cnt + 1, 1);
				line[cnt] = '\t';
			}
		}

        pos = line.find("\t");
        // Less than 2 words/columns
		if(pos == std::string::npos){
			if (started) done = true;
            continue;
		}
        end = line.find("\t", pos + 1);

		if(end == std::string::npos)
			end = line.length() - pos;
		else
			end = end - pos;

		if(end == 0) {	// There is no second word
			if (started) done = true;
			continue;
		}

        // Check to make sure the two words are doubles
		char *str = new char[line.length()];
		char *ptr;
		strcpy(str, line.substr(0, pos).c_str());
		strtod(str, &ptr);
		if(ptr == str) {
			if (started) done = true;
			delete[] str;
            continue;
		}
		strcpy(str, line.substr(pos + 1, end).c_str());
		strtod(str, &ptr);
		if(ptr == str) {
			if (started) done = true;
			delete[] str;
            continue;
		}
		delete[] str;

        curx = strtod(line.substr(0, pos).c_str(), NULL);
        cury = strtod(line.substr(pos + 1, end).c_str(), NULL);

		if(!started && !(fabs(cury) > 0.0)) continue;
		if(!started) started = true;
        x.push_back(curx);
        y.push_back(cury);
	}

	// Removes trailing zeroes
	if(y.empty()) return;
	while(y.back() == 0.0) {
		y.pop_back();
		x.pop_back();
	}
	//sort vectors
	for (unsigned int i = 0; i < x.size(); i++) {
		for(unsigned int j = i + 1; j < x.size(); j++) {
			if( x[j] < x[i]) {
				double a = x[j];
				x[j] = x[i];
				x[i] = a;
				a= y[j];
				y[j] = y[i];
				y[i] = a;
			}
		}
	}
}

void Read1DDataFile(const wchar_t *filename, vector<double>& x)
{
	int i = 0, n = -1;
	ifstream in (filename);

	if(!in) {
		fprintf(stderr, "Error opening file %s for reading\n",
				filename);
    }

	while(!in.eof()) {
		i++;
		if(n > 0 && i > n)
			break;

		std::string line;
        double curx = -1.0;
		getline(in, line);
		if(line.size() == 0)
			break;

		line = line.substr(0, line.find("#"));

		curx = strtod(line.c_str(), NULL);

		x.push_back(curx);
	}
}

void WriteDataFileWHeader(const wchar_t *filename, vector<double>& x,
				   vector<double>& y, std::stringstream& header)
{
	FILE *fp;

	if ((fp = _wfopen(filename, L"w")) == NULL) {
		fprintf(stderr, "Error opening file %s for writing\n",
                        filename);
		exit(1);
	}

	fprintf(fp, "%s\n", header.str().c_str());

	int size = (x.size() < y.size()) ? x.size() : y.size();
	for(int i = 0; i < size; i++)
		fprintf(fp,"%.8g\t%.8g\n", x.at(i), y.at(i));

	fclose(fp);
}

void WriteDataFile(const wchar_t *filename, vector<double>& x,
				   vector<double>& y)
{
	FILE *fp;

	if ((fp = _wfopen(filename, L"w")) == NULL) {
		fprintf(stderr, "Error opening file %s for writing\n",
                        filename);
		exit(1);
	}

	int size = (x.size() < y.size()) ? x.size() : y.size();
	for(int i = 0; i < size; i++)
		fprintf(fp,"%.8g\t%.8g\n", x.at(i), y.at(i));

	fclose(fp);
}

void Write3ColDataFile(const wchar_t *filename, vector<double>& x,
				   vector<double>& y, vector<double>& err)
{
	FILE *fp;

	if ((fp = _wfopen(filename, L"w")) == NULL) {
		fprintf(stderr, "Error opening file %s for writing\n",
                        filename);
		exit(1);
	}

	int size = (x.size() < y.size()) ? x.size() : y.size();
	size = (size < (int) err.size()) ? size : err.size();
	for(int i = 0; i < size; i++)
		fprintf(fp,"%.8g\t%e\t%e\n", x.at(i), y.at(i), err.at(i));

	fclose(fp);
}

void Write1DDataFile(const wchar_t *filename, vector<double>& x)
{
	FILE *fp;

	if ((fp = _wfopen(filename, L"w")) == NULL) {
		fprintf(stderr, "Error opening file %s for writing\n",
                        filename);
	}

	for(unsigned int i = 0; i < x.size(); i++)
		fprintf(fp,"%f\n", x.at(i));

	fclose(fp);
}

void GetDirectory(const wchar_t *file, wchar_t *result, int n) {
	const wchar_t *index = NULL;
	index = wcsrchr(file, '\\');

	if(!index)
		wcscpy(result, L"?ERROR?");
	else
		wcsncpy(result, file, (n < (int)(index - file)) ? n : (int)(index - file) );
}

void GetBasename(const wchar_t *file, wchar_t *result, int n) {
	const wchar_t *index = NULL, *ext = NULL;
	index = wcsrchr(file, '\\'); // find directory
	ext = wcsrchr(file, '.'); // find extension, if exists

	if(!index)
            wcscpy(result, L"?ERROR?");
	else {
		if(!ext)
			wcsncpy(result, index, n);
		else
			wcsncpy(result, index, (n < (int)(ext - index)) ? n : (int)(ext - index) );
	}
}

/*

std::string GetDirectory(const string file) {
	char temp[256] = {0};

	GetDirectory(file.c_str(), temp);

	return std::string(temp);
}

std::string GetBasename(const string file) {
	char temp[256] = {0};

	GetBasename(file.c_str(), temp);

	return std::string(temp);
}
*/
