#ifndef __INIMGT_H
#define __INIMGT_H

#pragma once

#include "simpleini/SimpleIni.h"
#include "Common.h"

// Defines the current version of INI files
#define INI_VERSION 2

double GetIniDouble (const std::wstring& file, const std::string& object, const std::string& param);
void   SetIniDouble (const std::wstring& file, const std::string& object, const std::string& param,
						     double value, int precision = 6);

int  GetIniInt (const std::wstring& file, const std::string& object, const std::string& param);
void SetIniInt (const std::wstring& file, const std::string& object, const std::string& param,
					    int value);

char GetIniChar (const std::wstring& file, const std::string& object, const std::string& param);
void SetIniChar (const std::wstring& file, const std::string& object, const std::string& param,
						 char value);

void GetIniString(const std::wstring& file, const std::string& section, const std::string& key, 
						   std::string& result);
void SetIniString (const std::wstring& file, const std::string& object, const std::string& param,
						   const std::string& value);



#endif
