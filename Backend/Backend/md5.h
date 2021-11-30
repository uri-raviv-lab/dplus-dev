/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * See md5.c for more information.
 */
 
#include <string>
#include "Common.h"

#ifdef HAVE_OPENSSL
#include <openssl/md5.h>
#elif !defined(_MD5_H)
#define _MD5_H
 
/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD5_u32plus;
 
typedef struct {
	MD5_u32plus lo, hi;
	MD5_u32plus a, b, c, d;
	unsigned char buffer[64];
	MD5_u32plus block[16];
} MD5_CTX;
 
EXPORTED_BE void MD5_Init(MD5_CTX *ctx);
EXPORTED_BE void MD5_Update(MD5_CTX *ctx, const void *data, unsigned long size);
EXPORTED_BE void MD5_Final(unsigned char *result, MD5_CTX *ctx);
 
inline std::string bin2hexstr(const unsigned char *data, unsigned long size)
{
	std::string res (size * 2, '\0');

	// Convert to a string
	for(unsigned long i = 0; i < size; i++)
	{
		char a[3] = {0};
		sprintf(a, "%02X", data[i]);
		res[2*i]   = a[0];
		res[2*i+1] = a[1];
	}

	return res;
}

inline std::string md5(const void *data, unsigned long size)
{
	MD5_CTX ctx;
	MD5_Init(&ctx);
	MD5_Update(&ctx, data, size);
	
	unsigned char d[16] = {0};
	MD5_Final(d, &ctx);

	return bin2hexstr(d, 16);
}

inline std::string md5(const std::string& str)
{
	return md5(str.c_str(), str.length());
}


#endif
