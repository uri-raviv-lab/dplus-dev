/**********************************************************************************
 * Copyright (c) 2008-2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 **********************************************************************************/

/* $Revision: 9283 $ on $Date: 2009-10-14 10:18:57 -0700 (Wed, 14 Oct 2009) $ */

#ifndef __CL_PLATFORM_H
#define __CL_PLATFORM_H

#ifdef __APPLE__
    /* Contains #defines for AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER below */
    #include <AvailabilityMacros.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CL_API_ENTRY
#define CL_API_CALL
#ifdef __APPLE__
#define CL_API_SUFFIX__VERSION_1_0   AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER
#define CL_EXTENSION_WEAK_LINK       __attribute__((weak_import))       
#else
#define CL_API_SUFFIX__VERSION_1_0
#define CL_EXTENSION_WEAK_LINK                         
#endif

#if (defined (_WIN32) && (_MSC_VER))

/* scalar types  */
typedef signed   __int8         cl_char;
typedef unsigned __int8         cl_uchar;
typedef signed   __int16        cl_short;
typedef unsigned __int16        cl_ushort;
typedef signed   __int32        cl_int;
typedef unsigned __int32        cl_uint;
typedef signed   __int64        cl_long;
typedef unsigned __int64        cl_ulong;

typedef unsigned __int16        cl_half;
typedef float                   cl_float;
typedef double                  cl_double;


/*
 * Vector types 
 *
 *  Note:   OpenCL requires that all types be naturally aligned. 
 *          This means that vector types must be naturally aligned.
 *          For example, a vector of four floats must be aligned to
 *          a 16 byte boundary (calculated as 4 * the natural 4-byte 
 *          alignment of the float).  The alignment qualifiers here
 *          will only function properly if your compiler supports them
 *          and if you don't actively work to defeat them.  For example,
 *          in order for a cl_float4 to be 16 byte aligned in a struct,
 *          the start of the struct must itself be 16-byte aligned. 
 *
 *          Maintaining proper alignment is the user's responsibility.
 */
typedef signed   __int8          cl_char2[2];
typedef signed   __int8          cl_char4[4];
typedef signed   __int8          cl_char8[8];
typedef signed   __int8          cl_char16[16];
typedef unsigned __int8         cl_uchar2[2];
typedef unsigned __int8         cl_uchar4[4];
typedef unsigned __int8         cl_uchar8[8];
typedef unsigned __int8         cl_uchar16[16];

typedef signed   __int16         cl_short2[2];
typedef signed   __int16         cl_short4[4];
typedef signed   __int16         cl_short8[8];
typedef signed   __int16         cl_short16[16];
typedef unsigned __int16        cl_ushort2[2];
typedef unsigned __int16        cl_ushort4[4];
typedef unsigned __int16        cl_ushort8[8];
typedef unsigned __int16        cl_ushort16[16];

typedef signed   __int32         cl_int2[2];
typedef signed   __int32         cl_int4[4];
typedef signed   __int32         cl_int8[8];
typedef signed   __int32         cl_int16[16];
typedef unsigned __int32        cl_uint2[2];
typedef unsigned __int32        cl_uint4[4];
typedef unsigned __int32        cl_uint8[8];
typedef unsigned __int32        cl_uint16[16];

typedef signed   __int64         cl_long2[2];
typedef signed   __int64         cl_long4[4];
typedef signed   __int64         cl_long8[8];
typedef signed   __int64         cl_long16[16];
typedef unsigned __int64        cl_ulong2[2];
typedef unsigned __int64        cl_ulong4[4];
typedef unsigned __int64        cl_ulong8[8];
typedef unsigned __int64        cl_ulong16[16];

typedef float           cl_float2[2];
typedef float           cl_float4[4];
typedef float           cl_float8[8];
typedef float           cl_float16[16];

typedef double          cl_double2[2];
typedef double          cl_double4[4];
typedef double          cl_double8[8];
typedef double          cl_double16[16];
/* There are no vector types for half */

#define CL_CHAR_BIT         8
#define CL_SCHAR_MAX        127
#define CL_SCHAR_MIN        (-127-1)
#define CL_CHAR_MAX         CL_SCHAR_MAX
#define CL_CHAR_MIN         CL_SCHAR_MIN
#define CL_UCHAR_MAX        255
#define CL_SHRT_MAX         32767
#define CL_SHRT_MIN         (-32767-1)
#define CL_USHRT_MAX        65535
#define CL_INT_MAX          2147483647
#define CL_INT_MIN          (-2147483647-1)
#define CL_UINT_MAX         0xffffffffU
#define CL_LONG_MAX         ((cl_long) 0x7FFFFFFFFFFFFFFFLL)
#define CL_LONG_MIN         ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define CL_ULONG_MAX        ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL)

#define CL_FLT_DIG          6
#define CL_FLT_MANT_DIG     24
#define CL_FLT_MAX_10_EXP   +38
#define CL_FLT_MAX_EXP      +128
#define CL_FLT_MIN_10_EXP   -37
#define CL_FLT_MIN_EXP      -125
#define CL_FLT_RADIX        2
#define CL_FLT_MAX          340282346638528859811704183484516925440.0f
#define CL_FLT_MIN          1.175494350822287507969e-38f
#define CL_FLT_EPSILON      0x1.0p-23f

#define CL_DBL_DIG          15
#define CL_DBL_MANT_DIG     53
#define CL_DBL_MAX_10_EXP   +308
#define CL_DBL_MAX_EXP      +1024
#define CL_DBL_MIN_10_EXP   -307
#define CL_DBL_MIN_EXP      -1021
#define CL_DBL_RADIX        2
#define CL_DBL_MAX          179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0
#define CL_DBL_MIN          2.225073858507201383090e-308
#define CL_DBL_EPSILON      2.220446049250313080847e-16


#else

#include <stdint.h>

/* scalar types  */
typedef int8_t          cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t         cl_short    __attribute__((aligned(2)));
typedef uint16_t        cl_ushort   __attribute__((aligned(2)));
typedef int32_t         cl_int      __attribute__((aligned(4)));
typedef uint32_t        cl_uint     __attribute__((aligned(4)));
typedef int64_t         cl_long     __attribute__((aligned(8)));
typedef uint64_t        cl_ulong    __attribute__((aligned(8)));

typedef uint16_t        cl_half     __attribute__((aligned(2)));
typedef float           cl_float    __attribute__((aligned(4)));
typedef double          cl_double   __attribute__((aligned(8)));

/*
 * Vector types 
 *
 *  Note:   OpenCL requires that all types be naturally aligned. 
 *          This means that vector types must be naturally aligned.
 *          For example, a vector of four floats must be aligned to
 *          a 16 byte boundary (calculated as 4 * the natural 4-byte 
 *          alignment of the float).  The alignment qualifiers here
 *          will only function properly if your compiler supports them
 *          and if you don't actively work to defeat them.  For example,
 *          in order for a cl_float4 to be 16 byte aligned in a struct,
 *          the start of the struct must itself be 16-byte aligned. 
 *
 *          Maintaining proper alignment is the user's responsibility.
 */
typedef int8_t          cl_char2[2]     __attribute__((aligned(2)));
typedef int8_t          cl_char4[4]     __attribute__((aligned(4)));
typedef int8_t          cl_char8[8]     __attribute__((aligned(8)));
typedef int8_t          cl_char16[16]   __attribute__((aligned(16)));
typedef uint8_t         cl_uchar2[2]    __attribute__((aligned(2)));
typedef uint8_t         cl_uchar4[4]    __attribute__((aligned(4)));
typedef uint8_t         cl_uchar8[8]    __attribute__((aligned(8)));
typedef uint8_t         cl_uchar16[16]  __attribute__((aligned(16)));

typedef int16_t         cl_short2[2]     __attribute__((aligned(4)));
typedef int16_t         cl_short4[4]     __attribute__((aligned(8)));
typedef int16_t         cl_short8[8]     __attribute__((aligned(16)));
typedef int16_t         cl_short16[16]   __attribute__((aligned(32)));
typedef uint16_t        cl_ushort2[2]    __attribute__((aligned(4)));
typedef uint16_t        cl_ushort4[4]    __attribute__((aligned(8)));
typedef uint16_t        cl_ushort8[8]    __attribute__((aligned(16)));
typedef uint16_t        cl_ushort16[16]  __attribute__((aligned(32)));

typedef int32_t         cl_int2[2]      __attribute__((aligned(8)));
typedef int32_t         cl_int4[4]      __attribute__((aligned(16)));
typedef int32_t         cl_int8[8]      __attribute__((aligned(32)));
typedef int32_t         cl_int16[16]    __attribute__((aligned(64)));
typedef uint32_t        cl_uint2[2]     __attribute__((aligned(8)));
typedef uint32_t        cl_uint4[4]     __attribute__((aligned(16)));
typedef uint32_t        cl_uint8[8]     __attribute__((aligned(32)));
typedef uint32_t        cl_uint16[16]   __attribute__((aligned(64)));

typedef int64_t         cl_long2[2]     __attribute__((aligned(16)));
typedef int64_t         cl_long4[4]     __attribute__((aligned(32)));
typedef int64_t         cl_long8[8]     __attribute__((aligned(64)));
typedef int64_t         cl_long16[16]   __attribute__((aligned(128)));
typedef uint64_t        cl_ulong2[2]    __attribute__((aligned(16)));
typedef uint64_t        cl_ulong4[4]    __attribute__((aligned(32)));
typedef uint64_t        cl_ulong8[8]    __attribute__((aligned(64)));
typedef uint64_t        cl_ulong16[16]  __attribute__((aligned(128)));

typedef float           cl_float2[2]    __attribute__((aligned(8)));
typedef float           cl_float4[4]    __attribute__((aligned(16)));
typedef float           cl_float8[8]    __attribute__((aligned(32)));
typedef float           cl_float16[16]  __attribute__((aligned(64)));

typedef double          cl_double2[2]   __attribute__((aligned(16)));
typedef double          cl_double4[4]   __attribute__((aligned(32)));
typedef double          cl_double8[8]   __attribute__((aligned(64)));
typedef double          cl_double16[16] __attribute__((aligned(128)));

/* There are no vector types for half */

/******************************************************************************/

// Macro names and corresponding values defined by OpenCL

#define CL_CHAR_BIT         8
#define CL_SCHAR_MAX        127
#define CL_SCHAR_MIN        (-127-1)
#define CL_CHAR_MAX         CL_SCHAR_MAX
#define CL_CHAR_MIN         CL_SCHAR_MIN
#define CL_UCHAR_MAX        255
#define CL_SHRT_MAX         32767
#define CL_SHRT_MIN         (-32767-1)
#define CL_USHRT_MAX        65535
#define CL_INT_MAX          2147483647
#define CL_INT_MIN          (-2147483647-1)
#define CL_UINT_MAX         0xffffffffU
#define CL_LONG_MAX         ((cl_long) 0x7FFFFFFFFFFFFFFFLL)
#define CL_LONG_MIN         ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define CL_ULONG_MAX        ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL)

#define CL_FLT_DIG          6
#define CL_FLT_MANT_DIG     24
#define CL_FLT_MAX_10_EXP   +38
#define CL_FLT_MAX_EXP      +128
#define CL_FLT_MIN_10_EXP   -37
#define CL_FLT_MIN_EXP      -125
#define CL_FLT_RADIX        2
#define CL_FLT_MAX          0x1.fffffep127f
#define CL_FLT_MIN          0x1.0p-126f
#define CL_FLT_EPSILON      0x1.0p-23f

#define CL_DBL_DIG          15
#define CL_DBL_MANT_DIG     53
#define CL_DBL_MAX_10_EXP   +308
#define CL_DBL_MAX_EXP      +1024
#define CL_DBL_MIN_10_EXP   -307
#define CL_DBL_MIN_EXP      -1021
#define CL_DBL_RADIX        2
#define CL_DBL_MAX          0x1.fffffffffffffp1023
#define CL_DBL_MIN          0x1.0p-1022
#define CL_DBL_EPSILON      0x1.0p-52

#endif

#include <stddef.h>
  
  
#ifdef __cplusplus
}
#endif

#endif  // __CL_PLATFORM_H
