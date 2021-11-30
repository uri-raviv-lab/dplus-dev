#pragma once
#include <cstdint>
#include <memory>
#include "ICompressionMethod.h"

#include "StoreMethod.h"

#define ZIP_METHOD_TABLE          \
  ZIP_METHOD_ADD(StoreMethod);    \

#define ZIP_METHOD_ADD(method_class)                                                            \
  if (compressionMethod == method_class::GetZipMethodDescriptorStatic().GetCompressionMethod()) \
    return method_class::Create()

struct ZipMethodResolver
{
  static ICompressionMethod::Ptr GetZipMethodInstance(uint16_t compressionMethod)
  {
    ZIP_METHOD_TABLE;
    return ICompressionMethod::Ptr();
  }
};

#undef ZIP_METHOD
