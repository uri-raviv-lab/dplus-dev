// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//#ifndef __MAPS_TEXREF_CUH_
//#define __MAPS_TEXREF_CUH_
//
//#include <cuda_runtime.h>
//#include <iterator>
//
//namespace maps
//{
//
//
//    /// @brief A workaround to use 1D texture references (pre-Kepler) in the 
//    /// library. (logic inspired by CUB library)
//    /// We must wrap it with two classes, so as to avoid commas in the CUDA 
//    /// generated code.
//    template <typename T>
//    struct UniqueTexRef1D
//    {
//        template <int TEXTURE_UID>
//        struct TexId
//        {
//            typedef texture<T, cudaTextureType1D,
//                cudaReadModeElementType> TexRefType;
//            static TexRefType tex;
//            
//            template<typename DiffType>
//            static __device__ __forceinline__ T read(DiffType offset) 
//            { 
//                return tex1Dfetch(tex, offset); 
//            }
//
//            /// Bind texture
//            static __host__ cudaError_t BindTexture(const void *d_in, 
//                                                    size_t size)
//            {
//                if (d_in)
//                {
//                    cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
//                    tex.channelDesc = tex_desc;
//                    return cudaBindTexture(NULL, &tex, d_in, &tex_desc, size);
//                }
//
//                return cudaSuccess;
//            }
//
//            /// Unbind texture
//            static __host__ cudaError_t UnbindTexture()
//            {
//                return cudaUnbindTexture(&tex);
//            }
//        };
//    };
//
//    template <typename T>
//    template <int TEXTURE_UID>
//    typename UniqueTexRef1D<T>::template TexId<TEXTURE_UID>::TexRefType 
//      UniqueTexRef1D<T>::template TexId<TEXTURE_UID>::tex = 0;
//
//    /// @brief A workaround to use 2D texture references (pre-Kepler) in the 
//    /// library. (logic inspired by CUB library)
//    /// We must wrap it with two classes, so as to avoid commas in the CUDA 
//    /// generated code.
//    template <typename T>
//    struct UniqueTexRef2D
//    {
//        template <int TEXTURE_UID>
//        struct TexId
//        {
//            typedef texture<T, cudaTextureType2D, 
//                            cudaReadModeElementType> TexRefType;
//            static TexRefType tex;
//
//            template<typename DiffType>
//            static __device__ __forceinline__ T read(DiffType x, DiffType y) 
//            { 
//                return tex2D(tex, x, y); 
//            }
//
//            /// Bind texture
//            static cudaError_t BindTexture(const void *d_in, size_t width, 
//                                           size_t height, size_t stride)
//            {
//                if (d_in)
//                {
//                    cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
//                    tex.channelDesc = tex_desc;
//                    return cudaBindTexture2D(NULL, &tex, d_in, &tex_desc, 
//                                             width, height, stride);
//                }
//
//                return cudaSuccess;
//            }
//
//            /// Unbind texture
//            static cudaError_t UnbindTexture()
//            {
//                return cudaUnbindTexture(&tex);
//            }
//        };
//    };
//
//    template <typename T>
//    template <int TEXTURE_UID>
//    typename UniqueTexRef2D<T>::template TexId<TEXTURE_UID>::TexRefType 
//      UniqueTexRef2D<T>::template TexId<TEXTURE_UID>::tex = 0;
//
//}  // namespace maps
//
//#endif  // __MAPS_TEXREF_CUH_


// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __MAPS_TEXREF_CUH_
#define __MAPS_TEXREF_CUH_

#include <cuda_runtime.h>
#include <iterator>

namespace maps
{
    /// @brief A workaround to use 1D texture objects in the library.
    template <typename T>
    struct UniqueTexRef1D
    {
        template <int TEXTURE_UID>
        struct TexId
        {
            static cudaTextureObject_t tex;

            template<typename DiffType>
            static __device__ __forceinline__ T read(DiffType offset)
            {
                return tex1Dfetch<T>(tex, offset);
            }

            /// Bind texture
            static __host__ cudaError_t BindTexture(const void* d_in, size_t size)
            {
                if (d_in)
                {
                    cudaResourceDesc resDesc = {};
                    resDesc.resType = cudaResourceTypeLinear;
                    resDesc.res.linear.devPtr = const_cast<void*>(d_in);
                    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
                    resDesc.res.linear.sizeInBytes = size;

                    cudaTextureDesc texDesc = {};
                    texDesc.readMode = cudaReadModeElementType;

                    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
                }

                return cudaSuccess;
            }

            /// Unbind texture
            static __host__ cudaError_t UnbindTexture()
            {
                return cudaDestroyTextureObject(tex);
            }
        };
    };

    template <typename T>
    template <int TEXTURE_UID>
    cudaTextureObject_t UniqueTexRef1D<T>::TexId<TEXTURE_UID>::tex = 0;

    /// @brief A workaround to use 2D texture objects in the library.
    template <typename T>
    struct UniqueTexRef2D
    {
        template <int TEXTURE_UID>
        struct TexId
        {
            static cudaTextureObject_t tex;

            template<typename DiffType>
            static __device__ __forceinline__ T read(DiffType x, DiffType y)
            {
                return tex2D<T>(tex, x, y);
            }

            /// Bind texture
            static cudaError_t BindTexture(const void* d_in, size_t width, size_t height, size_t pitch)
            {
                if (d_in)
                {
                    cudaResourceDesc resDesc = {};
                    resDesc.resType = cudaResourceTypePitch2D;
                    resDesc.res.pitch2D.devPtr = const_cast<void*>(d_in);
                    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
                    resDesc.res.pitch2D.width = width;
                    resDesc.res.pitch2D.height = height;
                    resDesc.res.pitch2D.pitchInBytes = pitch;

                    cudaTextureDesc texDesc = {};
                    texDesc.readMode = cudaReadModeElementType;

                    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
                }

                return cudaSuccess;
            }

            /// Unbind texture
            static cudaError_t UnbindTexture()
            {
                return cudaDestroyTextureObject(tex);
            }
        };
    };

    template <typename T>
    template <int TEXTURE_UID>
    cudaTextureObject_t UniqueTexRef2D<T>::TexId<TEXTURE_UID>::tex = 0;

}  // namespace maps

#endif  // __MAPS_TEXREF_CUH_

