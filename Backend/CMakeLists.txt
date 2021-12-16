set(PROJECT_NAME Backend)
aux_source_directory(Backend backend_files)
aux_source_directory(Backend/Externals backend_files2)
aux_source_directory(${dplus_SOURCE_DIR}/PDBReaderLib backend_files_pdb)
aux_source_directory(${dplus_SOURCE_DIR}/Common/lua-5.1.4/src backend_files3)
aux_source_directory(${dplus_SOURCE_DIR}/Common/lua-5.1.4/src lua_files)
aux_source_directory(${dplus_SOURCE_DIR}/Common/TinyThread backend_files4)
aux_source_directory(${dplus_SOURCE_DIR}/Common/ZipLib/Source/extlibs/zlib zip1)
aux_source_directory(${dplus_SOURCE_DIR}/Common/ZipLib/Source zip2)
aux_source_directory(DefaultModels models)
aux_source_directory(GPU gpumodels)
aux_source_directory(${dplus_SOURCE_DIR}/Conversions backend_files_conv)
aux_source_directory(Backend/liblbfgs backend_files_lbfgs)


link_directories(${dplus_SOURCE_DIR}/Common/ZipLib/Source/extlibs/zlib)
link_directories(${dplus_SOURCE_DIR}/Common/ZipLib/Source)


list(REMOVE_ITEM backend_files3 ${dplus_SOURCE_DIR}/Common/lua-5.1.4/src/lua.c 
     ${dplus_SOURCE_DIR}/Common/lua-5.1.4/src/luac.c)

#FILE(GLOB gpu_source_files GPU/*.cpp GPU/*.cu)
#list(REMOVE_ITEM gpu_source_files ${dplus_SOURCE_DIR}/Backend/GPU/PDBGenDirect-rcosbCaching.cu)

FILE(GLOB backend_cuda_files Backend/*.cu GPU/*.cu)
#list(REMOVE_ITEM backend_cuda_files ${dplus_SOURCE_DIR}/Backend/GPU/PDBGenDirect-#rcosbCaching.cu)

list(REMOVE_ITEM lua_files ${dplus_SOURCE_DIR}/Common/lua-5.1.4/src/luac.c)
add_library(lua ${lua_files})

# Hack to suppress (all) warnings from lua
# We get a lot of warnings when compiling lua because we used add_definitions(-std=c++11... )
# which is applied to bot C and C++ code. We did this because nvcc makes us (otherwise we get
# the same warning in the cuda code for which nvcc calls the C preprocessor...) 
set_target_properties(lua PROPERTIES COMPILE_FLAGS "-w -fPIC")

add_definitions(-DBACKEND)

FILE(GLOB gpu_aff_file GPU/A*.cpp)


include_directories(Backend)

# add_library(xplusmodels SHARED ${models})
add_library(backend SHARED ${dplus_SOURCE_DIR}/Common/mathfuncs.cpp
            ${dplus_SOURCE_DIR}/Common/Statistics.cpp
			${gpu_aff_file} ${backend_cuda_files}
            ${backend_files} ${models} ${gpumodels} ${backend_files2} ${backend_files4} ${backend_files_conv} 
${backend_files_pdb} ${backend_files_miniglog} ${backend_files_lbfgs})
#add_dependencies(backend gitrev)
target_link_libraries(backend lua zlib ZipLib ${Boost_LIBRARIES} pthread dl ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY})

#cuda_add_library(gpubackend SHARED ${gpu_source_files})
#target_link_libraries(gpubackend ${Boost_LIBRARIES} dl pthread ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY})
