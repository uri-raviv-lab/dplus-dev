cmake_minimum_required (VERSION 3.0) 
project (dplus C CXX CUDA) 

#########################
# Packages

find_package(CUDA 11 REQUIRED)
set (CUDA_NVCC_FLAGS -std=c++14 -lcurand -lcublas -lcudart --generate-code --generate-code arch=compute_52,code=sm_52)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-march=native -fPIC")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-march=native")

#set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS system filesystem thread program_options REQUIRED)

#########################

# Moving -std=c++11 to CMAKE_CXX_FLAGS (as it should be) makes nvcc cause a lot of warnings:
# "cc1: warning: command line option ‘-std=c++11’ is valid for C++/ObjC++ but not for C"
# See https://public.kitware.com/Bug/print_bug_page.php?bug_id=15240
# Since ubuntu 14.04 is stuck at cmake 2.8, we'll just have to ignore these warnings for now.
# Other option: try and remove the -std=c++11 flag (definition) from C projects (lua)
# add_definitions(-std=c++14)



include_directories(${dplus_SOURCE_DIR}/Common ${dplus_SOURCE_DIR}/Common/ZipLib/Source ${dplus_SOURCE_DIR}/PDBReaderLib ${dplus_SOURCE_DIR}/Common/lua-5.1.4/src ${dplus_SOURCE_DIR}/Backend
					${dplus_SOURCE_DIR}/Backend/Backend
					${dplus_SOURCE_DIR}/Backend/GPU
                    ${dplus_SOURCE_DIR}/Externals/ceres-for-windows/ceres-solver-1.12.0/include
                    ${dplus_SOURCE_DIR}/Common/zeromq-4.1.3/include
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#set(EIGEN_INCLUDE_DIR ${dplus_SOURCE_DIR}/Externals/ceres-for-windows/eigen)
set(EIGEN_INCLUDE_DIR ${dplus_SOURCE_DIR}/Common)


#########################
## CERES SOLVER v1.12.0
#add_subdirectory(Externals/ceres-for-windows/eigen)
add_subdirectory(Externals/ceres-for-windows/ceres-solver-1.12.0)
#########################

#build ziplib
 add_subdirectory(Common/ZipLib/Source/extlibs/zlib)
 add_subdirectory(Common/ZipLib/Source/)



############################

add_subdirectory (Backend)

add_executable(pdbgen pdbgen/pdbgen.cpp)

# DebyeCalculator
aux_source_directory(${dplus_SOURCE_DIR}/PDBReaderLib backend_files_pdb)
add_executable(debye DebyeCalculator/DebyeCalculator.cpp ${backend_files_pdb})
target_link_libraries(debye backend ${Boost_LIBRARIES} dl pthread)

# GetAllMetadata
aux_source_directory(Backend/CommandLine/GetAllMetadata getallmetadata_files)
add_executable(getallmetadata ${getallmetadata_files})
target_link_libraries(getallmetadata backend ${Boost_LIBRARIES} dl pthread)


# Generate
aux_source_directory(Backend/CommandLine/Generate generate_files)
add_executable(generate ${generate_files})
target_link_libraries(generate backend ${Boost_LIBRARIES} dl pthread)

# Fit
aux_source_directory(Backend/CommandLine/Fit fit_files)
add_executable(fit ${fit_files})
target_link_libraries(fit backend ${Boost_LIBRARIES} dl pthread)


# CheckCapabilities
aux_source_directory(Backend/CommandLine/CheckCapabilities checkCapabilities_files)
add_executable(checkCapabilities ${checkCapabilities_files})
target_link_libraries(checkCapabilities backend ${Boost_LIBRARIES} dl pthread)



target_link_libraries(pdbgen backend dl)

