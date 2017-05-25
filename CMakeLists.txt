cmake_minimum_required(VERSION 2.8)
project(OpenCloth_Co_rotation)

#Additional libraries
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/include/kinect/)
include_directories(${CMAKE_SOURCE_DIR}/AFEM/include/)
include_directories(${CMAKE_SOURCE_DIR}/AFEM/cuda/gpu_include/)
include_directories(${CMAKE_SOURCE_DIR}/Geometry)
link_directories(${CMAKE_SOURCE_DIR}/lib)
#link_libraries()
#opencv
#find_package(OPENCV REQUIRED)
#include_directories( ${OpenCV_INCLUDE_DIRS} )


#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)
#link_libraries(${CUDA_LIBRARIES})
message("${CUDA_LIBRARIES}")
add_executable(OpenCloth "OpenCloth/OpenCloth_CoRotated_Linear_FEM/main.cpp")

file(GLOB SOURCES AFEM/source/*.cpp )
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)
add_subdirectory(AFEM/cuda)


ADD_EXECUTABLE(AFEM ${SOURCES})


target_link_libraries(OpenCloth freeglutd glew32d )
#target_link_libraries( AFEM cudart cublas cuda cusparse)
target_link_libraries( AFEM AFEM_cuda_lib cudart freeglutd glew32d )



#set this to your opencv version and change depending if debug or release
SET(CV_VERSION 310)
SET(CV_FILE_PRE ${CV_VERSION}d.dll)





#Adding dll to the debug (assuming debug) folder
if(1)
add_custom_target(dll_includes

#opencv
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/opencv_core${CV_FILE_PRE}            			${CMAKE_BINARY_DIR}/Debug/
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/opencv_highgui${CV_FILE_PRE}            		${CMAKE_BINARY_DIR}/Debug/
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/opencv_imgcodecs${CV_FILE_PRE}            		${CMAKE_BINARY_DIR}/Debug/
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/opencv_imgproc${CV_FILE_PRE}            		${CMAKE_BINARY_DIR}/Debug/
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/opencv_videoio${CV_FILE_PRE}           		${CMAKE_BINARY_DIR}/Debug/


  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/freeglutd.dll            ${CMAKE_BINARY_DIR}/Debug/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/glew32d.dll          ${CMAKE_BINARY_DIR}/Debug/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/msvcp120.dll            ${CMAKE_BINARY_DIR}/Debug/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/msvcr120.dll          ${CMAKE_BINARY_DIR}/Debug/
 

 COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Geometry/FEM_Nodes.txt          ${CMAKE_BINARY_DIR}/Debug/
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Geometry/FEM_Nodes.txt          ${CMAKE_BINARY_DIR}/
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Geometry/FEM_Elem.txt          ${CMAKE_BINARY_DIR}/Debug/
 COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Geometry/FEM_Elem.txt          ${CMAKE_BINARY_DIR}/
 
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Geometry/FEM_Stationary.txt          ${CMAKE_BINARY_DIR}/

COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/AFEM_kinect_lib.dll         ${CMAKE_BINARY_DIR}/debug


)
add_dependencies(OpenCloth  dll_includes)
add_dependencies(AFEM        dll_includes)
endif(1)




