cmake_minimum_required(VERSION 2.8)
project( OpenCloth_Co_rotation)

if(0)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)
endif(0)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)
link_libraries(freeglutd glew32d)



#include_directories("OpenCloth/OpenCloth_CoRotated_Linear_FEM/")
add_executable( OpenCloth "OpenCloth/OpenCloth_CoRotated_Linear_FEM/main.cpp")



#target_link_libraries( OpenCloth freeglutd glew32d)

if(1)
add_custom_target(dll_includes

  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/freeglutd.dll            ${CMAKE_BINARY_DIR}/Debug
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bin/glew32d.dll          ${CMAKE_BINARY_DIR}/Debug
 
)

add_dependencies(OpenCloth  dll_includes)
endif(1)




