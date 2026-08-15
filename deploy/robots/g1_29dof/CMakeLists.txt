cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

if(NOT DEFINED ONNXRUNTIME_DIR)
  set(ONNXRUNTIME_DIR
      ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0)
endif()

if(NOT DEFINED UNITREE_SDK_INCLUDE_DIR)
  set(UNITREE_SDK_INCLUDE_DIR /usr/local/include)
endif()

if(NOT DEFINED DDS_WRAPPER_INCLUDE_DIR)
  set(DDS_WRAPPER_INCLUDE_DIR /usr/local/include)
endif()

if(NOT DEFINED UNITREE_SDK_LIB_DIR)
  set(UNITREE_SDK_LIB_DIR /usr/local/lib)
endif()

if(NOT DEFINED DDS_LIB_DIR)
  set(DDS_LIB_DIR /usr/local/lib)
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
  ${UNITREE_SDK_INCLUDE_DIR}
  ${DDS_WRAPPER_INCLUDE_DIR}
)

link_libraries(
  ${UNITREE_SDK_LIB_DIR}/libunitree_sdk2.a
  ${DDS_LIB_DIR}/libddscxx.so
  ${DDS_LIB_DIR}/libddsc.so
  rt pthread
  libboost_program_options.a libyaml-cpp.a
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

find_library(FMT_LIB fmt)
if(FMT_LIB)
  link_libraries(${FMT_LIB})
endif()

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)
