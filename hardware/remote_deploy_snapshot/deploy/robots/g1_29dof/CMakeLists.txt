cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)

set(UNITREE_SDK2_PREFIX
  "/home/mosense/guo/.unitree_sdk2"
  CACHE PATH "Local unitree_sdk2 install prefix")
set(CPP_DEPS_PREFIX
  "/home/mosense/guo/.cpp_deps"
  CACHE PATH "Local C++ dependency prefix")
set(ONNXRUNTIME_PREFIX
  "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0"
  CACHE PATH "ONNX Runtime prefix")

foreach(required
    "${UNITREE_SDK2_PREFIX}/lib/libunitree_sdk2.a"
    "${UNITREE_SDK2_PREFIX}/lib/libddsc.so"
    "${UNITREE_SDK2_PREFIX}/lib/libddscxx.so"
    "${CPP_DEPS_PREFIX}/lib/libboost_program_options.so"
    "${CPP_DEPS_PREFIX}/lib/libyaml-cpp.so"
    "${CPP_DEPS_PREFIX}/lib/libspdlog.so"
    "${ONNXRUNTIME_PREFIX}/lib/libonnxruntime.so.1.22.0")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing deployment dependency: ${required}")
  endif()
endforeach()

include_directories(
  ${CPP_DEPS_PREFIX}/include
  ${CPP_DEPS_PREFIX}/include/eigen3
  ${UNITREE_SDK2_PREFIX}/include
  ${UNITREE_SDK2_PREFIX}/include/ddscxx
  include
  ${ONNXRUNTIME_PREFIX}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  ${UNITREE_SDK2_PREFIX}/lib/libunitree_sdk2.a
  ${UNITREE_SDK2_PREFIX}/lib/libddsc.so
  ${UNITREE_SDK2_PREFIX}/lib/libddscxx.so
  rt pthread
  ${CPP_DEPS_PREFIX}/lib/libboost_program_options.so
  ${CPP_DEPS_PREFIX}/lib/libyaml-cpp.so
  ${CPP_DEPS_PREFIX}/lib/libspdlog.so
  ${ONNXRUNTIME_PREFIX}/lib/libonnxruntime.so.1.22.0
)

# Ubuntu 20.04's spdlog package uses its bundled fmt implementation, while
# newer local builds link against a separate libfmt.  Support both layouts so
# the same controller sources can be built natively on the G1 EDU2 computer.
if(EXISTS "${CPP_DEPS_PREFIX}/lib/libfmt.so")
  link_libraries(${CPP_DEPS_PREFIX}/lib/libfmt.so)
endif()

set(CMAKE_BUILD_RPATH
  "${UNITREE_SDK2_PREFIX}/lib;${CPP_DEPS_PREFIX}/lib;${ONNXRUNTIME_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_BUILD_RPATH}")

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)
