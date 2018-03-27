# date March 2018
# CMake flags in use:
#       -DBUILD_SHARED=ON                                (Default: compile a shared lib)
#       -DBUILD_STATIC=ON                                (Optional: compile a static lib)
#       -DBUILD_TESTS=ON                                (Optional: compile various tests - run with `make test`)
#       -DBUILD_EXAMPLES=ON                             (Optional: compile various examples)
#       -DCMAKE_BUILD_TYPE=Debug                        (Optional: enable debug symbols)
#       -DCMAKE_INSTALL_PREFIX=/usr                        (Optional: specify the library install directory
#
# After succesfull compilation, run as su/sudo: `make install`
# 
PROJECT(shoggoths_map)
cmake_minimum_required(VERSION 2.8)

# library paths
set(LIBRARY_PATH ${LIBRARY_PATH}
              /lib
              /usr/lib
              /usr/lib64
              /usr/local/lib
              /usr/local/lib64
              /usr/lib/x86_64-linux-gnu)

message("Build directory: ${CMAKE_CURRENT_BINARY_DIR}")

# include everything from "/includes"
include_directories(".")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "Building: ${CMAKE_BUILD_TYPE}. Ommit flag to change to `Release`")
else()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -O3")
    message(STATUS "Building: ${CMAKE_BUILD_TYPE}. Add flag `-DCMAKE_BUILD_TYPE=Debug` to switch to `Debug`")
endif()


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp
                               src/orb/extractor.cpp
                               src/orb/extra.cpp
              )

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT}
                                      ${OpenCV_LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
