cmake_minimum_required(VERSION "3.18.2")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_BUILD_TYPE Debug)

#indicate for cmake to look for packages in the vcpkg directory
include(C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake)

project("mini-projects")

file(GLOB SOURCES "./src/*.cpp" "./src/*.hpp")

#find packages needed for this project
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
#linking opencv directories
link_directories(${OpenCV_LIB_DIR})
link_libraries(${OpenCV_LIBS})

#Libraries offered by opencv
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include_directory: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

#linking dll to my project
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ${OpenCV_LIBS} )

