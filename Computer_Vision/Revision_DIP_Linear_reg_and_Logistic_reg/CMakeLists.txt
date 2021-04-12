cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_BUILD_TYPE Debug)
set(OpenCV_DIR "/usr/local/include/opencv_4.1.0/lib/cmake/opencv4")
#set(GCC_COVERAGE_COMPILE_FLAGS "-lstdc++fs")
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

project(DIP_LR_Logit)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


#This project is divided into sub-projects in which each sub-project have its own directory
    # spatial_filtering/
    # machine_learning_basics

#file(GLOB SOURCES "spatial_filtering/*.hpp" "spatial_filtering/*.cpp")
file(GLOB SOURCES "machine_learning_basics/*.hpp" "machine_learning_basics/*.cpp")


add_executable(${PROJECT_NAME} ${SOURCES})
add_definitions(${OpenCV_DEFINITIONS})

#target_compile_options(${PROJECT_NAME} PRIVATE )
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ${OpenCV_LIBS} -lstdc++fs)