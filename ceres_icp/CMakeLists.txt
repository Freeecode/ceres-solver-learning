cmake_minimum_required(VERSION 3.1.0)
project(ICP)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fpermissive -Wsign-compare")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV 3 REQUIRED)

include_directories(
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(ceres_icp src/ceres_icp.cpp
                         src/matcher.cpp                          
)
target_link_libraries(ceres_icp ${OpenCV_LIBS} ${CERES_LIBRARIES})
