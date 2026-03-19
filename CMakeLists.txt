cmake_minimum_required(VERSION 3.15)
project(tempo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
)

# Create Python module
pybind11_add_module(${PROJECT_NAME} py/python.cpp)

# Linux: static link libstdc++ for better portability
if(UNIX AND NOT APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE 
        -static-libstdc++ -static-libgcc
    )
    
    target_link_options(${PROJECT_NAME} PRIVATE -static-libstdc++)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    pybind11::headers
    Eigen3::Eigen
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -O3
    -DEIGEN_NO_ASSERT
    -DEIGEN_NO_DEBUG
    -funsafe-math-optimizations
    -ffast-math
    -march=native
    -fopenmp
)

# Installation
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${Python_SITELIB}
)
