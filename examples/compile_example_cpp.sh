#!/bin/bash

# Compile with as many optimizations as possible
# You may need to adapt the include paths to Eigen and teMPO, depending on their location

mkdir -p build
g++ example_cpp.cpp -I/usr/include/eigen3 -I../src -std=gnu++17 -march=native -O3 -DEIGEN_NO_DEBUG -DEIGEN_NO_ASSERT -fopenmp -funsafe-math-optimizations -o build/example_cpp
