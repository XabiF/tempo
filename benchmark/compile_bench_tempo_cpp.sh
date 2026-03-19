#!/bin/bash

mkdir -p build
g++ bench_tempo_cpp.cpp -I/usr/include/eigen3 -I../src -std=gnu++17 -march=native -O3 -DEIGEN_NO_DEBUG -DEIGEN_NO_ASSERT -fopenmp -funsafe-math-optimizations -o build/bench_tempo_cpp
