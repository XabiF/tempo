# teMPO

> Eigen-based utilities for MPS and MPO tensor networks: C++ library & Python layer

> [!note]
> The current state of the library is more of a PoC

**teMPO** is a (work-in-progress) custom header-only C++ library, whose only external dependencies are [Eigen](https://libeigen.gitlab.io/) and [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) for the Python layer. This library specializes in MPS/MPO tensor network (TN) contractions for both discrete optimization and quantum circuit simulation.

Unlike most MPS/MPO/TN libraries out there, this library does not primarily focus on quantum systems, condensed-matter physics or many-body simulations, but just serves as a general toolbox primarily aimed at solving certain discrete optimization problems or typical QML circuits blazingly fast.

According to [performed benchmarks](benchmark/benchmark.ipynb), this library is faster than other conventional methods to contract such kinds of tensor networks, even if used via Python. It is lightweight and exposes the bare minimum on its Python layer, so that most of the critical for-loops and operations can be optimized and vectorized thanks to Eigen and compiler optimizations.

## TODO

- Implement more well-known MPS and MPOs: things like typical initial states (GHZ, uniform superposition, and other memory-efficient ones), well-known QML ansatzë, and so on...

- Implement generic functions/module for optimization support: see, for instance, [here](https://arxiv.org/abs/2311.14344) and [here](https://arxiv.org/abs/2309.10509) for research papers utilizing these kinds of TN-based discrete optimization.

- Proper documentation

- Cross-platform support via precompiled library binaries

## Usage

> [!note]
> Only Linux platforms are currently supported

> Make sure you have `cmake` and Eigen libraries installed!

Clone the repository and run `pip install .`

Afterwards, the [examples](examples) or the benchmark [Python](benchmark/bench_tempo_py.py) or [C++](benchmark/bench_tempo_cpp.cpp) codes may be followed as code examples. Proper documentation will be made soon!
