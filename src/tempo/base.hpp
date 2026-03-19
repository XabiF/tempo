
#pragma once
#include <cstdint>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace tempo {

    using i32 = int32_t;
    using u64 = uint64_t;

    template<typename S, i32 N>
    using ETensor = Eigen::Tensor<S, N>;

    using indx = Eigen::Index;

    template<typename S, i32 N>
    struct Tensor {
        ETensor<S, N> raw;

        inline explicit Tensor(ETensor<S, N> &&tensor) : raw(std::move(tensor)) {}

        template<typename... Dims>
        static inline Tensor<S, N> zeroed(Dims... dims) {
            ETensor<S, N> raw(dims...);
            raw.setZero();
            return Tensor(std::move(raw));
        }

        static inline Tensor<S, N> zeroed_vec(const std::array<indx, N> &dims) {
            ETensor<S, N> raw(dims);
            raw.setZero();
            return Tensor(std::move(raw));
        }

        template<typename... IndexTypes>
        inline S &at(IndexTypes... indices) {
            return raw(indices...);
        }
    };

}
