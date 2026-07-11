
#pragma once
#include <tempo/base.hpp>
#include <tempo/core/mps.hpp>
#include <tempo/core/mpo.hpp>
#include <tempo/core/mputil.hpp>

namespace tempo {

    template<typename S>
    i32 ETensor1_arg_max(const ETensor<S, 1> &vec) {
        i32 cur_index = 0;
        S cur_max_value = vec(0);
        for(i32 i = 1; i < vec.dimension(0); i++) {
            const auto value = vec(i);
            if(value > cur_max_value) {
                cur_max_value = value;
                cur_index = i;
            }
        }
        return cur_index;
    }

    template<typename S>
    i32 ETensor1_arg_max(const ETensor<std::complex<S>, 1> &vec) {
        i32 cur_index = 0;
        S cur_max_value = std::norm(vec(0));
        for(i32 i = 1; i < vec.dimension(0); i++) {
            const auto value = std::norm(vec(i));
            if(value > cur_max_value) {
                cur_max_value = value;
                cur_index = i;
            }
        }
        return cur_index;
    }

    template<typename S>
    std::vector<indx> mps_argmax_ptraced(const MPS<S> &mps, bool normalize = false) {
        using ETensor1 = ETensor<S, 1>;
        using ETensor2 = ETensor<S, 2>;

        const auto dim = mps.top_tensor.raw.dimension(0);
        const auto size = mps.size();

        ETensor1 partial_trace_vec(dim);
        partial_trace_vec.setConstant(normalize ? (1.0f / dim) : 1.0f);

        std::vector<ETensor1> bottom_trace_vecs = {
            mps.bottom_tensor.raw.contract(partial_trace_vec, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) })
        };

        for(auto it = mps.middle_tensors.rbegin(); it != mps.middle_tensors.rend(); it++) {
            ETensor2 tmp_tsr = it->raw.contract(partial_trace_vec, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
            bottom_trace_vecs.push_back(tmp_tsr.contract(bottom_trace_vecs.back(), Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) }));
        }

        std::vector<indx> max_coeff_indices;
        max_coeff_indices.reserve(size);

        // x 0

        ETensor1 x0_extract_vec = mps.top_tensor.raw.contract(bottom_trace_vecs.back(), Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
        max_coeff_indices.push_back(ETensor1_arg_max(x0_extract_vec));

        // x 1 ... x n-2

        ETensor1 cur_top_trace_vec = mps.top_tensor.raw.chip(max_coeff_indices.back(), 0);
        const auto bottom_trace_count = bottom_trace_vecs.size();
        for(i32 i = 1; i < size - 1; i++) {
            ETensor2 tmp_tsr = cur_top_trace_vec.contract(mps.middle_tensors[i-1].raw, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(0, 0) });
            ETensor1 xi_extract_vec = tmp_tsr.contract(bottom_trace_vecs[bottom_trace_count-(i+1)], Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
            max_coeff_indices.push_back(ETensor1_arg_max(xi_extract_vec));
            cur_top_trace_vec = tmp_tsr.chip(max_coeff_indices.back(), 0);
        }

        // x n-1

        ETensor1 xnm1_extract_vec = mps.bottom_tensor.raw.contract(cur_top_trace_vec, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(0, 0) });
        max_coeff_indices.push_back(ETensor1_arg_max(xnm1_extract_vec));

        return max_coeff_indices;
    }

}
