
#pragma once
#include <tempo/core/mps.hpp>
#include <tempo/core/mpo.hpp>

namespace tempo {

    template<typename S>
    S contract_mps_mpos_mps_rowbyrow(const MPS<S> &mps_a, const std::vector<MPO<S>> &mpos, const MPS<S> &mps_b) {
        using ETensor0 = ETensor<S, 0>;
        using ETensor1 = ETensor<S, 1>;
        using ETensor2 = ETensor<S, 2>;
        using ETensor3 = ETensor<S, 3>;
        using ETensor4 = ETensor<S, 4>;

        const auto mps_b_t = mps_b.transpose();
        const auto height = mps_a.size();
        const auto mpo_count = mpos.size();

        // Bottom row
        ETensor3 b_row_first_unshaped = mps_a.bottom_tensor.raw.contract(mpos.at(0).bottom_tensor.raw, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
        ETensor2 b_row_cur = b_row_first_unshaped.reshape(std::array<indx, 2> { b_row_first_unshaped.dimension(0)*b_row_first_unshaped.dimension(1), b_row_first_unshaped.dimension(2) });
        for(i32 i = 1; i < mpo_count; i++) {
            ETensor3 b_row_cur_unshaped = b_row_cur.contract(mpos.at(i).bottom_tensor.raw, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
            b_row_cur = b_row_cur_unshaped.reshape(std::array<indx, 2> { b_row_cur_unshaped.dimension(0)*b_row_cur_unshaped.dimension(1), b_row_cur_unshaped.dimension(2) });
        }
        ETensor2 b_row_final_unshaped = b_row_cur.contract(mps_b_t.bottom_tensor.raw, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });

        ETensor1 m_row_cur = b_row_final_unshaped.reshape(std::array<indx, 1> { b_row_final_unshaped.dimension(0)*b_row_final_unshaped.dimension(1) });

        // Middle rows
        for(i32 j = height-2; j >= 1; j--) {
            const auto dim = m_row_cur.dimension(0);
            const auto dim_split = mps_a.middle_tensors.at(j-1).raw.dimension(2);
            ETensor2 m_row_j_cur_unshaped = m_row_cur.reshape(std::array<indx, 2> { dim_split, dim / dim_split });
            ETensor3 m_row_j_cur = mps_a.middle_tensors.at(j-1).raw.contract(m_row_j_cur_unshaped, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(2, 0) });

            for(i32 k = 0; k < mpo_count; k++) {
                const auto cur_dim = m_row_j_cur.dimension(2);
                const auto dim_split = mpos.at(k).middle_tensors.at(j-1).raw.dimension(3);
                ETensor4 m_row_j_adapted = m_row_j_cur.reshape(std::array<indx, 4> { m_row_j_cur.dimension(0), m_row_j_cur.dimension(1), dim_split, cur_dim / dim_split });
                ETensor4 m_row_j_cur_k_unshaped = m_row_j_adapted.contract(mpos.at(k).middle_tensors.at(j-1).raw, Eigen::array<Eigen::IndexPair<i32>, 2>{ Eigen::IndexPair<i32>(1, 0), Eigen::IndexPair<i32>(2, 3) });
                ETensor4 m_row_j_cur_k_unshuffled = m_row_j_cur_k_unshaped.shuffle(std::array<indx, 4> { 0, 2, 3, 1 });
                m_row_j_cur = m_row_j_cur_k_unshuffled.reshape(std::array<indx, 3> { m_row_j_cur_k_unshuffled.dimension(0)*m_row_j_cur_k_unshuffled.dimension(1), m_row_j_cur_k_unshuffled.dimension(2), m_row_j_cur_k_unshuffled.dimension(3) });
            }

            ETensor2 m_row_end_unshaped = m_row_j_cur.contract(mps_b_t.middle_tensors.at(j-1).raw, Eigen::array<Eigen::IndexPair<i32>, 2>{ Eigen::IndexPair<i32>(1, 0), Eigen::IndexPair<i32>(2, 2) });
            m_row_cur = m_row_end_unshaped.reshape(std::array<indx, 1> { m_row_end_unshaped.dimension(0)*m_row_end_unshaped.dimension(1) });
        }

        // Top row
        const auto t_dim = m_row_cur.dimension(0);
        const auto t_dim_split = mps_a.top_tensor.raw.dimension(1);
        ETensor2 m_row_cur_adapted = m_row_cur.reshape(std::array<indx, 2> { t_dim_split, t_dim / t_dim_split });
        ETensor2 t_row_cur = mps_a.top_tensor.raw.contract(m_row_cur_adapted, Eigen::array<Eigen::IndexPair<i32>, 1>{ Eigen::IndexPair<i32>(1, 0) });
        for(i32 k = 0; k < mpo_count; k++) {
            const auto cur_dim = t_row_cur.dimension(1);
            const auto dim_split = mpos.at(k).top_tensor.raw.dimension(2);
            ETensor3 t_row_cur_k = t_row_cur.reshape(std::array<indx, 3> { t_row_cur.dimension(0), dim_split, cur_dim / dim_split });
            t_row_cur = mpos.at(k).top_tensor.raw.contract(t_row_cur_k, Eigen::array<Eigen::IndexPair<i32>, 2>{ Eigen::IndexPair<i32>(0, 0), Eigen::IndexPair<i32>(2, 1) });
        }

        ETensor0 final_value = t_row_cur.contract(mps_b_t.top_tensor.raw, Eigen::array<Eigen::IndexPair<i32>, 2>{ Eigen::IndexPair<i32>(0, 0), Eigen::IndexPair<i32>(1, 1) });
        return final_value(0);
    }

}
