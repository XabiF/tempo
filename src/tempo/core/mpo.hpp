
#pragma once
#include <tempo/base.hpp>

namespace tempo {

    template<typename S>
    struct MPO {
        using TopTensor = Tensor<S, 3>;
        using MiddleTensor = Tensor<S, 4>;
        using BottomTensor = Tensor<S, 3>;

        TopTensor top_tensor;
        std::vector<MiddleTensor> middle_tensors;
        BottomTensor bottom_tensor;
        
        inline MPO(TopTensor top_tensor, std::vector<MiddleTensor> middle_tensors, BottomTensor bottom_tensor) : top_tensor(std::move(top_tensor)), middle_tensors(std::move(middle_tensors)), bottom_tensor(std::move(bottom_tensor)) {}

        inline size_t size() const {
            return this->middle_tensors.size() + 2;
        }

        MPO<S> transpose() const {
            auto t_top_tensor = TopTensor(this->top_tensor.raw.shuffle(std::array<indx, 3> { 1, 0, 2 }));
            std::vector<MiddleTensor> t_middle_tensors;
            t_middle_tensors.reserve(this->middle_tensors.size());
            for(const auto &middle_tensor : this->middle_tensors) {
                t_middle_tensors.push_back(MiddleTensor(middle_tensor.raw.shuffle(std::array<indx, 4> { 2, 1, 0, 3 })));
            }
            auto t_bottom_tensor = BottomTensor(this->bottom_tensor.raw.shuffle(std::array<indx, 3> { 2, 1, 0 }));
            return MPO<S>(std::move(t_top_tensor), std::move(t_middle_tensors), std::move(t_bottom_tensor));
        }

        indx bond_dimension() const {
            auto largest_dim = this->top_tensor.raw.dimension(2);
            for(const auto &mid_tensor: this->middle_tensors) {
                largest_dim = std::max(largest_dim, mid_tensor.raw.dimension(3));
            }
            return largest_dim;
        }

        S compress_factor() const {
            const auto max_edge_phys_dim = std::max(this->top_tensor.raw.dimension(1), this->bottom_tensor.raw.dimension(2));
            auto max_theo_bond_dim = max_edge_phys_dim;
            for(const auto &mid_tensor: this->middle_tensors) {
                max_theo_bond_dim *= mid_tensor.raw.dimension(2);
            }
            return log(this->bond_dimension()) / log(max_theo_bond_dim);
        }
    };

}
