
#pragma once
#include <tempo/core/mps.hpp>

namespace tempo {

    template<typename S>
    MPS<S> create_ghz_state_mps(i32 n) {
        auto top_tensor = MPS<S>::TopTensor::zeroed(2, 2);
        top_tensor.at(0, 0) = 1.0;
        top_tensor.at(1, 1) = 1.0;

        std::vector<typename MPS<S>::MiddleTensor> middle_tensors;
        middle_tensors.reserve(n-2);
        auto middle_tensor = MPS<S>::MiddleTensor::zeroed(2, 2, 2);
        middle_tensor.at(0, 0, 0) = 1.0;
        middle_tensor.at(1, 1, 1) = 1.0;
        for(i32 i = 0; i < n-2; i++) {
            middle_tensors.push_back(middle_tensor);
        }

        auto bottom_tensor = MPS<S>::BottomTensor::zeroed(2, 2);
        bottom_tensor.at(0, 0) = 1.0;
        bottom_tensor.at(1, 1) = 1.0;

        return MPS<S>(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
    }

}
