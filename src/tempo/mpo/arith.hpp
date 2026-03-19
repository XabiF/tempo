
#pragma once
#include <tempo/core/mpo.hpp>

namespace tempo {

    template<typename S>
    MPO<S> create_adder_mpo(i32 n, u64 a) {
        #define _ADD_RES_BIT(c, x, a) (i32)(((x + a + c) & 0b1) != 0)
        #define _ADD_CARRY_BIT(c, x, a) (i32)(((x + a + c) & 0b10) != 0)
        #define _A_BIT_I(a, i) (i32)((a >> i) & 0b1)

        auto top_tensor = MPO<S>::TopTensor::zeroed(2, 2, 2);
        for(i32 x = 0; x < 2; x++) {
            top_tensor.at(x, _ADD_RES_BIT(0, x, _A_BIT_I(a, 0)), _ADD_CARRY_BIT(0, x, _A_BIT_I(a, 0))) = 1.0;
        }

        std::vector<typename MPO<S>::MiddleTensor> middle_tensors;
        middle_tensors.reserve(n-2);
        for(i32 i = 1; i < n-1; i++) {
            auto middle_tensor = MPO<S>::MiddleTensor::zeroed(2, 2, 2, 2);
            for(i32 c = 0; c < 2; c++) {
                for(i32 x = 0; x < 2; x++) {
                    middle_tensor.at(x, c, _ADD_RES_BIT(c, x, _A_BIT_I(a, i)), _ADD_CARRY_BIT(c, x, _A_BIT_I(a, i))) = 1.0;
                }
            }
            middle_tensors.push_back(std::move(middle_tensor));
        }

        auto bottom_tensor = MPO<S>::BottomTensor::zeroed(2, 2, 2);
        for(i32 c = 0; c < 2; c++) {
            for(i32 x = 0; x < 2; x++) {
                bottom_tensor.at(x, c, _ADD_RES_BIT(c, x, _A_BIT_I(a, n-1))) = 1.0;
            }
        }

        return MPO<S>(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
    }

}
