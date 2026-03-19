#include <tempo/core/mputil.hpp>
#include <tempo/mps/common.hpp>
#include <tempo/mpo/arith.hpp>
#include <chrono>
#include <iostream>

#define TIME_FUNCTION_RUNS 30

#define TIME_FUNCTION(fn, out_time, out_var) { \
    const auto start = std::chrono::high_resolution_clock::now(); \
    for(int _tf_i = 0; _tf_i < TIME_FUNCTION_RUNS; _tf_i++) { \
        fn; \
    } \
    const auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> elapsed = (end - start) / TIME_FUNCTION_RUNS; \
    out_time = elapsed.count(); \
    out_var = fn; \
}

using FType = double;
using MPS = tempo::MPS<FType>;
using MPO = tempo::MPO<FType>;

MPS create_zero_state_mps(int n) {
    auto top_tensor = MPS::TopTensor::zeroed(2, 1);
    top_tensor.at(0, 0) = 1.0;

    std::vector<MPS::MiddleTensor> middle_tensors;
    middle_tensors.reserve(n-2);
    auto middle_tensor = MPS::MiddleTensor::zeroed(1, 2, 1);
    middle_tensor.at(0, 0, 0) = 1.0;
    for(int i = 0; i < n-2; i++) {
        middle_tensors.push_back(middle_tensor);
    }

    auto bottom_tensor = MPS::BottomTensor::zeroed(1, 2);
    bottom_tensor.at(0, 0) = 1.0;

    return MPS(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
}

MPO create_theta_ry_mpo(int n, const std::vector<double> &layer_theta, const size_t base_index) {
    const auto c0 = std::cos(layer_theta[base_index+0]/2);
    const auto s0 = std::sin(layer_theta[base_index+0]/2);
    auto top_tensor = MPO::TopTensor::zeroed(2, 2, 1);
    top_tensor.at(0, 0, 0) = c0;
    top_tensor.at(0, 1, 0) = s0;
    top_tensor.at(1, 0, 0) = -s0;
    top_tensor.at(1, 1, 0) = c0;

    std::vector<MPO::MiddleTensor> middle_tensors;
    middle_tensors.reserve(n-2);
    for(int j = 1; j < n-1; j++) {
        const auto cj = std::cos(layer_theta[base_index+j]/2);
        const auto sj = std::sin(layer_theta[base_index+j]/2);
        auto middle_tensor = MPO::MiddleTensor::zeroed(2, 1, 2, 1);
        middle_tensor.at(0, 0, 0, 0) = cj;
        middle_tensor.at(0, 0, 1, 0) = sj;
        middle_tensor.at(1, 0, 0, 0) = -sj;
        middle_tensor.at(1, 0, 1, 0) = cj;
        middle_tensors.push_back(middle_tensor);
    }

    const auto cf = std::cos(layer_theta[base_index+n-1]/2);
    const auto sf = std::sin(layer_theta[base_index+n-1]/2);
    auto bottom_tensor = MPO::BottomTensor::zeroed(2, 1, 2);
    bottom_tensor.at(0, 0, 0) = cf;
    bottom_tensor.at(0, 0, 1) = sf;
    bottom_tensor.at(1, 0, 0) = -sf;
    bottom_tensor.at(1, 0, 1) = cf;

    return MPO(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
}

MPO create_linear_entangl_mpo(int n) {
    auto top_tensor = MPO::TopTensor::zeroed(2, 2, 2);
    top_tensor.at(0, 0, 0) = 1.0;
    top_tensor.at(1, 1, 1) = 1.0;

    std::vector<MPO::MiddleTensor> middle_tensors;
    middle_tensors.reserve(n-2);
    auto middle_tensor = MPO::MiddleTensor::zeroed(2, 2, 2, 2);
    middle_tensor.at(0, 0, 0, 0) = 1.0;
    middle_tensor.at(1, 0, 1, 1) = 1.0;
    middle_tensor.at(0, 1, 1, 1) = 1.0;
    middle_tensor.at(1, 1, 0, 0) = 1.0;
    for(int i = 0; i < n-2; i++) {
        middle_tensors.push_back(middle_tensor);
    }

    auto bottom_tensor = MPO::BottomTensor::zeroed(2, 2, 2);
    bottom_tensor.at(0, 0, 0) = 1.0;
    bottom_tensor.at(1, 0, 1) = 1.0;
    bottom_tensor.at(0, 1, 1) = 1.0;
    bottom_tensor.at(1, 1, 0) = 1.0;

    return MPO(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
}

MPO create_Z_expval_mpo(int n) {
    auto top_tensor = MPO::TopTensor::zeroed(2, 2, 1);
    top_tensor.at(0, 0, 0) = 1.0;
    top_tensor.at(1, 1, 0) = -1.0;

    std::vector<MPO::MiddleTensor> middle_tensors;
    middle_tensors.reserve(n-2);
    auto middle_tensor = MPO::MiddleTensor::zeroed(2, 1, 2, 1);
    middle_tensor.at(0, 0, 0, 0) = 1.0;
    middle_tensor.at(1, 0, 1, 0) = -1.0;
    for(int i = 0; i < n-2; i++) {
        middle_tensors.push_back(middle_tensor);
    }

    auto bottom_tensor = MPO::BottomTensor::zeroed(2, 1, 2);
    bottom_tensor.at(0, 0, 0) = 1.0;
    bottom_tensor.at(1, 0, 1) = -1.0;

    return MPO(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
}

constexpr int NumThetaLayers = 3;
constexpr int NRangeStart = 2;
constexpr int NRangeEnd = 60;

double compute_expval(int n, const std::vector<double> &theta) {
    auto mps = create_zero_state_mps(n);

    std::vector<MPO> mpos = { create_theta_ry_mpo(n, theta, 0) };
    for(int j = 1; j < NumThetaLayers; j++) {
        mpos.push_back(create_linear_entangl_mpo(n));
        mpos.push_back(create_theta_ry_mpo(n, theta, j*n));
    }
    mpos.push_back(create_Z_expval_mpo(n));
    for(int j = NumThetaLayers-1; j >= 1; j--) {
        mpos.push_back(mpos[j*2].transpose());
        mpos.push_back(mpos[j*2-1].transpose());
    }
    mpos.push_back(mpos[0].transpose());

    return tempo::contract_mps_mpos_mps_rowbyrow(mps, mpos, mps);
}

std::vector<double> gen_theta(int n) {
    std::vector<double> theta(n*NumThetaLayers);
    for(int i = 0; i < n*NumThetaLayers; i++) {
        theta[i] = 2*M_PI*(static_cast<double>(i) / static_cast<double>(n*NumThetaLayers));
    }
    return std::move(theta);
}

int main() {
    for(int n = NRangeStart; n < NRangeEnd; n++) {
        std::vector<double> theta;
        double theta_time;
        TIME_FUNCTION(gen_theta(n), theta_time, theta);
        std::cout << "[theta] n=" << n << " theta_time=" << theta_time << std::endl;

        double expval;
        double expval_time;
        TIME_FUNCTION(compute_expval(n, theta), expval_time, expval);
        std::cout << "[expval] n=" << n << " expval=" << expval << " expval_time=" << expval_time << std::endl;
    }

    return 0;
}
