#include <tempo/core/mputil.hpp>
#include <tempo/mps/common.hpp>
#include <tempo/mpo/arith.hpp>
#include <chrono>
#include <iostream>

// Test it with 60 qubits and 64-bit floating point precision

using Scalar = double;
constexpr int N = 60;

int main() {
    const auto start_t = std::chrono::high_resolution_clock::now();

    // Quick test: contract <ghz| sub(3) add(3) |ghz> = 1

    // For the implemented (mod 2^n-based) adder, adding 2^n-1-3 is equivalent to subtracting 3
    // If you try any other add/sub combination that isn't of the form x and 2^n-1-x, the result value will be 0

    // Note that tensor-network/MPS orientation is the opposite to typical quantum circuit orientation!
    // *** the code below essentially computes: contract_mps_mpos_mps_rowbyrow(|ghz>, add(3), sub(3), <ghz|)

    const auto mps = tempo::create_ghz_state_mps<Scalar>(N);
    const auto mpo_1 = tempo::create_adder_mpo<Scalar>(N, 3);
    const auto mpo_2 = tempo::create_adder_mpo<Scalar>(N, (1ull << N) - 1 - 3);
    const auto res = tempo::contract_mps_mpos_mps_rowbyrow(mps, {mpo_1, mpo_2}, mps);

    const auto end_t = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration<double>(end_t - start_t).count();

    std::cout << "Result: " << res << std::endl;
    std::cout << "Time taken: " << duration << " s" << std::endl;
    return 0;
}
