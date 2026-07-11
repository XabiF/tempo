#include <tempo/core/mputil.hpp>
#include <tempo/core/ptrace.hpp>
#include <chrono>
#include <iostream>

/*

This examples solves the JSP (Job Selection Problem) optimization problem from the following paper:
https://arxiv.org/abs/2309.16522

The problem is migrated from QUBO to a QUDO formulation, so that part of the restrictions are encoded on the vartiable dimensions themselves.

The other remaining restrictions (so that a place is visit once at most) are imposed by filter MPOs.

For an explanation on how to solve discrete/combinatorial optimization problems using MPS/MPOs, check the following paper:
https://arxiv.org/abs/2603.28065

*/

using Scalar = double;

constexpr const char *PlaceNames[] = {
    "Bree", "Edoras", "Isengard", "Lórien", "Minas Tirith", "Pelargir", "Rivendel", "Tharbad", "Valle"
};

constexpr int N = std::size(PlaceNames);

const Eigen::Vector<Scalar, N> Priorities {
   15,
    150,
    35,
    75,
    170,
    50,
    40,
    5,
    15
};

const Eigen::Vector<Scalar, N> VisitTimes {
   3,
    5,
    4,
    4,
    7,
    3,
    5,
    2,
    4
};

constexpr Scalar TravelSpeed = 9.6;

constexpr Eigen::Matrix<Scalar, N, N> TravelDistances {
   { 0, 200, 150, 140, 285, 315, 100, 67, 225 },
    { 200, 0, 48, 100, 102, 117, 172, 133, 235 },
    { 150, 48, 0, 83, 150, 163, 135, 83, 225 },
    { 140, 100, 83, 0, 158, 192, 77, 100, 145 },
    { 285, 102, 150, 158, 0, 43, 200, 229, 245 },
    { 315, 117, 163, 192, 43, 0, 243, 252, 290 },
    { 100, 172, 135, 77, 200, 243, 0, 100, 125 },
    { 67, 133, 83, 100, 229, 252, 100, 0, 220 },
    { 225, 235, 225, 145, 245, 290, 125, 220, 0 }
};
const Eigen::Matrix<Scalar, N, N> TravelTimes = TravelDistances / TravelSpeed;

const Eigen::Vector<Scalar, N> RootTravelDistances {
   40,
    225,
    175,
    183,
    321,
    342,
    167,
    90,
    270
};
const Eigen::Vector<Scalar, N> RootTravelTimes = RootTravelDistances / TravelSpeed;

constexpr int Xi = 6;

Scalar compute_solution_cost(Scalar c_p, Scalar c_vt, Scalar c_tt, const std::vector<tempo::indx> &sol) {
    Scalar cost = c_tt*RootTravelTimes(sol.front()) + c_tt*RootTravelTimes(sol.back());
    for(int i = 0; i < Xi; i++) {
        cost += -c_p*Priorities(sol[i]) + c_vt*VisitTimes(sol[i]);
        if(i > 0) {
            cost += c_tt*TravelTimes(sol[i-1],sol[i]);
        }
    }
    return cost;
}

tempo::MPS<Scalar> create_problem_cost_mps(Scalar c_p, Scalar c_vt, Scalar c_tt, Scalar tau) {
    auto top_tensor = tempo::MPS<Scalar>::TopTensor::zeroed(N, N);
    for(int y0 = 0; y0 < N; y0++) {
        top_tensor.raw(y0, y0) = exp(-tau*(-c_p*Priorities(y0) + c_vt*VisitTimes(y0) + c_tt*RootTravelTimes(y0)));
    }

    std::vector<typename tempo::MPS<Scalar>::MiddleTensor> middle_tensors;
    middle_tensors.reserve(Xi-2);
    for(int i = 1; i < Xi-1; i++) {
        auto middle_tensor_i = tempo::MPS<Scalar>::MiddleTensor::zeroed(N, N, N);
        for(int yim1 = 0; yim1 < N; yim1++) {
            for(int yi = 0; yi < N; yi++) {
                middle_tensor_i.raw(yim1, yi, yi) = exp(-tau*(-c_p*Priorities(yi) + c_vt*VisitTimes(yi) + c_tt*TravelTimes(yim1, yi)));
            }
        }
        middle_tensors.push_back(middle_tensor_i);
    }

    auto bottom_tensor = tempo::MPS<Scalar>::BottomTensor::zeroed(N, N);
    for(int yxm2 = 0; yxm2 < N; yxm2++) {
        for(int yxm1 = 0; yxm1 < N; yxm1++) {
            bottom_tensor.raw(yxm2, yxm1) = exp(-tau*(-c_p*Priorities(yxm1) + c_vt*VisitTimes(yxm1) + c_tt*RootTravelTimes(yxm1) + c_tt*TravelTimes(yxm2, yxm1)));
        }    
    }

    return tempo::MPS<Scalar>(std::move(top_tensor), std::move(middle_tensors), std::move(bottom_tensor));
}

tempo::MPO<Scalar> create_constraint_filter_mpo(int k) {
    auto filter_top_tensor = tempo::MPO<Scalar>::TopTensor::zeroed(N, N, 2);
    for(int y0 = 0; y0 < N; y0++) {
        filter_top_tensor.raw(y0, y0, (int)(y0 == k)) = 1;
    }

    std::vector<typename tempo::MPO<Scalar>::MiddleTensor> filter_middle_tensors;
    filter_middle_tensors.reserve(Xi-2);
    auto filter_middle_tensor = tempo::MPO<Scalar>::MiddleTensor::zeroed(N, 2, N, 2);
    for(int yi = 0; yi < N; yi++) {
        if(yi == k) {
            filter_middle_tensor.raw(k, 0, k, 1) = 1;
        }
        else {
            filter_middle_tensor.raw(yi, 0, yi, 0) = 1;
            filter_middle_tensor.raw(yi, 1, yi, 1) = 1;
        }
    }
    for(int i = 0; i < Xi-2; i++) {
        filter_middle_tensors.push_back(filter_middle_tensor);
    }

    auto filter_bottom_tensor = tempo::MPO<Scalar>::BottomTensor::zeroed(N, 2, N);
    for(int yxm1 = 0; yxm1 < N; yxm1++) {
        filter_bottom_tensor.raw(yxm1, 0, yxm1) = 1;
        if(yxm1 != k) {
            filter_bottom_tensor.raw(yxm1, 1, yxm1) = 1;
        }
    }

    return tempo::MPO<Scalar>(std::move(filter_top_tensor), std::move(filter_middle_tensors), std::move(filter_bottom_tensor));
}

int main() {
    const auto t_max = std::max(RootTravelTimes.maxCoeff(), TravelTimes.maxCoeff());
    const auto c_p = 0.1;
    const auto c_tt = c_p * (300 / t_max);
    const auto c_vt = c_tt;
    constexpr auto tau = 0.9;

    std::cout << "Solving Hobbit routing problem..." << std::endl;

    const auto start_t = std::chrono::high_resolution_clock::now();
    //
    const auto optim_cost_mps = create_problem_cost_mps(c_p, c_vt, c_tt, tau);
    std::vector<tempo::MPO<Scalar>> optim_constraint_filter_mpos;
    optim_constraint_filter_mpos.reserve(N);
    for(int k = 0; k < N; k++) {
        optim_constraint_filter_mpos.push_back(create_constraint_filter_mpo(k));
    }
    const auto optim_space_mps = tempo::join_mps_mpos(optim_cost_mps, optim_constraint_filter_mpos);
    const auto optim_solution = tempo::mps_argmax_ptraced(optim_space_mps);
    //
    const auto end_t = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = end_t - start_t;
    std::cout << "Solved! elapsed time: " << std::scientific << elapsed.count() << std::defaultfloat << "s" << std::endl;

    std::cout << "Optimal route:" << std::endl;
    std::cout << "-- Hobbiton" << std::endl;
    for(const auto yi: optim_solution) {
        std::cout << "-- " << PlaceNames[yi] << std::endl;
    }
    std::cout << "-- Hobbiton" << std::endl;

    std::cout << "Route cost: " << compute_solution_cost(c_p, c_vt, c_tt, optim_solution) << std::endl;
    return 0;
}
