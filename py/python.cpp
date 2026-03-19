#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
namespace py = pybind11;

#include <tempo/core/mps.hpp>
#include <tempo/core/mpo.hpp>

#include <tempo/mps/common.hpp>
#include <tempo/mpo/arith.hpp>
#include <tempo/core/mputil.hpp>

using namespace tempo;

PYBIND11_MODULE(tempo, m) {
    m.doc() = "Main teMPO module";

    #define _DEFINE_FOR_PRECISION(name, desc, type) \
    auto sub_##name = m.def_submodule(#name, desc); \
    \
    py::class_<Tensor<type, 0>>(sub_##name, "Scalar") \
        .def("value", [](const Tensor<type, 0> &tensor) { return tensor.raw(0); }, "Returns the scalar value"); \
    \
    py::class_<Tensor<type, 1>>(sub_##name, "Tensor1") \
        .def_static("zeroed", &Tensor<type, 1>::zeroed_vec, py::arg("dims"), "Creates a zeroed 1D tensor with dimensions specified by a vector") \
        .def("__setitem__", [](Tensor<type, 1> &tensor, i32 i, type value) { tensor.at(i) = value; }, "Sets the element at index i"); \
    \
    py::class_<Tensor<type, 2>>(sub_##name, "Tensor2") \
        .def_static("zeroed", &Tensor<type, 2>::zeroed_vec, py::arg("dims"), "Creates a zeroed 2D tensor with dimensions specified by a vector") \
        .def("__setitem__", [](Tensor<type, 2> &tensor, std::tuple<i32, i32> indices, type value) { tensor.at(std::get<0>(indices), std::get<1>(indices)) = value; }, "Sets the element at indices (i, j)"); \
    \
    py::class_<Tensor<type, 3>>(sub_##name, "Tensor3") \
        .def_static("zeroed", &Tensor<type, 3>::zeroed_vec, py::arg("dims"), "Creates a zeroed 3D tensor with dimensions specified by a vector") \
        .def("__setitem__", [](Tensor<type, 3> &tensor, std::tuple<i32, i32, i32> indices, type value) { tensor.at(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices)) = value; }, "Sets the element at indices (i, j, k)"); \
    \
    py::class_<Tensor<type, 4>>(sub_##name, "Tensor4") \
        .def_static("zeroed", &Tensor<type, 4>::zeroed_vec, py::arg("dims"), "Creates a zeroed 4D tensor with dimensions specified by a vector") \
        .def("__setitem__", [](Tensor<type, 4> &tensor, std::tuple<i32, i32, i32, i32> indices, type value) { tensor.at(std::get<0>(indices), std::get<1>(indices), std::get<2>(indices), std::get<3>(indices)) = value; }, "Sets the element at indices (i, j, k, l)"); \
    \
    py::class_<MPS<type>>(sub_##name, "MPS") \
        .def(py::init<MPS<type>::TopTensor, std::vector<MPS<type>::MiddleTensor>, MPS<type>::BottomTensor>(), py::arg("top_tensor"), py::arg("middle_tensors"), py::arg("bottom_tensor"), "Initializes an MPS with the given top tensor, middle tensors, and bottom tensor") \
        .def("transpose", &MPS<type>::transpose, "Transposes the MPS"); \
    \
    py::class_<MPO<type>>(sub_##name, "MPO") \
        .def(py::init<MPO<type>::TopTensor, std::vector<MPO<type>::MiddleTensor>, MPO<type>::BottomTensor>(), py::arg("top_tensor"), py::arg("middle_tensors"), py::arg("bottom_tensor"), "Initializes an MPO with the given top tensor, middle tensors, and bottom tensor") \
        .def("transpose", &MPO<type>::transpose, "Transposes the MPO"); \
    \
    sub_##name.def("create_ghz_state_mps", &create_ghz_state_mps<type>, py::arg("n"), \
        "Creates an MPS GHZ object with parameter n"); \
    \
    sub_##name.def("create_adder_mpo", &create_adder_mpo<type>, py::arg("n"), py::arg("a"), \
        "Creates a MPO adder object with parameters n and a"); \
    \
    sub_##name.def("contract_mps_mpos_mps_rowbyrow", &contract_mps_mpos_mps_rowbyrow<type>, py::arg("mps"), py::arg("mpos"), py::arg("mps"), \
        "Contracts an MPS with a list of MPOs and another MPS row by row, returning the result as a scalar");

    _DEFINE_FOR_PRECISION(f32, "Single precision tensors and operations", float);
    _DEFINE_FOR_PRECISION(f64, "Double precision tensors and operations", double);
    _DEFINE_FOR_PRECISION(c64, "Double precision complex tensors and operations", std::complex<float>);
    _DEFINE_FOR_PRECISION(c128, "Double precision complex tensors and operations", std::complex<double>);

}
