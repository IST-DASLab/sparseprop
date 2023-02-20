#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lib/sparse_linear.cpp"
#include "lib/sparse_conv2d.cpp"
#include "lib/sparse_conv2d_over_on.cpp"
#include "lib/utils.cpp"

PYBIND11_MODULE(backend, m)
{
    // linear
    m.def("sparse_linear_vectorized_forward", &sparse_linear_vectorized_forward_wrapper);
    m.def("sparse_linear_vectorized_backward", &sparse_linear_vectorized_backward_wrapper);

    // conv2d
    m.def("sparse_conv2d_vectorized_forward_stride_1", &sparse_conv2d_vectorized_forward_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_stride_1", &sparse_conv2d_vectorized_backward_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_forward_stride_2", &sparse_conv2d_vectorized_forward_stride_2_wrapper);
    m.def("sparse_conv2d_vectorized_backward_stride_2", &sparse_conv2d_vectorized_backward_stride_2_wrapper);

    // conv2d over on
    m.def("sparse_conv2d_vectorized_forward_over_on_stride_1", &sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_1", &sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_2", &sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper);

    // utils
    m.def("transpose", &transpose_wrapper);
    m.def("sparsify_conv2d", &sparsify_conv2d_wrapper);
    m.def("further_sparsify_conv2d", &further_sparsify_conv2d_wrapper);
}