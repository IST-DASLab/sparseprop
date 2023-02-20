#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

// ====================================== Stride 1 ===========================================

void sparse_conv2d_vectorized_forward_over_on_stride_1(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                                       float* __restrict__ X, int* __restrict__ W_idx_OC,
                                                       int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                                       uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                                       float* __restrict__ O) {    
	const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < OC; oc++){
            for (int ic = 0; ic < IC; ic++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];
                    
                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(v);

                    int pdmi = padding - i;
                    int p_start = pdmi;
                    if (p_start < 0) p_start = 0;
                    int p_end = pdmi + M;
                    if (p_end > OM) p_end = OM;

                    int pdmj = padding - j;
                    int q_start = pdmj;
                    if (q_start < 0) q_start = 0;
                    int q_end = pdmj + N ;
                    if (q_end > ON) q_end = ON;
                    int q_end_div8 = q_end - ((q_end - q_start) % 8);

                    for (int p = p_start; p < p_end; p++) {
                        for (int q = q_start; q < q_end_div8; q+=8) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + p + i) * N + (-padding + q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            __m256 xv = _mm256_loadu_ps(X + Xi);

                            __m256 ov = _mm256_loadu_ps(O + Oi);
                            ov = _mm256_fmadd_ps(xv, vv, ov);
                            _mm256_storeu_ps(O + Oi, ov);
                        }

                        for (int q = q_end_div8; q < q_end; q++) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + p + i) * N + (-padding + q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            float x = X[Xi];

                            O[Oi] += x * v;
                        }
                    }
                }
            }
        }
    }
}

void sparse_conv2d_vectorized_backward_over_on_stride_1(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                                        float* __restrict__ X, int* __restrict__ W_idx_OC,
                                                        int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                                        uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                                        float* __restrict__ dLdO, float* __restrict__ dLdX,
                                                        float* __restrict__ dLdW_val) {    
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < IC; ic++){
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];
                    
                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    int pdmi = padding - i;
                    int p_start = pdmi;
                    if (p_start < 0) p_start = 0;
                    int p_end = pdmi + M;
                    if (p_end > OM) p_end = OM;

                    int pdmj = padding - j;
                    int q_start = pdmj;
                    if (q_start < 0) q_start = 0;
                    int q_end = pdmj + N;
                    if (q_end > ON) q_end = ON;
                    int q_end_div8 = q_end - ((q_end - q_start) % 8);
                

                    for (int p = p_start; p < p_end; p++) {
                        for (int q = q_start; q < q_end_div8; q+=8) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + p + i) * N + (-padding + q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            __m256 ov = _mm256_loadu_ps(dLdO + Oi);
                            __m256 xv = _mm256_loadu_ps(X + Xi);
                            
                            dwv = _mm256_fmadd_ps(ov, xv, dwv);
                            
                            __m256 dxv = _mm256_loadu_ps(dLdX + Xi);
                            dxv = _mm256_fmadd_ps(ov, vv, dxv);
                            _mm256_storeu_ps(dLdX + Xi, dxv);
                        }

                        // handle the end of the row
                        for (int q = q_end_div8; q < q_end; q++) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + p + i) * N + (-padding + q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            float o = dLdO[Oi];
                            float x = X[Xi];

                            dLdX[Xi] += o * v;

                            dw += o * x;
                        }
                    }

                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dw += _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(dwv), _mm256_extractf128_ps(dwv, 1)));

                    #pragma omp atomic 
                    dLdW_val[si] += dw;
                }
            }
        }
    }
}

// ====================================== Stride 2 ===========================================

void sparse_conv2d_vectorized_backward_over_on_stride_2(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                                        float* __restrict__ X, int* __restrict__ W_idx_OC,
                                                        int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                                        uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                                        float* __restrict__ dLdO, float* __restrict__ dLdX,
                                                        float* __restrict__ dLdW_val) {    
    int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);
    
    __m256i permutevar8x32_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    __m256 zv = _mm256_setzero_ps();
		__m256i permuteamask = _mm256_set_epi32(7,7,7,7,6,4,2,0);


    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < IC; ic++){
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];
                    
                    float v = W_val[si];
                    __m256 v0v = _mm256_set_ps(0., v, 0., v, 0., v, 0., v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    int pdmi = padding - i;
                    int p_start = (int) ceil((float) pdmi / 2);
                    if (p_start < 0) p_start = 0;
                    int p_end = (int) floor((float) (pdmi + M - 1) / 2) + 1;
                    if (p_end > OM) p_end = OM;

                    int pdmj = padding - j;
                    int q_start = (int) ceil((float) pdmj / 2);
                    if (q_start < 0) q_start = 0;
                    int q_end = (int) floor((float) (pdmj + N - 1) / 2) + 1;
                    if (q_end > ON) q_end = ON;
                    int q_end_div8 = q_end - ((q_end - q_start) % 8);
                

                    for (int p = p_start; p < p_end; p++) {
                        for (int q = q_start; q < q_end_div8; q+=8) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + 2 * p + i) * N + (-padding + 2 * q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            __m256 ov = _mm256_loadu_ps(dLdO + Oi);
                            
                            __m256 a = _mm256_loadu_ps(X + Xi);
                            __m256 b = _mm256_loadu_ps(X + Xi + 8);
														__m256 ap = _mm256_permutevar8x32_ps(a, permuteamask);
														__m256 bp = _mm256_permutevar8x32_ps(b, permuteamask);
														__m256 xv  = _mm256_insertf128_ps(ap, _mm256_castps256_ps128(bp), 1);
                            
                            dwv = _mm256_fmadd_ps(ov, xv, dwv);
                            
                            ov = _mm256_permutevar8x32_ps(ov, permutevar8x32_mask);

                            __m256 ov0 = _mm256_unpacklo_ps(ov, zv);
                            __m256 ov1 = _mm256_unpackhi_ps(ov, zv);

                            __m256 dxv0 = _mm256_loadu_ps(dLdX + Xi);
                            __m256 dxv1 = _mm256_loadu_ps(dLdX + Xi + 8);

                            dxv0 = _mm256_fmadd_ps(ov0, v0v, dxv0);
                            dxv1 = _mm256_fmadd_ps(ov1, v0v, dxv1);

                            _mm256_storeu_ps(dLdX + Xi, dxv0);
                            _mm256_storeu_ps(dLdX + Xi + 8, dxv1);
                        }

                        // handle the end of the row
                        for (int q = q_end_div8; q < q_end; q++) {
                            int Xi = b * IC * M * N + ic * M * N + (-padding + 2 * p + i) * N + (-padding + 2 * q + j);
                            int Oi = b * OC * OM * ON + oc * OM * ON + p * ON + q;

                            float o = dLdO[Oi];
                            float x = X[Xi];

                            dLdX[Xi] += o * v;

                            dw += o * x;
                        }
                    }

                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dw += _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(dwv), _mm256_extractf128_ps(dwv, 1)));

                    #pragma omp atomic 
                    dLdW_val[si] += dw;
                }
            }
        }
    }
}

// ====================================== Wrappers ===========================================

void sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC, 
                                                               py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                               py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                               py::array_t<float> O, int kernel_size, int padding) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = O.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    float* ptr_X = (float*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse_conv2d_vectorized_forward_over_on_stride_1(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_O);
}

void sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                                py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X, 
                                                                py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                                py::array_t<float> dLdO, py::array_t<float> dLdX,
                                                                py::array_t<float> dLdW_val, int kernel_size, int padding) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    float* ptr_X = (float*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_dLdO = (float*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse_conv2d_vectorized_backward_over_on_stride_1(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_dLdO, ptr_dLdX, ptr_dLdW_val);
}

void sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                                py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                                py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                                py::array_t<float> dLdO, py::array_t<float> dLdX,
                                                                py::array_t<float> dLdW_val, int kernel_size, int padding) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    float* ptr_X = (float*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_dLdO = (float*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse_conv2d_vectorized_backward_over_on_stride_2(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_dLdO, ptr_dLdX, ptr_dLdW_val);
}