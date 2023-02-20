#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

// ====================================== Stride 1 ===========================================
void sparse_conv2d_vectorized_forward_stride_1(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                               float* __restrict__ X, int* __restrict__ W_idx_OC, 
                                               int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                               uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                               float* __restrict__ O) {    
    
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;

    #pragma omp parallel
	{
    
        #pragma omp for 
        for (int oc = 0; oc < OC; oc++){
            for (int ic = 0; ic < IC; ic++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];
                    
                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(W_val[si]);

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);

                    for (int po = p_start, px = p_start - padding + i; po < p_end; po++, px++) {
                        int qo = q_start, qx = q_start - padding + j;
                        for (; qo < q_end-3; qo+=4, qx+=4) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o0 = _mm256_loadu_ps(O + Oi);
                                const __m256 o1 = _mm256_loadu_ps(O + Oi + B);
                                const __m256 o2 = _mm256_loadu_ps(O + Oi + 2 * B);
                                const __m256 o3 = _mm256_loadu_ps(O + Oi + 3 * B);
                                const __m256 x0 = _mm256_loadu_ps(X + Xi);
                                const __m256 x1 = _mm256_loadu_ps(X + Xi + B);
                                const __m256 x2 = _mm256_loadu_ps(X + Xi + 2 * B);
                                const __m256 x3 = _mm256_loadu_ps(X + Xi + 3 * B);

                                const __m256 r0 = _mm256_fmadd_ps(x0,vv,o0);
                                const __m256 r1 = _mm256_fmadd_ps(x1,vv,o1);
                                const __m256 r2 = _mm256_fmadd_ps(x2,vv,o2);
                                const __m256 r3 = _mm256_fmadd_ps(x3,vv,o3);

                                _mm256_storeu_ps(O + Oi, r0);
                                _mm256_storeu_ps(O + Oi + B, r1);
                                _mm256_storeu_ps(O + Oi + 2 * B, r2);
                                _mm256_storeu_ps(O + Oi + 3 * B, r3);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                        for (; qo < q_end; qo++, qx++) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o = _mm256_loadu_ps(O + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);

                                const __m256 r = _mm256_fmadd_ps(x,vv,o);

                                _mm256_storeu_ps(O + Oi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                    }
                }
            }
		}
	}
}

void sparse_conv2d_vectorized_backward_stride_1(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                                float* __restrict__ X, int* __restrict__ W_idx_OC,
                                                int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                                uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                                float* __restrict__ dLdO, float* __restrict__ dLdX,
                                                float* __restrict__ dLdW_val) {    
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    
    #pragma omp parallel
	{

        #pragma omp for reduction(+:dLdW_val[:W_nnz]) 
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

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);

                    for (int po = p_start, px = p_start - padding + i; po < p_end; po++, px++) {
                        int qo = q_start, qx = q_start - padding + j;
                        for (; qo < q_end; qo++, qx++) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o = _mm256_loadu_ps(dLdO + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);
                                const __m256 dx = _mm256_loadu_ps(dLdX + Xi);

                                const __m256 r = _mm256_fmadd_ps(o,vv,dx);
                                dwv = _mm256_fmadd_ps(o, x, dwv);

                                _mm256_storeu_ps(dLdX + Xi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                float o = dLdO[Oi];
                                float x = X[Xi];

                                dLdX[Xi] += o * v;

                                dw += o * x;
                            }
                        }
                    }

                    const __m128 hiQuad0 = _mm256_extractf128_ps(dwv, 1);
                    const __m128 loQuad0 = _mm256_castps256_ps128(dwv);
                    const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                    const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                    const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                    const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                    const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

                    dLdW_val[si] += dw + _mm_cvtss_f32(sum0);
                }
            }
		}
	}
}

// ====================================== Stride 2 ===========================================

void sparse_conv2d_vectorized_forward_stride_2(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                               float* __restrict__ X, int* __restrict__ W_idx_OC,
                                               int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                               uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                               float* __restrict__ O) {    
    
    const int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    const int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);

    #pragma omp parallel
	{
    
        #pragma omp for 
		for (int oc = 0; oc < OC; oc++){
            for (int ic = 0; ic < IC; ic++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];
                    
                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(W_val[si]);

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);

                    for (int po = p_start, px = 2 * p_start - padding + i; po < p_end; po++, px+=2) {
                        int qo = q_start, qx = 2 * q_start - padding + j;
                        for (; qo < q_end-3; qo+=4, qx+=8) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o0 = _mm256_loadu_ps(O + Oi);
                                const __m256 o1 = _mm256_loadu_ps(O + Oi + B);
                                const __m256 o2 = _mm256_loadu_ps(O + Oi + 2 * B);
                                const __m256 o3 = _mm256_loadu_ps(O + Oi + 3 * B);
                                const __m256 x0 = _mm256_loadu_ps(X + Xi);
                                const __m256 x1 = _mm256_loadu_ps(X + Xi + 2 * B);
                                const __m256 x2 = _mm256_loadu_ps(X + Xi + 4 * B);
                                const __m256 x3 = _mm256_loadu_ps(X + Xi + 6 * B);

                                const __m256 r0 = _mm256_fmadd_ps(x0,vv,o0);
                                const __m256 r1 = _mm256_fmadd_ps(x1,vv,o1);
                                const __m256 r2 = _mm256_fmadd_ps(x2,vv,o2);
                                const __m256 r3 = _mm256_fmadd_ps(x3,vv,o3);

                                _mm256_storeu_ps(O + Oi, r0);
                                _mm256_storeu_ps(O + Oi + B, r1);
                                _mm256_storeu_ps(O + Oi + 2 * B, r2);
                                _mm256_storeu_ps(O + Oi + 3 * B, r3);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }

                        for (; qo < q_end; qo++, qx+=2) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o = _mm256_loadu_ps(O + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);

                                const __m256 r = _mm256_fmadd_ps(x,vv,o);

                                _mm256_storeu_ps(O + Oi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                    }
                }
            }
		}
	}
}

void sparse_conv2d_vectorized_backward_stride_2(int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
                                                float* __restrict__ X, int* __restrict__ W_idx_OC,
                                                int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
                                                uint8_t* __restrict__ W_idx_Y, float* __restrict__ W_val,
                                                float* __restrict__ dLdO, float* __restrict__ dLdX,
                                                float* __restrict__ dLdW_val) {    

    const int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    const int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);
    
    #pragma omp parallel
	{

        #pragma omp for reduction(+:dLdW_val[:W_nnz])
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

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);
            

                    for (int po = p_start, px = 2 * p_start - padding + i; po < p_end; po++, px+=2) {
                        int qo = q_start, qx = 2 * q_start - padding + j;
                        for (; qo < q_end; qo++, qx+=2) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o = _mm256_loadu_ps(dLdO + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);
                                const __m256 dx = _mm256_loadu_ps(dLdX + Xi);

                                const __m256 r = _mm256_fmadd_ps(o,vv,dx);
                                dwv = _mm256_fmadd_ps(o, x, dwv);

                                _mm256_storeu_ps(dLdX + Xi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                float o = dLdO[Oi];
                                float x = X[Xi];

                                dLdX[Xi] += o * v;

                                dw += o * x;
                            }
                        }
                    }

                    const __m128 hiQuad0 = _mm256_extractf128_ps(dwv, 1);
                    const __m128 loQuad0 = _mm256_castps256_ps128(dwv);
                    const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                    const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                    const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                    const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                    const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

                    dLdW_val[si] += dw + _mm_cvtss_f32(sum0);
                }
            }
		}
	}
}

// ====================================== Wrappers ===========================================

void sparse_conv2d_vectorized_forward_stride_1_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                       py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                       py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                       py::array_t<float> O, int kernel_size, int padding) {
    int B = X.shape()[3];
    int IC = X.shape()[0];
    int M = X.shape()[1];
    int N = X.shape()[2];
    int OC = O.shape()[0];
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

    sparse_conv2d_vectorized_forward_stride_1(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC,
                                              ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_O);
}

void sparse_conv2d_vectorized_backward_stride_1_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                        py::array_t<float> dLdO, py::array_t<float> dLdX,
                                                        py::array_t<float> dLdW_val, int kernel_size, int padding) {
    int B = X.shape()[3];
    int IC = X.shape()[0];
    int M = X.shape()[1];
    int N = X.shape()[2];
    int OC = dLdO.shape()[0];
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

    sparse_conv2d_vectorized_backward_stride_1(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC,
                                               ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_dLdO, ptr_dLdX, ptr_dLdW_val);
}

void sparse_conv2d_vectorized_forward_stride_2_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                       py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                       py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                       py::array_t<float> O, int kernel_size, int padding) {
    int B = X.shape()[3];
    int IC = X.shape()[0];
    int M = X.shape()[1];
    int N = X.shape()[2];
    int OC = O.shape()[0];
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

    sparse_conv2d_vectorized_forward_stride_2(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC,
                                              ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_O);
}

void sparse_conv2d_vectorized_backward_stride_2_wrapper(py::array_t<float> X, py::array_t<int> W_idx_OC,
                                                        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
                                                        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
                                                        py::array_t<float> dLdO, py::array_t<float> dLdX,
                                                        py::array_t<float> dLdW_val, int kernel_size, int padding) {
    int B = X.shape()[3];
    int IC = X.shape()[0];
    int M = X.shape()[1];
    int N = X.shape()[2];
    int OC = dLdO.shape()[0];
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

    sparse_conv2d_vectorized_backward_stride_2(B, IC, OC, M, N, K, W_nnz, padding, ptr_X, ptr_W_idx_OC, ptr_W_idx_IC,
                                               ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_dLdO, ptr_dLdX, ptr_dLdW_val);
}