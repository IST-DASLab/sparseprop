#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;


// ====================================== Transpose ===========================================

void tran(float* mat, float* matT, const int lda, const int ldb) {
	__m256  r0, r1, r2, r3, r4, r5, r6, r7;
	__m256  t0, t1, t2, t3, t4, t5, t6, t7;

	r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0*lda+0])), _mm_load_ps(&mat[4*lda+0]), 1);
	r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1*lda+0])), _mm_load_ps(&mat[5*lda+0]), 1);
	r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2*lda+0])), _mm_load_ps(&mat[6*lda+0]), 1);
	r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3*lda+0])), _mm_load_ps(&mat[7*lda+0]), 1);
	r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0*lda+4])), _mm_load_ps(&mat[4*lda+4]), 1);
	r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1*lda+4])), _mm_load_ps(&mat[5*lda+4]), 1);
	r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2*lda+4])), _mm_load_ps(&mat[6*lda+4]), 1);
	r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3*lda+4])), _mm_load_ps(&mat[7*lda+4]), 1);

	t0 = _mm256_unpacklo_ps(r0,r1);
	t1 = _mm256_unpackhi_ps(r0,r1);
	t2 = _mm256_unpacklo_ps(r2,r3);
	t3 = _mm256_unpackhi_ps(r2,r3);
	t4 = _mm256_unpacklo_ps(r4,r5);
	t5 = _mm256_unpackhi_ps(r4,r5);
	t6 = _mm256_unpacklo_ps(r6,r7);
	t7 = _mm256_unpackhi_ps(r6,r7);

	r0 = _mm256_shuffle_ps(t0,t2, 0x44);
	r1 = _mm256_shuffle_ps(t0,t2, 0xEE);
	r2 = _mm256_shuffle_ps(t1,t3, 0x44);
	r3 = _mm256_shuffle_ps(t1,t3, 0xEE);
	r4 = _mm256_shuffle_ps(t4,t6, 0x44);
	r5 = _mm256_shuffle_ps(t4,t6, 0xEE);
	r6 = _mm256_shuffle_ps(t5,t7, 0x44);
	r7 = _mm256_shuffle_ps(t5,t7, 0xEE);

	_mm256_store_ps(&matT[0*ldb], r0);
	_mm256_store_ps(&matT[1*ldb], r1);
	_mm256_store_ps(&matT[2*ldb], r2);
	_mm256_store_ps(&matT[3*ldb], r3);
	_mm256_store_ps(&matT[4*ldb], r4);
	_mm256_store_ps(&matT[5*ldb], r5);
	_mm256_store_ps(&matT[6*ldb], r6);
	_mm256_store_ps(&matT[7*ldb], r7);
}

inline void transpose(float* __restrict__ X, float* __restrict__ XT, const int N, const int M, const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<N; i+=block_size) {
        for(int j=0; j<M; j+=block_size) {
            int max_i2 = i+block_size < N ? i + block_size : N;
            int max_j2 = j+block_size < M ? j + block_size : M;
            for(int i2=i; i2<max_i2; i2+=8) {
                for(int j2=j; j2<max_j2; j2+=8) {
                    tran(&X[i2*M +j2], &XT[j2*N + i2], M, N);
                }
            }
        }
    }
}


// ====================================== Sparsify Conv2d ===========================================

void sparsify_conv2d(int IC, int OC, int K, float* __restrict__ W, int* __restrict__ W_idx_OC,
					 int16_t* __restrict__ W_idx_IC, uint8_t* __restrict__ W_idx_X,
					 uint8_t* __restrict__ W_idx_Y,float* __restrict__ W_val){	

	int new_si = 0;
	W_idx_OC[0] = 0;
	for (int oc = 0; oc < OC; oc++){
		W_idx_IC[oc*(IC + 1)] = 0;
		for (int ic = 0; ic < IC; ic++){
			int counter = 0;
			for(int x = 0; x < K; x++){
				for(int y = 0; y < K; y++){
					int idx = oc * IC * K * K + ic * K * K + x * K + y;
					if(W[idx] != 0){
						W_val[new_si] = W[idx];
						W_idx_X[new_si] = x;
						W_idx_Y[new_si] = y;
						new_si++;
						counter++;
					}
				}
			}
			W_idx_IC[(IC + 1) * oc + ic + 1] = W_idx_IC[(IC + 1) * oc + ic] + counter;
		}
		W_idx_OC[oc+1] = new_si;
	}
}

void further_sparsify_conv2d(int IC, int OC, int* __restrict__ W_idx_OC, int16_t* __restrict__ W_idx_IC,
							 uint8_t* __restrict__ W_idx_X, uint8_t* __restrict__ W_idx_Y,float* __restrict__ W_val,
							 int* __restrict__ W_idx_OC_new, int16_t* __restrict__ W_idx_IC_new,
							 uint8_t* __restrict__ W_idx_X_new, uint8_t* __restrict__ W_idx_Y_new,
							 float* __restrict__ W_val_new, int* __restrict__ mask) {    

	int new_si = 0;
	W_idx_OC_new[0] = 0;

	for (int oc = 0; oc < OC; oc++){
		W_idx_IC_new[(IC + 1) * oc] = 0;
		int oc_s = W_idx_OC[oc];

		for (int ic = 0; ic < IC; ic++){
			int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
			int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];
			int counter = 0;

			for (int si = ic_s; si < ic_e; si++) {
				if(mask[si] != 0){
					uint8_t i = W_idx_X[si];
					uint8_t j = W_idx_Y[si];
					float v = W_val[si];

					W_idx_X_new[new_si] = i;
					W_idx_Y_new[new_si] = j;
					W_val_new[new_si] = v;
					new_si++;
					counter++;
				}
			}
			
			W_idx_IC_new[(IC + 1) * oc + ic + 1] = W_idx_IC_new[(IC + 1) * oc + ic] + counter;
		}

		W_idx_OC_new[oc+1] = W_idx_OC_new[oc] + W_idx_IC_new[(IC + 1) * (oc + 1) - 1];
	}
}


// ====================================== Wrappers ===========================================

void transpose_wrapper(py::array_t<float> X, py::array_t<float> XT, int block_size) {

    int N = X.shape()[0];
    int M = X.shape()[1];

    auto buf_X = X.request();
    auto buf_XT = XT.request();

    float* ptr_X = (float*) buf_X.ptr;
    float* ptr_XT = (float*) buf_XT.ptr;

	transpose(ptr_X, ptr_XT, N, M, block_size);
}

void sparsify_conv2d_wrapper(int OC, int IC, int K, py::array_t<float> W, py::array_t<int> W_idx_OC,
						 py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
						 py::array_t<uint8_t> W_idx_Y,py::array_t<float> W_val) {

    auto buf_W = W.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();

    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_W= (float*) buf_W.ptr;

    sparsify_conv2d(IC, OC, K, ptr_W, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val);
}

void further_sparsify_conv2d_wrapper(int OC, int IC, py::array_t<int> W_idx_OC, py::array_t<int16_t> W_idx_IC,
									 py::array_t<uint8_t> W_idx_X, py::array_t<uint8_t> W_idx_Y,
									 py::array_t<float> W_val, py::array_t<int> W_idx_OC_new,
									 py::array_t<int16_t> W_idx_IC_new, py::array_t<uint8_t> W_idx_X_new,
									 py::array_t<uint8_t> W_idx_Y_new, py::array_t<float> W_val_new,
									 py::array_t<int> mask) {

    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_W_idx_OC_new = W_idx_OC_new.request();
    auto buf_W_idx_IC_new = W_idx_IC_new.request();
    auto buf_W_idx_X_new = W_idx_X_new.request();
    auto buf_W_idx_Y_new = W_idx_Y_new.request();
    auto buf_W_val_new = W_val_new.request();
    auto buf_mask = mask.request();

    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    int* ptr_W_idx_OC_new = (int*) buf_W_idx_OC_new.ptr;
    int16_t* ptr_W_idx_IC_new = (int16_t*) buf_W_idx_IC_new.ptr;
    uint8_t* ptr_W_idx_X_new = (uint8_t*) buf_W_idx_X_new.ptr;
    uint8_t* ptr_W_idx_Y_new = (uint8_t*) buf_W_idx_Y_new.ptr;
    float* ptr_W_val_new = (float*) buf_W_val_new.ptr;
    int* ptr_mask = (int*) buf_mask.ptr;

    further_sparsify_conv2d(IC, OC, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y, ptr_W_val, ptr_W_idx_OC_new, ptr_W_idx_IC_new, ptr_W_idx_X_new, ptr_W_idx_Y_new, ptr_W_val_new, ptr_mask);
}