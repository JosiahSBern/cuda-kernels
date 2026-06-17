// matmul.cu
// CUDA Matrix Multiplication Benchmark: C = A @ B
//      Where A is (M x K) matrix, B is (K x N) matrix, C is (M x N) matrix

// nvcc matmul.cu -o matmul
// ./matmul

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

// CPU Matrix Multiplication
// Matrices are stored in in 1D arrays. where:
//  elements exist at row i, col j (index = i * cols + j)
// So  A[i][l] = A[i*K+l] and B[l][j] = B[l*N + j]
void matmul_cpu(float *A, float *B,float *C, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float sum = 0.0f;
            for(int l = 0; l < k; l++){ //dot product over K
                sum += A[i * k + l] * B[l * n  + j];
            }
            C[i * n + j] = sum;

            }
        }
    }

// GPU Kernel Matrix Multiplication
// Instead of looping like above we launch one thread per cell and run all simultaneously

__global__ void matmul_gpu(float *A )