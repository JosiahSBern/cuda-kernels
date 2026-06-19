// tiled_matmul.cu
// CUDA Tiled Matrix Multiplication with shared memory
//      C = A @ B,   A(MxK) @ B(KxN) = C(MxN)

// NAIVE: every thread fetches its own data independently from global memory
// vs
// TILED: all threads in the block load once into shared memory, then reuse it

// Tiled leads to 16x fewer global memory reads and scales with TILE_SIZE



#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define M          512
#define K          512
#define N          512
#define TILE_SIZE  16 


// TILED GPU KERNEL
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    // Shared Memory Intilization 
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    // Slide across K in steps of TILE_SIZE

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // LOAD
        //  sharedA[ty][tx] = A[row][tile * TILE_SIZE + tx]
        //  sharedB[ty][tx] = B[tile * TILE_SIZE + ty][col]
        //
        // Boundary Guards: K may not be a multiple of TILE_SIZE so if out of bounds set to 0.0

        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if(col < N && tile * TILE_SIZE + tx < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;
        
        // SYNC 
        // All threads stop here until blocks are done writing shared memory
        __syncthreads();

        // COMPUTE - same as in standard matmul 
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sharedA[ty][k] * sharedB[k][tx];
    
        
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (float)rand() / RAND_MAX;
}

double get_time() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // host memory
    float *h_A       = (float*)malloc(size_A);
    float *h_B       = (float*)malloc(size_B);
    float *h_C_cpu   = (float*)malloc(size_C);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // benchmark: 20 runs each
    const int RUNS = 20;
    double t_cpu = 0, t_naive = 0, t_tiled = 0;

    for (int r = 0; r < RUNS; r++) {
        double t;

        t = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        t_cpu += get_time() - t;

        t = get_time();
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        t_naive += get_time() - t;

        t = get_time();
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        t_tiled += get_time() - t;
    }

    // copy results back for correctness check
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);

    // max absolute error vs CPU
    float err_naive = 0, err_tiled = 0;
    for (int i = 0; i < M * N; i++) {
        err_naive = fmaxf(err_naive, fabsf(h_C_cpu[i] - h_C_naive[i]));
        err_tiled = fmaxf(err_tiled, fabsf(h_C_cpu[i] - h_C_tiled[i]));
    }

    printf("Matrix: %dx%d @ %dx%d  |  tile: %d\n\n", M, K, K, N, TILE_SIZE);
    printf("CPU:        %7.1f ms\n",                    (t_cpu   / RUNS) * 1e3);
    printf("Naive GPU:  %7.1f ms   max err: %.2e\n",   (t_naive / RUNS) * 1e3, err_naive);
    printf("Tiled GPU:  %7.1f ms   max err: %.2e\n",   (t_tiled / RUNS) * 1e3, err_tiled);
    printf("\nSpeedup naive -> tiled: %.2fx\n", t_naive / t_tiled);
    printf("Speedup CPU   -> tiled: %.2fx\n",   t_cpu   / t_tiled);

    // error check
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

        
''

