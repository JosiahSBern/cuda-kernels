// matmul.cu
// CUDA Matrix Multiplication Benchmark: C = A @ B
//      A (MxK) @ B (KxN) = C (MxN)

// compile: nvcc matmul.cu -o matmul
// run: ./matmul

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

// CPU Matrix Multiplication O(M*K*N)
// matrices stored flat: element [i][j] = arr[i*cols + j]
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
// one thread per output cell, all run in parallel
// row/col computed from thread position in the grid


__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k ,int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < m && col < n){
        float sum = 0.0f;
        for (int l = 0; l < k; l++){
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;

    }
}

// Initilaize with rand values
void init_matrix(float *mat, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        mat[i] = (float)rand()/RAND_MAX;
    }
}

// Measure execution time
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    //Allocate host memory    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Benchmark CPU + GPU
    double cpu_t = 0.0, gpu_t = 0.0;
    for (int i = 0; i < 20; i++) {
        double t = get_time();
        matmul_cpu(h_A, h_B, h_C, M, K, N);
        cpu_t += get_time() - t;

        t = get_time();
        matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        gpu_t += get_time() - t;
    }

    printf("CPU: %.1f us\n",    (cpu_t / 20.0) * 1e6);
    printf("GPU: %.1f us\n",    (gpu_t / 20.0) * 1e6);
    printf("Speedup: %.2fx\n",  (cpu_t / gpu_t));


    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}