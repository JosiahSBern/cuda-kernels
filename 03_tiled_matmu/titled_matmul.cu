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