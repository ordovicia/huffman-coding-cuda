#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define GRID_DIM (1u << 12)
#define BLOCK_DIM (1u << 10)

#define BLOCK_NUM 32
#define ONE_BLOCK (GRID_DIM * BLOCK_DIM)
#define N (BLOCK_NUM * ONE_BLOCK)

#define CUDA_SAFE_CALL(func)                                               \
    do {                                                                   \
        cudaError_t err = (func);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__);         \
            exit(err);                                                     \
        }                                                                  \
    } while (0)

__global__ void blocked(float* a, float* b, float* c);
__global__ void many_elements(float* a, float* b, float* c);

int main(void)
{
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        a_h[i] = (float)rand() / (float)RAND_MAX;
        b_h[i] = (float)rand() / (float)RAND_MAX;
    }

    float *a, *b, *c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&a, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(
        a, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        b, b_h, N * sizeof(float), cudaMemcpyHostToDevice));


    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);

#if 0
    for (size_t i = 0; i < BLOCK_NUM; i++)
        blocked<<<GRID_DIM, BLOCK_DIM>>>(
            a + i * ONE_BLOCK, b + i * ONE_BLOCK, c + i * ONE_BLOCK);
#else
    many_elements<<<GRID_DIM, BLOCK_DIM>>>(a, b, c);
#endif

    gettimeofday(&time_end, NULL);
    double sec = (double)(time_end.tv_sec - time_start.tv_sec)
                 + (double)(time_end.tv_usec - time_start.tv_usec) / 1e6;
    printf("%lf\n", sec);

    /*
    float* out_h = (float*)malloc(N * sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(
        out_h, c, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N; i++) {
        printf("%lf\n", out_h[i]);
    }
    */

    return 0;
}

__global__ void blocked(float* a, float* b, float* c)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    c[idx] = a[idx] * b[idx];
}

__global__ void many_elements(float* a, float* b, float* c)
{
    size_t idx = BLOCK_NUM * (blockDim.x * blockIdx.x + threadIdx.x);
    for (size_t i = 0; i < BLOCK_NUM; i++)
        c[idx + i] = a[idx + i] * b[idx + i];
}
