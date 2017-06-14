#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SYMBOL_NUM 256
#define MAX_CODE_LEN (SYMBOL_NUM - 1)

#define GRID_X (1u << 12)
#define GRID_Y 1
#define BLOCK_X (1u << 10)
#define BLOCK_Y 1

#define RAW_BUFF_SIZE (GRID_X * GRID_Y * BLOCK_X * BLOCK_Y)

#define DEF_IDX \
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

#define CUDA_SAFE_CALL(func)                                               \
    do {                                                                   \
        cudaError_t err = (func);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__);         \
            exit(err);                                                     \
        }                                                                  \
    } while (0)

__global__ void prefixSum(
    char* raw_data, size_t* code_len, size_t* len_ps, size_t raw_len);
__global__ void genByteStream(
    char* raw_data, bool* code, size_t* code_len, size_t* len_ps, bool* bytes,
    size_t raw_len);
__global__ void compressByteStream(bool* bytes, uint8_t* bits, size_t bits_len);

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s raw_data huffman_table\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read raw data
    char* raw_data_h = (char*)calloc(RAW_BUFF_SIZE, 1);
    size_t raw_len = 0;

    {
        printf("Reading raw data ... ");
        FILE* raw_file = fopen(argv[1], "r");
        size_t r;
        while ((r = fread(
                    raw_data_h + raw_len, 1, RAW_BUFF_SIZE - raw_len, raw_file))
               != 0) {
            raw_len += r;
        }
        fclose(raw_file);
        printf("done. Read %zu bytes.\n", raw_len);
    }

    char* raw_data;
    CUDA_SAFE_CALL(cudaMalloc((void**)&raw_data, raw_len));
    CUDA_SAFE_CALL(cudaMemcpy(
        raw_data, raw_data_h, raw_len, cudaMemcpyHostToDevice));

    // Read Huffman table
    bool* code_h = (bool*)calloc(MAX_CODE_LEN * SYMBOL_NUM, sizeof(bool));
    size_t* code_len_h = (size_t*)calloc(SYMBOL_NUM, sizeof(size_t));

    {
        printf("Reading Huffman table ... ");
        FILE* hufftable_file = fopen(argv[2], "r");
        int c;
        while (fscanf(hufftable_file, "%d", &c) != EOF) {
            size_t len;
            int _ = fscanf(hufftable_file, "%zu", &len);
            code_len_h[c] = len;
            for (size_t j = 0; j < len; j++) {
                int b;
                int _ = fscanf(hufftable_file, "%d", &b);
                code_h[c * MAX_CODE_LEN + j] = b;
            }
        }

        fclose(hufftable_file);
        printf("done.\n");
    }

    bool* code;
    CUDA_SAFE_CALL(cudaMalloc(
        (void**)&code, MAX_CODE_LEN * SYMBOL_NUM * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemcpy(
        code, code_h, MAX_CODE_LEN * SYMBOL_NUM * sizeof(bool),
        cudaMemcpyHostToDevice));
    size_t* code_len;
    CUDA_SAFE_CALL(cudaMalloc((void**)&code_len, SYMBOL_NUM * sizeof(size_t)));
    CUDA_SAFE_CALL(cudaMemcpy(
        code_len, code_len_h, SYMBOL_NUM * sizeof(size_t),
        cudaMemcpyHostToDevice));

    // free(code_h);

    // Run on CUDA
    dim3 grid(GRID_X, GRID_Y);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);

    // Prefix sum of code length
    size_t* len_ps;
    CUDA_SAFE_CALL(cudaMalloc((void**)&len_ps, raw_len * sizeof(size_t)));

    prefixSum<<<grid, block>>>(raw_data, code_len, len_ps, raw_len);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    /* {
        size_t* len_ps_h = (size_t*)malloc(sizeof(size_t) * raw_len);
        CUDA_SAFE_CALL(cudaMemcpy(
            len_ps_h, len_ps, sizeof(size_t) * raw_len, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < raw_len; i++)
            printf("%zu\n", len_ps_h[i]);
    } */

    size_t len_ps_end;
    CUDA_SAFE_CALL(cudaMemcpy(
        &len_ps_end, &len_ps[raw_len - 1], sizeof(size_t),
        cudaMemcpyDeviceToHost));
    size_t bytes_len = len_ps_end + code_len_h[raw_data_h[raw_len - 1]];
    printf("bytes len: %zu\n", bytes_len);

    // free(code_len_h);
    // free(raw_data_h);

    if (bytes_len > RAW_BUFF_SIZE) {
        fprintf(stderr, "Bytes stream overflowed\n");
        exit(EXIT_FAILURE);
    }

    bool* bytes;
    CUDA_SAFE_CALL(cudaMalloc((void**)&bytes, sizeof(bool) * bytes_len));
    genByteStream<<<grid, block>>>(
        raw_data, code, code_len, len_ps, bytes, raw_len);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    // CUDA_SAFE_CALL(cudaFree(len_ps));
    // CUDA_SAFE_CALL(cudaFree(code_len));
    // CUDA_SAFE_CALL(cudaFree(code));
    // CUDA_SAFE_CALL(cudaFree(raw_data));

    uint8_t* bits;
    size_t bits_len = bytes_len / 8 + 1;
    printf("bits len: %zu\n", bits_len);
    CUDA_SAFE_CALL(cudaMalloc((void**)&bits, bits_len));
    compressByteStream<<<grid, block>>>(bytes, bits, bits_len);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    // CUDA_SAFE_CALL(cudaFree(bits));
    // CUDA_SAFE_CALL(cudaFree(bytes));

    gettimeofday(&time_end, NULL);
    double sec = (double)(time_end.tv_sec - time_start.tv_sec)
                  + (double)(time_end.tv_usec - time_start.tv_usec) / 1e6;
    printf("bytes: %zu sec: %lf bytes/sec: %lf\n",
        raw_len, sec, raw_len / sec);

    return 0;
}

__global__ void prefixSum(
    char* raw_data, size_t* code_len, size_t* len_ps, size_t raw_len)
{
    DEF_IDX;

    if (2 * idx + 1 < raw_len) {
        len_ps[2 * idx] = code_len[raw_data[2 * idx]];
        len_ps[2 * idx + 1] = code_len[raw_data[2 * idx + 1]];
    }

    __syncthreads();

    // build sum in place up the tree
    size_t offset = 1;
    for (size_t d = raw_len >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            size_t ai = offset * (2 * idx + 1) - 1;
            size_t bi = offset * (2 * idx + 2) - 1;
            len_ps[bi] += len_ps[ai];
        }
        offset *= 2;
    }

    // clear the last element
    if (idx == 0)
        len_ps[raw_len - 1] = 0;

    // traverse down tree & build scan
    for (size_t d = 1; d < raw_len; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            size_t ai = offset * (2 * idx + 1) - 1;
            size_t bi = offset * (2 * idx + 2) - 1;
            size_t t = len_ps[ai];
            len_ps[ai] = len_ps[bi];
            len_ps[bi] += t;
        }
    }
}

__global__ void genByteStream(
    char* raw_data, bool* code, size_t* code_len, size_t* len_ps, bool* bytes,
    size_t raw_len)
{
    DEF_IDX;

    if (idx < raw_len) {
        size_t start_pos = len_ps[idx];
        char symbol = raw_data[idx];
        for (size_t i = 0, len = code_len[symbol]; i < len; i++) {
            bytes[start_pos + i] = code[symbol * MAX_CODE_LEN + i];
        }
    }
}

__global__ void compressByteStream(bool* bytes, uint8_t* bits, size_t bits_len)
{
    DEF_IDX;

    if (idx < bits_len) {
        for (size_t i = 0; i < 8; i++) {
            bits[idx] |= (uint8_t)bytes[8 * idx + i] << (7 - i);
        }
    }
}
