#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>

#include "utils.h"
#include "reference.cpp"


#define TPB 1024

__device__
unsigned int sum_reduce_warp(unsigned int * s_buf)
{
    unsigned int i = threadIdx.x;
    if ((i&0x1) == 0) s_buf[i] += s_buf[i|0x1];
    if ((i&0x3) == 0) s_buf[i] += s_buf[i|0x2];
    if ((i&0x7) == 0) s_buf[i] += s_buf[i|0x4];
    if ((i&0xf) == 0) s_buf[i] += s_buf[i|0x8];
    unsigned int k = i & ~0x1f;
    return s_buf[k] + s_buf[k|0x10];
}

__device__
unsigned int sum_reduce_block(unsigned int * s_buf)
{
    assert(blockDim.x <= 1024);
    
    unsigned int sum_warp = sum_reduce_warp(s_buf);
    
    __shared__ unsigned int s_sums[1024];
    
    unsigned int i = threadIdx.x;
    if (i<32 && i>=(blockDim.x>>5)) s_sums[i] = 0;
    if ((i&0x1f) == 0) s_sums[i>>5] = sum_warp;
    __syncthreads();
    // only valid in 1st warp of a block
    return sum_reduce_warp(s_sums);
}

__global__
void sum_reduce_block_kernel(unsigned int const * d_in, unsigned int * d_out, unsigned int n)
{
    assert(blockDim.x <= 1024);
    __shared__ unsigned int s_vals[1024];

    unsigned int const i = threadIdx.x;
    unsigned int const j = threadIdx.x + blockIdx.x * blockDim.x;
    s_vals[i] = 0;
    if (j < n)
        s_vals[i] = d_in[j];
    
    unsigned int const sum = sum_reduce_block(s_vals);
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sum;
    }
}

#define N 2048
#define NBLOCKS 2

void test_reduce_block()
{
    unsigned int h_in[N], h_out[NBLOCKS];
    std::srand(0);
    for (unsigned int i = 0; i < N; ++i)
        h_in[i] = std::rand() & 0x1;
    unsigned int * d_in, * d_out;
    unsigned int bufsz = N*sizeof(unsigned int);
    unsigned int outsz = NBLOCKS*sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void**)&d_in, bufsz));
    checkCudaErrors(cudaMalloc((void**)&d_out, outsz));

    checkCudaErrors(cudaMemcpy(d_in, h_in, bufsz, cudaMemcpyHostToDevice));
    unsigned int const bsz = TPB;
    unsigned int const nblocks = N / bsz + (N%bsz != 0);
    sum_reduce_block_kernel<<<nblocks,bsz>>>(d_in, d_out, N);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, outsz, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);

    unsigned int * p = h_in;
    for (int i = 0; i < nblocks; ++i) {
        unsigned int sum = 0;
        for (int j = 0; j < TPB && p < h_in + N; ++j) {
            sum += *p++;
        }
        printf("block %d: ", i);
        if (sum == h_out[i]) {
            printf("PASS\n");
        } else {
            printf("FAIL! expected: %u, got: %u\n", sum, h_out[i]);
        }
    }
    /*
    printf("input:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d: %u\n", i, h_in[i]);
    }
    printf("output:\n");
    for (int i = 0; i < NBLOCKS; ++i) {
        printf("%d: %u\n", i, h_out[i]);
    }
    */
}
