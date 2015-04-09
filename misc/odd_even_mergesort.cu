#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>

#include "utils.h"
#include "reference.cpp"


#define TPB 1024

__device__
void shared_from_global(unsigned int * s_dst, unsigned int const * d_src, unsigned int n, unsigned int def_val)
{
    int i = threadIdx.x;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    s_dst[i] = def_val;
    if (j < n) s_dst[i] = d_src[j];
}

__device__
void global_from_shared(unsigned int * d_dst, unsigned int const * s_src, unsigned int n)
{
    int i = threadIdx.x;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) d_dst[j] = s_src[i];
}

__device__
unsigned int block_slice(unsigned int const n)
{
    return min(blockDim.x, n - blockDim.x * blockIdx.x);
}

__device__ void swap(unsigned int& a, unsigned int b, unsigned int greater_flag)
{
    a = ((a<b)^(greater_flag)) ? a : b;
}
__device__ void swap_sync(unsigned int * s_vals, unsigned int ia, unsigned int ib, unsigned int greater_flag, unsigned int enable_flag)
{
    unsigned int aa;
    if (enable_flag)
        aa = ((s_vals[ia]<s_vals[ib])^(greater_flag)) ? s_vals[ia] : s_vals[ib];
    __syncthreads();
    if (enable_flag)
        s_vals[ia] = aa;
}

__device__ void odd_even_merge_warp(unsigned int * s_vals, unsigned int n)
{
    assert(__popc(n) == 1 && n <= 32);
    
    unsigned int tid = threadIdx.x;
    unsigned int stride = n>>1;
    
    swap(s_vals[tid], s_vals[tid^stride], (tid&stride)!=0);
    
    unsigned int m = stride;
    for (stride >>= 1; stride > 0; stride >>= 1) {
        if ((~tid)&m)
            swap(s_vals[tid+stride], s_vals[(tid^stride)+stride], (tid&stride)!=0);
        m |= stride;
    }
}

__device__ void odd_even_merge_block(unsigned int * s_vals, unsigned int n)
{
    assert(__popc(n) == 1 && n > 32);
    
    unsigned int tid = threadIdx.x;
    unsigned int stride = n>>1;
    
    swap_sync(s_vals, tid, tid^stride, (tid&stride)!=0, 1);
    
    unsigned int m = stride;
    for (stride >>= 1; stride >= 32; stride >>= 1) {
        swap_sync(s_vals, tid+stride, (tid^stride)+stride, (tid&stride)!=0, (~tid)&m);
        m |= stride;
    }
    for (; stride > 0; stride >>= 1) {
        if ((~tid)&m)
            swap(s_vals[tid+stride], s_vals[(tid^stride)+stride], (tid&stride)!=0);
        m |= stride;
    }
}

// has terminating syncthreads
__device__
void oem_sort_block(unsigned int * s_vals)
{
    for (int n = 2; n <= 32; n <<= 1) {
        odd_even_merge_warp(s_vals, n);
    }
    __syncthreads();
    for (int n = 64; n <= blockDim.x; n <<= 1) {
        odd_even_merge_block(s_vals, n);
        __syncthreads();
    }
}

__global__
void oem_sort_block_kernel(unsigned int const * d_in, unsigned int * d_out, unsigned int n)
{
    assert(blockDim.x <= TPB);
    __shared__ unsigned int s_vals[TPB];

    shared_from_global(s_vals, d_in, n, UINT_MAX);
    oem_sort_block(s_vals);
    global_from_shared(d_out, s_vals, n);
}

#define NBLOCKS 16
#define N (NBLOCKS*TPB-123)

unsigned int num_blocks(unsigned int n, unsigned int tpb)
{
    unsigned int m = n / tpb;
    return m + (m*tpb<n);
}

void test_oem_sort()
{
    unsigned int h_in[N], h_out[N];
    for (unsigned int i = 0; i < N; ++i)
        h_in[i] = i;
    std::srand(0);
    std::random_shuffle(h_in, h_in+N);
    unsigned int * d_in, * d_out;
    unsigned int bufsz = N*sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void**)&d_in, bufsz));
    checkCudaErrors(cudaMalloc((void**)&d_out, bufsz));
    checkCudaErrors(cudaMemcpy(d_in, h_in, bufsz, cudaMemcpyHostToDevice));
    
    unsigned int nblocks = num_blocks(N, TPB);
    oem_sort_block_kernel<<<nblocks,TPB>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy(h_out, d_out, bufsz, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    for (int i = 0; i < nblocks; ++i) {
        unsigned int beg = i*TPB;
        unsigned int end = min(N, (i+1)*TPB);
        std::sort(h_in+beg, h_in+end);
        unsigned int fail = 0;
        for (unsigned int i = beg; !fail && i < end; ++i) {
            if (h_in[i] != h_out[i]) {
                fail = 1;
            }
        }
        if (!fail) {
            printf("block %d: PASS\n", i);
        } else {
            printf("block %d: FAIL\n", i);
            for (unsigned int i = beg; i < end; ++i) {
                printf("%d: expected=%u, got=%u\n", i, h_in[i], h_out[i]);
            }
        }
    }
}
