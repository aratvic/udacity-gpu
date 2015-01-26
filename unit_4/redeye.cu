//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

#include <cstdio>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

// assume at least compute capability 2.0
#define SHM_BYTES 48*1024
#define THREADS_PER_BLOCK 1024

#define SCAN_SHM 1024

template <typename T> struct device_plus { __inline__ __device__ T operator()(T x, T y) const {return x+y;} };

template <typename T, typename Op>
__global__ void exclusive_scan_kernel(T const * const d_input, T * const d_output, T * const d_aux, unsigned int n, Op const op, T ne)
{
    assert(blockDim.x <= SCAN_SHM);

    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T s_buf[SCAN_SHM];

    if (j < n)
        s_buf[i] = d_input[j];

    __syncthreads();

    unsigned int m = min(blockDim.x, n - blockDim.x * blockIdx.x);
    if (i >= m) return;

    int s;
    for (s = 1; s < m; s <<= 1) {
        int k = (i+1) % (s<<1);
        int ii = (k == 0) ? i - s : i - k + s;

        if (k == 0 || (i == m - 1 && ii < i))
            s_buf[i] = op(s_buf[i], s_buf[ii]);
        __syncthreads();
    }

    if (i == m-1) {
        if (d_aux) d_aux[blockIdx.x] = s_buf[m-1];
        s_buf[m-1] = ne;
    }

    for (s >>= 1; s > 0; s >>= 1) {
        int k = (i+1) % (s<<1);
        int ii = (k == 0) ? i - s : i - k + s;

        if (k == 0 || (i == m - 1 && ii < i)) {
            T x = s_buf[i];
            s_buf[i] = op(x, s_buf[ii]);
            s_buf[ii] = x;
        }
        __syncthreads();
    }

    if (j < n)
         d_output[j] = s_buf[i];
}

template <typename T, typename Op>
__global__ void blockwise_op_kernel(T const * const d_input, T const * const d_buf, T * const d_output, unsigned int const n, Op const op)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    T x = d_buf[blockIdx.x];
    if (i < n)
        d_output[i] = op(x, d_input[i]);
}

template <typename T, typename Op>
void exclusive_scan(T const * const d_input, T * const d_output, unsigned int const n, Op const op, T ne = T())
{
    assert(n < 512*512);
    unsigned int bsz = min(512, n);
    unsigned int nblocks = n / bsz + (n%bsz != 0);

    T * d_buf;
    checkCudaErrors(cudaMalloc((void**)&d_buf, nblocks*sizeof(T)));

    exclusive_scan_kernel<<<nblocks, bsz>>>(d_input, d_output, d_buf, n, op, ne);
    if (nblocks > 1) {
        exclusive_scan_kernel<<<1, nblocks>>>(d_buf, d_buf, (T*)0, n, op, ne);
        blockwise_op_kernel<<<nblocks, bsz>>>(d_output, d_buf, d_output, n, op);
    }

    cudaDeviceSynchronize();

    cudaFree(d_buf);
}

__global__ void digit_kernel(unsigned int const * const d_input, unsigned int * const d_output, unsigned int const numElems, unsigned int const digit)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElems) {
        d_output[i] = (d_input[i] & (1u << digit)) >> digit;
    }
}

__global__ void compute_pos_kernel(unsigned int const * const d_digits, unsigned int * const d_scan, unsigned int * const d_pos, unsigned int const numElems)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int numOnes = d_scan[numElems-1] + d_digits[numElems-1];
    unsigned int numZeros = numElems - numOnes;
    
    if (i < numElems) {
        if (d_digits[i]) {
            d_pos[i] = d_scan[i] + numZeros;
        } else {
            d_pos[i] = i - d_scan[i];
        }
    }
}

__global__ void scatter_kernel(unsigned int* const d_inputVals,
                               unsigned int* const d_inputPos,
                               unsigned int* const d_outputVals,
                               unsigned int* const d_outputPos,
                               unsigned int* const d_pos,
                               unsigned int const numElems)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numElems) {
        d_outputVals[d_pos[i]] = d_inputVals[i];
        d_outputPos[d_pos[i]] = d_inputPos[i];
    }
}

void radix_step(unsigned int* const d_inputVals,
                unsigned int* const d_inputPos,
                unsigned int* const d_outputVals,
                unsigned int* const d_outputPos,
                unsigned int const numElems,
                unsigned int const stepNum,
                unsigned int* const d_digits,
                unsigned int* const d_buf)
{
    assert(numElems < 512u*512u);
    unsigned int bsz = min(512u, numElems);
    unsigned int nblocks = numElems / bsz + (numElems%bsz != 0);
    
    digit_kernel<<<nblocks, bsz>>>(d_inputVals, d_digits, numElems, stepNum);
    exclusive_scan(d_digits, d_buf, numElems, device_plus<unsigned int>());
    compute_pos_kernel<<<nblocks, bsz>>>(d_digits, d_buf, d_buf, numElems);
    scatter_kernel<<<nblocks, bsz>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_buf, numElems);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t const numElems)
{
    unsigned int * d_digits, * d_buf;
    unsigned int * d_val[2], * d_pos[2];
    checkCudaErrors(cudaMalloc((void**)&d_digits, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_buf, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_val[0], numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_pos[0], numElems*sizeof(unsigned int)));
    d_val[1] = d_outputVals;
    d_pos[1] = d_outputPos;
    
    radix_step(d_inputVals, d_inputPos, d_val[0], d_pos[0], numElems, 0, d_digits, d_buf);
    for (unsigned int j = 1; j < 8*sizeof(unsigned int); j++) {
        unsigned int k = j&1;
        radix_step(d_val[1-k], d_pos[1-k], d_val[k], d_pos[k], numElems, j, d_digits, d_buf);
    }
    
    cudaFree(d_pos[0]);
    cudaFree(d_val[0]);
    cudaFree(d_buf);
    cudaFree(d_digits);
}
