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

#define TPB 512
#define RADIX_BITS 8
#define RADIX (1u << RADIX_BITS)

__device__ void inclusive_sum_scan_warp(unsigned int * const s_buf)
{
    unsigned int i = threadIdx.x;
    unsigned int k = i & 0x1F;
    if (k >= 1) s_buf[i] += s_buf[i-1];
    if (k >= 2) s_buf[i] += s_buf[i-2];
    if (k >= 4) s_buf[i] += s_buf[i-4];
    if (k >= 8) s_buf[i] += s_buf[i-8];
    if (k >= 16) s_buf[i] += s_buf[i-16];
}

// no terminating __syncthreads
__device__ void inclusive_sum_scan_block(unsigned int * const s_buf)
{
    assert(blockDim.x<=1024);

    __shared__ unsigned int warp_sum[33];

    unsigned int i = threadIdx.x;
    
    // per-warp scan
    inclusive_sum_scan_warp(s_buf);
    warp_sum[0] = 0;
    warp_sum[(i>>5)+1] = s_buf[i|0x1F];
    __syncthreads();
    
    // scan per-warp sums (at most 1 warp: 32*32=1024)
    if (i<32) {
        inclusive_sum_scan_warp(warp_sum);
    }
    __syncthreads();
    
    s_buf[i] += warp_sum[i>>5];
}

__device__ unsigned int block_slice(unsigned int const n)
{
    return min(blockDim.x, n - blockDim.x * blockIdx.x);
}

__device__ unsigned int
diff_pred(unsigned int * const s_val, unsigned int const mask, unsigned int const i)
{
    if (i>0) {
        unsigned int digit_this = (s_val[i] & mask);
        unsigned int digit_prev = (s_val[i-1] & mask);
        return digit_prev != digit_this;
    } else {
        return 1u;
    }
}

// sort in shared memory according to current r-ary digit (r=2^RADIX_BITS)
__device__ void
radix_sort_block(unsigned int * const s_val,
                 unsigned int * const s_pos,
                 unsigned int const m,
                 unsigned int const sortDigit,
                 unsigned int * const s_scan)
{
    assert(!(RADIX_BITS&1)); // buffer ping-pong will end correctly

    unsigned int const i = threadIdx.x;

    s_scan[0] = 0;
    unsigned int const k_base = sortDigit * RADIX_BITS;
    unsigned int base_src = 0;
    unsigned int base_dst = TPB+1;
    for (unsigned int k = k_base; k < k_base + RADIX_BITS; ++k) {
        unsigned int const mask = 1u<<k;
        unsigned int const k_bit = s_scan[i+1] = (s_val[i+base_src] & mask) >> k;
        __syncthreads();

        inclusive_sum_scan_block(s_scan+1);
        __syncthreads();
        
        unsigned int const numZeros = m - s_scan[m];
        
        unsigned int scatter_pos = s_scan[i];
        if (k_bit) {
            scatter_pos += numZeros;
        } else {
            scatter_pos = i-scatter_pos;
        }

        s_val[base_dst+scatter_pos] = s_val[base_src+i];
        s_pos[base_dst+scatter_pos] = s_pos[base_src+i];
        unsigned int tmp = base_src;
        base_src = base_dst;
        base_dst = tmp;
        __syncthreads();
    }
}

__device__ unsigned int
radix_hist_block(unsigned int * const s_val,
                 unsigned int * const s_ofs,
                 unsigned int * const s_bin,
                 unsigned int const m,
                 unsigned int const sortDigit,
                 unsigned int * const s_scan)
{
    // compute histogram in shared memory:
    // pack together indices (offsets) where the digit is different from the
    // previous one, then subtract neighbor offsets to get the histogram
    
    unsigned int const i = threadIdx.x;
    unsigned int const k_base = sortDigit * RADIX_BITS;
    unsigned int const mask = (RADIX - 1) << k_base;
    
    s_scan[0] = 0;
    unsigned int const pred = s_scan[i+1] = diff_pred(s_val, mask, i);
    __syncthreads();
    
    inclusive_sum_scan_block(s_scan+1);
    __syncthreads();
    
    unsigned int const nonempty_bins = s_scan[m];
    if (pred) {
        s_ofs[s_scan[i]] = i;
        s_bin[s_scan[i]] = (s_val[i] & mask) >> k_base;
    }
    s_ofs[nonempty_bins] = m;
    __syncthreads();

    return nonempty_bins;
}

__global__ void
radix_block_step (unsigned int * const d_inputVals,
                  unsigned int * const d_inputPos,
                  unsigned int * const d_outputVals,
                  unsigned int * const d_outputPos,
                  unsigned int * const d_perBlockHist, // has blockDim.x * 2^RADIXBITS elems
                  unsigned int const n,
                  unsigned int const sortDigit)
{
  
    unsigned int const i = threadIdx.x;
    unsigned int const j = threadIdx.x + blockDim.x * blockIdx.x;
    if (j >= n) return;

    unsigned int const m = block_slice(n);

    __shared__ unsigned int s_val[2*(TPB+1)];
    __shared__ unsigned int s_pos[2*(TPB+1)];
    __shared__ unsigned int s_scan[TPB+1];
    
    s_val[i] = d_inputVals[j];
    s_pos[i] = d_inputPos[j];

    radix_sort_block(s_val, s_pos, m, sortDigit, s_scan);

    d_outputVals[j] = s_val[i];
    d_outputPos[j] = s_pos[i];

    __syncthreads();

    unsigned int * const s_bin = s_val+TPB+1;
    unsigned int * const s_ofs = s_pos+TPB+1;
    
    unsigned int nonempty_bins = radix_hist_block(s_val, s_ofs, s_bin, m, sortDigit, s_scan);
    if (i < nonempty_bins) {
        unsigned int const scatter_pos = blockIdx.x + s_bin[i] * gridDim.x;
        d_perBlockHist[scatter_pos] = s_ofs[i+1] - s_ofs[i];
    }
}

__global__ void blockwise_scan_histograms(unsigned int * const d_perBlockHist)
{
    unsigned int i = threadIdx.x;
    
    __shared__ unsigned int s_buf[TPB+1];
    unsigned int hist_idx = i+blockIdx.x*blockDim.x;
    
    s_buf[i] = d_perBlockHist[hist_idx];
    __syncthreads();

    inclusive_sum_scan_block(s_buf);
    __syncthreads();
    
    d_perBlockHist[hist_idx] = s_buf[i];
}

__global__ void binwise_scan_histograms(unsigned int * const d_perBlockHist)
{
    unsigned int i = threadIdx.x;
    
    __shared__ unsigned int s_buf[TPB+1];
    unsigned int hist_idx = blockIdx.x+i*gridDim.x;
    
    s_buf[i] = d_perBlockHist[hist_idx];
    __syncthreads();

    inclusive_sum_scan_block(s_buf);
    __syncthreads();
    
    d_perBlockHist[hist_idx] = s_buf[i];
}

__global__ void
scatter_step(unsigned int * const d_val_in,
             unsigned int * const d_pos_in,
             unsigned int * const d_val_out,
             unsigned int * const d_pos_out,
             unsigned int * const d_histSAT,
             unsigned int n,
             unsigned int const sortDigit)
{
    unsigned int nblocks = gridDim.x;
    unsigned int i = threadIdx.x;
    unsigned int j = blockIdx.x;
    unsigned int k = i + j*blockDim.x;
    if (k >= n) return;
    
    unsigned int val = d_val_in[k];
    unsigned int pos = d_pos_in[k];

    unsigned int bit_base = sortDigit * RADIX_BITS;
    unsigned int mask = (RADIX - 1) << bit_base;
    unsigned int bin = (val & mask) >> bit_base;
    
    unsigned int global_bin_ofs = 0;
    if (bin > 0) global_bin_ofs = d_histSAT[bin*nblocks-1];
    unsigned int local_block_ofs = 0;
    unsigned int local_bin_ofs = 0;
    if (j > 0)
        local_block_ofs = d_histSAT[j-1+bin*nblocks];
    if (bin > 0)
        local_bin_ofs =  d_histSAT[j+(bin-1)*nblocks];
    if (j > 0 && bin > 0) {
        unsigned int a = d_histSAT[j-1+(bin-1)*nblocks];
        local_block_ofs -= a;
        local_bin_ofs -= a;
    }
    
    unsigned int scatter_pos = global_bin_ofs + local_block_ofs + i - local_bin_ofs;
    d_val_out[scatter_pos] = val;
    d_pos_out[scatter_pos] = pos;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t const numElems)
{
    unsigned int const bsz = min(TPB, static_cast<unsigned int>(numElems));
    unsigned int const nblocks = numElems / bsz + (numElems%bsz != 0);

    unsigned int * d_h = 0;
    unsigned int const hist_size_bytes = RADIX * nblocks * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void**)&d_h, hist_size_bytes));

    unsigned int * d_val[2] = {d_inputVals, d_outputVals};
    unsigned int * d_pos[2] = {d_inputPos, d_outputPos};
    unsigned int bp = 0;
    for (unsigned int i = 0; i < 4; ++i) {
        checkCudaErrors(cudaMemset(d_h, 0, hist_size_bytes));
        if (i==0) {
            // buffer pinp-pong should end in d_output*
            radix_block_step<<<nblocks, bsz>>>(d_val[bp], d_pos[bp], d_val[1-bp], d_pos[1-bp], d_h, numElems, i);
            bp = 1 - bp;
        } else {
            radix_block_step<<<nblocks, bsz>>>(d_val[bp], d_pos[bp], d_val[bp], d_pos[bp], d_h, numElems, i);
        }
        blockwise_scan_histograms<<<RADIX, nblocks>>>(d_h);
        binwise_scan_histograms<<<nblocks, RADIX>>>(d_h);
        scatter_step<<<nblocks, bsz>>>(d_val[bp], d_pos[bp], d_val[1-bp], d_pos[1-bp], d_h, numElems, i);
        bp = 1 - bp;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(d_h);
}
