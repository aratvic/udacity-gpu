/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"

#define EPT 64
#define TPB 256
#define MAXBINS 1024

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
    assert(MAXBINS >= TPB);
    
    __shared__ unsigned int shm[MAXBINS+225];
    
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    shm[i] = 0;
    shm[i+TPB] = 0;
    shm[i+2*TPB] = 0;
    shm[i+3*TPB] = 0;

    __syncthreads();
    for (unsigned int k = 0; k < EPT; ++k) {
        unsigned int addr = j+k*stride;
        if (addr < numVals) {
            atomicAdd(shm+vals[addr], 1);
        }
    }
    __syncthreads();
    
    atomicAdd(histo+i, shm[i]);
    atomicAdd(histo+i+TPB, shm[i+TPB]);
    atomicAdd(histo+i+2*TPB, shm[i+2*TPB]);
    atomicAdd(histo+i+3*TPB, shm[i+3*TPB]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    checkCudaErrors(cudaMemset(d_histo, 0, numBins*sizeof(unsigned int)));
    
    unsigned int epb = EPT * TPB;
    unsigned int nblocks = numElems/epb;
    nblocks += (nblocks*epb < numElems);
    yourHisto<<<nblocks, TPB>>>(d_vals, d_histo, numElems);
        
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
