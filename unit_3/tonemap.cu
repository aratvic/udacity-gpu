/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <cstdio>
#include <vector>

#define REDUCE_SHM 1024
#define SCAN_SHM 1024

template <typename T, typename Op>
__global__ void reduce_kernel(T const * const d_input, T * const d_output, unsigned int const n, Op const op)
{
    assert(blockDim.x <= REDUCE_SHM);
    
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T s_buf[REDUCE_SHM];
    
    if (j < n)
        s_buf[i] = d_input[j];
    
    __syncthreads();
    
    for (unsigned int s = min(blockDim.x, n - blockDim.x * blockIdx.x); s > 1; s >>= 1) {
        T x = s_buf[i];
        if ((i & 1) == 0 && i + 1 < s)
            x = op(x, s_buf[i+1]);
        __syncthreads();
        if ((i & 1) == 0 && i + 1 <= s)
            s_buf[i>>1] = x;
        __syncthreads();
        s = s + (s&1);
    }
    
    if (i == 0) {
        d_output[blockIdx.x] = s_buf[0];
    }
}

template <typename T, typename Op>
T reduce(T const * const d_input, unsigned int const n, Op const op)
{
    unsigned int bsz = min(512, n);
    
    T * d_buf;
    T const * d_in = d_input;
    unsigned int nbuf = n / bsz + (n%bsz != 0);
    checkCudaErrors(cudaMalloc((void**)&d_buf, nbuf*sizeof(T)));
    
    for (unsigned int m = n; m > 1; m = m / bsz + (m%bsz != 0)) {
        reduce_kernel<<<m/bsz + (m%bsz != 0), bsz>>>(d_in, d_buf, m, op);
        d_in = d_buf;
    }
    cudaDeviceSynchronize();
    
    T res;
    checkCudaErrors(cudaMemcpy(&res, d_buf, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(d_buf);
    return res;
}

template <typename T> struct device_plus { __inline__ __device__ T operator()(T x, T y) const {return x+y;} };
template <typename T> struct device_max { __inline__ __device__ T operator()(T x, T y) const {return max(x,y);} };
template <typename T> struct device_min { __inline__ __device__ T operator()(T x, T y) const {return min(x,y);} };

template <typename T>
__global__ void histogram_kernel(T const * const d_sample, unsigned int const nsamples, T const sample_min, T const sample_max, unsigned int const nbins, unsigned int * const d_hist)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= nsamples) return;
    
    unsigned int bin = float(d_sample[i] - sample_min) / float(sample_max - sample_min) * nbins;
    bin = min(bin, nbins-1);
    atomicAdd(d_hist+bin, 1);
}

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
    
    //if (i==0) printf("m=%d\n", m);
    int s;
    for (s = 1; s < m; s <<= 1) {
        int k = (i+1) % (s<<1);
        int ii = (k == 0) ? i - s : i - k + s;
        //if (i == m-1) printf("> s: %d, i: %d, ii: %d\n", s, i, ii);
        
        if (k == 0 || (i == m - 1 && ii < i))
            s_buf[i] = op(s_buf[i], s_buf[ii]);            
        //__syncthreads();
        //if (k == 0 || i == m - 1)
        //    s_buf[i] = x;
        __syncthreads();
        //if (i < m)
        //    printf("%d %d\n", i, s_buf[i]);
        //if (i == 0) printf("===\n");
    }
    
    if (i == m-1) {
        if (d_aux) d_aux[blockIdx.x] = s_buf[m-1];
        s_buf[m-1] = ne;
    }
    
    //if (i < m)
    //    printf("%d %d\n", i, s_buf[i]);
    //if (i == 0) printf("===\n");
    
    for (s >>= 1; s > 0; s >>= 1) {
        int k = (i+1) % (s<<1);
        int ii = (k == 0) ? i - s : i - k + s;
        
        //T y = x;
        if (k == 0 || (i == m - 1 && ii < i)) {
            T x = s_buf[i];
            s_buf[i] = op(x, s_buf[ii]);
            s_buf[ii] = x;
        //__syncthreads();
        //if (k == 0 || i == m - 1) {
            //s_buf[ii] = x;
            //s_buf[i] = y;
        }
        __syncthreads();
        //if (i < m)
        //    printf("%d %d\n", i, s_buf[i]);
        //if (i == 0) printf("===\n");
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
__global__ void zip_kernel(T const * const d_in1, T const * const d_in2, T * const d_out, unsigned int const n, Op const op)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        T x = d_in1[i];
        T y = d_in2[i];
        d_out[i] = op(x, y);
    }
}

template <typename T, typename Op>
void exclusive_scan(T const * const d_input, T * const d_output, unsigned int const n, Op const op, T ne = T())
{
    assert(n < 512*512);
    unsigned int bsz = min(512, n);
    unsigned int nblocks = n / bsz + (n%bsz != 0);
    
    T * d_buf;
    //T const * d_in = d_input;
    checkCudaErrors(cudaMalloc((void**)&d_buf, nblocks*sizeof(T)));
    
    exclusive_scan_kernel<<<nblocks, bsz>>>(d_input, d_output, d_buf, n, op, ne);
    if (nblocks > 1) {
        exclusive_scan_kernel<<<1, nblocks>>>(d_buf, d_buf, (T*)0, n, op, ne);
        blockwise_op_kernel<<<nblocks, bsz>>>(d_output, d_buf, d_output, n, op);
    }
    
    //for (unsigned int m = n; m > 1; m = m / bsz + (m%bsz != 0)) {
    //    reduce_kernel<<<m/bsz + (m%bsz != 0), bsz>>>(d_in, d_buf, m, op);
    //    d_in = d_buf;
    //}
    cudaDeviceSynchronize();
    
    //T res;
    //checkCudaErrors(cudaMemcpy(&res, d_buf, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(d_buf);
    //return res;
}

template <typename T, typename Op>
void inclusive_scan(T const * const d_input, T * const d_output, unsigned int const n, Op const op, T ne = T())
{
    exclusive_scan(d_input, d_output, n, op);
    
    unsigned int bsz = min(512, n);
    unsigned int nblocks = n / bsz + (n%bsz != 0);
    zip_kernel<<<nblocks, bsz>>>(d_output, d_input, d_output, n, op);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    /*
    int h_a[10000];
    int n = 1111;
    for (int i = 0; i < n; ++i) h_a[i] = i;
    int * d_a, * d_b;
    cudaMalloc((void**)&d_a, n*sizeof(int));
    cudaMalloc((void**)&d_b, n*sizeof(int));
    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    //printf("%d\n", reduce(d_a, 10000, device_plus<int>()));
    inclusive_scan(d_a, d_b, n, device_plus<int>());
    
    cudaMemcpy(h_a, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < n; ++i)
    //    printf("%d %d %d\n", i, h_a[i], i*(i+1)/2);
    cudaFree(d_a);
    cudaFree(d_b);
    
    bool f = true;
    for (int i = 0; i < n; ++i)
        f = f && h_a[i] == i*(i+1)/2;
    
    printf("%d\n", f);
        
    return;
    */
    
    
    unsigned int nsamples = numRows * numCols;
        
    min_logLum = reduce(d_logLuminance, nsamples, device_min<float>());
    max_logLum = reduce(d_logLuminance, nsamples, device_max<float>());
        
    unsigned int * d_hist;
    checkCudaErrors(cudaMalloc((void**)&d_hist, numBins * sizeof(unsigned int)));

    {
        unsigned int bsz = min(512, nsamples);
        cudaMemset(d_hist, 0, numBins * sizeof(unsigned int));
        histogram_kernel<<<nsamples/bsz + (nsamples%bsz != 0), bsz>>>(d_logLuminance, nsamples, min_logLum, max_logLum, numBins, d_hist);
        //int * h = new int[numBins];
        //cudaMemcpy(d_hist, h, nbins*sizeof(int));
        //for (int i = 0; i < nsamples; ++i)
            
        inclusive_scan(d_hist, d_cdf, numBins, device_plus<int>());
        //exclusive_scan_kernel<<<nsamples/bsz + (nsamples%bsz != 0), bsz>>>(d_hist, d_cdf, nsamples, device_plus<int>());
        //delete[] h;
    }
    
    /*
    std::vector<unsigned int> h_hist(numBins);
    checkCudaErrors(cudaMemcpy(h_hist.data(), d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < numBins; ++i)
        printf("%u\n", h_hist[i]);
    */
    
    checkCudaErrors(cudaFree(d_hist));
}
