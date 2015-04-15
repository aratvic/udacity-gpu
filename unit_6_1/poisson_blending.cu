//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */


#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

#define THREADS_PER_DIM 32
#define THREADS_PER_BLOCK (THREADS_PER_DIM*THREADS_PER_DIM)
#define NITER 800

unsigned int num_blocks(unsigned int n, unsigned int tpb)
{
    unsigned int res = n/tpb;
    res += (res*tpb < n);
    return res;
}

__device__
bool is_mask_color(uchar4 const color, uchar3 mask)
{
    return color.x == mask.x && color.y == mask.y && color.z == mask.z;
}

template <typename T>
__device__
void load_tile(T const * d_src, T s_dst[][THREADS_PER_DIM+3], uint2 const sz, T const def)
{
    unsigned int const x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int const y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < sz.x && y < sz.y) {
        s_dst[threadIdx.y+1][threadIdx.x+1] = d_src[x+y*sz.x];
        if (x > 0 && threadIdx.x == 0) {
            s_dst[threadIdx.y+1][0] = d_src[x-1+y*sz.x];
        } else if (x == 0) {
            s_dst[threadIdx.y+1][0] = def;
        }

        if (x < sz.x-1 && threadIdx.x == blockDim.x-1) {
            s_dst[threadIdx.y+1][blockDim.x+1] = d_src[x+1+y*sz.x];
        } else if (x == sz.x-1) {
            s_dst[threadIdx.y+1][threadIdx.x+2] = def;
        }

        if (y > 0 && threadIdx.y == 0) {
            s_dst[0][threadIdx.x+1] = d_src[x+(y-1)*sz.x];
        } else if (y == 0) {
            s_dst[0][threadIdx.x+1] = def;
        }

        if (y < sz.y-1 && threadIdx.y == blockDim.y-1) {
            s_dst[blockDim.y+1][threadIdx.x+1] = d_src[x+(y+1)*sz.x];
        } else if (y == sz.y-1) {
            s_dst[threadIdx.y+2][threadIdx.x+1] = def;
        }
    }
}

__device__
float3 f3_from_uc4(uchar4 src)
{
    float3 res;
    res.x = src.x;
    res.y = src.y;
    res.z = src.z;
    return res;
}

__device__
uchar4 uc4_from_f3(float3 src)
{
    uchar4 res;
    res.x = src.x;
    res.y = src.y;
    res.z = src.z;
    res.w = 255u;
    return res;
}

__device__
float3 clamp3(float3 f)
{
    float3 res;
    res.x = min(255.f, max(0.f, f.x));
    res.y = min(255.f, max(0.f, f.y));
    res.z = min(255.f, max(0.f, f.z));
    return res;
}


__global__
void preproc_kernel(uchar4 const * d_src, uchar4 const * d_dst, unsigned char * d_inner, float3 * d_laplace, float3 * d_buf_a, float3 * d_buf_b, uint2 const sz, uchar3 const mask_color)
{
    assert(blockDim.x == THREADS_PER_DIM && blockDim.y == THREADS_PER_DIM);

    unsigned int const x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int const y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int i = x+y*sz.x;
    
    __shared__ uchar4 s_src[THREADS_PER_DIM+2][THREADS_PER_DIM+3];
    uchar4 def = {0u, 0u, 0u, 0u};
    load_tile(d_src, s_src, sz, def);
    __syncthreads();

    unsigned int sx = threadIdx.x + 1, sy = threadIdx.y + 1;
    uchar4 const c = s_src[sy][sx];
    uchar4 const l = s_src[sy][sx-1];
    uchar4 const r = s_src[sy][sx+1];
    uchar4 const u = s_src[sy-1][sx];
    uchar4 const d = s_src[sy+1][sx];

    if (x >= sz.x || y >= sz.y) return;
    
    unsigned int is_mask = is_mask_color(c, mask_color);
    unsigned int has_outer_neighbor =
        is_mask_color(l, mask_color) |
        is_mask_color(r, mask_color) |
        is_mask_color(u, mask_color) |
        is_mask_color(d, mask_color);
    unsigned int is_inner = !is_mask & !has_outer_neighbor;
    unsigned int is_border = !is_mask & has_outer_neighbor;
    
    d_inner[i] = is_inner;

    if (is_mask) return;
    
    float3 dst = f3_from_uc4(d_dst[i]);
    if (is_inner) {
        d_buf_a[i] = d_buf_b[i] = f3_from_uc4(c);
    } else if (is_border) {
        d_buf_a[i] = d_buf_b[i] = dst;
    }

    if (!is_inner) return;
    
    // Laplacian of src
    float3 laplace;
    laplace.x = -4.f*c.x + l.x + r.x + d.x + u.x;
    laplace.y = -4.f*c.y + l.y + r.y + d.y + u.y;
    laplace.z = -4.f*c.z + l.z + r.z + d.z + u.z;
    d_laplace[i] = laplace;
}

__device__
void add3(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__global__
void jacobi_iter(float3 * d_buf_src, float3 * d_buf_dst, float3 const * d_laplace, unsigned char const * d_inner, uint2 const sz)
{
    assert(blockDim.x == THREADS_PER_DIM && blockDim.y == THREADS_PER_DIM);

    unsigned int const x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int const y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float3 s_buf_src[THREADS_PER_DIM+2][THREADS_PER_DIM+3];
    float3 const def_f3 = {0.f, 0.f, 0.f};
    load_tile(d_buf_src, s_buf_src, sz, def_f3);
    __syncthreads();

    if (x >= sz.x || y >= sz.y) return;
    
    unsigned int i = x + y*sz.x;
    unsigned int sx = threadIdx.x + 1, sy = threadIdx.y + 1;

    if (!d_inner[i]) return;
    
    float3 sum_a = {0.f, 0.f, 0.f};
    add3(sum_a, s_buf_src[sy][sx-1]);
    add3(sum_a, s_buf_src[sy][sx+1]);
    add3(sum_a, s_buf_src[sy-1][sx]);
    add3(sum_a, s_buf_src[sy+1][sx]);
    
    float3 laplace = d_laplace[i];
    float3 res;
    res.x = (sum_a.x - laplace.x) / 4.f;
    res.y = (sum_a.y - laplace.y) / 4.f;
    res.z = (sum_a.z - laplace.z) / 4.f;
    d_buf_dst[i] = clamp3(res);
}


__global__
void copy_back(float3 const * d_buf, uchar4 * d_out, unsigned char const * d_inner, uint2 const sz)
{
    unsigned int const x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int const y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = x + y * sz.x;
    if (x < sz.x && y < sz.y && d_inner[i]) {
        d_out[i] = uc4_from_f3(d_buf[i]);
    }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

    /* To Recap here are the steps you need to implement
    
       1) Compute a mask of the pixels from the source image to be copied
          The pixels that shouldn't be copied are completely white, they
          have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

       2) Compute the interior and border regions of the mask.  An interior
          pixel has all 4 neighbors also inside the mask.  A border pixel is
          in the mask itself, but has at least one neighbor that isn't.

       3) Separate out the incoming image into three separate channels

       4) Create two float(!) buffers for each color channel that will
          act as our guesses.  Initialize them to the respective color
          channel of the source image since that will act as our intial guess.

       5) For each color channel perform the Jacobi iteration described 
          above 800 times.

       6) Create the output image by replacing all the interior pixels
          in the destination image with the result of the Jacobi iterations.
          Just cast the floating point values to unsigned chars since we have
          already made sure to clamp them to the correct range.

        Since this is final assignment we provide little boilerplate code to
        help you.  Notice that all the input/output pointers are HOST pointers.

        You will have to allocate all of your own GPU memory and perform your own
        memcopies to get data in and out of the GPU memory.

        Remember to wrap all of your calls with checkCudaErrors() to catch any
        thing that might go wrong.  After each kernel call do:

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        to catch any errors that happened while executing the kernel.
    */

    uchar3 const mask_color = {255, 255, 255};

    unsigned char * d_inner;
    uchar4 * d_src, * d_dst, * d_out;
    float3 * d_laplace, * d_buf[2];

    unsigned int const lin_sz = numRowsSource * numColsSource;
    unsigned int const mask_sz = lin_sz * sizeof(unsigned char);
    unsigned int const img_sz = lin_sz * sizeof(uchar4);
    unsigned int const buf_sz = lin_sz * sizeof(float3);
    
    checkCudaErrors(cudaMalloc((void**)&d_inner, mask_sz));
    checkCudaErrors(cudaMalloc((void**)&d_src, img_sz));
    checkCudaErrors(cudaMalloc((void**)&d_dst, img_sz));
    checkCudaErrors(cudaMalloc((void**)&d_out, img_sz));
    checkCudaErrors(cudaMalloc((void**)&d_laplace, buf_sz));
    checkCudaErrors(cudaMalloc((void**)&d_buf[0], buf_sz));
    checkCudaErrors(cudaMalloc((void**)&d_buf[1], buf_sz));
    checkCudaErrors(cudaMemcpy(d_src, h_sourceImg, img_sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dst, h_destImg, img_sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_out, h_destImg, img_sz, cudaMemcpyHostToDevice));

    uint2 const sz = {numColsSource, numRowsSource};
    dim3 const block_dim = dim3(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 const grid_dim = dim3(num_blocks(numColsSource, block_dim.x),
                               num_blocks(numRowsSource, block_dim.y));

    preproc_kernel<<<grid_dim, block_dim>>>(d_src, d_dst, d_inner, d_laplace, d_buf[0], d_buf[1], sz, mask_color);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    int k = 0;
    for (int i = 0; i < NITER; ++i)
    {
        jacobi_iter<<<grid_dim, block_dim>>>(d_buf[k], d_buf[k^1], d_laplace, d_inner, sz);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        k ^= 1;
    }

    copy_back<<<grid_dim, block_dim>>>(d_buf[k], d_out, d_inner, sz);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_out, img_sz, cudaMemcpyDeviceToHost));

    cudaFree(d_src); cudaFree(d_dst);
    cudaFree(d_inner); cudaFree(d_out);
    cudaFree(d_laplace);
    cudaFree(d_buf[0]); cudaFree(d_buf[1]);
    
    /* The reference calculation is provided below, feel free to use it
       for debugging purposes. 
     */

    /*
    unsigned int const srcSize = lin_sz;
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; 
    */
}
