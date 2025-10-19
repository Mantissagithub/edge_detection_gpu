#include "common.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

// command to compile this along with gaussian_blur.cu:
// # Compile each file to object code
// nvcc -c gaussian_blur.cu `pkg-config --cflags --libs opencv4` -o gaussian_blur.o
// nvcc -c sobel.cu `pkg-config --cflags --libs opencv4` -o sobel.o

// # Link them together
// nvcc -o sobel sobel.o gaussian_blur.o `pkg-config --cflags --libs opencv4`
// ./sobel

// optimization 1: using constant memory for sobel kernels, coz when it is in the kernel implementtion, it has so many copies and store themselves in registers which is waste of space, but wtith constnant mem, only one copy gets stores in the device dram, often backed by a constant cache that sits on-chip near the sms, making access very fast

// optimization 2: using shared memory to store tile of image being processed by a block, to reduce global memory accesses - left for you to implement, with also the 1-pixel halo memory which is just a ghost border around the tile to accomodate the 3x3 kernel

// optimzation 3: loop unrolling for the 3x3 kernel application loop, so the compiler knows to schedule instructions better

// optimization 4: boundary checks to avoid out-of-bounds memory accesses, and no else blocks to reduce divergence

// optimization 5: using faster sqrt function __fsqrt_rn for computing magnitude

// optimization 6: using __restrict__ pointers to tell the compiler that the pointers do not overlap in memory, allowing for better optimization, as in simple explanation:
// Without __restrict__, the compiler must assume pointer aliasing could occur. Consider this scenario:
// cuda
// outputImage[idx] = 0;
// float mag = gradMag[idx];  // Compiler can't optimize this!

// Without __restrict__: The compiler thinks "What if outputImage and gradMag point to overlapping memory? I must re-read gradMag[idx] from memory after writing to outputImage[idx] in case they're the same location".​
// With __restrict__: The compiler knows they're separate, so it can cache gradMag[idx] in registers and avoid redundant memory reads.

using namespace std;
using namespace cv;

__constant__ int SOBEL_GX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int SOBEL_GY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

#define TILE_SIZE 16
#define RADIUS 1
#define SHARED_SIZE (TILE_SIZE + 2 * RADIUS)

__global__ void sobelGradient(unsigned char* __restrict__ inputImage, float* __restrict__ gradMag, float* __restrict__ gradDir, int width, int height){
    __shared__ unsigned char shredMem[SHARED_SIZE][SHARED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    // Load data into shared memory with halo
    int imgX = min(max(x, 0), width - 1);
    int imgY = min(max(y, 0), height - 1);
    shredMem[ty][tx] = inputImage[imgY * width + imgX];

    // if(x<=1 || x>=width-1 || y<=1 || y>=height-1) return;

    // // Sobel kernels for edge detection
    // int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    // int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // float sumX = 0.0f;
    // float sumY = 0.0f;

    // load the halo regions
    if(threadIdx.x<RADIUS){
        // left halo
        int haloX = min(max(x-RADIUS, 0), width-1);
        shredMem[ty][tx - RADIUS] = inputImage[imgY * width + haloX];

        //right halo
        haloX = min(max(x+TILE_SIZE, 0), width-1);
        shredMem[ty][tx + TILE_SIZE] = inputImage[imgY * width + haloX];
    }

    if(threadIdx.y < RADIUS){
        // top halo
        int haloY = min(max(y-RADIUS, 0), height-1);
        shredMem[ty-RADIUS][tx] = inputImage[haloY * width + imgX];

        // bottom halo
        haloY = min(max(y+TILE_SIZE, 0), height-1);
        shredMem[ty+TILE_SIZE][tx] = inputImage[haloY * width + imgX];
    }

    __syncthreads();

    // for(int ky=-1;ky<=1;ky++){
    //   for(int kx=-1;kx<=1;kx++){
    //     int idx = (y+ky)*width + (x+kx);
    //     unsigned char pixel = inputImage[idx];

    //     sumX += pixel * Gx[ky+1][kx+1];
    //     sumY += pixel * Gy[ky+1][kx+1];
    //   }
    // }

    // float magnitude = sqrtf(sumX*sumX + sumY*sumY);

    // float direction = atan2f(sumY, sumX);

    // int idx = y*width + x;
    // gradMag[idx] = magnitude;
    // gradDir[idx] = direction;

    if(x>=1 && x<width-1 && y>=1 && y<height-1){
      float sumX = 0.0f;
      float sumY = 0.0f;

      #pragma unroll
      for(int ky=-1;ky<=1;ky++){
        for(int kx=-1;kx<=1;kx++){
          int pixel = shredMem[ty + ky][tx + kx];
          int kidx = (ky+1)*3 + (kx+1);
          sumX += pixel * SOBEL_GX[kidx];
          sumY += pixel * SOBEL_GY[kidx];
        }
      }

      float magnitude = __fsqrt_rn(sumX*sumX + sumY*sumY);
      float direction = atan2f(sumY, sumX);

      int outidx = y*width + x;
      gradMag[outidx] = magnitude;
      gradDir[outidx] = direction;
    }
}

void sobelGradientCUDA(const Mat& inputImage, Mat& gradMag, Mat& gradDir){
    int width = inputImage.cols;
    int height = inputImage.rows;

    unsigned char* d_inputImage;
    float *d_gradMag, *d_gradDir;

    size_t imgSize = width * height * sizeof(unsigned char);
    size_t gradSize = width * height * sizeof(float);

    cudaMalloc((void**)&d_inputImage, imgSize);
    cudaMalloc((void**)&d_gradMag, gradSize);
    cudaMalloc((void**)&d_gradDir, gradSize);

    cudaMemcpy(d_inputImage, inputImage.data, imgSize, cudaMemcpyHostToDevice);

    cudaMemset(d_gradMag, 0, gradSize);
    cudaMemset(d_gradDir, 0, gradSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sobelGradient<<<gridSize, blockSize>>>(d_inputImage, d_gradMag, d_gradDir, width, height);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        cout << "Sobel kernel error: " << cudaGetErrorString(err) << endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(gradMag.data, d_gradMag, gradSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradDir.data, d_gradDir, gradSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_gradMag);
    cudaFree(d_gradDir);
}

// int main(){
//   Mat inputImage = imread("assets/image1.jpeg", IMREAD_GRAYSCALE);
//   if(inputImage.empty()){
//     cout<<"Error loading image"<<endl;
//     return -1;
//   }

//   int width = inputImage.cols;
//   int height = inputImage.rows;

//   // first, gaussian blur
//   float sigma = 1.5f;
//   int kernelSize = 5;

//   vector<float> kernel = generateGaussianKernel(sigma, kernelSize);

//   Mat blurredImage(height, width, CV_8UC1);
//   gaussianBlurCUDA(inputImage, blurredImage, kernel);

//   //second, sobel gradient
//   Mat gradMag(height, width, CV_32FC1);
//   Mat gradDir(height, width, CV_32FC1);

//   sobelGradientCUDA(blurredImage, gradMag, gradDir);

//   // normalize and save results
//   Mat gradMagNorm;
//   normalize(gradMag, gradMagNorm, 0, 255, NORM_MINMAX);

//   imshow("Input Image", inputImage);
//   imshow("Blurred Image", blurredImage);
//   imshow("Sobel Gradient Magnitude", gradMagNorm);

//   imwrite("assets/blurred_image_cuda.jpg", blurredImage);
//   imwrite("assets/sobel_gradient_magnitude_cuda.jpg", gradMagNorm);
//   waitKey(0);

//   return 0;
// }