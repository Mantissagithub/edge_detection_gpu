#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "common.h"

// command to compile: nvcc -o gaussian_blur gaussian_blur.cu `pkg-config --cflags --libs opencv4` -diag-suppress 611
// ./gaussian_blur

 // optmiization 1: obviously shared mem tiling and __restrict__ keyword for pointer parameters to avoid unnecessary loads and stores

 // optmiization 2: unrolling the kernel loops for both horizontal and vertical blur kernels

 // optimization 3:

using namespace std;
using namespace cv;

__constant__ float d_kernel[64]; // reducing from 256 to 64 assuming max kernel size is 64

#define TILE_WIDTH 128 // threads per block for horizonatal blur
#define TILE_HEIGHT 16

vector<float> generateGaussianKernel(float sigma, int kernelSize) {
    int kernelRadius = kernelSize / 2;
    vector<float> kernel(kernelSize);
    float multiplier = 1.0f / (sqrt(2.0f * M_PI) * sigma);

    for(int i = 0; i < kernelSize; i++) {
        float distance = i - kernelRadius;
        float exponent = -(distance * distance) / (2.0f * sigma * sigma);
        kernel[i] = multiplier * exp(exponent);
    }

    float sum = accumulate(kernel.begin(), kernel.end(), 0.0f);
    for(int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

__global__ void horizontalBlurKernel(unsigned char* __restrict__ inputImage, unsigned char* __restrict__ outputImage, int width, int height, int kernelSize){
    __shared__ unsigned char sharedRow[TILE_HEIGHT][TILE_WIDTH + 64]; // extra 64 for halo pixels

    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y;

    if(y>=height) return;

    int kernelRadius = kernelSize / 2;
    int sharedidx = threadIdx.x + kernelRadius;

    if(x<width){
      sharedRow[threadIdx.y][sharedidx] = inputImage[y * width + x];
    }else{
      sharedRow[threadIdx.y][sharedidx] = 0;
    }

    if(threadIdx.x < kernelRadius){
      // left halo
      int haloX = x - kernelRadius;
      sharedRow[threadIdx.y][threadIdx.x] = (haloX >=0) ? inputImage[y * width + haloX] : 0;

      // right halo
      haloX = x + TILE_WIDTH;
      sharedRow[threadIdx.y][sharedidx + TILE_WIDTH] = (haloX < width) ? inputImage[y * width + haloX] : 0;
    }

    __syncthreads();

    if(x>=width) return;

    float sum = 0.0f;

    #pragma unroll 8
    for(int k=0;k<kernelSize;k++){
      sum += sharedRow[threadIdx.y][sharedidx - kernelRadius + k] * d_kernel[k];
    }

    outputImage[y*width+x] = __float2uint_rn(__saturatef(sum/255.0f) * 255.0f);

    // for(int k=0;k<kernelSize;k++){
    //     int offsetX = x + (k - kernelRadius);

    //     offsetX = max(0, min(offsetX, width - 1));

    //     int idx = y * width + offsetX;
    //     sum += inputImage[idx] * d_kernel[k];
    // }

    // outputImage[y*width+x] = (unsigned char)min(max(int(sum + 0.5f), 0), 255);
}

// just x, y are swapped in horizontal blue its x, and in vertical blur its y
__global__ void verticalBlurKernel(unsigned char* __restrict__ inputImage, unsigned char* __restrict__ outputImage, int width, int height, int kernelSize){
    __shared__ unsigned char sharedCol[TILE_HEIGHT + 64][TILE_WIDTH]; // extra 64 for halo pixels

    int x = blockIdx.x;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    if(x>=width) return;

    int kernelRadius = kernelSize / 2;
    int sharedidy = threadIdx.y + kernelRadius;

    if(y<height){
      sharedCol[sharedidy][threadIdx.x] = inputImage[y * width + x];
    }else{
      sharedCol[sharedidy][threadIdx.x] = 0;
    }

    if(threadIdx.y < kernelRadius){
      // top halo
      int haloY = y - kernelRadius;
      sharedCol[threadIdx.y][threadIdx.x] = (haloY >=0) ? inputImage[haloY * width + x] : 0;

      // bottom halo
      haloY = y + TILE_HEIGHT;
      sharedCol[sharedidy + TILE_HEIGHT][threadIdx.x] = (haloY < height) ? inputImage[haloY * width + x] : 0;
    }

    __syncthreads();

    if(y>=height || x>=width) return;

    float sum = 0.0f;

    #pragma unroll 8 // unroll for smaller kernel sizes
    for(int k=0;k<kernelSize;k++){
      sum += sharedCol[sharedidy - kernelRadius + k][threadIdx.x] * d_kernel[k];
    }

    // using the saturate arithmetic to clamp values between 0 and 255
    outputImage[y*width+x] = __float2uint_rn(__saturatef(sum/255.0f) * 255.0f);

    // for(int k=0;k<kernelSize;k++){
    //     int offsetY = y + (k - kernelRadius);

    //     offsetY = max(0, min(offsetY, height - 1));

    //     int idx = offsetY * width + x;
    //     sum += inputImage[idx] * d_kernel[k];
    // }

    // outputImage[y*width+x] = (unsigned char)min(max(int(sum + 0.5f), 0), 255);
}

void gaussianBlurCUDA(const Mat& inputImage, Mat& outputImage, const vector<float>& kernel){
    int width = inputImage.cols;
    int height = inputImage.rows;

    int kernelSize = kernel.size();

    unsigned char *d_input, *d_temp, *d_output;
    size_t imageSize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, kernel.data(), kernelSize * sizeof(float));

    dim3 hBlockSize(TILE_WIDTH, 1);
    dim3 hGridSize((width + TILE_WIDTH - 1) / TILE_WIDTH, height);

    dim3 vBlockSize(1, TILE_HEIGHT);
    dim3 vGridSize(width, (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    horizontalBlurKernel<<<hGridSize, hBlockSize>>>(d_input, d_temp, width, height, kernelSize);
    cudaDeviceSynchronize();
    verticalBlurKernel<<<vGridSize, vBlockSize>>>(d_temp, d_output, width, height, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

// int main(){
//     Mat inputImage = imread("assets/image1.jpeg", IMREAD_GRAYSCALE);
//     if(inputImage.empty()){
//         cout<<"Could not open or find the image!"<<endl;
//         return -1;
//     }

//     Mat outputImage(inputImage.size(), inputImage.type());

//     int width = inputImage.cols;
//     int height = inputImage.rows;

//     float sigma = 1.5f;
//     int kernelSize = 3;

//     vector<float> kernel = generateGaussianKernel(sigma, kernelSize);

//     Mat cpuOutput(height, width, CV_8UC1);
//     Mat gpuOutput(height, width, CV_8UC1);

//     gaussianBlurCUDA(inputImage, gpuOutput, kernel);

//     imwrite("assets/gaussian_blur_image1.png", gpuOutput);
//     imshow("Input Image", inputImage);
//     imshow("Gaussian Blurred Image (CUDA)", gpuOutput);
//     waitKey(0);
// }
// these are the cpp functions i wrote to just learn what to do, the above the kernel implementations of the same
// vector<float> generateGaussianKernel(float sigma, int kernelSize) {
//     int kernelRadius = kernelSize / 2;
//     vector<float> kernel(kernelSize);
//     float multiplier = 1.0f / (sqrt(2.0f * M_PI) * sigma);

//     for(int i = 0; i < kernelSize; i++) {
//         float distance = i - kernelRadius;
//         float exponent = -(distance * distance) / (2.0f * sigma * sigma);
//         kernel[i] = multiplier * exp(exponent);
//     }

//     float sum = accumulate(kernel.begin(), kernel.end(), 0.0f);
//     for(int i = 0; i < kernelSize; i++) {
//         kernel[i] /= sum;
//     }

//     return kernel;
// }

// void horizontalBlur(const Mat& inputImage, Mat& outputImage,
//                     const vector<float>& kernel, int width, int height){
//     int kernelSize = kernel.size();
//     int kernelRadius = kernelSize/2;

//     for(int y=0; y<height; y++){
//         for(int x=0; x<width; x++){
//             float sum = 0.0f;

//             for(int k=0; k<kernelSize; k++){
//                 int offsetX = x + (k - kernelRadius);

//                 if(offsetX < 0) offsetX = 0;
//                 if(offsetX >= width) offsetX = width - 1;

//                 uchar pixelValue = inputImage.at<uchar>(y, offsetX);
//                 sum += pixelValue * kernel[k];
//             }

//             outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum); // FIXED
//         }
//     }
// }

// void verticalBlur(const Mat& inputImage, Mat& outputImage,
//                   const vector<float>& kernel, int width, int height){
//     int kernelSize = kernel.size();
//     int kernelRadius = kernelSize/2;

//     for(int y=0; y<height; y++){
//         for(int x=0; x<width; x++){
//             float sum = 0.0f;

//             for(int k=0; k<kernelSize; k++){
//                 int offsetY = y + (k - kernelRadius);

//                 if(offsetY < 0) offsetY = 0;
//                 if(offsetY >= height) offsetY = height - 1;

//                 uchar pixelValue = inputImage.at<uchar>(offsetY, x);
//                 sum += pixelValue * kernel[k];
//             }

//             outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum); // FIXED
//         }
//     }
// }

