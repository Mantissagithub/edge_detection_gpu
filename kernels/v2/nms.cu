// this is the third step now
// non-maximum suppression implementation in CUDA


#include "common.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

// command to compile alternative as not in sobel to, this is easier but hard to remember: nvcc -o nms nms.cu gaussian_blur.cu sobel.cu `pkg-config --cflags --libs opencv4` -diag-suppress 611

// optimization 1: multiple if-else cause massive warp divergence, as in a single warp (32 threads), diff threads will likely take diff branches among the 4, forcing the serialization of execution, so we can reduce the divergence by using only if statements and no else statements

// optimization 2: shared tiling mem for gradmag, as it is accessed multiple times, but gradDir is accessed only once per pixel, so no need to store in shared mem, decreasing the amt of global mem reads, which are slow

// optimization 3: converting angle form rad to deg is bs, can work with radians directly

// optimization 4: precompute the index offsets for the 4 directions, to avoid recomputing them for every pixel, and store them like a lookup table

// optimization 5: instead of if-else, we can use directly predicate (ternanry) operators to reduce branching, modern gpus support most of them efficiently

// optimization 6: use __restrict__ keyword to tell the compiler that pointers do not alias, enabling better optimizations, refer sobel.cu for a better explanation

using namespace std;
using namespace cv;

#define TILE_SIZE 16
#define SHARED_SIZE (TILE_SIZE + 2) // 1 pixel halo

__global__ void nonMaxSuppression(const float* __restrict__ gradMag, const float* __restrict__ gradDir, unsigned char* __restrict__ outputImage, int width, int height){
  __shared__ float sharedGMag[SHARED_SIZE][SHARED_SIZE];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;

  if(x<width && y<height){
    int idx = y*width+x;
    sharedGMag[ty][tx] = gradMag[idx];
  }else{
    sharedGMag[ty][tx] = 0.0f;
  }

  if(threadIdx.x==0 && x>0 && y<height){
    sharedGMag[ty][0] = gradMag[y*width + (x-1)];
  }
  if(threadIdx.x==blockDim.x-1 && x<width-1 && y<height){
    sharedGMag[ty][tx+1] = gradMag[y*width + (x+1)];
  }
  if(threadIdx.y==0 && y>0 && x<width){
    sharedGMag[0][tx] = gradMag[(y-1)*width + x];
  }
  if(threadIdx.y==blockDim.y-1 && y<height-1 && x<width){
    sharedGMag[ty+1][tx] = gradMag[(y+1)*width + x];
  }

  __syncthreads();

  if(x<1 || x>=width-1 || y<1 || y>=height-1){
    if(x<width && y<height){
      outputImage[y*width + x] = 0;
    }
    return;
  }

  int idx = y * width + x;
  float angle = gradDir[idx];
  float mag = sharedGMag[ty][tx];

  // normalize angle to [0, pi]
  if(angle < 0.0f) angle += M_PI;

  // mapping the angle to direction 0-horizontal, 1-45deg, 2-vertical, 3-135deg
  int dir = (int)((angle + M_PI/8) / (M_PI/4)) % 4;

  int2 os;
  if(dir == 0) os = make_int2(-1, 0);   // horizontal
  else if(dir == 1) os = make_int2(-1, -1);  // 45 deg
  else if(dir == 2) os = make_int2(0, -1);   // vertical
  else os = make_int2(-1, 1);   // 135

  float mag1 = sharedGMag[ty + os.y][tx + os.x];
  float mag2 = sharedGMag[ty - os.y][tx - os.x];

  unsigned char res = (mag >= mag1 && mag >= mag2) ? (unsigned char)fminf(mag, 255.0f) : 0;

  outputImage[idx] = res;

  // angle = angle * 180.0 / M_PI; // convert to degrees
  // if(angle < 0) angle += 180.0f;

  // int idx1, idx2; // for checking the dir

  // if((angle>=0 && angle<22.5) || (angle>=157.7 && angle<=180)){
  //   // o degrees -> horozontal check- left and right
  //   idx1 = y*width + (x-1);
  //   idx2 = y*width + (x+1);
  // }else if(angle>=22.5 && angle<67.5){
  //   // 45 degrees -> check top-right and bottom-left
  //   idx1 = (y-1)*width + (x+1);
  //   idx2 = (y+1)*width + (x-1);
  // }else if(angle>=67.5 && angle<112.5){
  //   // 90 degrees -> vertical check - top and bottom
  //   idx1 = (y-1)*width + x;
  //   idx2 = (y+1)*width + x;
  // }else{
  //   // 135 degrees -> check top-left and bottom-right
  //   idx1 = (y-1)*width + (x-1);
  //   idx2 = (y+1)*width + (x+1);
  // }

  // float mag1 = gradMag[idx1];
  // float mag2 = gradMag[idx2];

  // if(mag >= mag1 && mag >= mag2){
  //   outputImage[idx] = (unsigned char)min(mag, 255.0f);
  // }else{
  //   outputImage[idx] = 0;
  // }
}

void nonMaxSuppressionCUDA(const Mat& gradMag, const Mat& gradDir, Mat& outputImage){
  int width = outputImage.cols;
  int height = outputImage.rows;

  float *d_gradMag, *d_gradDir;
  unsigned char *d_outputImage;

  size_t floatSize = width * height * sizeof(float);
  size_t ucharSize = width * height * sizeof(unsigned char);

  cudaMalloc((void**)&d_gradMag, floatSize);
  cudaMalloc((void**)&d_gradDir, floatSize);
  cudaMalloc((void**)&d_outputImage, ucharSize);

  cudaMemcpy(d_gradMag, gradMag.data, floatSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradDir, gradDir.data, floatSize, cudaMemcpyHostToDevice);
  cudaMemset(d_outputImage, 0, ucharSize);


  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  nonMaxSuppression<<<gridSize, blockSize>>>(d_gradMag, d_gradDir, d_outputImage, width, height);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
      cout << "NMS kernel error: " << cudaGetErrorString(err) << endl;
  }
  cudaDeviceSynchronize();

  cudaMemcpy(outputImage.data, d_outputImage, ucharSize, cudaMemcpyDeviceToHost);

  cudaFree(d_gradMag);
  cudaFree(d_gradDir);
  cudaFree(d_outputImage);
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

//   // third, non-maximum suppression
//   Mat nmsImage(height, width, CV_8UC1);
//   nonMaxSuppressionCUDA(gradMag, gradDir, nmsImage);

//   imshow("Input Image", inputImage);
//   imshow("Blurred Image", blurredImage);
//   imshow("Sobel Gradient Magnitude", gradMagNorm);
//   imshow("Non-Maximum Suppression Result", nmsImage);

//   imwrite("assets/blurred_image_cuda.jpg", blurredImage);
//   imwrite("assets/sobel_gradient_magnitude_cuda.jpg", gradMagNorm);
//   imwrite("assets/non_max_suppression_cuda.jpg", nmsImage);
//   waitKey(0);

//   return 0;
// }