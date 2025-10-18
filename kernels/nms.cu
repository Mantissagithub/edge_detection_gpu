// this is the third step now
// non-maximum suppression implementation in CUDA


#include "common.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

// command to compile alternative as not in sobel to, this is easier but hard to remember: nvcc -o nms nms.cu gaussian_blur.cu sobel.cu `pkg-config --cflags --libs opencv4` -diag-suppress 611
using namespace std;
using namespace cv;

__global__ void nonMaxSuppression(float *gradMag, float *gradDir, unsigned char* outputImage, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x<1 || x>=width-1 || y<1 || y>=height-1){
    if(x<width && y<height){
      outputImage[y*width + x] = 0;
    }
    return;
  }

  int idx = y * width + x;
  float angle = gradDir[idx];
  float mag = gradMag[idx];

  angle = angle * 180.0 / M_PI; // convert to degrees
  if(angle < 0) angle += 180.0f;

  int idx1, idx2; // for checking the dir

  if((angle>=0 && angle<22.5) || (angle>=157.7 && angle<=180)){
    // o degrees -> horozontal check- left and right
    idx1 = y*width + (x-1);
    idx2 = y*width + (x+1);
  }else if(angle>=22.5 && angle<67.5){
    // 45 degrees -> check top-right and bottom-left
    idx1 = (y-1)*width + (x+1);
    idx2 = (y+1)*width + (x-1);
  }else if(angle>=67.5 && angle<112.5){
    // 90 degrees -> vertical check - top and bottom
    idx1 = (y-1)*width + x;
    idx2 = (y+1)*width + x;
  }else{
    // 135 degrees -> check top-left and bottom-right
    idx1 = (y-1)*width + (x-1);
    idx2 = (y+1)*width + (x+1);
  }

  float mag1 = gradMag[idx1];
  float mag2 = gradMag[idx2];

  if(mag >= mag1 && mag >= mag2){
    outputImage[idx] = (unsigned char)min(mag, 255.0f);
  }else{
    outputImage[idx] = 0;
  }
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