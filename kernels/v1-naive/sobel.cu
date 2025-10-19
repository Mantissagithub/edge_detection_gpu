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

using namespace std;
using namespace cv;

__global__ void sobelGradient(unsigned char* inputImage, float* gradMag, float* gradDir, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<=1 || x>=width-1 || y<=1 || y>=height-1) return;

    // Sobel kernels for edge detection
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float sumX = 0.0f;
    float sumY = 0.0f;

    for(int ky=-1;ky<=1;ky++){
      for(int kx=-1;kx<=1;kx++){
        int idx = (y+ky)*width + (x+kx);
        unsigned char pixel = inputImage[idx];

        sumX += pixel * Gx[ky+1][kx+1];
        sumY += pixel * Gy[ky+1][kx+1];
      }
    }

    float magnitude = sqrtf(sumX*sumX + sumY*sumY);

    float direction = atan2f(sumY, sumX);

    int idx = y*width + x;
    gradMag[idx] = magnitude;
    gradDir[idx] = direction;
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