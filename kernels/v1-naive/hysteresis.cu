// this is the basic operation
// Example:
// [weak] [weak] [strong]  →  [edge] [edge] [edge]  (all kept, connected to strong)
// [weak] [weak] [  0   ]  →  [  0 ] [  0 ] [  0  ]  (all discarded, no strong neighbor)
// Rule of thumb: highThresh = 2 × lowThresh or highThresh = 3 × lowThresh

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "common.h"

// command to comile: nvcc -o canny hysteresis.cu gaussian_blur.cu sobel.cu nms.cu `pkg-config --cflags --libs opencv4` -diag-suppress 611
using namespace std;
using namespace cv;

// okay, this is a bit tricky, and leanred by asking perplexity, as i couldn't track, but now understood, will try to implement it by myself
// kernel 1 -> classificaton of edges into strong, mid, weak, our aim is to convert this mid things to either weak or strong based on the neighbors
__global__ void doubleThresholdKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, float lowThresh, float highThresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=height) return;

  int idx = y * width + x;
  float value = inputImage[idx];

  if(value >= highThresh){
    outputImage[idx] = 255; // strong edge
  }else if(value >= lowThresh){
    outputImage[idx] = 128; // mid edge
  }else{
    outputImage[idx] = 0; // weak edge
  }
}

// kernel 2 -> edge connecting, convert mid edges to strong if connected to strong edges
__global__ void edgeTrackingKernel(unsigned char* edges, unsigned char* outputImage, int* modified, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=width) return;

  int idx = y * width + x;
  if(edges[idx] !=128){
    outputImage[idx] = edges[idx];
    return;
  }

  bool hasStringNeighbot = false;

  for(int dy=-1;dy>=1;dy++){
    for(int dx=-1;dx>=1;dx++){
      if(dx==0 && dy==0) continue;
      int nx = x + dx;
      int ny = y + dy;
      if(nx>=0 && nx<width && ny>=0 && ny<height){
        int nIdx = ny * width + nx;
        if(edges[nIdx] == 255){
          hasStringNeighbot = true;
          break;
        }
      }
    }
    if(hasStringNeighbot) break;
  }

  if(hasStringNeighbot){
    outputImage[idx] = 255;
    *modified = 1;
  }else{
    outputImage[idx] = 0;
  }
}

// kernel 3 -> final cleanup, convert remaining mid edges to weak edges
__global__ void finaCleanup(unsigned char* edges, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=height) return;

  int idx = y * width + x;
  if(edges[idx] == 128){
    edges[idx] = 0;
  }
}

void hysteresisThresoldCUDA(const Mat& inputImage, Mat& outputImage, float lowThresh, float highThresh){
  int width = inputImage.cols;
  int height = inputImage.rows;

  unsigned char *d_input, *d_edges, *d_output;
  int *d_modified, h_modified;

  size_t imgSize = width * height * sizeof(unsigned char);

  cudaMalloc((void**)&d_input, imgSize);
  cudaMalloc((void**)&d_edges, imgSize);
  cudaMalloc((void**)&d_output, imgSize);
  cudaMalloc((void**)&d_modified, sizeof(int));

  cudaMemcpy(d_input, inputImage.data, imgSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  doubleThresholdKernel<<<gridSize, blockSize>>>(d_input, d_edges, width, height, lowThresh, highThresh);
  cudaDeviceSynchronize();

  int iterations = 0;
  int maxIterations = 100;

  do{
    h_modified = 0;
    cudaMemcpy(d_modified, &h_modified, sizeof(int), cudaMemcpyHostToDevice);

    edgeTrackingKernel<<<gridSize, blockSize>>>(d_edges, d_output, d_modified, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_modified, d_modified, sizeof(int), cudaMemcpyDeviceToHost);

    // Swap edges and output pointers for next iteration
    std::swap(d_edges, d_output);

    iterations++;
  } while(h_modified && iterations < maxIterations);

  cout<<"iterations taken for edge tracking: "<<iterations<<endl;

  finaCleanup<<<gridSize, blockSize>>>(d_edges, width, height);
  cudaDeviceSynchronize();


  cudaMemcpy(outputImage.data, d_edges, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_edges);
  cudaFree(d_output);
  cudaFree(d_modified);
}

int main(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float kernelTime = 0;

  cudaEventRecord(start);

  Mat inputImage = imread("../assets/image1.jpeg", IMREAD_GRAYSCALE);
  if(inputImage.empty()){
    cout<<"Error loading image"<<endl;
    return -1;
  }

  int width = inputImage.cols;
  int height = inputImage.rows;

  // first, gaussian blur
  float sigma = 1.5f;
  int kernelSize = 5;

  vector<float> kernel = generateGaussianKernel(sigma, kernelSize);

  Mat blurredImage(height, width, CV_8UC1);
  gaussianBlurCUDA(inputImage, blurredImage, kernel);

  //second, sobel gradient
  Mat gradMag(height, width, CV_32FC1);
  Mat gradDir(height, width, CV_32FC1);

  sobelGradientCUDA(blurredImage, gradMag, gradDir);

  // normalize and save results
  Mat gradMagNorm;
  normalize(gradMag, gradMagNorm, 0, 255, NORM_MINMAX);

  // third, non-maximum suppression
  Mat nmsImage(height, width, CV_8UC1);
  nonMaxSuppressionCUDA(gradMag, gradDir, nmsImage);

  // fourth, hysteresis thresholding
  Mat finalEdges(height, width, CV_8UC1);
  float lowThresh = 50.0f;
  float highThresh = 150.0f;
  hysteresisThresoldCUDA(nmsImage, finalEdges, lowThresh, highThresh);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernelTime, start, stop);

  cout << "Total CUDA kernel execution time: " << kernelTime << " ms" << endl;

  // imshow("Input Image", inputImage);
  // imshow("Blurred Image", blurredImage);
  // imshow("Sobel Gradient Magnitude", gradMagNorm);
  // imshow("Non-Maximum Suppression Result", nmsImage);
  // imshow("Final Edges after Hysteresis Thresholding", finalEdges);

  imwrite("../assets/blurred_image_cuda.jpg", blurredImage);
  imwrite("../assets/sobel_gradient_magnitude_cuda.jpg", gradMagNorm);
  imwrite("../assets/non_max_suppression_cuda.jpg", nmsImage);
  imwrite("../assets/final_edges_hysteresis_cuda.jpg", finalEdges);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  waitKey(0);

  return 0;
}
