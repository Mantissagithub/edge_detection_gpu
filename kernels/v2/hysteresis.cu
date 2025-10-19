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

// optimization 1: used __restrict__ keyword for pointer parameters to avoid aliasing and enable compiler optimizations

// optimization 2: bitwise operations instead of branching in doubleThresholdKernel to avoid branch divergence

// optimization 3: shared memory tiling in edgeTrackingKernel with proper halo loading for efficient neighbor access

// optimization 4: unrolled 8-neighbor check instead of nested loops for better performance and reduced instruction overhead

// optimization 5: block-level reduction using shared memory (sModified) to minimize global atomic operations

// optimization 6: iterative edge tracking with buffer swapping to propagate strong edges through weak edge chains
using namespace std;
using namespace cv;

#define TILE_SIZE 16

// okay, this is a bit tricky, and leanred by asking perplexity, as i couldn't track, but now understood, will try to implement it by myself
// kernel 1 -> classificaton of edges into strong, mid, weak, our aim is to convert this mid things to either weak or strong based on the neighbors
__global__ void doubleThresholdKernel(unsigned char* __restrict__ inputImage, unsigned char* __restrict__ outputImage, int width, int height, unsigned char lowThresh, unsigned char highThresh){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=height) return;

  int idx = y * width + x;

  unsigned char value = inputImage[idx];

  // float value = inputImage[idx];

  unsigned char isHigh = (value >= highThresh) ? 255 : 0;
  unsigned char isMid = ((value >= lowThresh) & (value < highThresh)) ? 128 : 0;
  unsigned char result = isHigh | isMid;  // OR combines results

  outputImage[idx] = result;
  // same resulting in branch divergence, so using if-else
  // if(value >= highThresh){
  //   outputImage[idx] = 255; // strong edge
  // }else if(value >= lowThresh){
  //   outputImage[idx] = 128; // mid edge
  // }else{
  //   outputImage[idx] = 0; // weak edge
  // }
}

// kernel 2 -> edge connecting, convert mid edges to strong if connected to strong edges
__global__ void edgeTrackingKernel(unsigned char* __restrict__ edges, unsigned char* __restrict__ outputImage, int* modified, int width, int height){
  __shared__ unsigned char sTile[TILE_SIZE + 2][TILE_SIZE + 2];  // Shared memory tile
  __shared__ int sModified;

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;

  if (x < width && y < height) {
      sTile[ty][tx] = edges[y * width + x];
  } else {
      sTile[ty][tx] = 0;
  }

  // load halo (borders) - essential for neighbor checking
  if (threadIdx.x == 0 && x > 0) {
      sTile[ty][0] = edges[y * width + (x - 1)];
  }
  if (threadIdx.x == TILE_SIZE - 1 && x < width - 1) {
      sTile[ty][TILE_SIZE + 1] = edges[y * width + (x + 1)];
  }
  if (threadIdx.y == 0 && y > 0) {
      sTile[0][tx] = edges[(y - 1) * width + x];
  }
  if (threadIdx.y == TILE_SIZE - 1 && y < height - 1) {
      sTile[TILE_SIZE + 1][tx] = edges[(y + 1) * width + x];
  }

  // Load corners
  if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0) {
      sTile[0][0] = edges[(y - 1) * width + (x - 1)];
  }
  if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == 0 && x < width - 1 && y > 0) {
      sTile[0][TILE_SIZE + 1] = edges[(y - 1) * width + (x + 1)];
  }
  if (threadIdx.x == 0 && threadIdx.y == TILE_SIZE - 1 && x > 0 && y < height - 1) {
      sTile[TILE_SIZE + 1][0] = edges[(y + 1) * width + (x - 1)];
  }
  if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == TILE_SIZE - 1 && x < width - 1 && y < height - 1) {
      sTile[TILE_SIZE + 1][TILE_SIZE + 1] = edges[(y + 1) * width + (x + 1)];
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  unsigned char value = sTile[ty][tx];
  int idx = y * width + x;

  if (value != 128) {
      outputImage[idx] = value;
      return;
  }

  // fixed 8-neighbor check is faster unrolled than looped
  bool hasStrongNeighbor =
      (sTile[ty-1][tx-1] == 255) || // Top-left
      (sTile[ty-1][tx]   == 255) || // Top
      (sTile[ty-1][tx+1] == 255) || // Top-right
      (sTile[ty][tx-1]   == 255) || // Left
      (sTile[ty][tx+1]   == 255) || // Right
      (sTile[ty+1][tx-1] == 255) || // Bottom-left
      (sTile[ty+1][tx]   == 255) || // Bottom
      (sTile[ty+1][tx+1] == 255);   // Bottom-right

  if (threadIdx.x == 0 && threadIdx.y == 0) {
      sModified = 0;
  }
  __syncthreads();

  unsigned char result = hasStrongNeighbor ? 255 : 0;
  outputImage[idx] = result;

  if (hasStrongNeighbor) {
      atomicMax(&sModified, 1);  // mark block as modified
  }

  __syncthreads();

  // single thread per block updates global flag
  if (threadIdx.x == 0 && threadIdx.y == 0 && sModified) {
      atomicMax(modified, 1);
  }
}

// kernel 3 -> final cleanup, convert remaining mid edges to weak edges
__global__ void finalCleanup(unsigned char* __restrict__ edges, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=height) return;

  int idx = y * width + x;
  unsigned char value = edges[idx];

  edges[idx] = (value != 128) ? value : 0;
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

  dim3 trackBlockSize(TILE_SIZE, TILE_SIZE);
  dim3 trackGridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

  do{
    h_modified = 0;
    cudaMemcpy(d_modified, &h_modified, sizeof(int), cudaMemcpyHostToDevice);

    edgeTrackingKernel<<<trackGridSize, trackBlockSize>>>(d_edges, d_output, d_modified, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_modified, d_modified, sizeof(int), cudaMemcpyDeviceToHost);

    std::swap(d_edges, d_output);

    iterations++;
  } while(h_modified && iterations < maxIterations);

  cout<<"iterations taken for edge tracking: "<<iterations<<endl;

  finalCleanup<<<gridSize, blockSize>>>(d_edges, width, height);
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

  imwrite("../assets/blurred_image_cuda_v2.jpg", blurredImage);
  imwrite("../assets/sobel_gradient_magnitude_cuda_v2.jpg", gradMagNorm);
  imwrite("../assets/non_max_suppression_cuda_v2.jpg", nmsImage);
  imwrite("../assets/final_edges_hysteresis_cuda_v2.jpg", finalEdges);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  waitKey(0);

  return 0;
}
