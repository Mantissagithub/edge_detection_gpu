#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__constant__ float d_kernel[256]; // assuming max kernel size of 256

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

__global__ void horizontalBlurKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=width || y>=height) return;

    int kernelRadius = kernelSize / 2;
    float sum = 0.0f;

    for(int k=0;k<kernelSize;k++){
        int offsetX = x + (k - kernelRadius);

        offsetX = max(0, min(offsetX, width - 1));

        int idx = y * width + offsetX;
        sum += inputImage[idx] * d_kernel[k];
    }

    outputImage[y*width+x] = (unsigned char)min(max(int(sum + 0.5f), 0), 255);
}

// just x, y are swapped in horizontal blue its x, and in vertical blur its y
__global__ void verticalBlurKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=width || y>=height) return;

    int kernelRadius = kernelSize / 2;
    float sum = 0.0f;

    for(int k=0;k<kernelSize;k++){
        int offsetY = y + (k - kernelRadius);

        offsetY = max(0, min(offsetY, height - 1));

        int idx = offsetY * width + x;
        sum += inputImage[idx] * d_kernel[k];
    }

    outputImage[y*width+x] = (unsigned char)min(max(int(sum + 0.5f), 0), 255);
}

void gaussianBlurCUDA(const Mat& inputImage, Mat& outputImage, const vector<float>& kernel){
    int width = inputImage.cols;
    int height = inputImage.rows;

    int kernelSize = kernel.size();

    unsigned char *d_input, *d_temp, *d_output; // so the d_temp here is for d_input->horizontalblue->d_temp->verticalblut->d_output
    size_t imageSize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, kernel.data(), kernelSize * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    horizontalBlurKernel<<<gridSize, blockSize>>>(d_input, d_temp, width, height, kernelSize);
    cudaDeviceSynchronize();
    verticalBlurKernel<<<gridSize, blockSize>>>(d_temp, d_output, width, height, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}

int main(){
    Mat inputImage = imread("assets/image1.jpeg", IMREAD_GRAYSCALE);
    if(inputImage.empty()){
        cout<<"Could not open or find the image!"<<endl;
        return -1;
    }

    Mat outputImage(inputImage.size(), inputImage.type());

    int width = inputImage.cols;
    int height = inputImage.rows;

    float sigma = 1.5f;
    int kernelSize = 3;

    vector<float> kernel = generateGaussianKernel(sigma, kernelSize);

    Mat cpuOutput(height, width, CV_8UC1);
    Mat gpuOutput(height, width, CV_8UC1);

    gaussianBlurCUDA(inputImage, gpuOutput, kernel);

    imwrite("assets/gaussian_blur_image1.png", gpuOutput);
    imshow("Input Image", inputImage);
    imshow("Gaussian Blurred Image (CUDA)", gpuOutput);
    waitKey(0);
}
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

