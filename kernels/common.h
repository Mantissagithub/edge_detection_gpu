// this is going to be the common file, as need to import and implement it as a pipeline of different kernels

#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

vector<float> generateGaussianKernel(float sigma, int kernelSize);
void gaussianBlurCUDA(const Mat& inputImage, Mat& outputImage,
                      const vector<float>& kernel);

void sobelGradientCUDA(const Mat& inputImage, Mat& gradMag, Mat& gradDir);

void nonMaxSuppressionCUDA(const Mat& gradMag, const Mat& gradDir, Mat& output);

#endif // COMMON_H

