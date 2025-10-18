#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
}

// this is the result
//  g++ test.cpp -o test `pkg-config --cflags --libs opencv4`
// ./test
// OpenCV version: 4.6.0