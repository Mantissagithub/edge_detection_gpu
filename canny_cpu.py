#!/usr/bin/env python3

import cv2
import sys
import os

def main():
    input_image_path = "kernels/assets/image1.jpeg"
    output_blurred = "kernels/assets/blurred_image_cpu.jpg"
    output_edges = "kernels/assets/final_edges_cpu.jpg"

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        sys.exit(1)

    print("=" * 60)
    print("CPU-based Canny Edge Detection (Python + OpenCV)")
    print("=" * 60)

    print("Reading image...")
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Failed to load image from {input_image_path}")
        sys.exit(1)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    print("Applying Gaussian blur...")
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

    print("Applying Canny edge detection...")
    edges = cv2.Canny(blurred, 50, 150)

    print("Writing results...")
    cv2.imwrite(output_blurred, blurred)
    cv2.imwrite(output_edges, edges)

    print("=" * 60)
    print(f"Output saved to:")
    print(f"  - {output_blurred}")
    print(f"  - {output_edges}")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    main()
