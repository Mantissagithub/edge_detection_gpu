#!/bin/bash

# Shell script to compile and execute CUDA Canny Edge Detection
# Measures compilation time and execution time separately

set -e  # Exit on error

echo "============================================================"
echo "GPU-based Canny Edge Detection (CUDA)"
echo "============================================================"

# Change to kernels directory
cd "$(dirname "$0")/kernels"

# Output executable name
OUTPUT_EXEC="canny_cuda"

# CUDA files to compile
CUDA_FILES="gaussian_blur.cu sobel.cu nms.cu hysteresis.cu"

# Compilation flags
NVCC_FLAGS="-O3"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"

echo ""
echo "Starting compilation..."
echo "------------------------------------------------------------"

# Start compilation timing
COMPILE_START=$(date +%s.%N)

# Compile CUDA kernels
nvcc $NVCC_FLAGS $CUDA_FILES $OPENCV_FLAGS -o $OUTPUT_EXEC

# End compilation timing
COMPILE_END=$(date +%s.%N)
COMPILE_TIME=$(echo "$COMPILE_END - $COMPILE_START" | bc)

echo "Compilation completed successfully!"
echo "Compilation time: ${COMPILE_TIME} seconds"
echo ""

echo "------------------------------------------------------------"
echo "Starting execution..."
echo "------------------------------------------------------------"

# Start execution timing
EXEC_START=$(date +%s.%N)

# Execute the compiled program
./$OUTPUT_EXEC

# End execution timing
EXEC_END=$(date +%s.%N)
EXEC_TIME=$(echo "$EXEC_END - $EXEC_START" | bc)

echo ""
echo "Execution completed!"
echo "------------------------------------------------------------"

# Calculate total time
TOTAL_TIME=$(echo "$COMPILE_TIME + $EXEC_TIME" | bc)

# Convert to milliseconds for better readability
COMPILE_TIME_MS=$(echo "$COMPILE_TIME * 1000" | bc)
EXEC_TIME_MS=$(echo "$EXEC_TIME * 1000" | bc)
TOTAL_TIME_MS=$(echo "$TOTAL_TIME * 1000" | bc)

# Print timing summary
echo ""
echo "============================================================"
echo "TIMING SUMMARY (GPU)"
echo "============================================================"
echo "Compilation time: ${COMPILE_TIME} seconds (${COMPILE_TIME_MS} ms)"
echo "Execution time:   ${EXEC_TIME} seconds (${EXEC_TIME_MS} ms)"
echo "------------------------------------------------------------"
echo "Total time (GPU): ${TOTAL_TIME} seconds (${TOTAL_TIME_MS} ms)"
echo "============================================================"
echo ""
echo "Output files saved in kernels/assets/ directory:"
echo "  - blurred_image_cuda.jpg"
echo "  - sobel_gradient_magnitude_cuda.jpg"
echo "  - non_max_suppression_cuda.jpg"
echo "  - final_edges_hysteresis_cuda.jpg"
echo "============================================================"
