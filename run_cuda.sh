#!/bin/bash

set -e

echo "============================================================"
echo "GPU-based Canny Edge Detection (CUDA)"
echo "============================================================"

cd "$(dirname "$0")/kernels"

OUTPUT_EXEC="canny_cuda"

CUDA_FILES="gaussian_blur.cu sobel.cu nms.cu hysteresis.cu"

NVCC_FLAGS="-O3"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4) -diag-suppress 611"

echo ""
echo "Starting compilation..."
echo "------------------------------------------------------------"

COMPILE_START=$(date +%s.%N)

nvcc $NVCC_FLAGS $CUDA_FILES $OPENCV_FLAGS -o $OUTPUT_EXEC

COMPILE_END=$(date +%s.%N)
COMPILE_TIME=$(echo "$COMPILE_END - $COMPILE_START" | bc)

echo "Compilation completed successfully!"
echo "Compilation time: ${COMPILE_TIME} seconds"
echo ""

echo "------------------------------------------------------------"
echo "Starting execution..."
echo "------------------------------------------------------------"

EXEC_START=$(date +%s.%N)

./$OUTPUT_EXEC

EXEC_END=$(date +%s.%N)
EXEC_TIME=$(echo "$EXEC_END - $EXEC_START" | bc)

echo ""
echo "Execution completed!"
echo "------------------------------------------------------------"

TOTAL_TIME=$(echo "$COMPILE_TIME + $EXEC_TIME" | bc)

COMPILE_TIME_MS=$(echo "$COMPILE_TIME * 1000" | bc)
EXEC_TIME_MS=$(echo "$EXEC_TIME * 1000" | bc)
TOTAL_TIME_MS=$(echo "$TOTAL_TIME * 1000" | bc)

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
echo "GPU Output files saved in kernels/assets/ directory:"
echo "  - blurred_image_cuda.jpg"
echo "  - sobel_gradient_magnitude_cuda.jpg"
echo "  - non_max_suppression_cuda.jpg"
echo "  - final_edges_hysteresis_cuda.jpg"
echo "============================================================"

echo ""
echo ""
echo "============================================================"
echo "Running CPU Implementation for Comparison..."
echo "============================================================"
echo ""

cd ..

CPU_START=$(date +%s.%N)

python3 canny_cpu.py

CPU_END=$(date +%s.%N)
CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
CPU_TIME_MS=$(echo "$CPU_TIME * 1000" | bc)

echo ""
echo ""
echo "============================================================"
echo "PERFORMANCE COMPARISON"
echo "============================================================"
echo "GPU (CUDA):"
echo "  Compilation:  ${COMPILE_TIME_MS} ms"
echo "  Execution:    ${EXEC_TIME_MS} ms"
echo "  Total:        ${TOTAL_TIME_MS} ms"
echo ""
echo "CPU (Python + OpenCV):"
echo "  Execution:    ${CPU_TIME_MS} ms"
echo "------------------------------------------------------------"

if (( $(echo "$CPU_TIME > 0" | bc -l) )); then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $EXEC_TIME" | bc)
    echo "GPU Speedup: ${SPEEDUP}x faster than CPU"

    IMPROVEMENT=$(echo "scale=2; (($CPU_TIME - $EXEC_TIME) / $CPU_TIME) * 100" | bc)
    echo "Performance Improvement: ${IMPROVEMENT}% faster"
else
    echo "Could not calculate speedup (CPU time is zero)"
fi
echo "============================================================"
