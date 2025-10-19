#!/bin/bash

set -e

echo "============================================================"
echo "Canny Edge Detection - Performance Comparison"
echo "============================================================"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NVCC_FLAGS="-O3"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4) -diag-suppress 611"

echo ""
echo "============================================================"
echo "V1 KERNELS (NAIVE)"
echo "============================================================"

cd "$BASE_DIR/kernels/v1-naive"

V1_COMPILE_START=$(date +%s.%N)
nvcc $NVCC_FLAGS gaussian_blur.cu sobel.cu nms.cu hysteresis.cu $OPENCV_FLAGS -o canny_v1
V1_COMPILE_END=$(date +%s.%N)
V1_COMPILE_TIME=$(echo "$V1_COMPILE_END - $V1_COMPILE_START" | bc)

echo "V1 Compilation: ${V1_COMPILE_TIME}s"

V1_EXEC_START=$(date +%s.%N)
./canny_v1
V1_EXEC_END=$(date +%s.%N)
V1_EXEC_TIME=$(echo "$V1_EXEC_END - $V1_EXEC_START" | bc)

echo "V1 Execution: ${V1_EXEC_TIME}s"

V1_TOTAL_TIME=$(echo "$V1_COMPILE_TIME + $V1_EXEC_TIME" | bc)

echo ""
echo "============================================================"
echo "V2 KERNELS (OPTIMIZED)"
echo "============================================================"

cd "$BASE_DIR/kernels/v2"

V2_COMPILE_START=$(date +%s.%N)
nvcc $NVCC_FLAGS gaussian_blur.cu sobel.cu nms.cu hysteresis.cu $OPENCV_FLAGS -o canny_v2
V2_COMPILE_END=$(date +%s.%N)
V2_COMPILE_TIME=$(echo "$V2_COMPILE_END - $V2_COMPILE_START" | bc)

echo "V2 Compilation: ${V2_COMPILE_TIME}s"

V2_EXEC_START=$(date +%s.%N)
./canny_v2
V2_EXEC_END=$(date +%s.%N)
V2_EXEC_TIME=$(echo "$V2_EXEC_END - $V2_EXEC_START" | bc)

echo "V2 Execution: ${V2_EXEC_TIME}s"

V2_TOTAL_TIME=$(echo "$V2_COMPILE_TIME + $V2_EXEC_TIME" | bc)

echo ""
echo "============================================================"
echo "CPU (PYTHON + OPENCV)"
echo "============================================================"

cd "$BASE_DIR"

CPU_START=$(date +%s.%N)
python3 canny_cpu.py
CPU_END=$(date +%s.%N)
CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)

echo "CPU Execution: ${CPU_TIME}s"

echo ""
echo "============================================================"
echo "PERFORMANCE SUMMARY"
echo "============================================================"

V1_COMPILE_MS=$(echo "$V1_COMPILE_TIME * 1000" | bc)
V1_EXEC_MS=$(echo "$V1_EXEC_TIME * 1000" | bc)
V1_TOTAL_MS=$(echo "$V1_TOTAL_TIME * 1000" | bc)

V2_COMPILE_MS=$(echo "$V2_COMPILE_TIME * 1000" | bc)
V2_EXEC_MS=$(echo "$V2_EXEC_TIME * 1000" | bc)
V2_TOTAL_MS=$(echo "$V2_TOTAL_TIME * 1000" | bc)

CPU_MS=$(echo "$CPU_TIME * 1000" | bc)

printf "%-20s %15s %15s %15s\n" "Implementation" "Compilation" "Execution" "Total"
echo "------------------------------------------------------------"
printf "%-20s %15s %15s %15s\n" "V1 (Naive)" "${V1_COMPILE_MS} ms" "${V1_EXEC_MS} ms" "${V1_TOTAL_MS} ms"
printf "%-20s %15s %15s %15s\n" "V2 (Optimized)" "${V2_COMPILE_MS} ms" "${V2_EXEC_MS} ms" "${V2_TOTAL_MS} ms"
printf "%-20s %15s %15s %15s\n" "CPU (Python)" "N/A" "${CPU_MS} ms" "${CPU_MS} ms"
echo "------------------------------------------------------------"

echo ""
echo "SPEEDUP ANALYSIS"
echo "------------------------------------------------------------"

if (( $(echo "$CPU_TIME > 0 && $V1_EXEC_TIME > 0" | bc -l) )); then
    V1_SPEEDUP=$(echo "scale=2; $CPU_TIME / $V1_EXEC_TIME" | bc)
    echo "V1 vs CPU: ${V1_SPEEDUP}x"
fi

if (( $(echo "$CPU_TIME > 0 && $V2_EXEC_TIME > 0" | bc -l) )); then
    V2_SPEEDUP=$(echo "scale=2; $CPU_TIME / $V2_EXEC_TIME" | bc)
    echo "V2 vs CPU: ${V2_SPEEDUP}x"
fi

if (( $(echo "$V1_EXEC_TIME > 0 && $V2_EXEC_TIME > 0" | bc -l) )); then
    V2_VS_V1=$(echo "scale=2; $V1_EXEC_TIME / $V2_EXEC_TIME" | bc)
    echo "V2 vs V1: ${V2_VS_V1}x"
fi

echo "============================================================"

echo ""
echo ""
echo "============================================================"
echo "PURE EXECUTION TIME TEST (NO COMPILATION)"
echo "============================================================"

echo ""
echo "Running V1 (Pre-compiled)..."
cd "$BASE_DIR/kernels/v1-naive"
V1_PURE_START=$(date +%s.%N)
./canny_v1
V1_PURE_END=$(date +%s.%N)
V1_PURE_TIME=$(echo "$V1_PURE_END - $V1_PURE_START" | bc)
V1_PURE_MS=$(echo "$V1_PURE_TIME * 1000" | bc)
echo "V1 Pure Execution: ${V1_PURE_TIME}s (${V1_PURE_MS} ms)"

echo ""
echo "Running V2 (Pre-compiled)..."
cd "$BASE_DIR/kernels/v2"
V2_PURE_START=$(date +%s.%N)
./canny_v2
V2_PURE_END=$(date +%s.%N)
V2_PURE_TIME=$(echo "$V2_PURE_END - $V2_PURE_START" | bc)
V2_PURE_MS=$(echo "$V2_PURE_TIME * 1000" | bc)
echo "V2 Pure Execution: ${V2_PURE_TIME}s (${V2_PURE_MS} ms)"

echo ""
echo "Running CPU (Python)..."
cd "$BASE_DIR"
CPU_PURE_START=$(date +%s.%N)
python3 canny_cpu.py
CPU_PURE_END=$(date +%s.%N)
CPU_PURE_TIME=$(echo "$CPU_PURE_END - $CPU_PURE_START" | bc)
CPU_PURE_MS=$(echo "$CPU_PURE_TIME * 1000" | bc)
echo "CPU Pure Execution: ${CPU_PURE_TIME}s (${CPU_PURE_MS} ms)"

echo ""
echo "============================================================"
echo "PURE EXECUTION COMPARISON"
echo "============================================================"
printf "%-20s %20s\n" "Implementation" "Execution Time"
echo "------------------------------------------------------------"
printf "%-20s %20s\n" "V1 (Naive)" "${V1_PURE_MS} ms"
printf "%-20s %20s\n" "V2 (Optimized)" "${V2_PURE_MS} ms"
printf "%-20s %20s\n" "CPU (Python)" "${CPU_PURE_MS} ms"
echo "------------------------------------------------------------"

echo ""
echo "PURE EXECUTION SPEEDUP"
echo "------------------------------------------------------------"

if (( $(echo "$CPU_PURE_TIME > 0 && $V1_PURE_TIME > 0" | bc -l) )); then
    V1_PURE_SPEEDUP=$(echo "scale=2; $CPU_PURE_TIME / $V1_PURE_TIME" | bc)
    echo "V1 vs CPU: ${V1_PURE_SPEEDUP}x"
fi

if (( $(echo "$CPU_PURE_TIME > 0 && $V2_PURE_TIME > 0" | bc -l) )); then
    V2_PURE_SPEEDUP=$(echo "scale=2; $CPU_PURE_TIME / $V2_PURE_TIME" | bc)
    echo "V2 vs CPU: ${V2_PURE_SPEEDUP}x"
fi

if (( $(echo "$V1_PURE_TIME > 0 && $V2_PURE_TIME > 0" | bc -l) )); then
    V2_VS_V1_PURE=$(echo "scale=2; $V1_PURE_TIME / $V2_PURE_TIME" | bc)
    echo "V2 vs V1: ${V2_VS_V1_PURE}x"
fi

echo "============================================================"
