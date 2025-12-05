#!/bin/bash

set -e

# Absolute project path
PROJECT_DIR="/home/ubuntu/asst5-kernels/problems/flashattention"

# Create profiles directory if it doesn't exist
mkdir -p "$PROJECT_DIR/local_profiles"

# Timestamp for filename (YYYY-MM-DD_HH-MM-SS)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
outfile="$PROJECT_DIR/local_profiles/${timestamp}.txt"

echo "Saving profile output to: $outfile"
echo "Changing directory to: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Running local profiler with eval.py..."
echo "-----------------------------------------------------" | tee -a "$outfile"
echo "START TIME: $(date)" | tee -a "$outfile"
echo "-----------------------------------------------------" | tee -a "$outfile"

# Record start time (seconds since epoch)
start_time=$(date +%s)

# Run command, capture stdout + stderr, print live + save
python3 ../eval.py profile test_cases/test.txt 2>&1 | tee -a "$outfile"

# Record end time
end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Nicely formatted duration
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo "" | tee -a "$outfile"
echo "-----------------------------------------------------" | tee -a "$outfile"
echo "END TIME:   $(date)" | tee -a "$outfile"
echo "TOTAL RUNTIME: ${hours}h ${minutes}m ${seconds}s" | tee -a "$outfile"
echo "-----------------------------------------------------" | tee -a "$outfile"

echo ""
echo "====================================================="
echo " ✅ Profiling complete."
echo " 👉 Output saved to: $outfile"
echo " ⏱  Runtime: ${hours}h ${minutes}m ${seconds}s"
echo "====================================================="