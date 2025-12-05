#!/bin/bash

# Create profiles directory if not exists
mkdir -p popcorn_profiles

# Timestamp for filename (YYYY-MM-DD_HH-MM-SS)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
outfile="popcorn_profiles/${timestamp}.txt"

echo "Saving profile output to: $outfile"
echo "Running popcorn-cli..."

# Run the command and pipe to tee so it prints AND saves
popcorn-cli submit --leaderboard flashattention --mode profile submission.py 2>&1 | tee "$outfile"

echo ""
echo "====================================================="
echo " Profiling complete. Output saved to: $outfile"
echo "====================================================="