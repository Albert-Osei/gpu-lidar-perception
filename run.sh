#!/bin/bash

# ============================================================================
# GPU LiDAR Processing Pipeline Runner
# ============================================================================

# Number of LiDAR frames to process
FRAMES=100

# Output CSV path
CSV_OUTPUT=lidar_results.csv

# Build project
make build

# Run executable
./bin/lidar_pipeline.exe $FRAMES $CSV_OUTPUT
