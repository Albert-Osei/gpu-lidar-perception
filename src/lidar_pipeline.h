#pragma once

// ── Standard headers ──────────────────────────────────────────────────────────
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cufft.h> // cuFFT — radial distance frequency analysis

using namespace std;

// ════════════════════════════════════════════════════════════════════════════
// LiDAR sensor simulation parameters
// Modelled after a Velodyne HDL-64E — the industry-standard sensor used in
// the DARPA Urban Challenge and early Waymo/Uber AV platforms.
// ════════════════════════════════════════════════════════════════════════════
#define NUM_BEAMS 64                                 // vertical beams (elevation channels)
#define AZIMUTH_STEPS 1800                           // horizontal resolution (0.2° steps × 360°)
#define POINTS_PER_FRAME (NUM_BEAMS * AZIMUTH_STEPS) // 115,200 points
#define NUM_FRAMES 50                                // number of LiDAR frames to process

#define MAX_RANGE_M 100.0f // maximum LiDAR range in metres
#define MIN_RANGE_M 0.5f   // minimum detectable range

// ════════════════════════════════════════════════════════════════════════════
// Voxel grid downsampling parameters
// A voxel is a 3-D cube cell.  All points falling into the same voxel are
// replaced by a single representative point (the voxel centroid).
// Standard pre-processing step before any AV perception algorithm.
// ════════════════════════════════════════════════════════════════════════════
#define VOXEL_SIZE_M 0.2f  // 20 cm cubic voxel side length
#define VOXEL_GRID_DIM 512 // grid extends ±51.2 m from the vehicle
#define VOXEL_GRID_CELLS (VOXEL_GRID_DIM * VOXEL_GRID_DIM * VOXEL_GRID_DIM)

// ════════════════════════════════════════════════════════════════════════════
// Ground plane RANSAC parameters
// RANSAC (Random Sample Consensus) robustly fits a plane to the ground
// points, even in the presence of obstacles and noise.
// ════════════════════════════════════════════════════════════════════════════
#define RANSAC_ITERATIONS 50     // number of hypothesis trials
#define RANSAC_THRESHOLD_M 0.15f // inlier distance threshold (15 cm)
#define GROUND_HEIGHT_MAX -0.5f  // only consider points below this height

// ════════════════════════════════════════════════════════════════════════════
// Occupancy grid parameters
// A 2-D bird's-eye-view grid used directly by the path planner.
// Each cell is OCCUPIED (1) or FREE (0).
// ════════════════════════════════════════════════════════════════════════════
#define OCC_GRID_SIZE 512    // 512×512 cells
#define OCC_CELL_SIZE_M 0.2f // 20 cm per cell → covers 102.4 m × 102.4 m

// ════════════════════════════════════════════════════════════════════════════
// cuFFT: radial distance analysis
// For each azimuth sector we build a histogram of radial distances, then
// apply cuFFT to detect periodic structures (fence posts, guard rails, etc.)
// ════════════════════════════════════════════════════════════════════════════
#define FFT_SECTORS 36                     // 10° sectors around the vehicle (36 × 10° = 360°)
#define FFT_BINS 256                       // radial histogram bins (each = MAX_RANGE/256 ≈ 0.39 m)
#define FFT_OUTPUT_BINS (FFT_BINS / 2 + 1) // output of real-to-complex FFT

// ════════════════════════════════════════════════════════════════════════════
// Data structures
// ════════════════════════════════════════════════════════════════════════════

// A single LiDAR return point in Cartesian coordinates
struct Point3f
{
    float x; // forward  (positive = ahead of vehicle)
    float y; // lateral  (positive = left)
    float z; // vertical (positive = upward)
};

// Per-frame processing statistics reported to CSV
struct FrameStats
{
    int frameId;
    int rawPoints;
    int voxelPoints;    // after voxel downsampling
    int obstaclePoints; // after ground removal
    float voxelTimeMs;
    float ransacTimeMs;
    float fftTimeMs;
    float occGridTimeMs;
    float totalGpuTimeMs;
    float dominantFrequency; // Hz — strongest periodic structure detected by FFT
};

// ════════════════════════════════════════════════════════════════════════════
// CUDA error-checking helper
// ════════════════════════════════════════════════════════════════════════════
inline void cudaCheck(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
inline void cufftCheck(cufftResult err, const char *file, int line)
{
    if (err != CUFFT_SUCCESS)
    {
        fprintf(stderr, "[cuFFT ERROR] %s:%d — code %d\n", file, line, err);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)
#define CUFFT_CHECK(call) cufftCheck((call), __FILE__, __LINE__)

// ════════════════════════════════════════════════════════════════════════════
// Kernel declarations
// ════════════════════════════════════════════════════════════════════════════

// Kernel 1: Voxel grid downsampling
// Each thread handles one input point; atomically marks its voxel cell as
// occupied.  A second pass computes per-voxel centroids.
__global__ void voxelMarkOccupied(const Point3f *__restrict__ points,
                                  int *voxelOccupied,
                                  float *voxelSumX,
                                  float *voxelSumY,
                                  float *voxelSumZ,
                                  int *voxelCount,
                                  int numPoints);

__global__ void voxelBuildCentroids(const int *voxelOccupied,
                                    const float *voxelSumX,
                                    const float *voxelSumY,
                                    const float *voxelSumZ,
                                    const int *voxelCount,
                                    Point3f *centroids,
                                    int *numCentroids,
                                    int numVoxels);

// Kernel 2: Ground plane labelling via RANSAC hypothesis scoring
// Each thread scores one RANSAC hypothesis (plane coefficients) against ALL
// input points and counts inliers.
__global__ void ransacScorePlanes(const Point3f *__restrict__ points,
                                  int numPoints,
                                  const float *planeA,
                                  const float *planeB,
                                  const float *planeC,
                                  const float *planeD,
                                  int *inlierCounts,
                                  int numHypotheses);

// Given the best plane, label each point as ground (0) or obstacle (1)
__global__ void labelGroundPoints(const Point3f *__restrict__ points,
                                  int *labels,
                                  int numPoints,
                                  float a, float b,
                                  float c, float d);

// Kernel 3: Build radial distance histograms per azimuth sector (for cuFFT input)
__global__ void buildRadialHistograms(const Point3f *__restrict__ points,
                                      const int *labels,
                                      float *histograms,
                                      int numPoints);

// Kernel 4: Build 2-D occupancy grid from obstacle points
__global__ void buildOccupancyGrid(const Point3f *__restrict__ points,
                                   const int *labels,
                                   int *grid,
                                   int numPoints);

// ════════════════════════════════════════════════════════════════════════════
// Host function declarations
// ════════════════════════════════════════════════════════════════════════════
__host__ vector<Point3f> generateLidarFrame(int frameId);
__host__ FrameStats processFrame(const vector<Point3f> &hostPoints,
                                 int frameId,
                                 cufftHandle fftPlan,
                                 int *d_occGrid);
__host__ void writeCsvReport(const string &path,
                             const vector<FrameStats> &stats);
__host__ void printSummary(const vector<FrameStats> &stats);
