/*
 * ============================================================================
 * GPU-Accelerated LiDAR Point Cloud Processing Pipeline
 * Capstone Project — Autonomous Driving / Robotics
 *
 * Pipeline stages per frame:
 *   1. Voxel Grid Downsampling     (CUDA kernel, uses atomic shared memory)
 *   2. Ground Plane Removal        (CUDA kernel, GPU-parallel RANSAC scoring)
 *   3. Radial Distance FFT         (cuFFT real-to-complex, periodic structure detection)
 *   4. 2-D Occupancy Grid          (CUDA kernel, direct path-planner input)
 *
 * Target hardware: NVIDIA L4  (Ada Lovelace, sm_89, CUDA 12.6)
 * ============================================================================
 */

#include "lidar_pipeline.h"

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 1a — Voxel Mark Occupied
//
// Each thread processes one input point.
//   • Converts (x,y,z) → a flat voxel index in a VOXEL_GRID_DIM³ grid.
//   • Atomically accumulates the coordinates for centroid computation.
//   • Marks the voxel as occupied.
//
// Memory strategy: per-voxel counters and sums live in global memory.
// atomicAdd is efficient on Ampere/Ada when many threads hit DIFFERENT voxels
// (which is typical for sparse LiDAR data).
// ════════════════════════════════════════════════════════════════════════════
__global__ void voxelMarkOccupied(const Point3f *__restrict__ points,
                                  int *voxelOccupied,
                                  float *voxelSumX,
                                  float *voxelSumY,
                                  float *voxelSumZ,
                                  int *voxelCount,
                                  int numPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;

    float px = points[i].x;
    float py = points[i].y;
    float pz = points[i].z;

    // Convert world coordinate to voxel index (origin centred on the vehicle)
    float half = (VOXEL_GRID_DIM * VOXEL_SIZE_M) * 0.5f;
    int vx = (int)((px + half) / VOXEL_SIZE_M);
    int vy = (int)((py + half) / VOXEL_SIZE_M);
    int vz = (int)((pz + half) / VOXEL_SIZE_M);

    // Discard points outside the voxel grid extent
    if (vx < 0 || vx >= VOXEL_GRID_DIM ||
        vy < 0 || vy >= VOXEL_GRID_DIM ||
        vz < 0 || vz >= VOXEL_GRID_DIM)
        return;

    int vIdx = vx + vy * VOXEL_GRID_DIM + vz * VOXEL_GRID_DIM * VOXEL_GRID_DIM;

    // Mark occupied and accumulate coordinates for centroid
    atomicAdd(&voxelOccupied[vIdx], 1);
    atomicAdd(&voxelSumX[vIdx], px);
    atomicAdd(&voxelSumY[vIdx], py);
    atomicAdd(&voxelSumZ[vIdx], pz);
    atomicAdd(&voxelCount[vIdx], 1);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 1b — Voxel Build Centroids
//
// One thread per voxel cell.  For each occupied voxel, compute the centroid
// (sum/count) and write it to the output array using an atomic counter for
// the compact output index.
// ════════════════════════════════════════════════════════════════════════════
__global__ void voxelBuildCentroids(const int *voxelOccupied,
                                    const float *voxelSumX,
                                    const float *voxelSumY,
                                    const float *voxelSumZ,
                                    const int *voxelCount,
                                    Point3f *centroids,
                                    int *numCentroids,
                                    int numVoxels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVoxels)
        return;
    if (voxelOccupied[i] == 0)
        return;

    int cnt = voxelCount[i];
    if (cnt == 0)
        return;

    // Atomically claim the next output slot
    int outIdx = atomicAdd(numCentroids, 1);

    centroids[outIdx].x = voxelSumX[i] / (float)cnt;
    centroids[outIdx].y = voxelSumY[i] / (float)cnt;
    centroids[outIdx].z = voxelSumZ[i] / (float)cnt;
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 2a — RANSAC Plane Hypothesis Scoring
//
// GPU-parallel RANSAC: the CPU generates RANSAC_ITERATIONS plane hypotheses
// from random triplets of low-height points.  This kernel scores ALL
// hypotheses in parallel — one thread per hypothesis.  Each thread counts
// how many points in the cloud lie within RANSAC_THRESHOLD_M of its plane.
//
// Parallelising across hypotheses (not points) keeps the kernel simple and
// avoids reduction across threads while still using the GPU.
// ════════════════════════════════════════════════════════════════════════════
__global__ void ransacScorePlanes(const Point3f *__restrict__ points,
                                  int numPoints,
                                  const float *planeA,
                                  const float *planeB,
                                  const float *planeC,
                                  const float *planeD,
                                  int *inlierCounts,
                                  int numHypotheses)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= numHypotheses)
        return;

    float a = planeA[h], b = planeB[h],
          c = planeC[h], d = planeD[h];

    // Normalisation factor for point-to-plane distance
    float norm = sqrtf(a * a + b * b + c * c);
    if (norm < 1e-6f)
    {
        inlierCounts[h] = 0;
        return;
    }

    int count = 0;
    for (int i = 0; i < numPoints; i++)
    {
        // Only test low-lying points for efficiency (obstacles are above ground)
        if (points[i].z > GROUND_HEIGHT_MAX)
            continue;
        float dist = fabsf(a * points[i].x + b * points[i].y +
                           c * points[i].z + d) /
                     norm;
        if (dist < RANSAC_THRESHOLD_M)
            count++;
    }
    inlierCounts[h] = count;
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 2b — Label Ground Points
//
// Given the best-fit plane (a,b,c,d) from RANSAC, label each point as:
//   0 = ground (within RANSAC_THRESHOLD_M of the plane AND below height limit)
//   1 = obstacle (everything else, including buildings, pedestrians, vehicles)
// ════════════════════════════════════════════════════════════════════════════
__global__ void labelGroundPoints(const Point3f *__restrict__ points,
                                  int *labels,
                                  int numPoints,
                                  float a, float b, float c, float d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;

    float norm = sqrtf(a * a + b * b + c * c);
    float dist = (norm > 1e-6f) ? fabsf(a * points[i].x + b * points[i].y + c * points[i].z + d) / norm
                                : 999.0f;

    // Ground: close to the plane AND in the low-height band
    labels[i] = (dist < RANSAC_THRESHOLD_M &&
                 points[i].z < GROUND_HEIGHT_MAX)
                    ? 0
                    : 1;
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 3 — Build Radial Distance Histograms (cuFFT input)
//
// Divides the 360° field of view into FFT_SECTORS sectors (10° each).
// For each obstacle point, accumulates its radial distance r = sqrt(x²+y²)
// into the histogram bin for its sector.
// The resulting FFT_SECTORS × FFT_BINS matrix is the cuFFT input.
//
// Used to detect PERIODIC structures (fence posts, guard rails, lane markers)
// that appear as peaks in the frequency domain — a standard technique in
// AV scene understanding.
// ════════════════════════════════════════════════════════════════════════════
__global__ void buildRadialHistograms(const Point3f *__restrict__ points,
                                      const int *labels,
                                      float *histograms,
                                      int numPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    if (labels[i] == 0)
        return; // skip ground points

    float x = points[i].x;
    float y = points[i].y;
    float r = sqrtf(x * x + y * y);

    if (r < MIN_RANGE_M || r > MAX_RANGE_M)
        return;

    // Determine azimuth sector [0, FFT_SECTORS)
    float angle = atan2f(y, x); // [-π, π]
    if (angle < 0.0f)
        angle += 2.0f * 3.14159265f; // [0, 2π]
    int sector = (int)(angle / (2.0f * 3.14159265f) * FFT_SECTORS);
    sector = min(sector, FFT_SECTORS - 1);

    // Determine radial bin
    int bin = (int)(r / MAX_RANGE_M * FFT_BINS);
    bin = min(bin, FFT_BINS - 1);

    // Accumulate into the sector's histogram (atomic for concurrent threads)
    atomicAdd(&histograms[sector * FFT_BINS + bin], 1.0f);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 4 — Build 2-D Occupancy Grid
//
// Projects all obstacle points onto the XY (bird's-eye-view) plane and marks
// cells in an OCC_GRID_SIZE × OCC_GRID_SIZE binary grid as OCCUPIED (1).
// Free cells remain 0.
//
// This grid is the direct output consumed by the vehicle's path planner —
// it answers the question "is there something in my way?" for every
// 20 cm × 20 cm patch of road within 51 m of the vehicle.
// ════════════════════════════════════════════════════════════════════════════
__global__ void buildOccupancyGrid(const Point3f *__restrict__ points,
                                   const int *labels,
                                   int *grid,
                                   int numPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    if (labels[i] == 0)
        return; // skip ground points

    // Project (x,y) into grid cell coordinates (origin at grid centre)
    float half = (OCC_GRID_SIZE * OCC_CELL_SIZE_M) * 0.5f;
    int gx = (int)((points[i].x + half) / OCC_CELL_SIZE_M);
    int gy = (int)((points[i].y + half) / OCC_CELL_SIZE_M);

    if (gx < 0 || gx >= OCC_GRID_SIZE ||
        gy < 0 || gy >= OCC_GRID_SIZE)
        return;

    // Mark cell as occupied (1).  atomicExch avoids redundant writes.
    atomicExch(&grid[gy * OCC_GRID_SIZE + gx], 1);
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Generate synthetic LiDAR frame
//
// Simulates a Velodyne HDL-64E style rotating LiDAR producing one 360° scan.
// The synthetic scene contains:
//   • A flat ground plane with Gaussian noise
//   • Two vehicle-shaped rectangular obstacles ahead
//   • A fence (periodic point pattern) on one side
//   • Random background points to simulate distant buildings
// ════════════════════════════════════════════════════════════════════════════
__host__ vector<Point3f> generateLidarFrame(int frameId)
{
    vector<Point3f> pts;
    pts.reserve(POINTS_PER_FRAME);

    srand((unsigned)frameId * 1234 + 7);

    // Elevation angles for 64 beams: -24.8° to +2° (Velodyne HDL-64E spec)
    float elevMin = -24.8f * 3.14159f / 180.0f;
    float elevMax = 2.0f * 3.14159f / 180.0f;

    for (int beam = 0; beam < NUM_BEAMS; beam++)
    {
        float elev = elevMin + (elevMax - elevMin) * beam / (NUM_BEAMS - 1);
        float cosE = cosf(elev);
        float sinE = sinf(elev);

        for (int az = 0; az < AZIMUTH_STEPS; az++)
        {
            float azAngle = (float)az / AZIMUTH_STEPS * 2.0f * 3.14159265f;
            float cosA = cosf(azAngle);
            float sinA = sinf(azAngle);

            // Default: ground return
            float range = MAX_RANGE_M * 0.5f + (rand() % 100) * 0.1f;

            // Ground plane (z ≈ -1.5 m, the LiDAR is mounted 1.5 m above ground)
            if (elev < -5.0f * 3.14159f / 180.0f)
            {
                float groundRange = -1.5f / sinE; // height = range * sinE
                if (groundRange > 0 && groundRange < MAX_RANGE_M)
                    range = groundRange + ((rand() % 100) - 50) * 0.002f;
            }

            // Obstacle 1: vehicle ahead at (15 m, 2 m) — box 4 m × 2 m × 1.5 m
            float tx = 15.0f + (frameId % 10) * 0.5f; // vehicle moves frame-to-frame
            float dist_to_v1 = sqrtf((cosA * range - tx) * (cosA * range - tx) +
                                     (sinA * range - 2.0f) * (sinA * range - 2.0f));
            if (dist_to_v1 < 2.0f)
            {
                range = sqrtf(tx * tx + 4.0f) + ((rand() % 20) - 10) * 0.01f;
            }

            // Obstacle 2: vehicle ahead at (25 m, -3 m)
            float dist_to_v2 = sqrtf((cosA * range - 25.0f) * (cosA * range - 25.0f) +
                                     (sinA * range + 3.0f) * (sinA * range + 3.0f));
            if (dist_to_v2 < 2.0f)
            {
                range = sqrtf(625.0f + 9.0f) + ((rand() % 20) - 10) * 0.01f;
            }

            // Fence posts: periodic pattern at y = 10 m (left side), every 3 m
            if (fabsf(sinA * range - 10.0f) < 0.3f)
            {
                float postX = (float)((int)(cosA * range / 3.0f)) * 3.0f;
                float postDist = sqrtf(postX * postX + 100.0f);
                if (fabsf(range - postDist) < 1.0f && beam > 20)
                {
                    range = postDist + ((rand() % 20) - 10) * 0.005f;
                }
            }

            range = fmaxf(MIN_RANGE_M, fminf(range, MAX_RANGE_M));

            Point3f p;
            p.x = range * cosE * cosA;
            p.y = range * cosE * sinA;
            p.z = range * sinE;
            pts.push_back(p);
        }
    }
    return pts;
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Process one LiDAR frame through the full GPU pipeline
// ════════════════════════════════════════════════════════════════════════════
__host__ FrameStats processFrame(const vector<Point3f> &hostPoints,
                                 int frameId,
                                 cufftHandle fftPlan,
                                 int *d_occGrid)
{
    FrameStats stats;
    stats.frameId = frameId;
    stats.rawPoints = (int)hostPoints.size();

    int N = (int)hostPoints.size();
    size_t ptBytes = N * sizeof(Point3f);

    // ── Allocate device buffers ───────────────────────────────────────────
    Point3f *d_pts = NULL, *d_centroids = NULL;
    int *d_voxelOcc = NULL, *d_voxelCnt = NULL;
    float *d_voxelSX = NULL, *d_voxelSY = NULL, *d_voxelSZ = NULL;
    int *d_numCentroids = NULL;
    float *d_planeA = NULL, *d_planeB = NULL,
          *d_planeC = NULL, *d_planeD = NULL;
    int *d_inlierCounts = NULL;
    int *d_labels = NULL;
    float *d_radHist = NULL;
    cufftComplex *d_fftOut = NULL;

    CUDA_CHECK(cudaMalloc(&d_pts, ptBytes));
    CUDA_CHECK(cudaMalloc(&d_centroids, ptBytes)); // upper bound
    CUDA_CHECK(cudaMalloc(&d_voxelOcc, VOXEL_GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_voxelCnt, VOXEL_GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_voxelSX, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxelSY, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxelSZ, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_numCentroids, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_planeA, RANSAC_ITERATIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_planeB, RANSAC_ITERATIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_planeC, RANSAC_ITERATIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_planeD, RANSAC_ITERATIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inlierCounts, RANSAC_ITERATIONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radHist, FFT_SECTORS * FFT_BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fftOut, FFT_SECTORS * FFT_OUTPUT_BINS * sizeof(cufftComplex)));

    // ── Copy input to device ──────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_pts, hostPoints.data(), ptBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_voxelOcc, 0, VOXEL_GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_voxelCnt, 0, VOXEL_GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_voxelSX, 0, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_voxelSY, 0, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_voxelSZ, 0, VOXEL_GRID_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_numCentroids, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_radHist, 0, FFT_SECTORS * FFT_BINS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_occGrid, 0, OCC_GRID_SIZE * OCC_GRID_SIZE * sizeof(int)));

    // ── Timing events ─────────────────────────────────────────────────────
    cudaEvent_t t[10];
    for (int i = 0; i < 10; i++)
        CUDA_CHECK(cudaEventCreate(&t[i]));

    int blockSize = 256;
    int gridN = (N + blockSize - 1) / blockSize;
    int gridV = (VOXEL_GRID_CELLS + blockSize - 1) / blockSize;
    int gridH = (RANSAC_ITERATIONS + blockSize - 1) / blockSize;

    // ════════════════════════════════════════════════════════════════════
    // STAGE 1: Voxel Grid Downsampling
    // ════════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaEventRecord(t[0]));

    voxelMarkOccupied<<<gridN, blockSize>>>(
        d_pts, d_voxelOcc, d_voxelSX, d_voxelSY, d_voxelSZ, d_voxelCnt, N);
    CUDA_CHECK(cudaGetLastError());

    voxelBuildCentroids<<<gridV, blockSize>>>(
        d_voxelOcc, d_voxelSX, d_voxelSY, d_voxelSZ, d_voxelCnt,
        d_centroids, d_numCentroids, VOXEL_GRID_CELLS);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t[1]));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_numCentroids = 0;
    CUDA_CHECK(cudaMemcpy(&h_numCentroids, d_numCentroids,
                          sizeof(int), cudaMemcpyDeviceToHost));
    stats.voxelPoints = h_numCentroids;

    float voxelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&voxelMs, t[0], t[1]));
    stats.voxelTimeMs = voxelMs;

    // ════════════════════════════════════════════════════════════════════
    // STAGE 2: Ground Plane Removal (GPU-parallel RANSAC)
    //
    // The CPU generates random plane hypotheses from the centroid cloud.
    // The GPU scores all of them simultaneously.
    // ════════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaEventRecord(t[2]));

    // CPU: generate RANSAC_ITERATIONS random plane hypotheses
    // Copy centroids to host for random sampling
    vector<Point3f> h_centroids(h_numCentroids);
    CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids,
                          h_numCentroids * sizeof(Point3f),
                          cudaMemcpyDeviceToHost));

    vector<float> h_pA(RANSAC_ITERATIONS), h_pB(RANSAC_ITERATIONS),
        h_pC(RANSAC_ITERATIONS), h_pD(RANSAC_ITERATIONS);

    // Filter to low-height candidate points for ground sampling
    vector<int> groundCandidates;
    for (int i = 0; i < h_numCentroids; i++)
    {
        if (h_centroids[i].z < GROUND_HEIGHT_MAX)
            groundCandidates.push_back(i);
    }

    if (groundCandidates.size() >= 3)
    {
        for (int h = 0; h < RANSAC_ITERATIONS; h++)
        {
            // Pick 3 random ground candidate points
            int idx0 = groundCandidates[rand() % groundCandidates.size()];
            int idx1 = groundCandidates[rand() % groundCandidates.size()];
            int idx2 = groundCandidates[rand() % groundCandidates.size()];

            // Compute plane normal via cross product
            float ux = h_centroids[idx1].x - h_centroids[idx0].x;
            float uy = h_centroids[idx1].y - h_centroids[idx0].y;
            float uz = h_centroids[idx1].z - h_centroids[idx0].z;
            float vx = h_centroids[idx2].x - h_centroids[idx0].x;
            float vy = h_centroids[idx2].y - h_centroids[idx0].y;
            float vz = h_centroids[idx2].z - h_centroids[idx0].z;

            h_pA[h] = uy * vz - uz * vy; // normal.x
            h_pB[h] = uz * vx - ux * vz; // normal.y
            h_pC[h] = ux * vy - uy * vx; // normal.z
            h_pD[h] = -(h_pA[h] * h_centroids[idx0].x +
                        h_pB[h] * h_centroids[idx0].y +
                        h_pC[h] * h_centroids[idx0].z);
        }
    }
    else
    {
        // Fallback: flat ground plane z = -1.5
        for (int h = 0; h < RANSAC_ITERATIONS; h++)
        {
            h_pA[h] = 0;
            h_pB[h] = 0;
            h_pC[h] = 1;
            h_pD[h] = 1.5f;
        }
    }

    // Copy plane hypotheses to device
    CUDA_CHECK(cudaMemcpy(d_planeA, h_pA.data(),
                          RANSAC_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planeB, h_pB.data(),
                          RANSAC_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planeC, h_pC.data(),
                          RANSAC_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planeD, h_pD.data(),
                          RANSAC_ITERATIONS * sizeof(float), cudaMemcpyHostToDevice));

    // GPU scores all hypotheses in parallel — one thread per hypothesis
    ransacScorePlanes<<<gridH, blockSize>>>(
        d_centroids, h_numCentroids,
        d_planeA, d_planeB, d_planeC, d_planeD,
        d_inlierCounts, RANSAC_ITERATIONS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU: pick the hypothesis with the most inliers
    vector<int> h_inliers(RANSAC_ITERATIONS);
    CUDA_CHECK(cudaMemcpy(h_inliers.data(), d_inlierCounts,
                          RANSAC_ITERATIONS * sizeof(int), cudaMemcpyDeviceToHost));

    int bestH = 0;
    for (int h = 1; h < RANSAC_ITERATIONS; h++)
    {
        if (h_inliers[h] > h_inliers[bestH])
            bestH = h;
    }

    // GPU: label every centroid point as ground or obstacle
    labelGroundPoints<<<gridN, blockSize>>>(
        d_centroids, d_labels, h_numCentroids,
        h_pA[bestH], h_pB[bestH], h_pC[bestH], h_pD[bestH]);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t[3]));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ransacMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ransacMs, t[2], t[3]));
    stats.ransacTimeMs = ransacMs;

    // Count obstacle points (labels == 1)
    vector<int> h_labels(h_numCentroids);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                          h_numCentroids * sizeof(int), cudaMemcpyDeviceToHost));
    int obstCount = 0;
    for (int x : h_labels)
        if (x == 1)
            obstCount++;
    stats.obstaclePoints = obstCount;

    // ════════════════════════════════════════════════════════════════════
    // STAGE 3: cuFFT — Radial Distance Frequency Analysis
    //
    // For each 10° azimuth sector, build a histogram of obstacle distances.
    // Apply real-to-complex FFT to detect periodically spaced obstacles.
    // The dominant non-DC frequency indicates the spacing of repeating
    // structures (fence posts, lane markings, jersey barriers).
    // ════════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaEventRecord(t[4]));

    int gridCentroids = (h_numCentroids + blockSize - 1) / blockSize;
    buildRadialHistograms<<<gridCentroids, blockSize>>>(
        d_centroids, d_labels, d_radHist, h_numCentroids);
    CUDA_CHECK(cudaGetLastError());

    // Execute batched real-to-complex 1D FFT: FFT_SECTORS transforms of length FFT_BINS
    // fftPlan was created in main() as R2C batch plan (FFT_SECTORS × FFT_BINS)
    CUFFT_CHECK(cufftExecR2C(fftPlan, d_radHist, d_fftOut));

    CUDA_CHECK(cudaEventRecord(t[5]));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fftMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fftMs, t[4], t[5]));
    stats.fftTimeMs = fftMs;

    // CPU: find the dominant non-DC frequency across all sectors
    vector<cufftComplex> h_fftOut(FFT_SECTORS * FFT_OUTPUT_BINS);
    CUDA_CHECK(cudaMemcpy(h_fftOut.data(), d_fftOut,
                          FFT_SECTORS * FFT_OUTPUT_BINS * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    float maxMag = 0.0f;
    int maxBin = 1; // start at 1 to skip DC
    for (int s = 0; s < FFT_SECTORS; s++)
    {
        for (int b = 1; b < FFT_OUTPUT_BINS; b++)
        {
            float re = h_fftOut[s * FFT_OUTPUT_BINS + b].x;
            float im = h_fftOut[s * FFT_OUTPUT_BINS + b].y;
            float mag = sqrtf(re * re + im * im);
            if (mag > maxMag)
            {
                maxMag = mag;
                maxBin = b;
            }
        }
    }
    // Convert bin index to spatial frequency (cycles per MAX_RANGE_M)
    // i.e., one cycle covers MAX_RANGE_M / maxBin metres between obstacles
    stats.dominantFrequency = (float)maxBin / MAX_RANGE_M;

    // ════════════════════════════════════════════════════════════════════
    // STAGE 4: Occupancy Grid Generation
    // ════════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaEventRecord(t[6]));

    buildOccupancyGrid<<<gridCentroids, blockSize>>>(
        d_centroids, d_labels, d_occGrid, h_numCentroids);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t[7]));
    CUDA_CHECK(cudaDeviceSynchronize());

    float occMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&occMs, t[6], t[7]));
    stats.occGridTimeMs = occMs;
    stats.totalGpuTimeMs = voxelMs + ransacMs + fftMs + occMs;

    // ── Clean up ───────────────────────────────────────────────────────
    cudaFree(d_pts);
    cudaFree(d_centroids);
    cudaFree(d_voxelOcc);
    cudaFree(d_voxelCnt);
    cudaFree(d_voxelSX);
    cudaFree(d_voxelSY);
    cudaFree(d_voxelSZ);
    cudaFree(d_numCentroids);
    cudaFree(d_planeA);
    cudaFree(d_planeB);
    cudaFree(d_planeC);
    cudaFree(d_planeD);
    cudaFree(d_inlierCounts);
    cudaFree(d_labels);
    cudaFree(d_radHist);
    cudaFree(d_fftOut);
    for (int i = 0; i < 10; i++)
        cudaEventDestroy(t[i]);

    return stats;
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Write CSV report
// ════════════════════════════════════════════════════════════════════════════
__host__ void writeCsvReport(const string &path,
                             const vector<FrameStats> &stats)
{
    ofstream csv(path);
    if (!csv.is_open())
    {
        fprintf(stderr, "Cannot write CSV: %s\n", path.c_str());
        return;
    }

    csv << "frame_id,raw_points,voxel_points,obstacle_points,"
        << "voxel_ms,ransac_ms,fft_ms,occ_grid_ms,total_gpu_ms,"
        << "dominant_freq_cyc_per_m\n";

    for (const auto &s : stats)
    {
        csv << s.frameId << ","
            << s.rawPoints << ","
            << s.voxelPoints << ","
            << s.obstaclePoints << ","
            << s.voxelTimeMs << ","
            << s.ransacTimeMs << ","
            << s.fftTimeMs << ","
            << s.occGridTimeMs << ","
            << s.totalGpuTimeMs << ","
            << s.dominantFrequency << "\n";
    }
    csv.close();
    printf("CSV report written to: %s\n", path.c_str());
}

// ════════════════════════════════════════════════════════════════════════════
// HOST — Print timing summary
// ════════════════════════════════════════════════════════════════════════════
__host__ void printSummary(const vector<FrameStats> &stats)
{
    if (stats.empty())
        return;

    double tVox = 0, tRan = 0, tFft = 0, tOcc = 0, tTot = 0;
    for (const auto &s : stats)
    {
        tVox += s.voxelTimeMs;
        tRan += s.ransacTimeMs;
        tFft += s.fftTimeMs;
        tOcc += s.occGridTimeMs;
        tTot += s.totalGpuTimeMs;
    }
    int n = (int)stats.size();

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║   GPU LiDAR Pipeline — Performance Summary           ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Frames processed        : %4d                       ║\n", n);
    printf("║  Points per frame        : %4d  (%.1fK)              ║\n",
           POINTS_PER_FRAME, POINTS_PER_FRAME / 1000.0f);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Stage               Total (ms)    Avg/frame (ms)    ║\n");
    printf("║  Voxel downsampling  %9.2f      %9.3f           ║\n",
           tVox, tVox / n);
    printf("║  RANSAC ground rem.  %9.2f      %9.3f           ║\n",
           tRan, tRan / n);
    printf("║  cuFFT analysis      %9.2f      %9.3f           ║\n",
           tFft, tFft / n);
    printf("║  Occupancy grid      %9.2f      %9.3f           ║\n",
           tOcc, tOcc / n);
    printf("║  ─────────────────────────────────────────────────  ║\n");
    printf("║  TOTAL GPU time      %9.2f      %9.3f           ║\n",
           tTot, tTot / n);
    printf("║  Equivalent FPS      %9.1f                         ║\n",
           1000.0 * n / tTot);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    printf("Dominant periodic spacing (last frame): %.2f cycles/m\n",
           stats.back().dominantFrequency);
    if (stats.back().dominantFrequency > 0.01f)
    {
        printf("  → Periodic obstacle spacing ≈ %.2f m  "
               "(e.g. fence posts / guard rails)\n",
               1.0f / stats.back().dominantFrequency);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════════════
int main(int argc, char **argv)
{
    int numFrames = (argc > 1) ? atoi(argv[1]) : NUM_FRAMES;
    const string csvPath = (argc > 2) ? argv[2] : "lidar_results.csv";

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  GPU LiDAR Processing Pipeline                       ║\n");
    printf("║  Autonomous Driving — Capstone Project               ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Sensor model  : Velodyne HDL-64E (simulated)        ║\n");
    printf("║  Beams         : %3d vertical channels               ║\n", NUM_BEAMS);
    printf("║  Azimuth steps : %4d (0.2° resolution)             ║\n", AZIMUTH_STEPS);
    printf("║  Points/frame  : %6d                               ║\n", POINTS_PER_FRAME);
    printf("║  Frames        : %3d                                 ║\n", numFrames);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Pipeline stages:                                    ║\n");
    printf("║  1. Voxel grid downsampling   (CUDA kernel)          ║\n");
    printf("║  2. Ground plane removal      (GPU-parallel RANSAC)  ║\n");
    printf("║  3. Radial FFT analysis       (cuFFT R2C batched)    ║\n");
    printf("║  4. Occupancy grid generation (CUDA kernel)          ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ── Create reusable cuFFT plan ────────────────────────────────────────
    // Batched 1-D real-to-complex FFT: FFT_SECTORS transforms, each of
    // length FFT_BINS.  The plan is created once and reused across all frames.
    cufftHandle fftPlan;

    int fftDims[1] = {FFT_BINS};
    CUFFT_CHECK(cufftPlanMany(
        &fftPlan,
        1,                        // rank (1-D transforms)
        fftDims,                  // transform length
        NULL, 1, FFT_BINS,        // input:  stride=1, dist=FFT_BINS
        NULL, 1, FFT_OUTPUT_BINS, // output: stride=1, dist=FFT_OUTPUT_BINS
        CUFFT_R2C,                // real-to-complex
        FFT_SECTORS               // batch count
        ));

    // ── Allocate persistent occupancy grid (reused across frames) ─────────
    int *d_occGrid = NULL;
    CUDA_CHECK(cudaMalloc(&d_occGrid,
                          OCC_GRID_SIZE * OCC_GRID_SIZE * sizeof(int)));

    // ── Process all frames ────────────────────────────────────────────────
    vector<FrameStats> allStats;
    allStats.reserve(numFrames);

    for (int f = 0; f < numFrames; f++)
    {
        printf("[Frame %3d/%d] Generating LiDAR scan ... ", f + 1, numFrames);
        fflush(stdout);

        vector<Point3f> pts = generateLidarFrame(f);
        FrameStats s = processFrame(pts, f, fftPlan, d_occGrid);
        allStats.push_back(s);

        printf("voxel→%5d pts  obstacles→%5d  "
               "total=%.2fms\n",
               s.voxelPoints, s.obstaclePoints, s.totalGpuTimeMs);
    }

    // ── Results ───────────────────────────────────────────────────────────
    printSummary(allStats);
    writeCsvReport(csvPath, allStats);

    // ── Free persistent resources ─────────────────────────────────────────
    CUFFT_CHECK(cufftDestroy(fftPlan));
    cudaFree(d_occGrid);

    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
