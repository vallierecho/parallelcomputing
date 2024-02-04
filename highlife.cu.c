#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// Include new headers
#include <cuda.h>
#include <cuda_runtime.h>

    // Result from last compute of world.
    unsigned char *g_resultData = NULL;

// Current state of world.
unsigned char *g_data = NULL;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// Current data length (product of width and height)
size_t g_dataLength = 0; // g_worldWidth * g_worldHeight

static inline void HL_initAllZeros(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initAllOnes(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    for (i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }
}

static inline void HL_initOnesInMiddle(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));

    // set first 1 rows of world to true
    for (i = 10 * g_worldWidth; i < 11 * g_worldWidth; i++)
    {
        if ((i >= (10 * g_worldWidth + 10)) && (i < (10 * g_worldWidth + 20)))
        {
            g_data[i] = 1;
        }
    }
}

static inline void HL_initOnesAtCorners(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1;                                                 // upper left
    g_data[worldWidth - 1] = 1;                                    // upper right
    g_data[(worldHeight * (worldWidth - 1))] = 1;                  // lower left
    g_data[(worldHeight * (worldWidth - 1)) + worldWidth - 1] = 1; // lower right
}

static inline void HL_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1;              // upper left
    g_data[1] = 1;              // upper left +1
    g_data[worldWidth - 1] = 1; // upper right
}

static inline void HL_initReplicator(size_t worldWidth, size_t worldHeight)
{
    size_t x, y;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Replace all the calloc calls with cudaMallocManaged
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Zero out all data elements
    cudaMemset(g_data, 0, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, (g_dataLength * sizeof(unsigned char)));

    x = worldWidth / 2;
    y = worldHeight / 2;

    g_data[x + y * worldWidth + 1] = 1;
    g_data[x + y * worldWidth + 2] = 1;
    g_data[x + y * worldWidth + 3] = 1;
    g_data[x + (y + 1) * worldWidth] = 1;
    g_data[x + (y + 2) * worldWidth] = 1;
    g_data[x + (y + 3) * worldWidth] = 1;
}

static inline void HL_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight)
{
    switch (pattern)
    {
    case 0:
        HL_initAllZeros(worldWidth, worldHeight);
        break;

    case 1:
        HL_initAllOnes(worldWidth, worldHeight);
        break;

    case 2:
        HL_initOnesInMiddle(worldWidth, worldHeight);
        break;

    case 3:
        HL_initOnesAtCorners(worldWidth, worldHeight);
        break;

    case 4:
        HL_initSpinnerAtCorner(worldWidth, worldHeight);
        break;

    case 5:
        HL_initReplicator(worldWidth, worldHeight);
        break;

    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}

static inline void HL_swap(unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

__device__ inline unsigned int HL_countAliveCells(unsigned char *data,
                                                  size_t x0,
                                                  size_t x1,
                                                  size_t x2,
                                                  size_t y0,
                                                  size_t y1,
                                                  size_t y2)
{

    return data[x0 + y0] + data[x1 + y0] + data[x2 + y0] + data[x0 + y1] + data[x2 + y1] + data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
}
/*
Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);

    for( i = 0; i < g_worldHeight; i++)
    {
    printf("Row %2d: ", i);
    for( j = 0; j < g_worldWidth; j++)
    {
        printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
    }
    printf("\n");
    }

    printf("\n\n");
}
*/

/// Parallel version of standard byte-per-cell life.
// HL kernel: main CUDA kernel function
__global__ void HL_kernel(unsigned char *d_data,
                          unsigned int worldWidth,
                          unsigned int worldHeight,
                          unsigned char *d_resultData)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Iterate over the cells in the world
    for (size_t i = index; i < worldWidth * worldHeight; i += stride)
    {
        size_t x = i % worldWidth;
        size_t y = i / worldWidth;

        size_t x0 = (x + worldWidth - 1) % worldWidth;
        size_t x2 = (x + 1) % worldWidth;

        size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % worldHeight) * worldWidth;

        unsigned int aliveCells = HL_countAliveCells(d_data, x0, x, x2, y0, y1, y2);

        // rule B36/S23
        d_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !d_data[x + y1]) ||
                                       (aliveCells == 2 && d_data[x + y1])
                                   ? 1
                                   : 0;
    }
    // Synchronize the threads?
    __syncthreads();
}

// HL_kernelLaunch: invoke the HL_kernel and swap the worlds
bool HL_kernelLaunch(unsigned char **d_data,
                     unsigned char **d_resultData,
                     size_t worldWidth,
                     size_t worldHeight,
                     size_t iterationsCount,
                     ushort threadsCount)
{
    // Declare number of blocks
    size_t blockCount = (worldHeight * worldWidth) / threadsCount;
    // Iterate for the specified number of iterations
    for (size_t iter = 0; iter < iterationsCount; ++iter)
    {

        // Invoke the CUDA kernel
        HL_kernel<<<blockCount, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData);

        // Synchronize
        cudaDeviceSynchronize();
        // Swap the worlds
        HL_swap(d_data, d_resultData);
    }
    // Synchronize
    cudaDeviceSynchronize();
    // Return success
    return true;
}

int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int threadsCount = 0;

    printf("This is the HighLife running in parallel on a GPU.\n");

    if (argc != 5)
    {
        printf("HighLife requires 4 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations, and 4th is the thread count, e.g. ./highlife 0 32 2 32 \n");
        exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threadsCount = atoi(argv[4]);

    // Initialize on GPU
    HL_initMaster(pattern, worldSize, worldSize);

    // printf("AFTER INIT IS............\n");
    // HL_printWorld(0);

    // Launch CUDA kernel
    HL_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, iterations, threadsCount);

    // printf("######################### FINAL WORLD IS ###############################\n");
    // HL_printWorld(iterations);

    // Free allocated memory on GPU
    cudaFree(g_data);
    cudaFree(g_resultData);

    return true;
}
