// taken from https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
    #ifdef DEBUG
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    #endif
    return result;
}


#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100

__global__ void copy(float *odata, const float *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        odata[(y + j) * width + x] = idata[(y + j)*width + x];
    }
}

__global__ void transposeNaive(float *odata, const float *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
        odata[x*width + (y+j)] = idata[(y+j)*width + x];
    } 
}

__global__ void transposeCoalesced(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // load tile into shared memory
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    // transpose the block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeNoBankConflicts(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // load tile into shared memory
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    // transpose the block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


// copy kernel that uses shared memory to test if any overhead with filling and extracting 
// shared memory, as well as __syncthreads
__global__ void copySharedMem(float *odata, float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    // load tile into shared memory
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    
    __syncthreads(); // technically not needed as read and write performed by same thread

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.y+j][threadIdx.x];

}


void postProcess(const float *ref, const float *res, int n, float ms) {
    bool passed = true;
    for(int i = 0; i < n; i++) {
        if(res[i] != ref[i]) { 
            printf("%d\t%f != %f\n", i, res[i], ref[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if (passed) 
        printf("%25.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

int main(int argc, char **argv) {
    const int nx = 1024;
    const int ny = 1024;

    const int mem_size = nx*ny*sizeof(float);

    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM,1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;
    for(int i = 0; i < argc; i++)  {
        if (!strncmp(argv[i], "dev=", 4))
            devId = atoi((char*)(&argv[i][4]));
    }


    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));

    printf("Device: %s\n", prop.name);
    printf("Martix size: %dx%d\n", nx, ny);
    printf("Block size: %dx%d\n", TILE_DIM, BLOCK_ROWS);
    printf("Tile size: %dx%d\n", TILE_DIM, TILE_DIM);
    printf("dimGrid: %dx%dx%d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock: %dx%dx%d\n", dimBlock.x, dimBlock.y, dimBlock.z);

    checkCuda(cudaSetDevice(devId));

    float *h_idata = (float*)malloc(mem_size);
    float *h_cdata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);
    float *h_correct = (float*)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    checkCuda(cudaMalloc(&d_idata, mem_size));
    checkCuda(cudaMalloc(&d_tdata, mem_size));
    checkCuda(cudaMalloc(&d_cdata, mem_size));


    if (nx % TILE_DIM || ny % TILE_DIM) {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        goto error_exit;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        goto error_exit;
    }

    // fill host array
    for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
            h_idata[j*nx + i] = j*nx + i;

    for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
            h_correct[j*nx + i] = i*nx + j;


    checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // timing 
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    float ms;

    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");


    printf("--------------------------------------------------\n");
    
    // ---------------
    // cudaMemcpy
    // ---------------
    printf("%25s", "cudaMemcpy");
    // transfer input data to device
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        checkCuda(cudaMemcpy(d_cdata, d_idata, mem_size, cudaMemcpyDeviceToDevice));
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    postProcess(h_idata, h_idata, nx*ny, ms);
    
    // ---------------
    // sendReceivePCIe
    // ---------------
    printf("%25s", "sendReceivePCIe");
    // transfer input data to device
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(h_idata, d_idata, mem_size, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    postProcess(h_idata, h_idata, nx*ny, ms);

    printf("--------------------------------------------------\n");

    // ----
    // copy
    // ----
    printf("%25s", "copy");
    // transfer input data to device
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    // warm up
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postProcess(h_cdata, h_idata, nx*ny, ms);

    // ----
    // copy
    // ----
    printf("%25s", "copyShared");
    // transfer input data to device
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    // warm up
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postProcess(h_cdata, h_idata, nx*ny, ms);

    // --------------
    // transposeNaive
    // --------------
    printf("%25s", "transposeNaive");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postProcess(h_tdata, h_correct, nx*ny, ms);

    // ------------------
    // transposeCoalesced
    // ------------------
    printf("%25s", "transposeCoalesced"); 
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postProcess(h_tdata, h_correct, nx*ny, ms);

    
    // ------------------
    // transposeNoBankConflicts
    // ------------------
    printf("%25s", "transposeNoBankConflicts"); 
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent));
    for(int i = 0; i < NUM_REPS; i++)
        transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postProcess(h_tdata, h_correct, nx*ny, ms);

    // cleanup
    error_exit:
        checkCuda(cudaEventDestroy(startEvent));
        checkCuda(cudaEventDestroy(stopEvent));
        checkCuda(cudaFree(d_idata));
        checkCuda(cudaFree(d_cdata));
        checkCuda(cudaFree(d_tdata));
        free(h_idata);
        free(h_cdata);
        free(h_tdata);
        free(h_correct);
    return 0;
}
