#include <stdio.h>

#define M 16
#define N 8
#define P 4

#define numThread 8
#define numBlock 4
#define blockSize 4

__global__ void matMul(float *A, float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;

    if(row < M && col < P) {
        for(int ii = 0; ii < N; ii++) {
            sum += A[row*N + ii] * B[ii*P + col];
        }
        C[row*P + col] = sum;
    }    
}

void print2DArray(float *arr, int cols, int rows) {
    for(int i=0; i < cols*rows; i++) {
        if ((i % rows == 0) && (i != 0)) {
            printf("\n");
        }
        printf("%0.2e ", arr[i]);
    }
    printf("\n\n");
}

int main(int argc, char* argv[]) {

    float *ha, *hb, *hc; // host (CPU) arrays
    float *da, *db, *dc; // device (GPU) arrays
    float gpu_time;

    printf("Creating host and device arrays...\n\n");
    ha = (float*)malloc(M*N*sizeof(float));
    hb = (float*)malloc(N*P*sizeof(float));
    hc = (float*)malloc(M*P*sizeof(float));


    for (int i=0; i < M*N; i++) {
        ha[i] = 1.0*(i);
    }

    for (int i=0; i < N*P; i++) {
        hb[i] = 2.0*(i);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&da, M*N*sizeof(float));
    cudaMalloc((void**)&db, N*P*sizeof(float));
    cudaMalloc((void**)&dc, M*P*sizeof(float));

    // copy host arrays over to device
    cudaMemcpy(da, ha, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N*P*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int grid_rows = (M + blockSize - 1) / blockSize;
    unsigned int grid_cols = (P + blockSize - 1) / blockSize;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(blockSize, blockSize);

    // output input matrices
    print2DArray(ha, M, N);
    print2DArray(hb, N, P);
    
    printf("Performing [%dx%d]*[%dx%d] matrix multiplication...\n\n", M, N, N, P);


    // perform parralised addition of matrics
    cudaEventRecord(start,0);
    matMul<<<dimGrid, dimBlock>>>(da, db, dc);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // output resultant matrix
    cudaMemcpy(hc, dc, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    print2DArray(hc, M, P);
    
    // calculate elapsed time
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU elapsed time: %f ms\n\n", gpu_time);

    // free host arrays
    free(ha);
    free(hb);
    free(hc);

    // free device arrays
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

}