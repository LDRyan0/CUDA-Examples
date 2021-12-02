#include <stdio.h>

#define N 8 //number of rows

#define numThread 16
#define numBlock 4

__global__ void matAdd(float *A, float *B, float *C) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = x + y*N;

    printf("%d ", idx);

    C[idx] = A[idx] + B[idx];
}

void print2DArray(float *arr) {
    for(int i=0; i < N*N; i++) {
        if ((i % N == 0) && (i != 0)) {
            printf("\n");
        }
        printf("%0.2e ", arr[i]);
    }
    printf("\n\n");
}

int main(int argc, char* argv[]) {

    float *ha, *hb, *hc;
    float *da, *db, *dc;

    printf("Creating host and device arrays...\n\n");
    ha = (float*)malloc(N*N*sizeof(float));
    hb = (float*)malloc(N*N*sizeof(float));
    hc = (float*)malloc(N*N*sizeof(float));

    for (int i=0; i < N*N; i++) {
        ha[i] = 1.0*i;
        hb[i] = 2.0*i;
    }

    cudaMalloc((void**)&da, N*N*sizeof(float));
    cudaMalloc((void**)&db, N*N*sizeof(float));
    cudaMalloc((void**)&dc, N*N*sizeof(float));

    // copy host arrays over to device
    cudaMemcpy(da, ha, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // output input matrices
    print2DArray(ha);
    print2DArray(hb);
    
    printf("Performing elementwise addition...\n\n");
    printf("Thread sequence: ");
    // perform parralised addition of matrics
    matAdd<<<numBlock, numThread>>>(da, db, dc);

    cudaMemcpy(hc, dc, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n\n");

    // output resultant matrix
    print2DArray(hc);
    
    // check that the addition has been performed correctly
    // absolute error 1e-5
    bool success = true;
    int total=0;
    printf("Checking %d values in the array...\n\n", N*N);
    for (int i=0; i<N*N; i++) {
        if (abs((ha[i] + hb[i]) - hc[i]) > 1e-5) {
            printf( "Error:  %0.2e + %0.2e != %0.2e\n", ha[i], hb[i], hc[i] );
            success = false;
        } else {
            total += 1;
        }
    }
    if (success) {
        printf("%d out of %d values correct!\n", total, N*N);
    }

    // free host arrays
    free(ha);
    free(hb);
    free(hc);

    // free device arrays
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

}