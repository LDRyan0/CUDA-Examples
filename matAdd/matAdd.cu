#include <stdio.h>

#define M 6 // number of rows
#define N 8 // number of columns

#define numThread 16
#define numBlock 4

__global__ void matHad(float *A, float *B, float *C) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = x + y*N;

    printf("%d ", idx);

    C[idx] = A[idx] + B[idx];
}

void print2DArray(float *arr) {
    for(int i=0; i < M*N; i++) {
        if ((i % N == 0) && (i != 0)) {
            printf("\n");
        }
        printf("%0.2e ", arr[i]);
    }
    printf("\n\n");
}

int main(int argc, char* argv[]) {

    float *ha, *hb, *hc; // host (CPU) arrays
    float *da, *db, *dc; // device (GPU) arrays

    printf("Creating host and device arrays...\n\n");
    ha = (float*)malloc(M*N*sizeof(float));
    hb = (float*)malloc(M*N*sizeof(float));
    hc = (float*)malloc(M*N*sizeof(float));

    for (int i=0; i < M*N; i++) {
        ha[i] = 1.0*i;
        hb[i] = 2.0*i;
    }

    cudaMalloc((void**)&da, M*N*sizeof(float));
    cudaMalloc((void**)&db, M*N*sizeof(float));
    cudaMalloc((void**)&dc, M*N*sizeof(float));

    // copy host arrays over to device
    cudaMemcpy(da, ha, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, M*N*sizeof(float), cudaMemcpyHostToDevice);

    // output input matrices
    print2DArray(ha);
    print2DArray(hb);
    
    printf("Performing elementwise multiplication...\n\n");
    printf("Thread sequence: ");
    // perform parralised addition of matrics
    matHad<<<numBlock, numThread>>>(da, db, dc);

    cudaMemcpy(hc, dc, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n\n");

    // output resultant matrix
    print2DArray(hc);
    
    // check that the addition has been performed correctly
    // absolute error 1e-5
    bool success = true;
    int total=0;
    printf("Checking %d values in the array...\n\n", M*N);
    for (int i=0; i<M*N; i++) {
        if (abs((ha[i] + hb[i]) - hc[i]) > 1e-5) {
            printf( "Error:  %0.2e + %0.2e != %0.2e\n", ha[i], hb[i], hc[i] );
            success = false;
        } else {
            total += 1;
        }
    }
    if (success) {
        printf("%d out of %d values correct!\n", total, M*N);
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