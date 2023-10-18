// This piece of code aims to execute a basic elementwise multiplication 
// of two matrices A and B using CUDA

#include <stdio.h> // standard input/output

// declaring matrix dimensions
#define N 8
#define numThread 2 // 2 threads in each block
#define numBlock 4 // 4 blocks

// __global__ means that function is accessible from host (CPU)
//      can be called with <<..>> CUDA notation and threads will independtly execute 
__global__ void vecAdd(float *A, float *B, float *C) {
    // each thread first finds the correct index
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    printf("%d ", index);

    // perform the multiplication on the assigned index 
    C[index] = A[index] + B[index];
}

void printArray(float* arr) {
    for(int i=0; i < N; i++) {
        printf("%0.2e ", arr[i]);
    }
    printf("\n\n");
}

int main(int argc, char* argv[]) {
    float *a, *b, *c;               // arrays for host CPU machine
    float *dev_a, *dev_b, *dev_c;   // arrays for GPU device
    
    printf("Creating host and device arrays...\n\n");
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));

    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // create the arrays using void**
    // stored contiguously in memory so can access like a 1D array
    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_c, N * sizeof(float));

    // copy host arrays to device
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    // output the input vectors
    // printArray(a);
    // printArray(b);

    printf("Performing elementwise addition...\n\n");
    printf("Thread sequence: ");
    // perform elementwise addition on all elements of arrays a and b
    vecAdd<<<numBlock, numThread>>>(dev_a, dev_b, dev_c);

    // copy output array back to host
    cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n\n");

    // output the resultant vector
    // printArray(c);

    // check that the addition has been performed correctly
    // absolute error 1e-5
    bool success = true;
    int total=0;
    printf("Checking %d values in the array...\n\n", N);
    for (int i=0; i<N; i++) {
        if (abs((a[i] + b[i]) - c[i]) > 1e-5) {
            printf( "Error:  %0.2e + %0.2e != %0.2e\n", a[i], b[i], c[i] );
            success = false;
        } else {
            total += 1;
        }
    }
    if (success) {
        printf("%d out of %d values correct!\n", total, N);
    }

    // free CPU allocated memory
    free(a);    
    free(b);    
    free(c);    

    // free GPU allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
