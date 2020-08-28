// This code produces segmentation fault.
// I have intentionally written the code to print out GPU array element directly, which is NOT possible
#include <stdio.h>

__global__ void cube(float *d_out, float *d_in) {
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f * f;
    return;
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 96;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        h_in[i] = float(i);
    }

    // declare GPU memory pointers
    float *d_in;
    float *d_out;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    // launch the kernel
    cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // TODO - DO NOT DO THIS
    // Instead create a new array to store the results on the CPU and use cudaMemcpy to transfer from d_out to this new array
    // Then, print out the results from that CPU array
    // print out the resulting array
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%f", d_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
  
    return 0;
}
