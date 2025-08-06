#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32;
#define BLOCK_SIZE 64

__global__ void demonstrate_shuffle_functions(float* input, float* output, int n ){

    int idx = blockIdx.x  * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    

    if (idx<n){

        float value = input[idx];

        float from_lane_0 = __shfl_synx(0xFFFFFFFF, value, 0);

        float from_lane_1 = __shfl_synx(0xFFFFFFFF, value, 1);


        float value_from_lane_3 = __shfl_synx(0xFFFFFFFF, value, 15);

    }
}

