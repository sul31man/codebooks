#include <stdio.h>
#include <cuda_runtime.h>


__global__ void basic_kernel(int* A, int n ){


    int idx = blockDim * blockIdx + threadIdx;

    if (idx < n){

        A[idx] = A[idx] + 1;

    }

    //super basic kernel that we will use to set an example of a cuda graph
}

void checkCudaError(cudaError_t error){

    if (error != cudaSuccess){

        printf("brav");
    }
}

int main(){

    int* A; //intialise the pointe to our array
    int* A_d; 
    int n = 1024; 

    cudaGraph_t graph ;
    cudaGraphExec_t graph_exec; 

    cudaStream_t stream; 
    cudaStreamCreate(&stream);

    cudaHostAllocate(&A, n*sizeof(int) , cudaHostAllocDefault);//this frees up the memory efficiently for our array and assigns it to A
    cudaMalloc(&A_d, n*sizeof(int));
    //now lets assign some random numbers to the array;

    for(int i=0; i < n; i++){

        A[i] = i;
    }

    
    
    int threadPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1)/(threadsPerBlock) ; 

    checkCudaError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    cudaMemcpyAsync(A_d, A, n*sizeof(int), cudaHostToDevice, stream);

    basic_kernel<<<blocks, threadsPerBlock, 0, stream>>>(A_d, n);

    cudaMemcpyAsync(A, A_d, n*sizeof(int), cudaDeviceToHost, stream);

    checkCudaError(cudaStreamEndCapture(stream, &graph));

    checkCudaError(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    cudaGraphLaunch(graph_exec, stream);


    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGLobal);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graph_exec, stream);


    

    



}