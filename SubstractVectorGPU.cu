#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU Kernel to perform vector substraction.
__global__ void VectorSubsKernel(float* ad, float* bd, float* cd, int size)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	// Perform vector substracion.
	if (threadId<size) 
		cd[threadId] = ad[threadId] - bd[threadId];
}


bool subtractVectorGPU( float* ad, float* bd, float* cd, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector.
	int bytes = size * sizeof(float);

	dim3 dimBlock(128, 1);   //Block dimension initialization. 
	
	int gridx = 1;		//x dimension. 
	int gridy = 1;		//y dimension. 
	if(size/128 < 65536)
		gridx = ceil((float)size/128);
	else{
		gridx = 65535;
		gridy = ceil((float)size/(128*65535));
	}
	dim3 dimGrid(gridx, gridy); // Grid initilization. 

	// Launch the kernel on a size-by-size block of threads
	VectorSubsKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	
	// Wait for completion
	cudaThreadSynchronize();
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << 
		std::endl;
		return false;
	}
	// Success
	return true;
}
