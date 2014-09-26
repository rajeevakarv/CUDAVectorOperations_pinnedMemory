#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU Kernel to perform vector addition.
__global__ void VectorAddKernel(float* ad, float* bd, float* cd, int size)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	
	//Addition of the vectors. 
	if (threadId<size) 
		cd[threadId] = ad[threadId] + bd[threadId];
}

bool addVectorGPU( float* ad, float* bd, float* cd, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector.
	int bytes = size * sizeof(float);

	dim3 dimBlock(128, 1);   //block dims. 
	int gridx = 1;	  // x dim for grid.
	int gridy = 1;	 // y dim for grid.
	if(size/128 < 65536)
		gridx = ceil((float)size/128);
	else{
		gridx = 65535;
		gridy = ceil((float)size/(128*65535));
	}
	dim3 dimGrid(gridx, gridy); // Grid call for dimension. 

	// Launch the kernel on a size-by-size block of threads
	VectorAddKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	
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
