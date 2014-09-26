#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU Kernel to perform vector scaling.
__global__ void VectorScaleKernel(float* ad, float* cd, float scaleFactord, int size)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	float scaleFactor = scaleFactord;

	// Perform scaling 
	if (threadId<size) 
	cd[threadId] = ad[threadId] * scaleFactor;
}

bool scaleVectorGPU( float* ad, float* cd, float scaleFactor, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector.
	int bytes = size * sizeof(float);
	float scaleFactord = scaleFactor;   //Scale factor. 

	dim3 dimBlock(128, 1);   //Block dimensions.
	int gridx = 1;   // Grid X co-ordinate.
	int gridy = 1;	 // Grid Y co-ordinate. 
	if(size/128 < 65536)
		gridx = ceil((float)size/128);
	else{
		gridx = 65535;
		gridy = ceil((float)size/(128*65535));
	}
	dim3 dimGrid(gridx, gridy);    //Grid dimensions.

	// Launch the kernel on a size-by-size block of threads
	VectorScaleKernel<<<dimGrid, dimBlock>>>(ad, cd, scaleFactord, size);
	
	// Wait for completion
	cudaThreadSynchronize();
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << 
		std::endl;
		return false;
	}
	//Return success.
	return true;
}
