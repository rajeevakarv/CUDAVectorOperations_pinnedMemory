#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <cstdlib> // malloc(), free() 
#include <iostream> // cout, stream 
#include "common.h"
#include <bitset>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


const int ITERS = 1000;   //No of iterations we want to execute.
const int SIZE = 65536; // set the size of vectors to use.
/******
CPU function for addition of the vectors.
******/
void addVectorCPU( float* a, float* b, float* c, int size ){
	for (int i=0; i<size; i++)
		c[i] = a[i] + b[i];
}
/******
CPU function for substration of the vectors.
******/
void subtractVectorCPU( float* a, float* b, float* c, int size ){

	for (int i=0; i<size; i++){
		c[i] = a[i] - b[i];
	}
}
/******
CPU function for scaling of the vectors.
******/
void scaleVectorCPU( float* a, float* c, float scaleFactor, int size )
{
	for (int i = 0; i < size; i++) {
		c[i] = a[i] * scaleFactor; 
	}
}


int main() 
{ 
	cudaSetDeviceFlags( cudaDeviceMapHost );   //Instructed by manual to use pinned memory.

	float* a, *b, *cgpu;     // a, b, cgpu represents the input vectors and gpu output vector.
	float* ccpu = new float[SIZE];  //CPU output vector.
	float scaleFactor = 10;   //This is the scaling factor.
	clock_t start, end;    //Clock veriable
	float tcpu, tgpu;      //timing verialbles.
	float sum, L2norm1, delta, L2norm2, L2norm3;    //L2 norm and delta error arialbles.

	// these three calls allocated pinned memory to a, b and cgpu
	cudaHostAlloc((void**)&a, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&b, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&cgpu, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);

	// Initialize a and b to random integers 
	for (int i = 0; i < SIZE; i++) {
		a[i] = ((float) rand()) / (float) 1;
		b[i] = ((float) rand()) / (float) 1;
	} 

	std::cout << "Operating on a vector of length " << SIZE << std::endl;
	// Adition of the two vectors on the host
	start = clock();     //start of clock.
	for (int i = 0; i < ITERS; i++) {
		addVectorCPU(a, b, ccpu, SIZE);
	}
	end = clock();   // end of timing clock after 1000 iterations. 
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(8);
	std::cout << "CPU Addition took " << tcpu << " ms:" << std::endl;

	float *dA, *dB, *dCGPU;   //dA, dB and dCGPU are pointers to pass for kernel calls. 

	// Allocating all the pointers to original data.
	cudaHostGetDevicePointer( (void**)&dA, a, 0 );
	cudaHostGetDevicePointer( (void**)&dB, b, 0 );
	cudaHostGetDevicePointer( (void**)&dCGPU, cgpu, 0 );

	//One call to warm-up GPU. 
	bool success = addVectorGPU(dA, dB, dCGPU, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// Now we run the iterations. 
	start = clock();     //Timer starts	
	for (int i = 0; i < ITERS; i++) {
		addVectorGPU(dA, dB, dCGPU, SIZE);
	}
	end = clock();      //End timer.
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(9);
	std::cout << "GPU Addition took " << tgpu << " ms:" << std::endl;

	std::cout << "Addition speedup = " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);   //delta counts the error in two outputs.
		sum += (ccpu[i] * cgpu[i]);							  
	}

	L2norm1 = sqrt(delta / sum);
	std::cout << "Addition error = " << L2norm1 << "\n" << std::endl;


	//Substraction of the two vectors on host.
	
	start = clock();                         //timer start
	for (int i = 0; i < ITERS; i++) {
		subtractVectorCPU(a, b, ccpu, SIZE);	//CPU function call.
	}
	end = clock();							 //timer ends
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(8);
	std::cout << "CPU Subtraction took " << tcpu << " ms:" << std::endl;

	// Perform one warm-up pass and validate
	success = subtractVectorGPU(dA, dB, dCGPU, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// All the iteration are called now.
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		subtractVectorGPU(dA, dB, dCGPU, SIZE);			//GPU functions. 
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the result. 
	std::cout << "GPU Subtraction took " << tgpu << " ms:" << std::endl;
	std::cout << "Subtraction speedup = " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);   //Delta for the results. 
		sum += (ccpu[i] * cgpu[i]);
	}

	L2norm2 = sqrt(delta / sum);
	std::cout << "Subtraction Error = " << L2norm2 << "\n" << std::endl;
	
	//Scale code for CPU. 
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		scaleVectorCPU( a, ccpu, scaleFactor, SIZE );
	}
	end = clock();   //end the timer.
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(8);
	std::cout << "CPU Scale function took " << tcpu << " ms:" << std::endl;

	// Perform one warm-up pass and validate
	success = subtractVectorGPU(dA, dB, dCGPU, SIZE);
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}

	// All the iterations being called now. 
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		scaleVectorGPU( dA, dCGPU, scaleFactor, SIZE );     
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(8);
	std::cout << "GPU Scaling took " << tgpu << " ms:" << std::endl;
	std::cout << "Scaling speedup = " << (float)tcpu/tgpu << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
		sum += (ccpu[i] * cgpu[i]);
	}

	L2norm3 = sqrt(delta / sum);
	std::cout << "Scaleing Error = " << L2norm3 << "\n\n"<< std::endl;

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(ccpu);
	cudaFreeHost(cgpu);
	getchar();
	return 0;
} 
