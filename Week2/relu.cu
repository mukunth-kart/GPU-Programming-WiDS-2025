#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

//Kernel Definition
__global__ void relu(float* input, float* output, int vectorlength)
{
	int work_thread = blockDim.x*blockIdx.x + threadIdx.x;

	if (work_thread<vectorlength)
	{
		output[work_thread] = max(input[work_thread], 0.0);
	}
}

int main(){
	int n = 1000;//Length of array
	int num_threads = 256;
	size_t size = n*sizeof(float);

	float *A,*B;
	float *cuA,*cuB;

	//Allocate Memory
	cudaMallocHost(&A, size);
	cudaMallocHost(&B, size);

	cudaMalloc(&cuA, size);
	cudaMalloc(&cuB, size);

	//Random Generator Setup
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dist(-2.0f, 2.0f);

	//Initialize Array
	for(int i=0; i<n; ++i)
	{
		A[i] = dist(gen);
	}

	//From CPU to GPU
	cudaMemcpy(cuA, A, size, cudaMemcpyHostToDevice);
	cudaMemset(cuB, 0, size);

	//Kernel Launch
	int num_blocks  = (n+num_threads-1)/num_threads;
	relu<<<num_blocks, num_threads>>>(cuA, cuB, n);

	//Copy back to CPU
	cudaMemcpy(B, cuB, size, cudaMemcpyDeviceToHost);

	//Verify
	int verify = 0;
	for (int i=0; i<n; ++i)
	{
		verify = verify + (B[i]==max(0.0, A[i])); 
	}
	cout<<"No of correct: "<<verify<<endl;

	//Free Up Memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFree(cuA);
	cudaFree(cuB);

}
