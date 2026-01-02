#include <iostream>
#include <random>
#include <cuda_runtime.h>

using namespace std;

//Kernel Definition
__global__ void mult_scale(float* A, float* B, float* C, float scale, int n)
{
    extern __shared__ float shared_mem[];
    float* a = shared_mem;
    float* b = shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        a[threadIdx.x] = A[idx];
        b[threadIdx.x] = B[idx];
        __syncthreads();
        C[idx] = a[threadIdx.x] * b[threadIdx.x] * scale;
    }
}

int main()
{
	int n = 1000; //Size of array
	int num_threads = 256; //num_threads
	size_t size = n*sizeof(float);
	float scale = 0.1; //alpha value

	float *A, *B, *C;
	float *cuA, *cuB, *cuC;

	//To record time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//Allocate Memory

	cudaMallocHost(&A, size);
	cudaMallocHost(&B, size);
	cudaMallocHost(&C, size);

	cudaMalloc(&cuA, size);
	cudaMalloc(&cuB, size);
	cudaMalloc(&cuC, size);

	//Random Generation Setup
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	//Initialize Array
	for (int i=0; i<n; ++i)
	{
		A[i] = dist(gen);
		B[i] = dist(gen);
	}

	//Copy to GPU
	cudaMemcpy(cuA, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuB, B, size, cudaMemcpyHostToDevice);
	cudaMemset(cuC, 0.0, size);

	//Kernel Launch
	int num_blocks  = (n+num_threads-1)/num_threads;
	size_t shared_mem_size = 2 * num_threads * sizeof(float);

	//Warm-up run (to avoid last week's mishaps)
	mult_scale<<<num_blocks, num_threads, shmem_size>>>(
    cuA, cuB, cuC, scale, n
	);
	cudaDeviceSynchronize();

	cudaEventRecord(start);
	mult_scale<<<num_blocks, num_threads, shared_mem_size>>>(cuA, cuB, cuC, scale, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Kernel execution time: " << milliseconds << " ms" << endl;

	//Copy Back To CPU
	cudaMemcpy(C, cuC, size, cudaMemcpyDeviceToHost);


	//Verify
	int verify = 0;
	float epsilon = 1e-3;
	for(int i=0; i<n; ++i)
	{
		verify = verify+(fabs(C[i]-scale*A[i]*B[i])<epsilon);
	}

	cout<<"No of Correct = "<<verify<<endl;

	//Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFreeHost(&A);
	cudaFreeHost(&B);
	cudaFreeHost(&C);
	cudaFree(&cuA);
	cudaFree(&cuB);
	cudaFree(&cuC);
}

