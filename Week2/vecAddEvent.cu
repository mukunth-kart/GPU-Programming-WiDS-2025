#include <iostream>
#include <random>
#include <cuda_runtime.h>

using namespace std;

//Kernel Definition
__global__ void vecADD(float* cuA, float* cuB, float* cuC, int n)
{
    int work_thread = blockDim.x*blockIdx.x + threadIdx.x;
    //Bounds Checking
    if (work_thread<n){
        cuC[work_thread] = cuA[work_thread] + cuB[work_thread];
    }
}

int main()
{
    float *A, *B, *C; //Pointers for CPU
    int n = 100000; /*Change to change length of arrays*/
    int num_threads = 32;
    size_t size = n*sizeof(float);

    cudaEvent_t start, stop;
    float time_elapsed;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *cuA, *cuB, *cuC; //Pointers for GPU

    //Allocate Memory to GPU floats
    cudaMalloc(&cuA, size);
    cudaMalloc(&cuB, size);
    cudaMalloc(&cuC, size);

    //Allocate Memory to CPU floats
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    //Random float generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    //Initializing the arrays
    for (int i=0; i<10; ++i){
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    //From CPU to GPU
    cudaMemcpy(cuA, A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuB, B, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(cuC, 0, n*sizeof(float));

    int num_blocks = (n+num_threads-1)/num_threads;
    
    cudaEventRecord(start, 0);
    vecADD<<<num_blocks, num_threads>>>(cuA, cuB, cuC, n); //Kernel Launch
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Calc elapsed time
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cout<<"Kernel execution time: "<<time_elapsed<<" ms"<<endl<<"Block Size: "<<num_threads<<endl;
    //cudaDeviceSynchronize(); //Wait for Kernel to complete
    cudaMemcpy(C, cuC, n*sizeof(float), cudaMemcpyDeviceToHost);

    //Verification
    int verify = 0;
    for(int i=0; i<n; ++i){
	verify = verify+(C[i] == A[i] + B[i]);
    }
    cout<<"No Of Correct = "<<verify<<endl;

    //Clean up
    cudaFree(cuA);
    cudaFree(cuB);
    cudaFree(cuC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

