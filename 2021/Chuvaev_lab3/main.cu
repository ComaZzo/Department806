#include <stdio.h>
#include <ctime>

#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define E 10e-4
#define BLOCKS 32
#define THREADS 128
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__device__ __host__ float func(float x){
    return log(3.66*x) - 4.12 * x + 1.5;
}


__device__ __host__ float func_derivative(float x){
    return 1 / (3.66*x) - 4.12;
}


__device__ __host__ float func_derivative_second(float x){
    return -1 / (3.66*x*x);
}


__global__ void newton_method(float *c, double step, float A){   
    float a = A + blockIdx.x * threadIdx.x * step;
    float b = A + (blockIdx.x * threadIdx.x + 1) * step;
    if (func(a) * func(b) > 0)
        return;
    double calc;
    do{
        calc = calc - func(calc) / func_derivative(calc);
    }while (fabs(func(calc)) >= E);
    c[blockIdx.x * threadIdx.x] = calc;
}


void cpu_execute(){
    int start = clock(), time;
    double c=2; 
    int n=0;
    while (fabs(func(c))>=E)
    {
        c = c-func(c) / func_derivative(c);
        n++;
    }
    time = clock() - start;
 	printf("==============================   CPU TIME   ===============================\n");
    printf("Equation root = %lf\n",c);
    printf("Iteration number: n = %d\n",n); 
 	printf("\nCPU compute time: %.5f microseconds\n\n", time*1000);
}

void gpu_execute(){
    float B = 10, A = 0.01; // common borders

    cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    const int n = THREADS * BLOCKS;
    float step = fabs(A - B) / n;
    // float c[n];
    // float* dev_c;
    thrust::host_vector<float> c (n);
    thrust::device_vector<float> dev_c (n);

    // cudaMalloc(&dev_c, n * sizeof(float));

    newton_method <<<BLOCKS, THREADS >>> (thrust::raw_pointer_cast(dev_c.data()), step, A);
    // cudaMemcpy(c, 
    //                         dev_c, 
    //                         n * sizeof(unsigned int), 
    //                         cudaMemcpyDeviceToHost); 
    c = dev_c;

    HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
  	HANDLE_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
    printf("==============================   GPU TIME   ===============================\n");
 	printf("\nGPU compute time: %.5f microseconds\n\n", gpuTime);

    for (unsigned int i = 0; i < c.size(); i++)
    {
        if (c[i] > E)
        {
            printf("GPU root %f \n", c[i]);
        }
    }
    // cudaFree(dev_c);
    // cudaDeviceReset();
}

int main(void){
    gpu_execute();
    cpu_execute();
    return 0;
}