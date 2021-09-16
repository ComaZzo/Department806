#include <stdio.h>
#include <ctime>

#define ARRAY_LEN 1000000


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void cpu_execute(){
    int start2, time2;
    start2 = clock();

    int b[ARRAY_LEN];
    memset(b, 1, sizeof(int)*ARRAY_LEN);
    time2 = clock() - start2;
	printf("====================   CPU TIME   ====================\n");
	printf("\nCPU compute time: %f milliseconds\n\n", time2);

}

__global__ void initialize(int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ARRAY_LEN)
        c[tid] = 1;
}

int main(int argc, char* argv[]){
    int c[ARRAY_LEN];
    int *dev_c;

    cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, 
                             sizeof(int) * ARRAY_LEN));

    initialize<<<ceil(ARRAY_LEN / 1024), 1024>>>(dev_c);

    HANDLE_ERROR(cudaMemcpy(c, 
                            dev_c, 
                            ARRAY_LEN * sizeof(int), 
                            cudaMemcpyDeviceToHost));

    cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&gpuTime, start, stop);

    printf("\n====================   GPU TIME   ====================\n");
  	printf("\nGPU compute time: %.2f milliseconds\n\n", gpuTime);
    cudaFree(dev_c);
    //cpu_execute();
    return 0;
}