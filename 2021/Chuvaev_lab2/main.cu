#include <stdio.h>
#include <ctime>
#include <cmath>

//longint позволяет рассчитать только первые 31 числа Мерсенна
#define N 31
#define n 1241415

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


__global__ void mersen_num_global(long int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int temp = 0;
    if (tid < N)
        temp = pow(2, tid + 1) - 1;
    if (temp < n)
        c[tid] = temp;
}

__global__ void mersen_num_shared(long int *c){
    __shared__ int cache[N];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    if (tid < N)
        cache[cacheIndex] = pow(2, tid + 1) - 1;
    __syncthreads();
    if (cache[cacheIndex] < n){
        c[tid] = cache[cacheIndex];
    }
}

void global_mem_execute(){
    cudaEvent_t startg, stopg;
	float gpuTimeg = 0.0f;
	cudaEventCreate(&startg);
	cudaEventCreate(&stopg);
    cudaEventRecord(startg, 0);
    long int c[N];
    long int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, 
                             sizeof(long int) * N));

    mersen_num_global<<<N, N>>>(dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, 
                        dev_c, 
                        N * sizeof(long int), 
                        cudaMemcpyDeviceToHost));
    cudaEventRecord(stopg, 0);
	cudaEventSynchronize(stopg);
  	cudaEventElapsedTime(&gpuTimeg, startg, stopg);

    printf("\n====================   GPU TIME (global only memory)   ====================\n");
  	printf("\nGPU compute time: %.2f milliseconds\n\n", gpuTimeg);
    printf("Mersenn numbers less than %d: ", n);
    for(int i=0; i<N && c[i] != 0; i++)
    printf("%d ", c[i]);
    cudaFree(dev_c);
}

void shared_mem_execute(){
    cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    long int c[N];
    long int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, 
                             sizeof(long int) * N));

    mersen_num_shared<<<N, N>>>(dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, 
                        dev_c, 
                        N * sizeof(long int), 
                        cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&gpuTime, start, stop);

    printf("\n====================   GPU TIME (with shared memory)   ====================\n");
  	printf("\nGPU compute time: %.2f milliseconds\n\n", gpuTime);
    printf("Mersenn numbers less than %d: ", n);                        
    for(int i=0; i<N && c[i] != 0; i++){
    printf("%d ", c[i]);}
    printf("\n");
    cudaFree(dev_c);
}
void cpu_execute(){
    int start2, time2;
    start2 = clock();

    int b[N];
    int temp = 0;
    for (int i = 0; i < N; i++)
        b[i] = pow(2, i + 1) - 1;

    time2 = clock() - start2;
	printf("==============================   CPU TIME   ===============================\n");
	printf("\nCPU compute time: %.2f milliseconds\n\n", time2);
    printf("Mersenn numbers less than %d: ", n);
    for(int i=0; i<N && b[i] < n; i++)
    printf("%d ", b[i]);
    printf("\n");
}

int main(void){
    shared_mem_execute();
    global_mem_execute();
    cpu_execute();
    return 0;
}