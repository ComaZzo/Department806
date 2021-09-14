// System includes
#include <assert.h>
#include <stdio.h>
#include<math.h>
#include <time.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "device_launch_parameters.h"

//���������� ������� (<= 1024!)
#define N 1024
//���������� ������
#define BL 7

#define M_PI acos(-1.0)

//��������
#define A 0
#define B 2*M_PI
 

//��� ����������� �� GPU
__global__ void kernel(double* x_values, double* y_values){
    //�������� ������ ������ � �����
    uint16_t idx = threadIdx.x;
    uint16_t bx = blockIdx.x;

    //��������� x � y
    double x = A + bx * 1.0 + ((idx * 1.0)/ N);

    if (x > B)
        return;

    double y = tan(x);

    //��������� ���������� ����� � �������
    uint16_t position = bx * N + idx;

    //���������� � ������
    x_values[position] = x;
    y_values[position] = y;

	//printf("x=>%.10f y=>%.10f t=>%d b=>%d \n",xv, yv, idx, bx);
}

int main(int argc, char* argv)
{
    //---���������� �� ����������---
	printf("[Tan computing Using CUDA] - Starting...\n");

    cudaStream_t stream;

    // ��������� ������ �� ����������
    uint16_t size = N * BL;
    uint16_t mem_size = size * sizeof(double);
    double *p_h_X, *p_h_Y;
    checkCudaErrors(cudaMallocHost(&p_h_X, mem_size));
    checkCudaErrors(cudaMallocHost(&p_h_Y, mem_size));

    // ��������� ������ �� ���������� � ������������� ���� �������
    double *p_d_X, *p_d_Y;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&p_d_X), mem_size));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&p_d_Y), mem_size));

    checkCudaErrors(cudaDeviceSynchronize());

    // �������� ������� � ������ ��� �������
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    
    printf("Computing result using CUDA Kernel...\n");

    // ������ ������ �������
    checkCudaErrors(cudaEventRecord(start, stream));

    // ���������� ���� �� ���������� � �������� ���������� ���� �������

    kernel <<<BL, N>>> (p_d_X, p_d_Y);

    checkCudaErrors(cudaStreamSynchronize(stream));

    // ������ ��������� �������
    checkCudaErrors(cudaEventRecord(stop, stream));

    // ������������� � �������� ������������ �������
    checkCudaErrors(cudaEventSynchronize(stop));

    // ������ � ����� ������������������

    float m_sec_total = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&m_sec_total, start, stop));

    float mc_sec_total = m_sec_total * 1000;
    printf(
        "Time GPU = %.10f microsec\n",
        mc_sec_total);

    // ����������� ����������� � GPU �� CPU
    checkCudaErrors(
        cudaMemcpyAsync(p_h_X, p_d_X, mem_size, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(
        cudaMemcpyAsync(p_h_Y, p_d_Y, mem_size, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));


    //---���������� �� ����������---

    //������� ���������� ����� ��������� �� ��������� � ��������� �����
    uint16_t points_count = (uint16_t) ceil(N*(B-A));

    // ��������� ������ ��� ������
    float* p_Y = (float*)malloc(points_count * sizeof(float));
    float* p_X = (float*)malloc(points_count * sizeof(float));

    // �������� ����� �������
    auto begin = std::chrono::high_resolution_clock::now();

    uint16_t z = 0;
    for (float i = A; i <= B; i += 1.0 / N)
    {
        if (z > points_count)
            break;
        p_X[z] = i;
        p_Y[z] = tan(i);
        z++;
    }

    // ������������� ������ � ������� ����� ����������
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf(
        "Time CPU = %.10f microsec\n",
        elapsed.count() * 1e-3);

    //�������� ��������

    uint16_t wrong_values_count = 0;

    for (z = 0; z < points_count; z++)
    {
        if (abs(p_Y[z] - p_h_Y[z]) > 0.1)
        {
            printf(
                "z = %d, x_CPU = %.10f x_GPU = %.10f y_CPU = %.10f, y_GPU = %.10f\n",
                z, p_X[z], p_h_X[z], p_Y[z], p_h_Y[z]);
            wrong_values_count++;
        }

        //printf(
        //    "z = %d, x = %.10f CPU = %.10f, GPU = %.10f\n",
        //    z, p_h_X[z], p_Y[z], p_h_Y[z]);
    }

    printf("Total values: %d\n", points_count);
    printf("Total wrong values: %d\n", wrong_values_count);
    if (wrong_values_count == 0)
    {
        printf("TEST PASSED!\n");
    }
    else
    {
        printf("TEST FAILED!\n");
        exit(1);
    }

    //������ � ����
    FILE *p_file = fopen("values.txt", "w");

    if (p_file == NULL)
    {
        printf("Error creating file!");
        exit(1);
    }
    for (z = 0; z < points_count; z++)
    {
        fprintf(p_file, "%.10f %.10f\n", p_h_X[z], p_h_Y[z]);
    }

    fclose(p_file);
    printf("Writing to file complete");

    // ������������ ������
    checkCudaErrors(cudaFreeHost(p_h_X));
    checkCudaErrors(cudaFreeHost(p_h_Y));
    checkCudaErrors(cudaFree(p_d_X));
    checkCudaErrors(cudaFree(p_d_Y));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    cudaDeviceReset();

    free(p_Y);
    free(p_X);

	return 0;
}
