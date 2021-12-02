#include <cuda.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}   

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        printf("cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }    
#endif

    return;
}
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

texture <float,2,cudaReadModeElementType> tex1;

static cudaArray *cuArray = NULL;


__global__ void sobel_kernel(float* output,int width,int height,int widthStep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<width && y<height)
    {
        float output_value_x = (-1*tex2D(tex1,x-1,y-1)) + (0*tex2D(tex1,x,y-1)) + (1*tex2D(tex1,x+1,y-1))
                           + (-2*tex2D(tex1,x-1,y))   + (0*tex2D(tex1,x,y))   + (2*tex2D(tex1,x+1,y))
                           + (-1*tex2D(tex1,x-1,y+1)) + (0*tex2D(tex1,x,y+1)) + (1*tex2D(tex1,x+1,y+1));

        float output_value_y = (-1*tex2D(tex1,x-1,y-1)) + (-2*tex2D(tex1,x,y-1)) + (-1*tex2D(tex1,x+1,y-1))
                           + (0*tex2D(tex1,x-1,y))   + (0*tex2D(tex1,x,y))   + (0*tex2D(tex1,x+1,y))
                           + (1*tex2D(tex1,x-1,y+1)) + (2*tex2D(tex1,x,y+1)) + (1*tex2D(tex1,x+1,y+1));
        
        float output_value = pow(pow(output_value_x,2)+pow(output_value_y,2),0.5);

        output[y*widthStep+x]=output_value;
    }

}


void gpu_execute(float* input,float* output,int width,int height,int widthStep)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    CudaSafeCall(cudaMallocArray(&cuArray,&channelDesc,width,height));

    cudaMemcpy2DToArray(cuArray,0,0,input,widthStep,width * sizeof(float),height,cudaMemcpyHostToDevice);

    cudaBindTextureToArray(tex1,cuArray,channelDesc);

    float * D_output_x;
    CudaSafeCall(cudaMalloc(&D_output_x,widthStep*height)); 

    dim3 blocksize(16,16);
    dim3 gridsize;
    gridsize.x=(width+blocksize.x-1)/blocksize.x;
    gridsize.y=(height+blocksize.y-1)/blocksize.y;

    sobel_kernel<<<gridsize,blocksize>>>(D_output_x,width,height,widthStep/sizeof(float));

    cudaThreadSynchronize();
    CudaCheckError();

    cudaUnbindTexture(tex1);

    CudaSafeCall(cudaMemcpy(output,D_output_x,height*widthStep,cudaMemcpyDeviceToHost));

    cudaFree(D_output_x);
    cudaFreeArray(cuArray);
}



int main(int argc, char** argv) 
{
    IplImage* image;

    image = cvLoadImage("table.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    if(!image )
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }


    IplImage* image_sobel = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,image->nChannels);
    IplImage* image1 = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,image->nChannels);

    cvConvert(image,image1);

    float *output = (float*)image_sobel->imageData;
    float *input =  (float*)image1->imageData;

    cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    gpu_execute(input, output, image->width,image->height, image1->widthStep);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
  	HANDLE_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
    printf("==============================   GPU TIME   ===============================\n");
 	printf("\nGPU compute time: %.5f microseconds\n\n", gpuTime);
    
    cvScale(image_sobel,image_sobel,1.0/255.0);

    cvShowImage("Original Image", image );
    cvShowImage("Sobeled Image", image_sobel);
    cvWaitKey(0);
    return 0;
}
