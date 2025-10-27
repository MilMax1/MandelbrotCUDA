#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------------------
// Dll Exports
// --------------------------------------------------------------------
extern "C" __declspec(dllexport)
cudaError_t setCudaDevice(int device);

extern "C" __declspec(dllexport)
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

extern "C" __declspec(dllexport)
cudaError_t updateMandelCuda(
	unsigned int* pixelData, int width, int height,
	double mandelCenterX, double mandelCenterY,
	double mandelWidth, double mandelHeight,
	int mandelDepth);
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__device__ int IterCount(double cx, double cy, int mandelDepth)
{
	int result = 0;
	double x = 0.0;
	double y = 0.0;
	double xx = 0.0, yy = 0.0;

	while (xx + yy <= 4.0 && result < mandelDepth)
	{
		xx = x * x;
		yy = y * y;
		double xtmp = xx - yy + cx;
		y = 2.0 * x * y + cy;
		x = xtmp;
		result++;
	}
	return result;
}

__device__ int ClampInt(double val)
{
	if (val < 0.0) return 0;
	if (val > 255.0) return 255;
	return (int)val;
}

__device__ void HsvToRgb(int h, double S, double V, int* r, int* g, int* b)
{
	double H = h;
	while (H < 0) H += 360;
	while (H >= 360) H -= 360;

	double R, G, B;

	if (V <= 0) { R = G = B = 0; }
	else if (S <= 0) { R = G = B = V; }
	else
	{
		double hf = H / 60.0;
		int i = (int)floor(hf);
		double f = hf - i;
		double pv = V * (1 - S);
		double qv = V * (1 - S * f);
		double tv = V * (1 - S * (1 - f));

		switch (i)
		{
		case 0: R = V; G = tv; B = pv; break;
		case 1: R = qv; G = V; B = pv; break;
		case 2: R = pv; G = V; B = tv; break;
		case 3: R = pv; G = qv; B = V; break;
		case 4: R = tv; G = pv; B = V; break;
		case 5: R = V; G = pv; B = qv; break;
		default: R = G = B = V; break;
		}
	}

	*r = ClampInt(R * 255.0);
	*g = ClampInt(G * 255.0);
	*b = ClampInt(B * 255.0);
}


__global__ void mandelKernel(
	unsigned int* pixelData, int width, int height,
	double mandelCenterX, double mandelCenterY,
	double mandelWidth, double mandelHeight,
	int mandelDepth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int totalPixels = width * height;

	if (idx >= totalPixels) return;

	int row = idx / width;
	int col = idx % width;

	double cx = mandelCenterX - mandelWidth + col * ((mandelWidth * 2.0) / width);
	double cy = mandelCenterY - mandelHeight + row * ((mandelHeight * 2.0) / height);

	int light = IterCount(cx, cy, mandelDepth);

	int R, G, B;
	HsvToRgb(light, 1.0, light < mandelDepth ? 1.0 : 0.0, &R, &G, &B);

	pixelData[idx] = (R << 16) | (G << 8) | B;
}

extern "C" __declspec(dllexport)
cudaError_t updateMandelCuda(
	unsigned int* pixelData, int width, int height,
	double mandelCenterX, double mandelCenterY,
	double mandelWidth, double mandelHeight,
	int mandelDepth)
{
	cudaError_t cudaStatus;
	unsigned int* dev_pixels = nullptr;
	size_t totalSize = width * height * sizeof(unsigned int);

	// allocate GPU memory
	cudaStatus = cudaMalloc((void**)&dev_pixels, totalSize);
	if (cudaStatus != cudaSuccess) goto Error;

	// copy host pixel buffer to device 
	cudaStatus = cudaMemcpy(dev_pixels, pixelData, totalSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) goto Error;

	// launch kernel
	int threadsPerBlock = 256;
	int blocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	mandelKernel << <blocks, threadsPerBlock >> > (dev_pixels, width, height,
		mandelCenterX, mandelCenterY, mandelWidth, mandelHeight, mandelDepth);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) goto Error;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) goto Error;

	// copy result back
	cudaStatus = cudaMemcpy(pixelData, dev_pixels, totalSize, cudaMemcpyDeviceToHost);
Error:
	cudaFree(dev_pixels);
	return cudaStatus;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// --------------------------------------------------------------------
// CUDA Kernels
// --------------------------------------------------------------------

// Accepts pointers to three arrays and calculates c = a + b.
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i] + 1;
}



// --------------------------------------------------------------------
// Helper Functions
// --------------------------------------------------------------------

// This function accepts a CUDA device ID, and sets the CUDA device (GPU).
cudaError_t setCudaDevice(int device)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	// Note! Can be omitted if the default device (0) is used.
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	return cudaStatus;
}

// This function uses CUDA to add vectors in parallel.
// It accepts pointers to three arrays, sets the CUDA device (GPU),
// allocates device buffers and copied the host buffers to them, 
// launches a vector addition CUDA kernel, copied the device output
// buffer to the host output buffer (array), and finally
// frees the device buffers.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output) .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
			"cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
			cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	

	return cudaStatus;

	


}