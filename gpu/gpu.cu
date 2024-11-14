#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

// Here we hold the number of cells we have in the x and y directions
int nx, ny;

// This is where all of our points are. We need to keep track of our active
// height and velocity grids, but also the corresponding derivatives. The reason
// we have 2 copies for each derivative is that our multistep method uses the
// derivative from the last 2 time steps.
double *h;
double H, g, dx, dy, dt;

// GPU device pointers
double *gpu_h, *gpu_u, *gpu_v;
double *gpu_dh, *gpu_du, *gpu_dv;
double *gpu_dh1, *gpu_du1, *gpu_dv1;
double *gpu_dh2, *gpu_du2, *gpu_dv2;

/**
 * This is your initialization function! We pass in h0, u0, and v0, which are
 * your initial height, u velocity, and v velocity fields. You should send these
 * grids to the GPU so you can do work on them there, and also these other fields.
 * Here, length and width are the length and width of the domain, and nx and ny are
 * the number of grid points in the x and y directions. H is the height of the water
 * column, g is the acceleration due to gravity, and dt is the time step size.
 * The rank and num_procs variables are unused here, but you will need them
 * when doing the MPI version.
 */
void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    // @TODO: your code here
    // TODO: Your code here
    // We set the pointers to the arrays that were passed in
    nx = nx_;
    ny = ny_;

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;

    // Allocate GPU memory for h, u, v and their derivatives
    cudaMalloc((void **)&gpu_h, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_u, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_v, nx * ny * sizeof(double));

    // Transfer data from host to GPU
    cudaMemcpy(gpu_h, h0, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_u, u0, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v, v0, nx * ny * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&gpu_dh, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_du, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_dv, nx * ny * sizeof(double));

    cudaMalloc((void **)&gpu_dh1, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_du1, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_dv1, nx * ny * sizeof(double));

    cudaMalloc((void **)&gpu_dh2, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_du2, nx * ny * sizeof(double));
    cudaMalloc((void **)&gpu_dv2, nx * ny * sizeof(double));
}

void swap_buffers()
{
    double *gpu_tmp;

    gpu_tmp = gpu_dh2;
    gpu_dh2 = gpu_dh1;
    gpu_dh1 = gpu_dh;
    gpu_dh = gpu_tmp;

    gpu_tmp = gpu_du2;
    gpu_du2 = gpu_du1;
    gpu_du1 = gpu_du;
    gpu_du = gpu_tmp;

    gpu_tmp = gpu_dv2;
    gpu_dv2 = gpu_dv1;
    gpu_dv1 = gpu_dv;
    gpu_dv = gpu_tmp;
}

__global__ void compute_ghost_horizontal_gpu(double *h, int nx, int ny)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < ny; j += stride)
    {
        h(nx, j) = h(0, j);
    }
}

__global__ void compute_ghost_vertical_gpu(double *h, int nx, int ny)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nx; i += stride)
    {
        h(i, ny) = h(i, 0);
    }
}

__global__ void compute_dh_gpu(double *dh, double *u, double *v, double dx, double dy, int nx, int ny, double H)
{
    // get index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // get stride
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int x = i; x < nx; x += stride_x)
    {
        for (int y = j; y < ny; y += stride_y)
        {
            dh(x, y) = -H * (du_dx(x, y) + dv_dy(x, y));
        }
    }
}

__global__ void compute_du_gpu(double *du, double *h, double dx, double dy, int nx, int ny, double g)
{
    // get index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // get stride
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int x = i; x < nx; x += stride_x)
    {
        for (int y = j; y < ny; y += stride_y)
        {
            du(x, y) = -g * dh_dx(x, y);
        }
    }
}

__global__ void compute_dv_gpu(double *dv, double *h, double dx, double dy, int nx, int ny, double g)
{
    // get index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // get stride
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int x = i; x < nx; x += stride_x)
    {
        for (int y = j; y < ny; y += stride_y)
        {
            dv((x, y) = -g * dh_dy(x, y);
        }
    }
}

__global__
void update_fields_gpu(double* h, double* u, double* v, double* dh, double* du, double* dv,
                       double* dh1, double* du1, double* dv1, double* dh2, double* du2, double* dv2,
                       double a1, double a2, double a3, double dt, int nx, int ny)
{
    // Calculate the initial position of the thread in the grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the stride for both x and y directions
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // Loop over the elements using the stride
    for (int x = i; x < nx; x += stride_x) {
        for (int y = j; y < ny; y += stride_y) {
            h(x, y) += (a1 * dh(x, y) + a2 * dh1(x, y) + a3 * dh2(x, y)) * dt;
            u(x + 1, y) += (a1 * du(x, y) + a2 * du1(x, y) + a3 * du2(x, y)) * dt;
            v(x, y + 1) += (a1 * dv(x, y) + a2 * dv1(x, y) + a3 * dv2(x, y)) * dt;
        }
    }
}

__global__ void compute_boundaries_horizontal_gpu(double *u, int nx, int ny)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < ny; j += stride)
    {
        u(0, j) = u(nx, j);
    }
}

__global__ void compute_boundaries_vertical_gpu(double *v, int nx, int ny)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nx; i += stride)
    {
        v(i, 0) = v(i, ny);
    }
}

int t = 0;

/**
 * This is your transfer function! You should copy the h field back to the host
 * so that the CPU can check the results of your computation.
 */
void transfer(double *h_host)
{
    // @TODO: Your code here
    cudaMemcpy(h_host, gpu_h, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * This is your step function! Here, you will actually numerically solve the shallow
 * water equations. You should update the h, u, and v fields to be the solution after
 * one time step has passed.
 */
void step()
{
    int blockSize = 256;
    int numBlocks;
    cudaDeviceGetAttribute(&numBlocks, cudaDevAttrMultiProcessorCount, 0);

    dim3 blockSize_2D(16, 16);
    // dim3 numBlocks_2D(numBlocks * 32, numBlocks * 32);
    dim3 numBlocks_2D(numBlocks, numBlocks);

    // First
    compute_ghost_horizontal_gpu<<<numBlocks, blockSize>>>(gpu_h, nx, ny);

    compute_ghost_vertical_gpu<<<numBlocks, blockSize>>>(gpu_h, nx, ny);

    // Next, compute the derivatives of fields
    compute_dh_gpu<<<numBlocks_2D, blockSize_2D>>>(gpu_dh, gpu_u, gpu_v, dx, dy, nx, ny, H);
    compute_du_gpu<<<numBlocks_2D, blockSize_2D>>>(gpu_du, gpu_h, dx, dy, nx, ny, g);
    compute_dv_gpu<<<numBlocks_2D, blockSize_2D>>>(gpu_dv, gpu_h, dx, dy, nx, ny, g);

    // We set the coefficients for our multistep method
    double a1, a2, a3;

    if (t == 0)
    {
        a1 = 1.0;
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    // Finally, compute the next time step using multistep method  
    update_fields_gpu<<<numBlocks_2D, blockSize_2D>>>(gpu_h, gpu_u, gpu_v, gpu_dh, gpu_du, gpu_dv,
                                                  gpu_dh1, gpu_du1, gpu_dv1,
                                                  gpu_dh2, gpu_du2, gpu_dv2,
                                                  a1, a2, a3, dt, nx, ny);
    

    // We compute the boundaries for our fields, as they are (1) needed for
    // the next time step, and (2) aren't explicitly set in our multistep method
    compute_boundaries_horizontal_gpu<<<numBlocks, blockSize>>>(gpu_u, nx, ny);
    compute_boundaries_horizontal_gpu<<<numBlocks, blockSize>>>(gpu_v, nx, ny);
    

    // We swap the buffers for our derivatives so that we can use the derivatives
    // from the previous time steps in our multistep method, then increment
    // the time step counter
    swap_buffers();

    // write back
    transfer(h);

    t++;
}


/**
 * This is your finalization function! You should free all of the memory that you
 * allocated on the GPU here.
 */
void free_memory()
{
    // Free GPU memory
    cudaFree(gpu_h);
    cudaFree(gpu_u);
    cudaFree(gpu_v);

    cudaFree(gpu_dh);
    cudaFree(gpu_du);
    cudaFree(gpu_dv);

    cudaFree(gpu_dh1);
    cudaFree(gpu_du1);
    cudaFree(gpu_dv1);

    cudaFree(gpu_dh2);
    cudaFree(gpu_du2);
    cudaFree(gpu_dv2);
}