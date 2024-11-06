#include <mpi.h>

#include <stdlib.h>
// #include <cstring>
#include <string>
#include <iostream>

#include "../common/common.hpp"
#include "../common/solver.hpp"

// Here we hold the number of cells we have in the x and y directions
// rank 0 data
int nx_all;
double *h_all, *v_all, *u_all;
int *sendcounts = nullptr;
int *displs = nullptr;

int *recvcounts = nullptr;
int *recdispls = nullptr;
int *recdispls_hv = nullptr;

// This is where all of our points are. We need to keep track of our active
// height and velocity grids, but also the corresponding derivatives. The reason
// we have 2 copies for each derivative is that our multistep method uses the
// derivative from the last 2 time steps.
double *h, *u, *v;
double *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

int rank;
int num_procs;
int nx_in, nx, nx_out, ny;

void print_int(const std::string &s, const int x1, const int x2 = -1)
{
    std::cout << "print " << s << ", rank=" << rank << ", x1=" << x1;
    if (x2 != -1)
        std::cout << ", x2=" << x2;
    std::cout << std::endl;
}

void print_double(const std::string &s, const double x1, const double x2 = -10001)
{
    std::cout << "print " << s << ", rank=" << rank << ", x1=" << x1;
    if (x2 > -10000)
        std::cout << ", x2=" << x2;
    std::cout << std::endl;
}

void print_vec(const std::string &s, const int *x, const int size)
{
    std::cout << "print " << s << ", rank=" << rank << ", ";
    for (int i = 0; i < size; i++)
        std::cout << x[i] << ", ";
    std::cout << std::endl;
}

void print_vec_d(const std::string &s, const double *x, const int size)
{
    std::cout << "print " << s << ", rank=" << rank << ", ";
    for (int i = 0; i < size; i++)
        std::cout << x[i] << ", ";
    std::cout << std::endl;
}

/**
 * This is your initialization function! It is very similar to the one in
 * serial.cpp, but with some difference. Firstly, only the process with rank 0
 * is going to actually generate the initial conditions h0, u0, and v0, so all
 * other processes are going to get nullptrs. Therefore, you'll need to find some
 * way to scatter the initial conditions to all processes. Secondly, now the
 * rank and num_procs arguments are passed to the function, so you can use them
 * to determine which rank the node running this process has, and how many
 * processes are running in total. This is useful to determine which part of the
 * domain each process is going to be responsible for.
 */
void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    // Initialize MPI and distribute data among processes
    rank = rank_;
    num_procs = num_procs_;

    nx_all = nx_;
    ny = ny_;

    H = H_;
    g = g_;

    dx = length_ / nx_;
    dy = width_ / ny_;

    dt = dt_;

    print_int("nx_all-ny", nx_all, ny_);

    // We set the pointers to the arrays that were passed in
    if (rank == 0)
    {
        h_all = h0;
        u_all = u0;
        v_all = v0;

        // memcpy(&h_all[nx_all * (ny + 1)], &h_all[0], sizeof(double) * (ny + 1));
        // Scatter the data from rank 0 to all processes
        sendcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));

        int offset = 0;
        for (int i = 1; i < num_procs; ++i)
        {
            int temp_nx = nx_all / num_procs + (nx_all % num_procs > (i - 1) ? 1 : 0);
            sendcounts[i] = (temp_nx + 1) * (ny + 1);
            displs[i] = (offset) * (ny + 1);
            offset += temp_nx;
        }
        nx = nx_all / num_procs;
        sendcounts[0] = (nx + 1) * (ny + 1);
        displs[0] = offset;
    }

    // We allocate memory for the local arrays for each process
    if (rank != 0)
    {
        nx = nx_all / num_procs + (nx_all % num_procs > rank - 1 ? 1 : 0);
    }
    nx_in = nx + 1;

    print_int("nx_nxin", nx, nx_in);

    h = (double *)calloc(nx_in * (ny + 1), sizeof(double));
    u = (double *)calloc(nx_in * (ny + 1), sizeof(double));
    v = (double *)calloc(nx_in * (ny + 1), sizeof(double));

    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc(nx * ny, sizeof(double));
    dv = (double *)calloc(nx * ny, sizeof(double));

    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc(nx * ny, sizeof(double));
    dv1 = (double *)calloc(nx * ny, sizeof(double));

    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc(nx * ny, sizeof(double));
    dv2 = (double *)calloc(nx * ny, sizeof(double));

    // print_int("num_pro", num_procs);

    // print_vec("sc", sendcounts, num_procs);
    // print_vec("dis", displs, num_procs);
    // print_vec_d("h", h, 9);
}

/**
 * This is your step function! It is very similar to the one in serial.cpp, but
 * now the domain is divided among the processes, so you'll need to find some
 * way to communicate the ghost cells between processes.
 */

/**
 * This function computes the derivative of the height field
 * with respect to time. This is done by taking the divergence
 * of the velocity field and multiplying by -H.
 */
void compute_dh()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

/**
 * This function computes the derivative of the x-component of the
 * velocity field with respect to time. This is done by taking the
 * derivative of the height field with respect to x and multiplying
 * by -g.
 */
void compute_du()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

/**
 * This function computes the derivative of the y-component of the
 * velocity field with respect to time. This is done by taking the
 * derivative of the height field with respect to y and multiplying
 * by -g.
 */
void compute_dv()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

/**
 * This function computes the next time step using a multistep method.
 * The coefficients a1, a2, and a3 are used to determine the weights
 * of the current and previous time steps.
 */
void multistep(double a1, double a2, double a3)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }
}

/**
 * This function computes the ghost cells for the horizontal boundaries.
 * This is done by copying the values from the opposite side of the domain.
 */
void compute_ghost_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        h(nx, j) = h_all(0, j);
    }
}

/**
 * This function computes the ghost cells for the vertical boundaries.
 * This is done by copying the values from the opposite side of the domain.
 */
void compute_ghost_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        h(i, ny) = h(i, 0);
    }
}

/**
 * This function computes the boundaries for the horizontal boundaries.
 * We do this by copying the values from the opposite side of the domain.
 */
void compute_boundaries_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        u_all(0, j) = u(nx, j);
    }
}

/**
 * This function computes the boundaries for the vertical boundaries.
 * We do this by copying the values from the opposite side of the domain.
 */
void compute_boundaries_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        v(i, 0) = v(i, ny);
    }
}

/**
 * This function swaps the buffers for the derivatives of our different fields.
 * This is done so that we can use the derivatives from the previous time steps
 * in our multistep method.
 */
void swap_buffers()
{
    double *tmp;

    tmp = dh2;
    dh2 = dh1;
    dh1 = dh;
    dh = tmp;

    tmp = du2;
    du2 = du1;
    du1 = du;
    du = tmp;

    tmp = dv2;
    dv2 = dv1;
    dv1 = dv;
    dv = tmp;
}

/**
 * This is your transfer function! Similar to what you did in gpu.cu, you'll
 * need to get the data from the computers you're working on (there it was
 * the GPU, now its a bunch of CPU nodes), and send them all back to the process
 * which is actually running the main function (then it was the CPU, not it's
 * the node with rank 0).
 */
void transfer(double *h)
{
    return;
}

void gather()
{
    if (rank == 0)
    {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        recdispls = (int *)malloc(num_procs * sizeof(int));
        recdispls_hv = (int *)malloc(num_procs * sizeof(int));

        int offset = 0;
        for (int i = 1; i < num_procs; ++i)
        {
            int temp_nx = nx_all / num_procs + (nx_all % num_procs > (i - 1) ? 1 : 0);
            recvcounts[i] = temp_nx * (ny + 1);
            recdispls[i] = (offset+1) * (ny + 1);
            recdispls_hv[i] = offset * (ny + 1);
            offset += temp_nx;
        }

        nx_out = nx_all / num_procs;
        recvcounts[0] = nx_out * (ny + 1);
        recdispls[0] = (offset + 1)* (ny + 1);
        recdispls_hv[0] = offset* (ny + 1);
    }

    if (rank != 0)
    {
        nx_out = nx_all / num_procs + (nx_all % num_procs > (rank - 1) ? 1 : 0);
    }

    MPI_Gatherv(h, nx_out * (ny + 1), MPI_DOUBLE, h_all, recvcounts, recdispls_hv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&u[ny + 1], nx_out * (ny + 1), MPI_DOUBLE, u_all, recvcounts, recdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(v, nx_out * (ny + 1), MPI_DOUBLE, v_all, recvcounts, recdispls_hv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int t = 0;

void step()
{
    MPI_Scatterv(h_all, sendcounts, displs, MPI_DOUBLE, h, nx_in * (ny + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(u_all, sendcounts, displs, MPI_DOUBLE, u, nx_in * (ny + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(v_all, sendcounts, displs, MPI_DOUBLE, v, nx_in * (ny + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        compute_ghost_horizontal();
    }

    compute_ghost_vertical();

    compute_dh();
    compute_du();
    compute_dv();

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

    multistep(a1, a2, a3);

    if (rank == 0)
    {
        compute_boundaries_horizontal();
    }

    compute_boundaries_vertical();

    gather();

    swap_buffers();

    t++;


}

/**
 * This is your finalization function! Since different nodes are going to be
 * initializing different chunks of memory, make sure to check which node
 * is running the code before you free some memory you haven't allocated, or
 * that you've actually freed memory that you have.
 */
void free_memory()
{
    if (rank == 0)
    {
        free(sendcounts);
        free(displs);

        free(recvcounts);
        free(recdispls);
        free(recdispls_hv);
    }
    
    free(h);
    free(u);
    free(v);

    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);

    free(dh2);
    free(du2);
    free(dv2);
}