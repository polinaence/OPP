#include "stdio.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include "mpi.h"

const double eps = 0.00000001;
const int a = 100000;

const double Dx = 2;
const double Dy = 2;
const double Dz = 2;

const int Nx = 320;
const int Ny = 320;
const int Nz = 320;

const double hx = (double)(Dx / (Nx - 1));
const double hy = (double)(Dy / (Ny - 1));
const double hz = (double)(Dz / (Nz - 1));

const double multiplier = 1 / (2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a);

double phi(double x, double y, double z)
{
    double phi = 0;
    phi = x * x + y * y + z * z;
    return phi;
}

double rho(double x, double y, double z)
{
    double rho = 0;
    rho = 6 - a * phi(x, y, z);
    return rho;
}

void ÑalculateÑenter(double layer_height, double* prev_Phi, double* Phi, int rank, bool* flag)
{
    double x_component, y_component, z_component;

    for (int k = 1; k < layer_height - 1; k++)
    {
        for (int i = 1; i < Ny - 1; i++)
        {
            for (int j = 1; j < Nx - 1; j++)
            {

                x_component = (prev_Phi[Nx * Ny * k + Nx * i + (j - 1)] + prev_Phi[Nx * Ny * k + Nx * i + (j + 1)]) / (hx * hx);
                y_component = (prev_Phi[Nx * Ny * k + Nx * (i - 1) + j] + prev_Phi[Nx * Ny * k + Nx * (i + 1) + j]) / (hy * hy);
                z_component = (prev_Phi[Nx * Ny * (k - 1) + Nx * i + j] + prev_Phi[Nx * Ny * (k + 1) + Nx * i + j]) / (hz * hz);

                Phi[Nx * Ny * k + Nx * i + j] = multiplier * (x_component + y_component + z_component - rho(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + (k + layer_height * rank) * hz));

                if (fabs(Phi[Nx * Ny * k + Nx * i + j] - prev_Phi[Nx * Ny * k + Nx * i + j]) > eps)
                {
                    (*flag) = true;
                }

            }
        }
    }
}

void  CalculateEdges(int layer_height, double* prev_Phi, double* Phi, int rank, bool* flag, double* down_layer, double* up_layer, int proc_num)
{
    double x_component, y_component, z_component;

    for (int i = 1; i < Ny - 1; i++)
    {
        for (int j = 1; j < Nx - 1; j++)
        {
            if (rank != 0)
            {
                x_component = (prev_Phi[Nx * i + (j - 1)] + prev_Phi[Nx * i + (j + 1)]) / (hx * hx);
                y_component = (prev_Phi[Nx * (i - 1) + j] + prev_Phi[Nx * (i + 1) + j]) / (hy * hy);
                z_component = (down_layer[Nx * i + j] + prev_Phi[Nx * Ny + Nx * i + j]) / (hz * hz);

                Phi[Nx * i + j] = multiplier * (x_component + y_component + z_component - rho(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + (layer_height * rank) * hz));

                if (fabs(Phi[Nx * i + j] - prev_Phi[Nx * i + j]) > eps)
                {
                    (*flag) = true;
                }
            }

            if (rank != proc_num - 1)
            {
                x_component = (prev_Phi[Nx * Ny * (layer_height - 1) + Nx * i + (j - 1)] + prev_Phi[Nx * Ny * (layer_height - 1) + Nx * i + (j + 1)]) / (hx * hx);
                y_component = (prev_Phi[Nx * Ny * (layer_height - 1) + Nx * (i - 1) + j] + prev_Phi[Nx * Ny * (layer_height - 1) + Nx * (i + 1) + j]) / (hy * hy);
                z_component = (prev_Phi[Nx * Ny * (layer_height - 2) + Nx * i + j] + up_layer[Nx * i + j]) / (hz * hz);

                Phi[Nx * Ny * (layer_height - 1) + Nx * i + j] = multiplier * (x_component + y_component + z_component - rho(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + ((layer_height - 1) + layer_height * rank) * hz));

                if (fabs(Phi[Nx * i + j] - prev_Phi[Nx * i + j]) > eps)
                {
                    (*flag) = true;
                }
            }
        }
    }
}

void CalcMaxDiff(int rank, int layer_height, double* Phi)
{
    double max = 0;
    double diff = 0;

    for (int k = 0; k < layer_height; k++)
    {
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                diff = fabs(Phi[k * Nx * Ny + i * Nx + j] - phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + (k + layer_height * rank) * hz));
                if (diff > max)
                {
                    max = diff;
                }
            }
        }
    }

    double tmp = 0;
    MPI_Allreduce(&max, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0)
    {
        max = tmp;
        std::cout << "Max difference: " << max << std::endl;
    }
}

int main(int argc, char** argv)
{
    int rank, proc_num;
    double start, finish;
    bool delta_larger_eps = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    int layer_height = Nz / proc_num;
    double* Phi = new double[Nx * Ny * layer_height]();
    double* prev_Phi = new double[Nx * Ny * layer_height]();
    double* down_layer = new double[Nx * Ny];
    double* up_layer = new double[Nx * Ny];

    if (rank == 0) start = MPI_Wtime();

    for (int k = 0; k < layer_height; k++)
    {
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                if (i == 0 || j == 0 || i == Ny - 1 || j == Nx - 1)
                {
                    Phi[Nx * Ny * k + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + (k + layer_height * rank) * hz);
                    prev_Phi[Nx * Ny * k + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2 + (k + layer_height * rank) * hz);
                }
                else
                {
                    Phi[Nx * Ny * k + Nx * i + j] = 0;
                    prev_Phi[Nx * Ny * k + Nx * i + j] = 0;
                }
            }
        }
    }
    if (rank == 0)
    {
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                Phi[0 + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2);
                prev_Phi[0 + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, -Dz / 2);
            }
        }
    }

    if (rank == proc_num - 1)
    {
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                Phi[Nx * Ny * (layer_height - 1) + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, Dz / 2);
                prev_Phi[Nx * Ny * (layer_height - 1) + Nx * i + j] = phi(-Dx / 2 + j * hx, -Dy / 2 + i * hy, Dz / 2);

            }
        }
    }

    double* tmp;
    bool lor_res;
    double counter = 0;
    MPI_Request requests[4];

    while (delta_larger_eps)
    {
        delta_larger_eps = false;

        tmp = prev_Phi;
        prev_Phi = Phi;
        Phi = tmp;

        if (rank != 0)
        {
            MPI_Isend(&prev_Phi[0], Nx * Ny, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &requests[0]); //MPI_Request
            MPI_Irecv(down_layer, Nx * Ny, MPI_DOUBLE, rank - 1, 20, MPI_COMM_WORLD, &requests[1]);
        }

        if (rank != proc_num - 1)
        {
            MPI_Isend(&prev_Phi[(layer_height - 1) * Nx * Ny], Nx * Ny, MPI_DOUBLE, rank + 1, 20, MPI_COMM_WORLD, &requests[2]); //MPI_Request
            MPI_Irecv(up_layer, Nx * Ny, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &requests[3]);
        }

        ÑalculateÑenter(layer_height, prev_Phi, Phi, rank, &delta_larger_eps);

        if (rank != 0)
        {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        }

        if (rank != proc_num - 1)
        {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

        CalculateEdges(layer_height, prev_Phi, Phi, rank, &delta_larger_eps, down_layer, up_layer, proc_num);

        MPI_Allreduce(&delta_larger_eps, &lor_res, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        delta_larger_eps = lor_res;

        counter++;
    }

    if (rank == 0) finish = MPI_Wtime();

    double count_tmp = 0;
    MPI_Allreduce(&counter, &count_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    counter = count_tmp;

    CalcMaxDiff(rank, layer_height, Phi);

    if (rank == 0)
    {
        std::cout << "Number of iterations:" << counter << std::endl;
        std::cout << "Time:" << (finish - start) << std::endl;

    }

    MPI_Finalize();
    return 0;
}
