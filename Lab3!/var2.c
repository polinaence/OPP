#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RANDOM_MIN -10
#define RANDOM_MAX 10

typedef double value_t;

// 24
int sizes[][2] = {
    {1, 24},
    {2, 12},
    {3, 8},
    {4, 6},
    {6, 4},
    {8, 3},
    {12, 2},
    {24, 1},
};

#define ARR_SIZE(x) ((sizeof(x) / sizeof(x[0])))

value_t frand(value_t min, value_t max)
{
    return (max - min) * ((value_t)rand() / (value_t)RAND_MAX) + min;
}

value_t* GetRandomMatrix(size_t rowsN, size_t colsN)
{
    value_t* m = (value_t*)malloc(rowsN * colsN * sizeof(*m));
    if (!m)
    {
        perror("malloc() error");
        return NULL;
    }
    for (size_t i = 0; i < rowsN; i++)
    {
        for (size_t j = 0; j < colsN; j++)
        {
            m[i * colsN + j] = frand(RANDOM_MIN, RANDOM_MAX);
        }
    }
    return m;
}

void PrintMatrix(value_t* m, size_t rowsN, size_t colsN)
{
    if (!m)
    {
        return;
    }
    for (size_t i = 0; i < rowsN; i++)
    {
        for (size_t j = 0; j < colsN; j++)
        {
            printf("%.3lf ", m[i * colsN + j]);
        }
        printf("\n");
    }
}

void Do_MPI(value_t* A_full, value_t* B_full, value_t* C_full, size_t A_rowsN, size_t A_colsN, size_t B_colsN, int procsX, int procsY)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = { procsY, procsX };

    int periods[2] = { 0, 0 };
    int reorder = 1;
    MPI_Comm comm2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    int sizeY = dims[0];
    int sizeX = dims[1];

    if (A_rowsN % sizeY != 0)
    {
        fprintf(stderr, "A_rowsN %% sizeY != 0\n");
        return;
    }

    if (B_colsN % sizeX != 0)
    {
        fprintf(stderr, "B_colsN %% sizeX != 0\n");
        return;
    }

    size_t A_rowsPerProc = A_rowsN / sizeY;
    size_t B_colsPerProc = B_colsN / sizeX;

    int coords[2];
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    int rankY = coords[0];
    int rankX = coords[1];

    MPI_Comm colComm;
    MPI_Comm rowComm;
    MPI_Comm_split(MPI_COMM_WORLD, rankX, rank, &colComm);
    MPI_Comm_split(MPI_COMM_WORLD, rankY, rank, &rowComm);

    value_t* A_local = (value_t*)malloc(A_rowsPerProc * A_colsN * sizeof(*A_local));
    if (!A_local)
    {
        fprintf(stderr, "malloc() error");
        return;
    }

    value_t* B_local = (value_t*)malloc(A_colsN * B_colsPerProc * sizeof(*B_local));
    if (!B_local)
    {
        free(A_local);
        fprintf(stderr, "malloc() error");
        return;
    }

    bool isMaster = rankX == 0 && rankY == 0;

    // start bcast A
    if (rankX == 0)
    {
        MPI_Scatter(A_full, A_rowsPerProc * A_colsN, MPI_DOUBLE, A_local, A_rowsPerProc * A_colsN, MPI_DOUBLE, 0, colComm);
    }

    MPI_Bcast(A_local, A_rowsPerProc * A_colsN, MPI_DOUBLE, 0, rowComm);
    // end bcast A

    // start bcast B
    MPI_Datatype bPart;
    MPI_Datatype bPartType;
    MPI_Type_vector(A_colsN, B_colsPerProc, B_colsN, MPI_DOUBLE, &bPart);
    MPI_Type_commit(&bPart);
    MPI_Type_create_resized(bPart, 0, B_colsPerProc * sizeof(value_t), &bPartType);
    MPI_Type_commit(&bPartType);

    if (rankY == 0)
    {
        MPI_Scatter(B_full, 1, bPartType, B_local, A_colsN * B_colsPerProc, MPI_DOUBLE, 0, rowComm);
    }

    MPI_Bcast(B_local, A_colsN * B_colsPerProc, MPI_DOUBLE, 0, colComm);
    // end bcast B

    value_t* C_local = (value_t*)calloc(A_rowsPerProc * B_colsPerProc, sizeof(*C_local));
    if (!C_local)
    {
        fprintf(stderr, "malloc() error");
        return;
    }

    // calculate local matrix prod
    for (size_t i = 0; i < A_rowsPerProc; i++)
    {
        for (size_t k = 0; k < A_colsN; k++)
        {
            double A_mul = A_local[i * A_colsN + k];
            double* row = &B_local[k * B_colsPerProc];
            double* C_row = &C_local[i * B_colsPerProc];
            for (size_t j = 0; j < B_colsPerProc; j++)
            {
                // A_local - A_rowsPerProc * A_colsN
                // B_local - A_colsN * B_colsPerProc
                // C_local - A_rowsPerProc * B_colsPerProc

                // C[i][j] += A[i][k] * B[k][j];
                C_row[j] += A_mul * row[j];
            }
        }
    }
    // end calculate

    // gather result matrix

    MPI_Datatype cPart;
    MPI_Datatype cPartType;
    MPI_Type_vector(A_rowsPerProc, B_colsPerProc, B_colsN, MPI_DOUBLE, &cPart);
    MPI_Type_commit(&cPart);
    MPI_Type_create_resized(cPart, 0, B_colsPerProc * sizeof(value_t), &cPartType);
    MPI_Type_commit(&cPartType);

    int C_recvcounts[size];
    int C_displs[size];
    int index = 0;
    for (size_t y = 0; y < sizeY; y++)
    {
        for (size_t x = 0; x < sizeX; x++)
        {
            C_recvcounts[index] = 1;
            C_displs[index] = x + y * (A_rowsPerProc * sizeX);
            index++;
        }
    }

    MPI_Gatherv(C_local, A_rowsPerProc * B_colsPerProc, MPI_DOUBLE, C_full, C_recvcounts, C_displs, cPartType, 0, comm2d);
    // end gather result

#ifdef PRINT_MATRICES
    if (isMaster)
    {
        PrintMatrix(C_full, A_rowsN, B_colsN);
    }
#endif

    free(A_local);
    free(B_local);
    free(C_local);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&comm2d);
    MPI_Type_free(&bPart);
    MPI_Type_free(&bPartType);
    MPI_Type_free(&cPart);
    MPI_Type_free(&cPartType);
}

bool str2ul(const char* str, unsigned long* num)
{
    char* end;
    unsigned long long conv;
    errno = 0;
    conv = strtoull(str, &end, 10);
    if ((errno == ERANGE && conv == ULLONG_MAX) || conv > ULONG_MAX)
    {
        errno = ERANGE;
        return false;
    }
    if (errno == ERANGE && conv == 0)
    {
        return false;
    }
    if (*str == '\0' || *end != '\0')
    {
        errno = EINVAL;
        return false;
    }
    *num = (unsigned long)conv;
    return true;
}

int main(int argc, char* argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s A_rowsN A_colsN B_colsN\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    size_t A_rowsN, A_colsN, B_colsN;
    if (!str2ul(argv[1], &A_rowsN) ||
        !str2ul(argv[2], &A_colsN) ||
        !str2ul(argv[3], &B_colsN))
    {
        fprintf(stderr, "Error converting arguments to numbers\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    srand(19202);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    value_t* A_full = NULL;
    value_t* B_full = NULL;
    value_t* C_full = NULL;

    if (rank == 0)
    {
        printf("A_rowsN=%zu\nA_colsN=%zu\nB_rowsN=%zu\nB_colsN=%zu\nsize=%d\n", A_rowsN, A_colsN, A_colsN, B_colsN, size);
        A_full = GetRandomMatrix(A_rowsN, A_colsN);
        if (!A_full)
        {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
            return EXIT_FAILURE;
        }
        B_full = GetRandomMatrix(A_colsN, B_colsN);
        if (!B_full)
        {
            free(A_full);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
            return EXIT_FAILURE;
        }
        C_full = (value_t*)calloc(A_rowsN * B_colsN, sizeof(*C_full));
        if (!C_full)
        {
            free(A_full);
            free(B_full);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
            return EXIT_FAILURE;
        }

#ifdef PRINT_MATRICES
        printf("A=\n");
        PrintMatrix(A_full, A_rowsN, A_colsN);
        printf("B=\n");
        PrintMatrix(B_full, A_colsN, B_colsN);
        printf("\n\n\n");
#endif
    }

    for (int i = 0; i < ARR_SIZE(sizes); i++)
    {
        int procsX = sizes[i][0];
        int procsY = sizes[i][1];

        if (rank == 0)
        {
            printf("[procsX=%2d][procsY=%2d] ", procsX, procsY);
        }

        if (procsX * procsY != size)
        {
            if (rank == 0)
            {
                fprintf(stderr, "procsX * procsY != size (%d %d, size=%d)\n", procsX, procsY, size);
            }
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            return EXIT_FAILURE;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        Do_MPI(A_full, B_full, C_full, A_rowsN, A_colsN, B_colsN, procsX, procsY);

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        if (rank == 0)
        {
            printf("Time taken:\t%lf sec.\n", (end.tv_sec - start.tv_sec) + 0.000000001 * (end.tv_nsec - start.tv_nsec));
            memset(C_full, 0, A_rowsN * B_colsN * sizeof(*C_full));
        }
    }

    free(A_full);
    free(B_full);
    free(C_full);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

