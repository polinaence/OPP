#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
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

void Do_No_MPI(size_t A_rowsN, size_t A_colsN, size_t B_colsN)
{
    value_t* A = GetRandomMatrix(A_rowsN, A_colsN);
    if (!A)
    {
        return;
    }

    value_t* B = GetRandomMatrix(A_colsN, B_colsN);
    if (!B)
    {
        free(A);
        return;
    }

    value_t* C = (value_t*)calloc(A_rowsN * B_colsN, sizeof(*C));
    if (!C)
    {
        free(A);
        free(B);
        return;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (size_t i = 0; i < A_rowsN; i++)
    {
        for (size_t k = 0; k < A_colsN; k++)
        {
            value_t mul = A[i * A_colsN + k];
            value_t* arr = &B[k * B_colsN];
            value_t* resArr = &C[i * B_colsN];
            for (size_t j = 0; j < B_colsN; j++)
            {
                resArr[j] += mul * arr[j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf("[NO MPI] Time taken: %lf sec.\n", (end.tv_sec - start.tv_sec) + 0.000000001 * (end.tv_nsec - start.tv_nsec));

#ifdef PRINT_MATRICES
    printf("\nA:");
    PrintMatrix(A, A_rowsN, A_colsN);
    printf("\nB:");
    PrintMatrix(B, A_colsN, B_colsN);
    printf("\nC:");
    PrintMatrix(C, A_rowsN, B_colsN);
#endif
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

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s A_rowsN A_colsN B_colsN\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t A_rowsN, A_colsN, B_colsN;
    if (!str2ul(argv[1], &A_rowsN) ||
        !str2ul(argv[2], &A_colsN) ||
        !str2ul(argv[3], &B_colsN))
    {
        fprintf(stderr, "Error converting arguments to numbers\n");
        return EXIT_FAILURE;
    }

    srand(19202);

    Do_No_MPI(A_rowsN, A_colsN, B_colsN);

    return EXIT_SUCCESS;
}
