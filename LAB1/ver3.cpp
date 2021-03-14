#include "mpi/mpi.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unistd.h>

const float EPS = 0.00001f;
const float TAU = 0.00001f;
const float START_X = 0.0f;

void printVector(const float* vector, int vectorSize, const std::string& vectorName) {

    std::cout << vectorName;
    for (int i = 0; i < vectorSize; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void mul(const float* partA, const float* x, float* res, int matrixSize, int procRank, int procSize) {

    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);

    for (int i = 0; i < matrixSize; ++i) {
        res[i] = 0.0f;
        for (int j = 0; j < numberOfLines; ++j) {
            res[i] += partA[i * numberOfLines + j] * x[j + startLine];
        }
    }

}

float vectorFullAbs(const float* x, int size) {
    float len = 0.0f;
    for (int i = 0; i < size; ++i)
        len += x[i] * x[i];
    return len;
}

float vectorAbs(const float* x, int matrixSize, int procRank, int procSize) {
    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);
    float len = 0.0f;
    for (int i = startLine; i < startLine + numberOfLines; ++i) {
        len += x[i] * x[i];
    }
    return len;
}

void vectorSub(const float* x, const float* y, float* res, float k, int matrixSize, int procRank, int procSize) {

    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);
    for (int i = startLine; i < startLine + numberOfLines; ++i) {
        res[i] = x[i] - k * y[i];
    }

}

bool ending(float* x, float* partA, float* b, float* Ax, int matrixSize, int procRank, int procSize, float* buffer) {
    mul(partA, x, Ax, matrixSize, procRank, procSize); // частичная Ax

    vectorSub(Ax, b, Ax, 1, matrixSize, procRank, procSize); // частичная Ax-b

    MPI_Allreduce(Ax, buffer, matrixSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // Собираем Ax - b

    std::copy(buffer, buffer + matrixSize, Ax);

    float absOfAxb;
    if (procRank == 0) {
        absOfAxb = vectorFullAbs(buffer, matrixSize); // считаем ||Ax - b|| ^ 2
    }

    float partialAbs = vectorAbs(b, matrixSize, procRank, procSize); // считаем bk^2 + .. + bm^2

    float div;
    MPI_Reduce(&partialAbs, &div, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // собираем ||b|| ^ 2
    if (procRank == 0) {
        div = absOfAxb / div;
    }
    MPI_Bcast(&div, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    bool res = (div < EPS* EPS);

    if (procRank == 0) {
        std::cout << div << std::endl;
        std::cout << "==================================" << std::endl;
    }
    //usleep(1000000);
    //MPI_Barrier(MPI_COMM_WORLD);

    return res;
}

void init_1(float*& x, float*& partA, float*& b, float*& Ax, int matrixSize, int procRank, int procSize) {

    int startLine; //номер строки с которой процесс считает матрицу A
    int numberOfLines; //количествно строк которые обрабатывает каждый процесс

    numberOfLines = matrixSize / procSize;
    startLine = (procRank) * (numberOfLines);

    partA = new float[matrixSize * numberOfLines];

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < numberOfLines; j++) {
            partA[i * numberOfLines + j] = 1.0f;
            if (i == j + startLine) {
                partA[i * numberOfLines + j] = 2.0f;
            }
        }
    }

    b = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        b[i] = 0.0f;
        if (i >= startLine && i < startLine + numberOfLines) {
            b[i] = (float)matrixSize + 1;
        }
    }
    x = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        x[i] = 0.0f;
        if (i >= startLine && i < startLine + numberOfLines) {
            x[i] = START_X;
        }
    }

    Ax = new float[matrixSize];
}

void printRes(float* x, int size) {
    for (int i = 0; i < size; ++i)
        std::cout << x[i] << " ";;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size = 2048 * 2; //Размер матрицы и вектора
    float* partA; //часть матрицы коэффицентов для каждого процесса она своя
    float* b; //вектор правых частей
    float* x; //вектор значений

    float* Ax; //вспомогательный вектор, хранит в себе результат умножения матрицы A на вектор x

    int procSize; //количество выполняемых процессов
    int procRank; //номер текущего процесса(нумерация с нуля)

    float* buffer = new float[size]; //вектор, который будет использоваться как буфер для промежуточных расчетов

    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    init_1(x, partA, b, Ax, size, procRank, procSize);

    double startTime, endTime;
    if (procRank == 0) {
        startTime = MPI_Wtime();
    }

    while (!ending(x, partA, b, Ax, size, procRank, procSize, buffer)) {
        vectorSub(x, Ax, x, TAU, size, procRank, procSize);
    }

    if (procRank == 0) {
        endTime = MPI_Wtime();
        printRes(x, size);
        std::cout << "\nElapsed time is " << endTime - startTime << '\n';
    }

    delete[] partA;
    delete[] b;
    delete[] x;
    delete[] Ax;
    delete[] buffer;

    MPI_Finalize();
    return 0;
}
