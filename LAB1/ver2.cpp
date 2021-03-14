#include "mpi/mpi.h"
#include <iostream>
#include <cmath>
#include <algorithm>

const float EPS = 0.00001f;
const float TAU = 0.00001f;

void mul(const float* partA, const float* x, float* res, int matrixSize, int procRank, int procSize, float* buffer) {
    int numberOfLines = matrixSize / procSize;
    int shift = procRank * numberOfLines;
    for (int i = 0; i < numberOfLines; ++i) {
        res[shift + i] = 0.0f;
        for (int j = 0; j < matrixSize; ++j) {
            res[shift + i] += partA[i * matrixSize + j] * x[j];
        }
    }

    MPI_Allgather(res + shift, numberOfLines, MPI_FLOAT, buffer, numberOfLines, MPI_FLOAT, MPI_COMM_WORLD);

    std::copy(buffer, buffer + matrixSize, res);
}


float vectorAbs(const float* x, int size) {
    float len = 0.0f;
    for (int i = 0; i < size; ++i)
        len += x[i] * x[i];
    return len;
}

void vectorSub(const float* x, const float* y, float* res, float k, int size) {
    for (int i = 0; i < size; ++i)
        res[i] = x[i] - k * y[i];
}


float* nextX(float* x, const float* b, float* Ax, int matrixSize, float* buffer) {
    //Ax уже умноженный после проверки (... < EPS) - используеv его
    vectorSub(Ax, b, buffer, 1, matrixSize);
    vectorSub(x, buffer, x, TAU, matrixSize);

    return x;
}

bool ending(float* x, float* partA, float* b, float* Ax, int matrixSize, int procRank, int procSize, float* buffer) {
    mul(partA, x, Ax, matrixSize, procRank, procSize, buffer);

    vectorSub(Ax, b, buffer, 1, matrixSize);

    float div;
    if (procRank == 0) {
        div = (vectorAbs(buffer, matrixSize) / vectorAbs(b, matrixSize));
    }

    MPI_Bcast(&div, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    bool res = (div < EPS* EPS);

    if (procRank == 0) {
        std::cout << div << std::endl;
    }

    return res;
}

void init_1(float*& x, float*& partA, float*& b, float*& Ax, int matrixSize, int procRank, int procSize) {

    int startLine; //номер строки с которой процесс считает матрицу A
    int numberOfLines; //количествно строк которые обрабатывает каждый процесс

    numberOfLines = matrixSize / procSize;
    startLine = (procRank) * (numberOfLines);

    partA = new float[numberOfLines * matrixSize];

    for (int i = 0; i < numberOfLines; i++) {
        for (int j = 0; j < matrixSize; j++) {
            partA[i * matrixSize + j] = 1.0f;
        }
        partA[i * matrixSize + startLine + i] = 2.0f;
    }

    b = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        b[i] = (float)matrixSize + 1;
    }
    x = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        x[i] = 0.0f;
    }

    Ax = new float[matrixSize];
}

void printRes(float* x, int size) {
    for (int i = 0; i < size; ++i)
        std::cout << x[i] << " ";;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size = 2 * 2048; //Размер матрицы и вектора
    float* partA; //часть матрицы коэффицентов для каждого процесса она своя
    float* b; //вектор правых частей
    float* x; //вектор значений

    float* Ax; //вспомогательный вектор, хранит в себе результат умножения матрицы A на вектор x

    float* buffer = new float[size]; //вектор, который будет использоваться как буфер для промежуточных расчетов

    int procSize; //количество выполняемых процессов
    int procRank; //номер текущего процесса(нумерация с нуля)

    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    init_1(x, partA, b, Ax, size, procRank, procSize);

    double startTime, endTime;
    if (procRank == 0) {
        startTime = MPI_Wtime();
    }

    while (!ending(x, partA, b, Ax, size, procRank, procSize, buffer)) {
        x = nextX(x, b, Ax, size, buffer);
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
