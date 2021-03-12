#include "mpi/mpi.h"
#include <iostream>
#include <cmath>
#include <unistd.h>

const float EPS = 0.00001f;
float TAU = 0.00001f;
float lastDiv = 1000000.0f;
const float START_X = 0.0f;

void printVector(const float *vector, int vectorSize, const std::string &vectorName) {

    std::cout << vectorName;
    for (int i = 0; i < vectorSize; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void copyFirstVectorToSecond(const float *vector1, float *vector2, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        vector2[i] = vector1[i];
    }
}

void mul(const float *partA, const float *x, float *res, int matrixSize, int procRank, int procSize) {

    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);

    for (int i = 0; i < matrixSize; ++i) {
        res[i] = 0.0f;
        for (int j = 0; j < numberOfLines; ++j) {
            res[i] += partA[i * numberOfLines + j] * x[j + startLine];
        }
    }

}

float vectorFullAbs(const float *x, int size) {
    float len = 0.0f;
    for (int i = 0; i < size; ++i)
        len += x[i] * x[i];
    return len;
}

float vectorAbs(const float *x, int matrixSize, int procRank, int procSize) {
    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);
    float len = 0.0f;
    for (int i = startLine; i < startLine + numberOfLines; ++i) {
        len += x[i] * x[i];
    }
    return len;
}

void vectorSub(const float *x, const float *y, float *res, float k, int matrixSize, int procRank, int procSize) {

    int numberOfLines = matrixSize / procSize;
    int startLine = (procRank) * (numberOfLines);
    for (int i = startLine; i < startLine + numberOfLines; ++i) {
        res[i] = x[i] - k * y[i];
    }

}

void nextX(float *x, const float *b, float *Ax, int matrixSize, int procRank, int procSize) {
    //Ax - b уже после проверки (... < EPS) - используем его
    //vectorSub(Ax, b, Ax, 1, matrixSize, procRank, procSize);
    vectorSub(x, Ax, x, TAU, matrixSize, procRank, procSize);
}

bool ending(float *x, float *partA, float *b, float *Ax, int matrixSize, int procRank, int procSize) {
    mul(partA, x, Ax, matrixSize, procRank, procSize); // частичная Ax

    //float *firstTemp = (float*) calloc(matrixSize, sizeof(float));

    vectorSub(Ax, b, Ax, 1, matrixSize, procRank, procSize); // частичная Ax-b

    float *tempBuf = new float[matrixSize];

    MPI_Allreduce(Ax, tempBuf, matrixSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // Собираем Ax - b

    copyFirstVectorToSecond(tempBuf, Ax, matrixSize);

    float absOfAxb;
    //std::cout << procRank << " ";
    //printVector(Ax, 10, "First elements Ax-b not collected: ");
    if (procRank == 0) {
        //printVector(firstTemp, 10, "First elements Ax-b not collected: ");
        //printVector(tempBuf, 10, "First elements Ax-b collected: ");
        absOfAxb = vectorFullAbs(tempBuf, matrixSize); // считаем ||Ax - b|| ^ 2
        //std::cout << "Abs of AX - B " << absOfAxb << std::endl;
        //printVector(tempBuf, matrixSize, "||Ax - b||: ");
    }

    float partialAbs = vectorAbs(b, matrixSize, procRank, procSize); // считаем bk^2 + .. + bm^2

    float div;
    MPI_Reduce(&partialAbs, &div, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // собираем ||b|| ^ 2
    if (procRank == 0) {
        div = absOfAxb / div;
    }
    MPI_Bcast(&div, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    bool res = (div < EPS * EPS);
    if (div > lastDiv) {
        if (procRank == 0) {
            std::cout << "Reverse TAU" << std::endl;
        }
        TAU = TAU * -1.0f;
    }

    if (procRank == 0) {
        std::cout << div << " " << lastDiv << std::endl;
        std::cout << "==================================" << std::endl;
    }
    //usleep(1000000);
    //MPI_Barrier(MPI_COMM_WORLD);

    lastDiv = div;

    return res;
}

void init_1(float *&x, float *&partA, float *&b, float *&Ax, int matrixSize, int procRank, int procSize) {

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
            b[i] = (float) matrixSize + 1;
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

void printRes(float *x, int size, int procRank) {
    if (procRank == 0) {
        for (int i = 0; i < size; ++i)
            std::cout << x[i] << " ";;
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int size = 2048 * 2; //Размер матрицы и вектора
    float *partA; //часть матрицы коэффицентов для каждого процесса она своя
    float *b; //вектор правых частей
    float *x; //вектор значений

    float *Ax; //вспомогательный вектор, хранит в себе результат умножения матрицы A на вектор x

    int procSize; //количество выполняемых процессов
    int procRank; //номер текущего процесса(нумерация с нуля)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    init_1(x, partA, b, Ax, size, procRank, procSize);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double startTime, endTime;
    if (procRank == 0) {
        startTime = MPI_Wtime();
    }

    while (!ending(x, partA, b, Ax, size, procRank, procSize)) {
        nextX(x, b, Ax, size, procRank, procSize);
    }

    if (procRank == 0) {
        endTime = MPI_Wtime();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    printRes(x, size, procRank);

    if (procRank == 0) {
        std::cout << "\nElapsed time is " << endTime - startTime << '\n';
    }

    delete[] partA;
    delete[] b;
    delete[] x;
    delete[] Ax;

    MPI_Finalize();
    return 0;
}
