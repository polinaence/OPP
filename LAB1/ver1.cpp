#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <time.h>

const float EPS = 0.00001f;
const float TAU = 0.00001f;

void copyFirstVectorToSecond(const float *vector1, float *vector2, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        vector2[i] = vector1[i];
    }
}

void mul(const float *partA, const float *x, float *res, int matrixSize) {
    for (int i = 0; i < matrixSize; ++i) {
        res[i] = 0.0f;
        for (int j = 0; j < matrixSize; ++j) {
            res[i] += partA[i * matrixSize + j] * x[j];
        }
    }
}

float vectorAbs(const float *x, int size) {
    float len = 0.0f;
    for (int i = 0; i < size; ++i)
        len += x[i] * x[i];
    return len;
}

void vectorSub(const float *x, const float *y, float *res, float k, int size) {
    for (int i = 0; i < size; ++i)
        res[i] = x[i] - k * y[i];
}


float *nextX(float *x, const float *b, float *Ax, int matrixSize) {
    //Ax уже умноженный после проверки (... < EPS) - используеv его
    float *y = new float[matrixSize];
    vectorSub(Ax, b, y, 1, matrixSize);

    float *part_x_plus_one = new float[matrixSize];
    vectorSub(x, y, part_x_plus_one, TAU, matrixSize);
    copyFirstVectorToSecond(part_x_plus_one, x, matrixSize);
    delete[] part_x_plus_one;

    delete[] y;
    return x;
}

bool ending(float *x, float *partA, float *b, float *Ax, int matrixSize) {
    mul(partA, x, Ax, matrixSize);

    float *sub = new float[matrixSize];
    vectorSub(Ax, b, sub, 1, matrixSize);

    float div = (vectorAbs(sub, matrixSize) / vectorAbs(b, matrixSize));

    bool res = (div < EPS * EPS);

    delete[] sub;

    return res;
}

void init_1(float *&x, float *&partA, float *&b, float *&Ax, int matrixSize) {

    partA = new float[matrixSize * matrixSize];

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            partA[i * matrixSize + j] = 1.0f;
        }
        partA[i * matrixSize + i] = 2.0f;
    }

    b = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        b[i] = (float) matrixSize + 1;
    }
    x = new float[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        x[i] = 0.0f;
    }

    Ax = new float[matrixSize];
}

void printRes(float *x, int size) {
    for (int i = 0; i < size; ++i)
        std::cout << x[i] << " ";
}


int main(int argc, char *argv[]) {

    int size = 2048 * 2; //Размер матрицы и вектора
    float *partA; //часть матрицы коэффицентов для каждого процесса она своя
    float *b; //вектор правых частей
    float *x; //вектор значений

    float *Ax; //вспомогательный вектор, хранит в себе результат умножения матрицы A на вектор x

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    init_1(x, partA, b, Ax, size);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);

    while (!ending(x, partA, b, Ax, size)) {
        x = nextX(x, b, Ax, size);
    }

    gettimeofday(&end_time, 0);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    printRes(x, size);


    std::cout << "\nElapsed time is " << (int) ((end_time.tv_sec - start_time.tv_sec) * 1000000 + end_time.tv_usec - start_time.tv_usec) << '\n';

    delete[] partA;
    delete[] b;
    delete[] x;
    delete[] Ax;
    return 0;
}
