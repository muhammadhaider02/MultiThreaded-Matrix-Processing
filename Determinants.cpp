#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

static const int MATRIX_SIZE = 1024;
static const int NUM_THREADS = 6;

static vector<vector<double>> originalMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
static vector<double> parallelDeterminants(NUM_THREADS);
static vector<double> sequentialDeterminants(NUM_THREADS);

struct ThreadData {
    int threadId;
    int startRow;
    int startCol;
    int size;
};

// Compute the determinant of a square matrix using LU decomposition
double determinant(vector<vector<double>> matrix, int size) {
    double det = 1.0;

    for (int k = 0; k < size; ++k) {
        int maxRow = k;
        for (int i = k + 1; i < size; ++i) {
            if (fabs(matrix[i][k]) > fabs(matrix[maxRow][k])) {
                maxRow = i;
            }
        }
        if (fabs(matrix[maxRow][k]) < 1e-14) {
            return 0.0;
        }
        // Swap rows if needed
        if (maxRow != k) {
            swap(matrix[k], matrix[maxRow]);
            det = -det;
        }
        det *= matrix[k][k];
        // Eliminate below pivot
        for (int i = k + 1; i < size; ++i) {
            matrix[i][k] /= matrix[k][k];
            for (int j = k + 1; j < size; ++j) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
        }
    }

    return det;
}

// Parallel worker: compute determinant of a sub-block
void* computeDeterminant(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int startRow = data->startRow;
    int startCol = data->startCol;
    int size     = data->size;

    // Copy submatrix into a buffer
    vector<vector<double>> submatrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            submatrix[i][j] = originalMatrix[startRow + i][startCol + j];
        }
    }

    // Compute determinant
    double detVal = determinant(submatrix, size);
    parallelDeterminants[data->threadId] = detVal;

    pthread_exit(NULL);
}

// Sequential: compute determinant of each block
void sequentialDeterminant() {
    int blockSize = MATRIX_SIZE / NUM_THREADS;
    int remainder = MATRIX_SIZE % NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * blockSize;
        int startCol = i * blockSize;
        int size = (i == NUM_THREADS - 1) ? (blockSize + remainder) : blockSize;

        // Copy submatrix
        vector<vector<double>> submatrix(size, vector<double>(size));
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                submatrix[r][c] = originalMatrix[startRow + r][startCol + c];
            }
        }
        // Compute determinant
        sequentialDeterminants[i] = determinant(submatrix, size);
    }
}

// Correctness check
bool checkDeterminantCorrectness() {
    for (int i = 0; i < NUM_THREADS; ++i) {
        double diff = fabs(parallelDeterminants[i] - sequentialDeterminants[i]);
        if (diff > 1e-6) {
            return false;
        }
    }
    return true;
}


int main() {
    srand((unsigned)time(NULL));

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            originalMatrix[i][j] = rand() % 100 + 1;
        }
    }

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    int blockSize  = MATRIX_SIZE / NUM_THREADS;
    int remainder  = MATRIX_SIZE % NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].threadId = i;
        threadData[i].startRow = i * blockSize;
        threadData[i].startCol = i * blockSize;
        threadData[i].size     = (i == NUM_THREADS - 1) 
                                 ? (blockSize + remainder)
                                 : blockSize;
        pthread_create(&threads[i], NULL, computeDeterminant, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    sequentialDeterminant();

    if (checkDeterminantCorrectness()) {
        cout << "Determinant Correctness: Passed" << endl;
    } else {
        cout << "Determinant Correctness: Failed" << endl;
    }

    return 0;
}
