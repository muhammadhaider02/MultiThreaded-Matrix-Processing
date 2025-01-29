#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 6;

vector<vector<double>> originalMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));

vector<double> parallelDeterminants(NUM_THREADS);
vector<double> sequentialDeterminants(NUM_THREADS);

vector<vector<double>> transposedMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
vector<vector<double>> sequentialTransposed(MATRIX_SIZE, vector<double>(MATRIX_SIZE));

vector<vector<double>> logTransformedMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
vector<vector<double>> sequentialLogTransformed(MATRIX_SIZE, vector<double>(MATRIX_SIZE));

struct ThreadData {
    int threadId;
    int startRow;
    int startCol;
    int size;
};

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
        if (maxRow != k) {
            swap(matrix[k], matrix[maxRow]);
            det = -det;
        }
        det *= matrix[k][k];
        for (int i = k + 1; i < size; ++i) {
            matrix[i][k] /= matrix[k][k];
            for (int j = k + 1; j < size; ++j) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
        }
    }
    return det;
}

// Parallel Determinant (Block,Block Distribution)
void* computeDeterminant(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int startRow = data->startRow;
    int startCol = data->startCol;
    int size     = data->size;

    vector<vector<double>> submatrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            submatrix[i][j] = originalMatrix[startRow + i][startCol + j];
        }
    }
    parallelDeterminants[data->threadId] = determinant(submatrix, size);

    pthread_exit(NULL);
}

void sequentialDeterminant() {
    int blockSize = MATRIX_SIZE / NUM_THREADS;
    int remainder = MATRIX_SIZE % NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * blockSize;
        int startCol = i * blockSize;
        int size = (i == NUM_THREADS-1) ? (blockSize + remainder) : blockSize;

        vector<vector<double>> submatrix(size, vector<double>(size));
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                submatrix[r][c] = originalMatrix[startRow + r][startCol + c];
            }
        }
        sequentialDeterminants[i] = determinant(submatrix, size);
    }
}

// Parallel Transpose (Row-wise Cyclic)
void* transposeRows(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->threadId;

    for (int r = tid; r < MATRIX_SIZE; r += NUM_THREADS) {
        for (int c = 0; c < MATRIX_SIZE; ++c) {
            transposedMatrix[c][r] = originalMatrix[r][c];
        }
    }
    pthread_exit(NULL);
}

void sequentialTranspose() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            sequentialTransposed[j][i] = originalMatrix[i][j];
        }
    }
}

// Parallel Log Transform (Row & Column-wise Cyclic)
void* computeLog(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->threadId;

    for (int i = tid; i < MATRIX_SIZE; i += NUM_THREADS) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            logTransformedMatrix[i][j] = log(originalMatrix[i][j]);
        }
    }
    pthread_exit(NULL);
}

void sequentialLogTransform() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            sequentialLogTransformed[i][j] = log(originalMatrix[i][j]);
        }
    }
}

// Correctness Checks
bool checkDeterminantCorrectness() {
    for (int i = 0; i < NUM_THREADS; ++i) {
        double diff = fabs(parallelDeterminants[i] - sequentialDeterminants[i]);
        if (diff > 1e-6) {
            return false;
        }
    }
    return true;
}

bool checkTransposeCorrectness() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            double diff = fabs(transposedMatrix[i][j] - sequentialTransposed[i][j]);
            if (diff > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

bool checkLogCorrectness() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            double diff = fabs(logTransformedMatrix[i][j] - sequentialLogTransformed[i][j]);
            if (diff > 1e-9) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    srand((unsigned)time(NULL));

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            originalMatrix[i][j] = rand() % 100;
        }
    }

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    // Determinant
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

    // Transpose
    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].threadId = i;
        pthread_create(&threads[i], NULL, transposeRows, &threadData[i]);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Log
    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].threadId = i;
        pthread_create(&threads[i], NULL, computeLog, &threadData[i]);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    sequentialDeterminant();
    sequentialTranspose();
    sequentialLogTransform();

    cout << "Determinant Correctness: " << (checkDeterminantCorrectness() ? "Passed" : "Failed") << endl;
    cout << "Transpose Correctness:   " << (checkTransposeCorrectness()   ? "Passed" : "Failed") << endl;
    cout << "Log Correctness:         " << (checkLogCorrectness()         ? "Passed" : "Failed") << endl;

    return 0;
}
