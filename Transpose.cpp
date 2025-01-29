#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <ctime>

using namespace std;

static const int MATRIX_SIZE = 1024;
static const int NUM_THREADS = 6;

static vector<vector<double>> originalMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
static vector<vector<double>> transposedMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
static vector<vector<double>> sequentialTransposed(MATRIX_SIZE, vector<double>(MATRIX_SIZE));

struct ThreadData {
    int threadId;
};

// Parallel worker: Transpose rows in a cyclic fashion
void* transposeRows(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->threadId;

    // Each thread handles rows with stride of NUM_THREADS
    for (int r = tid; r < MATRIX_SIZE; r += NUM_THREADS) {
        for (int c = 0; c < MATRIX_SIZE; ++c) {
            transposedMatrix[c][r] = originalMatrix[r][c];
        }
    }
    pthread_exit(NULL);
}

// Sequential transpose
void sequentialTranspose() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            sequentialTransposed[j][i] = originalMatrix[i][j];
        }
    }
}

// Correctness check
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

int main() {
    srand((unsigned)time(NULL));

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            originalMatrix[i][j] = rand() % 100 + 1;
        }
    }

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].threadId = i;
        pthread_create(&threads[i], NULL, transposeRows, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    sequentialTranspose();

    if (checkTransposeCorrectness()) {
        cout << "Transpose Correctness: Passed" << endl;
    } else {
        cout << "Transpose Correctness: Failed" << endl;
    }

    return 0;
}
