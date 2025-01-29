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
static vector<vector<double>> logTransformedMatrix(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
static vector<vector<double>> sequentialLogTransformed(MATRIX_SIZE, vector<double>(MATRIX_SIZE));

struct ThreadData {
    int threadId;
};

// Parallel worker: Row & column-wise cyclic distribution
void* computeLog(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->threadId;

    // Each thread works on rows tid, tid+NUM_THREADS, ...
    for (int i = tid; i < MATRIX_SIZE; i += NUM_THREADS) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            logTransformedMatrix[i][j] = log(originalMatrix[i][j] + 1e-9); 
            // Adding a small offset (1e-9) to avoid log(0) if random data includes 0
        }
    }
    pthread_exit(NULL);
}

// Sequential log transform
void sequentialLogTransform() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            sequentialLogTransformed[i][j] = log(originalMatrix[i][j] + 1e-9);
        }
    }
}

// Correctness check
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
            originalMatrix[i][j] = rand() % 100 + 1; 
        }
    }

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        threadData[i].threadId = i;
        pthread_create(&threads[i], NULL, computeLog, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    sequentialLogTransform();

    if (checkLogCorrectness()) {
        cout << "Log Transformation Correctness: Passed" << endl;
    } else {
        cout << "Log Transformation Correctness: Failed" << endl;
    }

    return 0;
}
