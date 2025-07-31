#ifndef THREAD_H
#define THREAD_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

// Structure to pass data to threads
typedef struct {
    long *array;           // Pointer to the array
    int start_idx;         // Starting index for this thread
    int end_idx;           // Ending index for this thread
    long partial_sum;      // Result from this thread
} ThreadData;

// Timing function declarations
double get_time_diff(struct timespec start, struct timespec end);

// Array summation function declarations
long serial_sum(long *array, int size);
long threaded_sum(long *array, int size, int num_threads);
void* thread_sum(void* arg);

// Utility function declarations
void initialize_array(long *array, int size);
void run_performance_test(int array_size, int max_threads);
void demonstrate_threading(void);

// Global timing variables (external declarations)
extern struct timespec start_time, end_time;

#endif 