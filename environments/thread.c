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

// Global variables for timing
struct timespec start_time, end_time;

// Function to get time difference in milliseconds
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// Serial array summation function
long serial_sum(long *array, int size) {
    long sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

// Thread function for partial summation
void* thread_sum(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->partial_sum = 0;
    
    for (int i = data->start_idx; i < data->end_idx; i++) {
        data->partial_sum += data->array[i];
    }
    
    return NULL;
}

// Threaded array summation function
long threaded_sum(long *array, int size, int num_threads) {
    
    pthread_t *threads;
    ThreadData *thread_data;
    long total_sum = 0;
    
    // Allocate memory for threads and thread data
    threads = malloc(num_threads * sizeof(pthread_t));
    thread_data = malloc(num_threads * sizeof(ThreadData));
    
    if (!threads || !thread_data) {
        printf("Memory allocation failed!\n");
        return -1;
    }
    
    int chunk_size = size / num_threads;
    int remainder = size % num_threads;
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].array = array;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i + 1) * chunk_size;
        
        // Handle remainder for the last thread
        if (i == num_threads - 1) {
            thread_data[i].end_idx += remainder;
        }
        
        if (pthread_create(&threads[i], NULL, thread_sum, &thread_data[i]) != 0) {
            printf("Error creating thread %d\n", i);
            free(threads);
            free(thread_data);
            return -1;
        }
    }
    
    // Wait for all threads to complete and sum results
    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            printf("Error joining thread %d\n", i);
        }
        total_sum += thread_data[i].partial_sum;
    }

    free(threads);
    free(thread_data);
    return total_sum;
}

// Function to initialize array with random values
void initialize_array(long *array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100 + 1; // Random values between 1-100
    }
}

// Function to run performance test
void run_performance_test(int array_size, int max_threads) {
    long *array = malloc(array_size * sizeof(long));
    if (!array) {
        printf("Failed to allocate memory for array of size %d\n", array_size);
        return;
    }
    
    initialize_array(array, array_size);
    
    printf("\n=== Performance Test: Array Size = %d ===\n", array_size);
    printf("%-12s %-15s %-15s %-10s\n", "Method", "Time (ms)", "Sum", "Speedup");
    printf("-------------------------------------------------------\n");
    
    // Serial version
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    long serial_result = serial_sum(array, array_size);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double serial_time = get_time_diff(start_time, end_time);
    
    printf("%-12s %-15.3f %-15ld %-10s\n", "Serial", serial_time, serial_result, "1.00x");
    
    // Threaded versions with different thread counts
    for (int num_threads = 2; num_threads <= max_threads; num_threads *= 2) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        long threaded_result = threaded_sum(array, array_size, num_threads);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double threaded_time = get_time_diff(start_time, end_time);
        
        double speedup = serial_time / threaded_time;
        char speedup_str[20];
        sprintf(speedup_str, "%.2fx", speedup);
        
        printf("%-12s %-15.3f %-15ld %-10s", 
               (num_threads == 2) ? "2 Threads" :
               (num_threads == 4) ? "4 Threads" :
               (num_threads == 8) ? "8 Threads" : "16 Threads",
               threaded_time, threaded_result, speedup_str);
        
        // Verify correctness
        if (threaded_result != serial_result) {
            printf(" [ERROR: Results don't match!]");
        }
        printf("\n");
    }
    
    free(array);
}

// Function to demonstrate basic threading concept
void demonstrate_threading() {
    printf("=== Threading Demonstration ===\n");
    
    int demo_size = 1000;
    long *demo_array = malloc(demo_size * sizeof(long));
    
    // Initialize with simple values for easy verification
    for (int i = 0; i < demo_size; i++) {
        demo_array[i] = i + 1; // 1, 2, 3, ..., 1000
    }
    
    printf("Array size: %d\n", demo_size);
    printf("Expected sum: %ld\n", (long)demo_size * (demo_size + 1) / 2);
    
    // Test with different thread counts
    printf("\nTesting with different thread counts:\n");
    for (int threads = 1; threads <= 8; threads *= 2) {
        long result = (threads == 1) ? 
            serial_sum(demo_array, demo_size) : 
            threaded_sum(demo_array, demo_size, threads);
        
        printf("Threads: %d, Sum: %ld\n", threads, result);
    }
    
    free(demo_array);
    printf("\n");
}

int main() {
    printf("Threaded Array Summation Performance Comparison\n");
    printf("===============================================\n");
    
    // Get number of CPU cores
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    printf("System has %d CPU cores\n", num_cores);
    
    // Demonstrate basic threading
    demonstrate_threading();
    
    // Run performance tests with different array sizes
    int test_sizes[] = {100000, 1000000, 10000000, 50000000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    int max_threads = (num_cores > 8) ? 16 : 8; // Test up to 8 or 16 threads
    
    for (int i = 0; i < num_tests; i++) {
        run_performance_test(test_sizes[i], max_threads);
    }
    
    printf("\n=== Analysis ===\n");
    printf("1. Speedup depends on array size and number of threads\n");
    printf("2. Small arrays may show overhead from thread creation\n");
    printf("3. Optimal thread count often matches CPU core count\n");
    printf("4. Memory bandwidth can become a bottleneck\n");
    printf("5. Context switching overhead increases with more threads\n");
    
    return 0;
}
