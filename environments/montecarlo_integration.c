#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h> 
#include <time.h>  // Added for timing

// Include your custom libraries
#include "vectorlib.h"     // For Vector and Matrix operations
#include "randomnumber.h"  // For XorShift random number generation

//this MC simulator will only work exclusively for regions starting from x = 0, f(x)>= 0.
double x_range = 4.0;
double y_range = 16.0;
int num_samples = 10000000;  // Increased to 10 million samples for heavier workload

// Function to evaluate: y = x^2
double function_to_integrate(double x) {
    return x * x;
}

// Function to get time difference in milliseconds
double get_time_diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// Optimized thread data structure that returns hit count
typedef struct{
    int thread_samples; 
    int thread_id;
    int hits;  // Each thread tracks its own hits
    uint32_t seed;  // Each thread gets its own seed
} ThreadData_Optimized; 

// Optimized thread function - no mutex, no vector allocation
void* thread_sample_optimized(void* arg){ 
    ThreadData_Optimized* data = (ThreadData_Optimized*) arg;
    
    // Initialize thread-local random state
    uint32_t local_state = data->seed;
    
    int local_hits = 0;
    
    for (int i = 0; i < data->thread_samples; i++){
        // Generate random points directly without vector allocation
        
        // Simple XorShift inline (avoid function call overhead)
        local_state ^= local_state << 13;
        local_state ^= local_state >> 17;
        local_state ^= local_state << 5;
        
        double x = ((double)local_state / 4294967295.0) * x_range;
        
        local_state ^= local_state << 13;
        local_state ^= local_state >> 17;
        local_state ^= local_state << 5;
        
        double y = ((double)local_state / 4294967295.0) * y_range;
        
        // Check if point is under curve
        if (x >= 0 && x <= x_range && y >= 0 && y <= (x * x)) {
            local_hits++;
        }
    }
    
    data->hits = local_hits;
    return NULL;
}

void VectorWithin_Serial(Vector* vec, int* hits){
    // Serial version - no mutex needed
    double x_point = vec->data[0];
    double y_point = vec->data[1];
    
    // Check if point is within bounds AND under the curve
    if (x_point >= 0 && x_point <= x_range && 
        y_point >= 0 && y_point <= function_to_integrate(x_point)) {
        (*hits)++;
    }
}

Vector* random_vector(void){
    //this method generates a random vector in the range we want
    Vector* vec = create_vector(2);
    
    // Use floats to get precise coordinates
    double x_value = xorshift_float() * x_range;  // 0 to x_range
    double y_value = xorshift_float() * y_range;  // 0 to y_range

    vec->data[0] = x_value;
    vec->data[1] = y_value;

    return vec;
}

// Serial sampling function for comparison
int serial_sampling(){
    int hits = 0;
    
    for (int i = 0; i < num_samples; i++){
        Vector* random_gen = random_vector();
        VectorWithin_Serial(random_gen, &hits);
        free_vector(random_gen);
    }
    
    return hits;
}

// Optimized serial version without vector allocation
int serial_sampling_optimized(){
    int hits = 0;
    
    for (int i = 0; i < num_samples; i++){
        double x = xorshift_float() * x_range;
        double y = xorshift_float() * y_range;
        
        if (x >= 0 && x <= x_range && y >= 0 && y <= (x * x)) {
            hits++;
        }
    }
    
    return hits;
}

// Optimized threaded sampling without mutex contention
int threaded_sampling_optimized(int num_threads){
    int thread_samples = num_samples / num_threads; 
    int remainder = num_samples % num_threads;

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData_Optimized* thread_data = malloc(num_threads * sizeof(ThreadData_Optimized));

    if (!threads || !thread_data) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Create threads with different seeds
    for (int i = 0; i < num_threads; i++){
        thread_data[i].thread_samples = thread_samples;
        thread_data[i].thread_id = i;
        thread_data[i].hits = 0;
        thread_data[i].seed = 1234 + i * 1000;  // Different seed per thread
        
        // Give remainder samples to the last thread
        if (i == num_threads - 1) {
            thread_data[i].thread_samples += remainder;
        }
       
       if (pthread_create(&threads[i], NULL, thread_sample_optimized, &thread_data[i]) != 0){
        printf("Error creating thread %d\n", i);
        free(threads);
        free(thread_data);
        return -1;
       }
    }

    // Wait for all threads and sum results
    int total_hits = 0;
    for (int i = 0; i < num_threads; i++){
        if (pthread_join(threads[i], NULL) != 0){
            printf("Error joining thread %d\n", i);
        }
        total_hits += thread_data[i].hits;
    }
    
    free(threads);
    free(thread_data);
    return total_hits;
}

void run_performance_comparison() {
    struct timespec start_time, end_time;
    
    printf("=== Performance Comparison ===\n");
    printf("Monte Carlo Integration of y = x^2 from 0 to %.1f\n", x_range);
    printf("Using %d samples\n\n", num_samples);
    
    // Test different thread counts
    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    double serial_time = 0;
    double serial_opt_time = 0;
    int serial_hits = 0;
    int serial_opt_hits = 0;
    
    printf("%-18s %-15s %-15s %-12s %-12s %-10s\n", 
           "Method", "Time (ms)", "Hits", "Hit Ratio", "Area", "Speedup");
    printf("-------------------------------------------------------------------------------\n");
    
    // Original Serial version with vectors
    xorshift_seed(1234);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    serial_hits = serial_sampling();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    serial_time = get_time_diff_ms(start_time, end_time);
    
    double serial_area = (x_range * y_range) * ((double)serial_hits / (double)num_samples);
    
    printf("%-18s %-15.3f %-15d %-12.4f %-12.6f %-10s\n", 
           "Serial (vectors)", serial_time, serial_hits, 
           (double)serial_hits / num_samples, serial_area, "1.00x");
    
    // Optimized Serial version without vectors
    xorshift_seed(1234);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    serial_opt_hits = serial_sampling_optimized();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    serial_opt_time = get_time_diff_ms(start_time, end_time);
    
    double serial_opt_area = (x_range * y_range) * ((double)serial_opt_hits / (double)num_samples);
    double opt_speedup = serial_time / serial_opt_time;
    
    printf("%-18s %-15.3f %-15d %-12.4f %-12.6f %-10.2fx\n", 
           "Serial (optimized)", serial_opt_time, serial_opt_hits, 
           (double)serial_opt_hits / num_samples, serial_opt_area, opt_speedup);
    
    // Optimized Threaded versions
    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        
        if (num_threads == 1) continue; // Skip 1 thread (that's serial)
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        int threaded_hits = threaded_sampling_optimized(num_threads);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double threaded_time = get_time_diff_ms(start_time, end_time);
        
        double threaded_area = (x_range * y_range) * ((double)threaded_hits / (double)num_samples);
        double speedup = serial_opt_time / threaded_time;  // Compare against optimized serial
        
        char method_name[20];
        sprintf(method_name, "%d Threads (opt)", num_threads);
        
        printf("%-18s %-15.3f %-15d %-12.4f %-12.6f %-10.2fx\n", 
               method_name, threaded_time, threaded_hits,
               (double)threaded_hits / num_samples, threaded_area, speedup);
    }
    
    // Show analytical result
    double analytical_result = pow(x_range, 3) / 3.0;
    printf("\nAnalytical result: %.6f\n", analytical_result);
}

int main(void) {
    
    run_performance_comparison();
    
    return 0;
}
