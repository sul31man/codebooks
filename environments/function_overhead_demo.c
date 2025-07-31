#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

// Simple function version
uint32_t xorshift_function(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

// Function that does the same math as our integration
double function_check_under_curve(double x, double y) {
    return (x >= 0 && x <= 4.0 && y >= 0 && y <= (x * x)) ? 1.0 : 0.0;
}

void test_function_call_overhead() {
    const int iterations = 10000000;
    struct timespec start, end;
    uint32_t state = 1234;
    double total = 0;
    
    printf("=== Function Call Overhead Test ===\n");
    printf("Testing %d iterations\n\n", iterations);
    
    // Test 1: With function calls
    state = 1234;
    total = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        uint32_t rand1 = xorshift_function(&state);
        uint32_t rand2 = xorshift_function(&state);
        
        double x = ((double)rand1 / 4294967295.0) * 4.0;
        double y = ((double)rand2 / 4294967295.0) * 16.0;
        
        total += function_check_under_curve(x, y);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_with_functions = (end.tv_sec - start.tv_sec) * 1000.0 + 
                                (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("With function calls: %.3f ms, result: %.0f\n", time_with_functions, total);
    
    // Test 2: Inline version (same logic, no function calls)
    state = 1234;
    total = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        // Inline XorShift
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        double x = ((double)state / 4294967295.0) * 4.0;
        
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        double y = ((double)state / 4294967295.0) * 16.0;
        
        // Inline curve check
        if (x >= 0 && x <= 4.0 && y >= 0 && y <= (x * x)) {
            total += 1.0;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_inline = (end.tv_sec - start.tv_sec) * 1000.0 + 
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Inline version:     %.3f ms, result: %.0f\n", time_inline, total);
    printf("Speedup from inlining: %.2fx\n\n", time_with_functions / time_inline);
}

void test_memory_allocation_overhead() {
    const int iterations = 1000000;
    struct timespec start, end;
    
    printf("=== Memory Allocation Overhead Test ===\n");
    printf("Testing %d iterations\n\n", iterations);
    
    // Test 1: With malloc/free every iteration (like creating vectors)
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    double total = 0;
    for (int i = 0; i < iterations; i++) {
        double* vec = malloc(2 * sizeof(double));
        vec[0] = (double)i;
        vec[1] = (double)(i * i);
        
        total += vec[0] + vec[1];
        
        free(vec);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_with_malloc = (end.tv_sec - start.tv_sec) * 1000.0 + 
                             (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("With malloc/free:   %.3f ms, result: %.0f\n", time_with_malloc, total);
    
    // Test 2: Stack variables (no allocation)
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    total = 0;
    for (int i = 0; i < iterations; i++) {
        double vec[2];
        vec[0] = (double)i;
        vec[1] = (double)(i * i);
        
        total += vec[0] + vec[1];
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_stack = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Stack variables:    %.3f ms, result: %.0f\n", time_stack, total);
    printf("Speedup from avoiding malloc: %.2fx\n\n", time_with_malloc / time_stack);
}

void explain_function_call_costs() {
    printf("=== Why Function Calls Are Expensive ===\n\n");
    
    printf("1. STACK OPERATIONS:\n");
    printf("   - Push parameters onto stack\n");
    printf("   - Save caller's registers\n");
    printf("   - Create new stack frame\n");
    printf("   - Return value handling\n");
    printf("   - Restore registers\n");
    printf("   - Pop stack frame\n\n");
    
    printf("2. CPU CACHE MISSES:\n");
    printf("   - Jump to different memory location\n");
    printf("   - Instruction cache may miss\n");
    printf("   - CPU pipeline may stall\n");
    printf("   - Branch prediction may fail\n\n");
    
    printf("3. COMPILER OPTIMIZATION BARRIERS:\n");
    printf("   - Can't optimize across function boundaries\n");
    printf("   - Loop unrolling blocked\n");
    printf("   - Register allocation suboptimal\n");
    printf("   - Dead code elimination blocked\n\n");
    
    printf("4. MEMORY ALLOCATION (malloc/free):\n");
    printf("   - System call overhead\n");
    printf("   - Heap traversal to find free blocks\n");
    printf("   - Memory fragmentation\n");
    printf("   - Cache pollution\n\n");
    
    printf("In tight loops with millions of iterations,\n");
    printf("these small costs multiply dramatically!\n");
}

int main() {
    test_function_call_overhead();
    test_memory_allocation_overhead();
    explain_function_call_costs();
    
    return 0;
} 