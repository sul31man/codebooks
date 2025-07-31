#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

// Simple thread pool implementation
typedef struct {
    pthread_t* threads;
    int num_threads;
    int active;
    pthread_mutex_t queue_mutex;
    pthread_cond_t work_available;
    pthread_cond_t work_done;
    int pending_work;
    
    // Work queue (simple for demo)
    void (*work_function)(int);
    int* work_data;
    int work_head;
    int work_tail;
    int work_count;
    int max_work;
} ThreadPool;

ThreadPool* pool = NULL;

// Worker function for thread pool
void* thread_pool_worker(void* arg) {
    while (pool->active) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        // Wait for work
        while (pool->work_count == 0 && pool->active) {
            pthread_cond_wait(&pool->work_available, &pool->queue_mutex);
        }
        
        if (!pool->active) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        // Get work
        int work_id = pool->work_data[pool->work_head];
        pool->work_head = (pool->work_head + 1) % pool->max_work;
        pool->work_count--;
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        // Do the work
        pool->work_function(work_id);
        
        // Signal work done
        pthread_mutex_lock(&pool->queue_mutex);
        pool->pending_work--;
        pthread_cond_signal(&pool->work_done);
        pthread_mutex_unlock(&pool->queue_mutex);
    }
    return NULL;
}

// Initialize thread pool
void init_thread_pool(int num_threads, int max_work_items) {
    pool = malloc(sizeof(ThreadPool));
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->num_threads = num_threads;
    pool->active = 1;
    pool->work_data = malloc(max_work_items * sizeof(int));
    pool->work_head = 0;
    pool->work_tail = 0;
    pool->work_count = 0;
    pool->max_work = max_work_items;
    pool->pending_work = 0;
    
    pthread_mutex_init(&pool->queue_mutex, NULL);
    pthread_cond_init(&pool->work_available, NULL);
    pthread_cond_init(&pool->work_done, NULL);
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, thread_pool_worker, NULL);
    }
}

// Add work to thread pool
void add_work(void (*work_func)(int), int work_id) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    pool->work_function = work_func;
    pool->work_data[pool->work_tail] = work_id;
    pool->work_tail = (pool->work_tail + 1) % pool->max_work;
    pool->work_count++;
    pool->pending_work++;
    
    pthread_cond_signal(&pool->work_available);
    pthread_mutex_unlock(&pool->queue_mutex);
}

// Wait for all work to complete
void wait_for_completion() {
    pthread_mutex_lock(&pool->queue_mutex);
    while (pool->pending_work > 0) {
        pthread_cond_wait(&pool->work_done, &pool->queue_mutex);
    }
    pthread_mutex_unlock(&pool->queue_mutex);
}

// Cleanup thread pool
void cleanup_thread_pool() {
    pool->active = 0;
    pthread_cond_broadcast(&pool->work_available);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    free(pool->threads);
    free(pool->work_data);
    pthread_mutex_destroy(&pool->queue_mutex);
    pthread_cond_destroy(&pool->work_available);
    pthread_cond_destroy(&pool->work_done);
    free(pool);
}

// Example work functions
void light_work(int work_id) {
    // Simulate light work (like a small calculation)
    usleep(1000);  // 1ms of work
    printf("Light work %d completed\n", work_id);
}

void heavy_work(int work_id) {
    // Simulate heavy work (like Monte Carlo with millions of samples)
    usleep(50000);  // 50ms of work
    printf("Heavy work %d completed\n", work_id);
}

// Traditional approach: create thread for each task
void* traditional_light_worker(void* arg) {
    int work_id = *(int*)arg;
    light_work(work_id);
    return NULL;
}

void* traditional_heavy_worker(void* arg) {
    int work_id = *(int*)arg;
    heavy_work(work_id);
    return NULL;
}

void test_light_work_scenario() {
    printf("=== LIGHT WORK SCENARIO (1ms per task) ===\n");
    printf("Thread creation overhead becomes significant!\n\n");
    
    const int num_tasks = 100;
    struct timespec start, end;
    
    // Test 1: Traditional approach (create thread per task)
    printf("Traditional approach (create thread per task):\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_t threads[num_tasks];
    int work_ids[num_tasks];
    
    for (int i = 0; i < num_tasks; i++) {
        work_ids[i] = i;
        pthread_create(&threads[i], NULL, traditional_light_worker, &work_ids[i]);
    }
    
    for (int i = 0; i < num_tasks; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double traditional_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                             (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Time: %.3f ms\n\n", traditional_time);
    
    // Test 2: Thread pool approach
    printf("Thread pool approach:\n");
    init_thread_pool(4, 200);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_tasks; i++) {
        add_work(light_work, i);
    }
    
    wait_for_completion();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double pool_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                      (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Time: %.3f ms\n", pool_time);
    printf("Speedup: %.2fx\n\n", traditional_time / pool_time);
    
    cleanup_thread_pool();
}

void test_heavy_work_scenario() {
    printf("=== HEAVY WORK SCENARIO (50ms per task) ===\n");
    printf("Thread creation overhead is negligible!\n\n");
    
    const int num_tasks = 8;
    struct timespec start, end;
    
    // Test 1: Traditional approach
    printf("Traditional approach (create thread per task):\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_t threads[num_tasks];
    int work_ids[num_tasks];
    
    for (int i = 0; i < num_tasks; i++) {
        work_ids[i] = i;
        pthread_create(&threads[i], NULL, traditional_heavy_worker, &work_ids[i]);
    }
    
    for (int i = 0; i < num_tasks; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double traditional_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                             (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Time: %.3f ms\n\n", traditional_time);
    
    // Test 2: Thread pool approach
    printf("Thread pool approach:\n");
    init_thread_pool(4, 20);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_tasks; i++) {
        add_work(heavy_work, i);
    }
    
    wait_for_completion();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double pool_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                      (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("Time: %.3f ms\n", pool_time);
    printf("Speedup: %.2fx\n\n", traditional_time / pool_time);
    
    cleanup_thread_pool();
}

void explain_when_to_use_what() {
    printf("=== WHEN TO USE THREAD POOLS VS SIMPLE THREADS ===\n\n");
    
    printf("USE THREAD POOLS WHEN:\n");
    printf("✅ Many small tasks (< 10ms each)\n");
    printf("✅ Continuous/repeated work\n");
    printf("✅ Web servers, database connections\n");
    printf("✅ Task queue systems\n");
    printf("✅ Thread creation overhead > 5%% of work time\n\n");
    
    printf("USE SIMPLE THREADS WHEN:\n");
    printf("✅ Few large tasks (> 50ms each)\n");
    printf("✅ One-shot computations\n");
    printf("✅ Monte Carlo simulations\n");
    printf("✅ Scientific computing\n");
    printf("✅ Thread creation overhead < 1%% of work time\n\n");
    
    printf("YOUR MONTE CARLO CASE:\n");
    printf("• Thread creation: ~0.1ms\n");
    printf("• Thread work: ~25ms\n");
    printf("• Overhead: 0.4%% ← NEGLIGIBLE!\n");
    printf("• Verdict: Simple threads are PERFECT!\n\n");
}

int main() {
    test_light_work_scenario();
    test_heavy_work_scenario();
    explain_when_to_use_what();
    
    return 0;
} 