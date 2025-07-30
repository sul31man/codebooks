#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

typedef struct {

    long* array;
    int start_idx;
    int end_idx;
    int partial_sum; 
} ThreadData; 


struct timepsec start_time, end_time ; 

void* thread_sum(void* arg){
    
    ThreadData* data = (ThreadData*) arg;

    data->partial_sum = 0;

    for(int i = data->start_idx; i < data->end_idx; i++){

       data->partial_sum += data->array[i];
    }

    return NULL;
}


int serial_sum(long* array, int size){

    int sum = 0;

    for (int i=0; i < size; i++){

        sum += array[i];
    }

}

void* threaded_sum(long* array, int num_threads, int array_size){

    pthread_t* threads = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    ThreadData* threads_data = (ThreadData*)malloc(num_threads*sizeof(ThreadData));
    
    int chunk_size = array_size / num_threads;

    for(int i = 0; i < num_threads; i++){
         
         threads_data[i].array = array;
         threads_data[i].start_idx = i*chunk_size;
         threads_data[i].end_idx = (i+1)*chunk_size;
         threads_data[i].partial_sum = NULL;
          
    }//now all of the threads are initiliased 

    //now we need to pass the threads, the 

}