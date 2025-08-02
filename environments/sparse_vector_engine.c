//now i need to create a sparse vector inner product engine with efficient representations and multiplications. 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vectorlib.h"

#define CACHE_LINE_SIZE 64

typedef struct {
    int* indices;
    float* values;
    int nnz; // number of non-zero elements
    int capacity; // maximum number of elements that can be stored
    // Padding to prevent false sharing between different vectors in multithreaded environments
    char padding[CACHE_LINE_SIZE - sizeof(int*) - sizeof(float*) - 2*sizeof(int)];
} SparseVector;

// Create a cache-aligned sparse vector
SparseVector* create_sparse_vector(int capacity) {
    SparseVector* vec = malloc(sizeof(SparseVector));
    if (!vec) return NULL;
    
    // Use posix_memalign for cache-line aligned arrays
    if (posix_memalign((void**)&vec->indices, CACHE_LINE_SIZE, capacity * sizeof(int)) != 0) {
        free(vec);
        return NULL;
    }
    
    if (posix_memalign((void**)&vec->values, CACHE_LINE_SIZE, capacity * sizeof(float)) != 0) {
        free(vec->indices);
        free(vec);
        return NULL;
    }
    
    vec->nnz = 0;
    vec->capacity = capacity;
    return vec;
}

// Free a sparse vector and its aligned memory
void free_sparse_vector(SparseVector* vec) {
    if (vec) {
        free(vec->indices);
        free(vec->values);
        free(vec);
    }
}

float sparse_sparse_product(SparseVector* vec1, SparseVector* vec2){

   //this will be more efficient for computing inner products between sparse vectors than using the usual inner products; 

   int* indices1 = vec1->indices;
   int* indices2 = vec2->indices;
   
   // now we have access to the indices in which the sparse vector is non zero

   int nnz1 = vec1->nnz;
   int nnz2 = vec2->nnz;

   // we will iterate using two pointers
   int idx1 = 0;
   int idx2 = 0;
   float result = 0.0f; 

   // Fixed loop condition - check array bounds instead of values
   while(idx1 < nnz1 && idx2 < nnz2){

     if (indices1[idx1] == indices2[idx2]){

       result += vec1->values[idx1] * vec2->values[idx2]; 
       idx1++;
       idx2++;

     }

     else if( indices1[idx1] > indices2[idx2]){

        idx2++;
     }

     else{

        idx1++;
     }
      
   }

   
return result; 

}

float sparse_dense_product(SparseVector* vec1, Vector* vec2){
  
  float result = 0.0f;

  // Fixed loop - iterate through all non-zero elements
  for(int i = 0; i < vec1->nnz; i++) {
     int index = vec1->indices[i];
     result += vec1->values[i] * (float)vec2->data[index];
  }

  return result; 

}

void sort_sparse_vector(SparseVector* vec){

   
   //we need to go through vec->indices, sort them and change vec->values accordingly.
   int temp_idx;
   float temp_val;

   for(int i = 0; i < vec->nnz; i++){

    for (int j = 0; j < vec->nnz-i-1; j++){


        if(vec->indices[j] > vec->indices[j+1]){
           
            temp_idx = vec->indices[j];
            vec->indices[j] = vec->indices[j+1];
            vec->indices[j+1] = temp_idx; 

            temp_val = vec->values[j]; 
            vec->values[j] = vec->values[j+1];
            vec->values[j+1] = temp_val;
            

        }
    }
   }

}

// Helper function to add an element to sparse vector
int add_sparse_element(SparseVector* vec, int index, float value) {
    if (vec->nnz >= vec->capacity) {
        return -1; // No space
    }
    
    vec->indices[vec->nnz] = index;
    vec->values[vec->nnz] = value;
    vec->nnz++;
    return 0; // Success
}

// Helper function to check alignment (for debugging)
void check_alignment(void* ptr, const char* name) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % CACHE_LINE_SIZE == 0) {
        printf("%s is cache-line aligned\n", name);
    } else {
        printf("%s is NOT cache-line aligned (offset: %lu)\n", name, addr % CACHE_LINE_SIZE);
    }
}





