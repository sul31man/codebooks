// lets create a matrix and vector library 

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

typedef struct{

    int size; 
    double* data;
} Vector; 

typedef struct{

    int rows; 
    int cols; 
    double** data; 
} Matrix; 




Vector* create_vector(int size){


    double* data = (double*)calloc(size, sizeof(double));

    Vector* vec = (Vector*)malloc(sizeof(Vector)); 
    
    if (!vec){

        printf("Fucked up !\n");
        
        return NULL;
    }


    vec->size = size; 
    vec->data = data; 

    if (!vec->data){
        printf("memory allocation failed");
        free(vec);
        return NULL; 
    }

    return vec; 
}


void set_vector_values(Vector* vec, double* values){

    for (int i = 0; i < vec->size; i++){
     
      vec->data[i] = values[i];

    }

}

void print_vector(Vector* vec){

    for (int i = 0; i < vec->size; i++){

        printf(vec->data[i]);

    }

}

double dot_product(Vector* vec1, Vector* vec2){

    if (vec1->size != vec2->size){

        printf("fucked up!");
        return NULL;
    }

    double sum = 0;

    for(int i = 0; i < vec1->size; i++){

        sum += vec1->data[i] * vec2->data[j]; 
    }

    return sum;
}

double vector_magnitude(Vector* vec){

    double result = 0; 

    for (int i =0; i<vec->size; i++){

        result += vec->data[i] * vec->data[i];
    }

    return sqrt(result);

}

Matrix* create_matrix(int rows, int cols){

    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    
    mat->data = (double**)malloc(rows*sizeof(double*)); 

    mat->rows = rows; 

    mat->cols = cols;

    if (!mat->data){

        free(mat);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++){

        mat->data[i] = (double*)calloc(cols, sizeof(double));

        if(!mat->data[i]){

            for (int j = 0; j < i; j++){

             free(mat->data[j]);
            }

            free(mat->data);

            free(mat);

            return NULL;
        }
    }
    return mat; 
}


Matrix* matrix_multiply(Matrix* mat1, Matrix* mat2){

    if (mat1->cols != mat2->rows){


        printf("brav");
        
        return NULL; 
    }
    
    Matrix* result = create_matrix(mat1->rows, mat2->cols);

    for (int i = 0; i < mat1->rows; i++){
        
        

        for (int j = 0; j < mat2->cols; j++){
           
            double sum = 0;

            for (int k = 0; k < mat1->cols; k++){

                sum += mat1->data[i][k] * mat2->data[k][j]; 
                

            }
            result->data[i][j] = sum; 
        }

    }
}

void free_matrix(Matrix* mat){

   for (int i =0; i < mat->rows; i++){
       
     double* data = mat->data[i]; // this is the pointer to each vector in the matrix
    
     free(data);
    }

    free(mat->data);

    free(mat);
}


Matrix* transpose_matrix(Matrix* mat){
  
  Matrix* trans = create_matrix(mat->cols,mat->rows);

  double value; 

  for (int i = 0; i < mat->rows; i++){


    for ( int j = 0; j < mat->cols; j++){
        
        value = mat->data[i][j];

        trans->data[j][i] = value; 
       
    }
  }

  return trans;

}

Vector* matrix_vector_multiply(Matrix* mat, Vector* vec){

    // Check if matrix columns match vector size
    if (mat->cols != vec->size) {
        printf("Matrix-vector multiplication error: dimension mismatch\n");
        return NULL;
    }

    Vector* result = create_vector(mat->rows);
    if (!result) {
        return NULL;
    }
    
    double sum; 
    
    for (int i = 0; i < mat->rows; i++){
        sum = 0;
        for (int k = 0; k < mat->cols; k++){
          
         sum += mat->data[i][k] * vec->data[k];      
            
        }

        result->data[i] = sum; 

    }

    return result; 
}

