#ifndef VECTORLIB_H
#define VECTORLIB_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Structure definitions
typedef struct {
    int size; 
    double* data;
} Vector; 

typedef struct {
    int rows; 
    int cols; 
    double** data; 
} Matrix; 

// Vector function declarations
Vector* create_vector(int size);
void set_vector_values(Vector* vec, double* values);
void print_vector(Vector* vec);
void free_vector(Vector* vec);
double dot_product(Vector* vec1, Vector* vec2);
double vector_magnitude(Vector* vec);

// Matrix function declarations
Matrix* create_matrix(int rows, int cols);
Matrix* matrix_multiply(Matrix* mat1, Matrix* mat2);
void free_matrix(Matrix* mat);
Matrix* transpose_matrix(Matrix* mat);
Vector* matrix_vector_multiply(Matrix* mat, Vector* vec);
void print_matrix(Matrix* mat);
void set_matrix_values(Matrix* mat, double** values);

#endif
