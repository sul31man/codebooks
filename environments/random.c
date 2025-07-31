


// test_gsl.c
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main() {
    // Initialize random number generator
    gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937);
    
    // Generate some random numbers
    printf("GSL is working!\n");
    for (int i = 0; i < 5; i++) {
        double x = gsl_ran_gaussian(r, 1.0);
        printf("Random Gaussian: %.6f\n", x);
    }
    
    gsl_rng_free(r);
    return 0;
}