//implement a simple repetition encoder and decoder in C..

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>  // Added missing include
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "randomnumber.h"

int generate_bit(void){
    return xorshift_next() & 1;
}

char* generate_bits(int size){
    char* output_bits = malloc(size*sizeof(char));
    int random_bit; 

    for (int i = 0; i < size; i++){
        random_bit = generate_bit();
        output_bits[i] = random_bit + '0';
    }

    return output_bits; 
}

char* encoder(char* input_bits, int size){
    char passing_bit;
    char* output_bits = malloc(size * 3 * sizeof(char));

    for(int i=0; i < size; i++){
        passing_bit = input_bits[i];
        
        if(passing_bit == '1'){
            output_bits[i*3] = '1';
            output_bits[i*3 + 1] = '1';
            output_bits[i*3 + 2] = '1';
        }
        else{
            output_bits[i*3] = '0';
            output_bits[i*3 + 1] = '0';
            output_bits[i*3 + 2] = '0';
        }
    }
    
    return output_bits;
}

void add_noise(char* input_bits, int size){
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL)); 

    double noise_sample;
    double bit_value; 

    for(int i=0; i < size; i++){
        noise_sample = gsl_ran_gaussian(rng, 0.5);  // Reduced noise
        bit_value = input_bits[i] - '0';
        bit_value += noise_sample; 
        
        // Keep value as float but store back as char for now
        if (bit_value > 1.0) bit_value = 1.0;
        if (bit_value < 0.0) bit_value = 0.0;
        
        // Store the noisy value (will be properly decoded later)
        input_bits[i] = (char)(bit_value * 100) + '0';  // Scale for storage
    } 
    
    gsl_rng_free(rng);
}

char* decoder(char* input_bits, int size){
    // First, convert back to clean 0s and 1s based on threshold
    for(int i = 0; i < size; i++){
        double bit_value = (input_bits[i] - '0') / 100.0;  // Unscale
        if(bit_value > 0.5){
            input_bits[i] = '1';
        } else {
            input_bits[i] = '0';
        }
    }

    // Now decode using majority voting
    int decoded_size = size / 3;
    char* output_bit = malloc(decoded_size * sizeof(char)); 

    for(int i = 0; i < decoded_size; i++){
        int j = i * 3;  // Fixed: declared j
        
        char bit1 = input_bits[j];
        char bit2 = input_bits[j+1];
        char bit3 = input_bits[j+2];

        // Majority voting
        int count_ones = 0;
        if (bit1 == '1') count_ones++;
        if (bit2 == '1') count_ones++;
        if (bit3 == '1') count_ones++;
        
        if (count_ones >= 2) {
            output_bit[i] = '1';
        } else {
            output_bit[i] = '0';
        }
    }

    return output_bit;  // Fixed: added return
}

double decoder_accuracy(char* decoded_bits, char* original_bits, int size){  // Fixed: return double
    double accuracy = 0.0; 

    for(int i=0; i < size; i++){
        if(decoded_bits[i] == original_bits[i]){
            accuracy++; 
        }
    }

    accuracy = accuracy / size; 
    return accuracy; 
}

int main(){
    xorshift_seed(12345);  // Initialize random number generator
    
    printf("=== Repetition Code Test ===\n");
    
    char* generated_bits = generate_bits(10);
    printf("Original bits:  ");
    for(int i = 0; i < 10; i++) printf("%c", generated_bits[i]);
    printf("\n");

    char* encoded_bits = encoder(generated_bits, 10);  // Fixed: added size
    printf("Encoded bits:   ");
    for(int i = 0; i < 30; i++) printf("%c", encoded_bits[i]);
    printf("\n");

    add_noise(encoded_bits, 30);  // Fixed: use encoded size
    printf("Noisy bits:     ");
    for(int i = 0; i < 30; i++) printf("%c", encoded_bits[i]);
    printf("\n");

    char* decoded_bits = decoder(encoded_bits, 30);  // Fixed: added size
    printf("Decoded bits:   ");
    for(int i = 0; i < 10; i++) printf("%c", decoded_bits[i]);
    printf("\n");

    double accuracy = decoder_accuracy(decoded_bits, generated_bits, 10);  // Fixed: added size
    printf("Accuracy: %.2f%%\n", accuracy * 100);  // Fixed: use printf

    // Cleanup
    free(generated_bits);
    free(encoded_bits);
    free(decoded_bits);

    return 0;  // Fixed: return 0 for success
}
