//Improved repetition encoder and decoder in C
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "randomnumber.h"

int generate_bit(void){
    return xorshift_next() & 1;
}

char* generate_bits(int size){
    char* output_bits = malloc((size + 1) * sizeof(char));  // +1 for null terminator
    
    for (int i = 0; i < size; i++){
        output_bits[i] = generate_bit() + '0';
    }
    output_bits[size] = '\0';  // Null terminate
    
    return output_bits; 
}

char* encoder(char* input_bits, int size){
    char* output_bits = malloc((size * 3 + 1) * sizeof(char));
    
    for(int i = 0; i < size; i++){
        char bit = input_bits[i];
        
        // Repeat each bit 3 times
        output_bits[i*3] = bit;
        output_bits[i*3 + 1] = bit;
        output_bits[i*3 + 2] = bit;
    }
    output_bits[size * 3] = '\0';
    
    return output_bits;
}

// Better noise function using bit flip probability
void add_noise(char* input_bits, int size, double bit_error_rate){
    for(int i = 0; i < size; i++){
        if (xorshift_float() < bit_error_rate) {
            // Flip this bit
            if (input_bits[i] == '0') {
                input_bits[i] = '1';
            } else {
                input_bits[i] = '0';
            }
        }
    }
}

char* decoder(char* input_bits, int size){
    int decoded_size = size / 3;
    char* output_bits = malloc((decoded_size + 1) * sizeof(char));
    
    for(int i = 0; i < decoded_size; i++){
        int j = i * 3;
        
        // Count 1s and 0s in the triplet
        int count_ones = 0;
        if (input_bits[j] == '1') count_ones++;
        if (input_bits[j+1] == '1') count_ones++;
        if (input_bits[j+2] == '1') count_ones++;
        
        // Majority voting
        if (count_ones >= 2) {
            output_bits[i] = '1';
        } else {
            output_bits[i] = '0';
        }
    }
    output_bits[decoded_size] = '\0';
    
    return output_bits;
}

double calculate_accuracy(char* decoded_bits, char* original_bits, int size){
    int correct = 0;
    
    for(int i = 0; i < size; i++){
        if(decoded_bits[i] == original_bits[i]){
            correct++; 
        }
    }
    
    return (double)correct / size; 
}

void test_repetition_code(int data_size, double error_rate) {
    printf("\n=== Testing with %d bits, %.1f%% error rate ===\n", data_size, error_rate * 100);
    
    // Generate random data
    char* original_bits = generate_bits(data_size);
    printf("Original:  %s\n", original_bits);
    
    // Encode using repetition code
    char* encoded_bits = encoder(original_bits, data_size);
    printf("Encoded:   %s\n", encoded_bits);
    
    // Add noise
    add_noise(encoded_bits, data_size * 3, error_rate);
    printf("With noise: %s\n", encoded_bits);
    
    // Decode
    char* decoded_bits = decoder(encoded_bits, data_size * 3);
    printf("Decoded:   %s\n", decoded_bits);
    
    // Calculate accuracy
    double accuracy = calculate_accuracy(decoded_bits, original_bits, data_size);
    printf("Accuracy: %.1f%%\n", accuracy * 100);
    
    // Count errors corrected
    int original_errors = 0;
    for (int i = 0; i < data_size * 3; i++) {
        char original_encoded = (i % 3 == 0) ? original_bits[i/3] : 
                               (i % 3 == 1) ? original_bits[i/3] : original_bits[i/3];
        if (encoded_bits[i] != original_encoded) {
            original_errors++;
        }
    }
    printf("Errors introduced: %d, Errors corrected: %d\n", 
           original_errors, original_errors - (data_size - (int)(accuracy * data_size)));
    
    // Cleanup
    free(original_bits);
    free(encoded_bits);
    free(decoded_bits);
}

int main(){
    xorshift_seed(12345);
    
    printf("=== Repetition Code Error Correction Demo ===\n");
    
    // Test with different error rates
    test_repetition_code(10, 0.05);  // 5% error rate
    test_repetition_code(10, 0.10);  // 10% error rate
    test_repetition_code(10, 0.20);  // 20% error rate
    test_repetition_code(10, 0.30);  // 30% error rate
    
    return 0;
}
