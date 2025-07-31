//implement a simple reptition codeer and decoder in C..

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "randomnumber.h"

//we need a function which encodes data, which decodes data. We also need a function whcih adds noise to the data and which applies a method of turning the float back into binary.
//how do we represent this data ? are we going to create a structure which has this ? because we could technically just feed in 1s and 0s and then the function just goes through each digit
//turns the digit into some

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
    gsl_rng_set(rng, time(NULL)); // seed with the currnet time. now the AWGN is setup.

    double noise_sample = gsl_ran_gaussian(rng, 1.0);
    double bit_value; 
    // now we go through all the bits and add it through

    for(int i=0; i < size; i++){
         
        noise_sample = gsl_ran_gaussian(rng, 1.0);
        bit_value = input_bits[i] - '0';
        bit_value += noise_sample ; 
        input_bits[i] = bit_value + '0';
    } 
}


char* decoder(char* input_bits, int size){
    
    double bit_value; 

    for(int i = 0; i < size; i++){

       bit_value = input_bits[i] - '0';
       if(bit_value > 0.5){

        bit_value = 1;
       }
       else{

        bit_value = 0;
       }

       input_bits[i] = bit_value + '0';
    }

    //now we should have the bits only consisting of 1s and 0s
    double bit1;
    double bit2; 
    double bit3;
    char* output_bit = malloc(size*sizeof(char)); 

    for(int i = 0; i < size; i++){
        
       j = i*3; 

       bit1 = input_bits[j] ;
       bit2 = input_bits[j+1];
       bit3 = input_bits[j+2];

       if (bit1==0 && bit2==0 || bit1==0 && bit3==0 || bit2==0 && bit3==0){
          
          output_bit[i] = '0';
 
       }
       else{
        output_bit[i] = '1';
       }


    }


}

int decoder_accuracy(char* decoded_bits, char* original_bits, int size){
     
    double accuracy; 

    for(int i=0; i < size; i++){
 
       if(decoded_bits[i] == original_bits[i]){
        
         accuracy++; 
       }


    }

    accuracy = accuracy / size; 
    
    return accuracy; 

}




int main (){

    char* generated_bits = generate_bits(10);

    char* encoded_bits = encoder(generated_bits);

    add_noise(encoded_bits, 10);

    char* decoded_bits = decoder(encoded_bits);

    int accuracy = decoder_accuracy(decoded_bits, generated_bits);

    print(accuracy);



    return -1;
}