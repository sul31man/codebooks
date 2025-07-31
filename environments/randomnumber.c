#include <stdint.h>
#include <stdio.h>

static uint32_t xorshift_state = 1;

void xorshift_seed(uint32_t seed){
    // Check if seed is zero (XorShift cannot use zero as state)
    if (seed == 0) {
        // Set state to 1 if seed is zero
        xorshift_state = 1;
    } else {
        // Set state to the provided seed
        xorshift_state = seed;
    }
}

uint32_t xorshift_next(void){
    // Get current state into a local variable
    uint32_t x = xorshift_state;
    
    // Apply first XOR-shift: x ^= x << 13
    // This shifts x left by 13 bits and XORs with original x

    uint32_t shifted_number = x << 13;

    x = x ^ shifted_number ; 
    
    // Apply second XOR-shift: x ^= x >> 17  
    // This shifts the result right by 17 bits and XORs with current x
    
    shifted_number = x >> 17; 
    x = x ^ shifted_number;
    

    // Apply third XOR-shift: x ^= x << 5
    // This shifts the result left by 5 bits and XORs with current x
    
    shifted_number = x<<5; 
    x = x ^ shifted_number;    

    // Update the global state with the new value
    xorshift_state = x;
    
    // Return the new random number
    return x;
}

float xorshift_float(void){
    // Call xorshift_next() to get a random 32-bit integer
    
    uint32_t random_number = xorshift_next();
    // Convert the integer to float by dividing by the maximum uint32_t value

    float random_float = (float)random_number / 4294967295.0f;

    // Use 4294967295.0f (which is 2^32 - 1) as divisor
    // This gives you a float between 0.0 and 1.0
    
    // Return the normalized float

    return random_float;
}

int xorshift_range(int min, int max){
    // Validate that min <= max

    if (min > max){

        printf("max is smaller than min, retry");
        return -1;
    }
    // If not, you could swap them or return an error
    
    // Calculate the range size: (max - min + 1)
    
    int range_size = max-min + 1 ;
    // Get a random number using xorshift_next()
    
    uint32_t random_number = xorshift_next();

    // Use modulo to map the random number to the range size

    random_number = random_number % range_size ; 
    // Then add min to shift the range to [min, max]

    random_number = min + random_number; 
    // Formula: (random_number % range_size) + min
    
    // Return the result

    return random_number;
}  

// Test function removed to avoid duplicate main() when linking with other files