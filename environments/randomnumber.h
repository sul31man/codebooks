#ifndef RANDOMNUMBER_H
#define RANDOMNUMBER_H

#include <stdint.h>

// Function declarations (prototypes)
void xorshift_seed(uint32_t seed);
uint32_t xorshift_next(void);
float xorshift_float(void);
int xorshift_range(int min, int max);

#endif