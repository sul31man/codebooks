# Makefile for CUDA environment
NVCC = nvcc
CC = gcc
CFLAGS = -fPIC -O3
NVCCFLAGS = -shared -Xcompiler -fPIC -O3

# Target library
TARGET = libenvironment.so

# Source files
CUDA_SOURCES = environment.cu
C_SOURCES = 

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Compile CUDA source to object file
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

# Compile C source to object file
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Link objects into shared library
$(TARGET): $(CUDA_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcudart

# Clean build artifacts
clean:
	rm -f *.o $(TARGET)

# Install (copy to current directory)
install: $(TARGET)
	cp $(TARGET) .

# Test compilation
test: $(TARGET)
	@echo "Compilation successful! Library created: $(TARGET)"

.PHONY: all clean install test 