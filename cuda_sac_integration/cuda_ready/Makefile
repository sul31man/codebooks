# Makefile for fixed parallel CUDA environment

NVCC = nvcc
CC = gcc

# Stable flags for reliable compilation
NVCC_FLAGS = -shared -Xcompiler -fPIC -O2 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -Xcompiler -Wall -Xcompiler -Wextra \
             --ptxas-options=-v \
             -maxrregcount=40

# Target library
TARGET = libenvironment_fixed.so
SOURCE = environment_parallel_fixed.cu

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Fixed library compiled successfully: $(TARGET)"
	@echo "Features enabled:"
	@echo "  - One block per Ka value (optimal distribution)" 
	@echo "  - Safe shared memory with __syncthreads()"
	@echo "  - Simple reduction pattern (no complex cooperation)"
	@echo "  - Direct global memory access (no hanging)"
	@echo "  - Reduced memory per thread (~4KB)"

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	@echo "Running basic library test..."
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); print('✅ Library loads successfully')"

# Help target
help:
	@echo "Available targets:"
	@echo "  all    - Build fixed library (default)"
	@echo "  clean  - Remove compiled files"
	@echo "  test   - Test library loading"
	@echo "  help   - Show this help" 