from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='cuda_gpu_environment',
    ext_modules=[
        CUDAExtension(
            name='env',  # This will be the import name
            sources=[
                'environment.cu',
                'amp_decoder.cu'
            ],
            include_dirs=[
                # Add any additional include directories here if needed
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_70',  # Adjust for your GPU architecture
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            },
            libraries=['curand'],  # Link against curand for random number generation
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'numpy',
    ],
    description='CUDA-accelerated GPU environment for RL training',
    author='Your Name',
    version='0.1.0',
)
