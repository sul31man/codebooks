import torch 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup 


setup(
    name="env2", 
    ext_modules=[
        CUDAExtension(
            name="env2", 
            sources=["env2.cu"], 
            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]}
        )
    ], 
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)}
)







