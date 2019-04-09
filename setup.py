from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='unfold',
      ext_modules=[
            CUDAExtension('unfold', ['unfold.cpp', 'unfold_backward_cuda.cu'])
            ],
      cmdclass={'build_ext': BuildExtension})