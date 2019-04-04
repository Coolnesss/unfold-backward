from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='unfold',
      ext_modules=[CppExtension('unfold', ['unfold.cpp'])],
      cmdclass={'build_ext': BuildExtension})