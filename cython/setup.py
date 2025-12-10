from distutils.core import setup
from Cython.Build import cythonize
import numpy

# com
setup(
    ext_modules = cythonize("tropicalpy.pyx"),
    include_dirs=[numpy.get_include()]
)