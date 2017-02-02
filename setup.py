from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "My word2vec_inner",
    ext_modules = cythonize('fm_ngram_inner.pyx'),
    include_dirs=[numpy.get_include()]
)