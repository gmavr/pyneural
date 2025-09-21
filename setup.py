import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

cython_extensions = [
    Extension("pyneural.embedding_cy",
              sources=["pyneural/embedding_cy.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("pyneural.misc_cy",
              sources=["pyneural/misc_cy.pyx"],
              include_dirs=[numpy.get_include()])
]

# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://setuptools.readthedocs.io/en/latest/setuptools.html

setup(
    packages=['pyneural', 'pyneural.test', "samples"],
    ext_modules=cythonize(cython_extensions, compiler_directives={'language_level': "3"})
)
