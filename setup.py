import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# for cython compilation with setuptools in the most flexible manner see:
# https://github.com/Technologicat/setup-template-cython
cython_extensions = [
    Extension("pyneural.embedding_cy",
              sources=["pyneural/embedding_cy.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("pyneural.misc_cy",
              sources=["pyneural/misc_cy.pyx"],
              include_dirs=[numpy.get_include()])
]

if __name__ == '__main__':

    # https://packaging.python.org/guides/distributing-packages-using-setuptools/
    # https://setuptools.readthedocs.io/en/latest/setuptools.html

    setup(
        name="pyneural",
        version="0.1.0",
        description="Neural Networks Framework using Python and Numpy",
        packages=['pyneural', 'pyneural.test'],

        author="George Mavromatis",
        author_email="gmavrom@",
        url='https://github.com/gmavr/pyneural',

        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Programming Language :: Python :: Only',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],

        license="AGPL-3.0",

        platforms=["Linux", "Unix", "Mac OS-X"],

        python_requires=">=3.6",
        setup_requires=[
            "setuptools>=18.0",  # minimum for Cython
            "cython>=0.29.16",
            "numpy>=1.18.2",
            "nose>=1.3.7"
        ],
        install_requires=["numpy>=1.18.2"],

        ext_modules=cythonize(cython_extensions)
    )
