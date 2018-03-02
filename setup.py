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
            'Programming Language :: Python :: 2.7',
        ],

        license="AGPL-3.0",

        platforms=["Linux", "Unix", "Mac OS-X"],

        # https://setuptools.readthedocs.io/en/latest/setuptools.html
        python_requires=">=2.7",
        setup_requires=[
            "setuptools>=18.0",  # minimum for Cython
            "cython>=0.28.1",
            "numpy>=1.12.0"
        ],
        install_requires=["numpy>=1.12.0"],

        test_suite="pyneural.test",

        ext_modules=cythonize(cython_extensions)
    )
