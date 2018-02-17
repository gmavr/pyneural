from setuptools import setup


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
        python_requires=">=2.7",
        install_requires=["numpy>=1.12.0"],

        test_suite="pyneural.test"
    )
