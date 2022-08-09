from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pdebench',
    version='0.0.1',
    install_requires=['scipy',
                        'matplotlib',
                        'h5py',
                        'pandas',
                        'python-dotenv',
                        'hydra-core',
                        'omegaconf',
                        'deepxde'
                        ],
    packages=['pdebench'],
    author='Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger, Mathias Niepert',
    author_email='Makoto.Takamoto@neclab.eu, timothy.praditia@iws.uni-stuttgart.de',
    url='https://github.com/pdebench/PDEBench',
    zip_safe=False,
    description='PDEBench: An Extensive Benchmark for Scientific Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: see licence file",
    ],
)