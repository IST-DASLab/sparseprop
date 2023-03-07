from setuptools import setup, find_packages, Extension
import subprocess

subprocess.run(["pip install pybind11"], shell=True)

proc = subprocess.Popen(["python3 -m pybind11 --includes"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode('ascii').strip().split()

setup(
    name='sparseprop',
    version='0.1.11',    
    description='SparseProp: Efficient Sparse Backpropagation for Faster Training of Neural Networks',
    url='https://github.com/IST-DASLab/sparseprop',
    author='Mahdi Nikdan, Tommaso Pegolotti, Eugenia Iofinova, Eldar Kurtic, Dan Alistarh',
    author_email='mahdi.nikdan@ist.ac.at, tommaso.pegolotti@inf.ethz.ch, eugenia.iofinova@ist.ac.at, eldar.kurtic@ist.ac.at, dan.alistarh@ist.ac.at',
    license='Apache License 2.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    ext_modules=[Extension(
        'backend',
        [
            'sparseprop/backend.cpp',
        ],
        extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++11', '-fPIC', *out, '-march=native', '-fopenmp', '-ffast-math'],
        extra_link_args=['-lgomp']
    )],
    install_requires=[
        'setuptools>=59.0',
        'pybind11>=2.0.0',
        'scipy',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
)