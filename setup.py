from setuptools import setup, find_packages, Extension, Feature
import setuptools
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

#########################################
### inspired by simplejson's setup.py ###
#########################################
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, ValueError)

class BuildFailed(Exception):
    pass

class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()

#########################################

def run_setup(gpu_support):
    if gpu_support:
        setup_kw = {
            ext_modules : [
                Extension(
                    'ecGAN.gpuman',
                    ['src/gpuman.c'],
                    include_dirs=['/usr/local/cuda/include'],
                    libraries=['nvidia-ml'],
                    # WARNING driver branch can be other than 384!!
                    library_dirs=['/usr/lib/nvidia-384']
                )
            ],
            cmdclass : {
                build_ext : ve_build_ext,
            }
        }
    else:
        setup_kw = {}

    setup(
        name="ecGAN",
        version="0.1",
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'ecgan = ecGAN.cli:main'
            ]
        },
        install_requires=[
            ['astroid==1.6.1'],
            ['certifi==2018.1.18'],
            ['chardet==3.0.4'],
            ['cycler==0.10.0'],
            ['graphviz==0.8.1'],
            ['greenlet==0.4.13'],
            ['h5py==2.7.1'],
            ['idna==2.6'],
            ['imageio==2.2.0'],
            ['lazy-object-proxy==1.3.1'],
            ['matplotlib==2.1.0'],
            ['msgpack==0.5.4'],
            ['mxnet==1.1.0'],
            ['numpy==1.13.3'],
            ['olefile==0.45.1'],
            ['parso==0.1.1'],
            ['Pillow==4.3.0'],
            ['pyparsing==2.2.0'],
            ['python-dateutil==2.6.1'],
            ['pytz==2017.3'],
            ['PyYAML==3.12'],
            ['requests==2.18.4'],
            ['six==1.11.0'],
            ['urllib3==1.22'],
            ['wrapt==1.10.11'],
        ],
        **setup_kw)

try:
    setup(gpu_support=True)
except BuildFailed:
    print("Building extension Failed, retrying without gpu support.")
    setup(gpu_support=False)
