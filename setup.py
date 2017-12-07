from setuptools import setup, find_packages, Extension
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
        ['certifi==2017.7.27.1'],
        ['chardet==3.0.4'],
        ['cycler==0.10.0'],
        ['graphviz==0.8.1'],
        ['idna==2.6'],
        ['matplotlib==2.1.0'],
        ['mxnet==0.12.0'],
        ['numpy==1.13.3'],
        ['pyparsing==2.2.0'],
        ['python-dateutil==2.6.1'],
        ['pytz==2017.3'],
        ['requests==2.18.4'],
        ['six==1.11.0'],
        ['urllib3==1.22'],
        ['PyYAML==3.12'],
        ['Pillow==4.3.0'],
        ['imageio==2.2.0']
    ],
    ext_modules=[
        Extension(
            'ecGAN.gpuman',
            ['src/gpuman.c'],
            include_dirs=['/usr/local/cuda/include'],
            libraries=['nvidia-ml'],
            # WARNING driver branch can be other than 384!!
            library_dirs=['/usr/lib/nvidia-384']
        )
    ]
)
