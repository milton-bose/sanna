#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
        'numpy>=1.10.0',
        'scipy',
        'theano>=0.7.0',
    # TODO: put package requirements here x
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='sanna',
    version='0.1.0',
    description="Simple Artificial Neural Network Architectures",
    long_description=readme + '\n\n' + history,
    author="Milton Bose",
    author_email='milton.bose@gmail.com',
    url='https://github.com/milton-bose/sanna',
    packages=[
        'sanna',
    ],
    package_dir={'sanna':
                 'sanna'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='sanna',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
