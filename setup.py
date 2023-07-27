#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'flax>=0.4.1',
    'jax>=0.4.13',
    'jaxlib>=0.4.13',
    'matplotlib>=3.5.1',
    'numpy>=1.22.2',
    'optax>=0.1.1',
    'scipy>=1.8.0',
    'wandb>=0.12.11',
    'termcolor>=1.1.0',
    'distrax~=0.1.2',
    'argparse-dataclass>=0.2.1',
    'jaxutils',
    'chex',
    'brax>=0.9.1'
]

extras = {}
setup(
    name='bsm',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.9',
    include_package_data=True,
    install_requires=required,
    extras_require=extras
)
