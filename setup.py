  
from setuptools import setup,find_packages
import sys, os

setup(name="FlowGMM",
      description="Flow Gaussian Mixture Model",
      version='0.1',
      author='Pavel Izmailov, Polina Kirichenko, Marc Finzi',
      author_email='pi390@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml'],
      packages=find_packages(),
      long_description=open('README.md').read(),
)