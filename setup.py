# pylint: disable=missing-docstring
from setuptools import setup
from setuptools import find_packages

setup(name='mltils',
      version='0.1',
      description='A package with utility code for Machine Learning practice',
      author='Rafael Ladeira',
      author_email='rwladeira@gmail.com',
      license='MIT',
      install_requires=[
          'tqdm', 'numpy', 'scipy', 'scikit-learn', 'pandas',
          'xgboost', 'statsmodels'
      ],
      packages=find_packages())
