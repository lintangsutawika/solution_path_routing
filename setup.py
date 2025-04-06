from setuptools import setup, find_packages

setup(
    name='routing',
    version='0.1.0',
    packages=find_packages(include=['routing', 'routing.*']),
    install_requires=[],  # List dependencies here, if any
)