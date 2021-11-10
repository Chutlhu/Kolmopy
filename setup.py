from setuptools import setup, find_packages

USERNAME = 'chutlhu'
NAME = 'kolmopy'

with open('requirements.txt') as f:
    REQUIREMENTS = f.readlines()

setup(
    name='kolmopy',
    version='0.1.0',
    description='A library for Turbulence data analysis and visualization',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Diego Di Carlo',
    author_email='diego.dicarlo89@gmail.com',
    url='https://github.com/{}/{}'.format(USERNAME, NAME),
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    extras_require={},
    keywords=['turbulence', 'python', 'kolmogorov', 'analysis'],
    license='CC BY-NC-SA 4.0',
)