import os

from setuptools import setup

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='ottodiff',
    version='0.1',
    packages=['ottodiff'],
    description='Graph base autodifferentiation framework.',
    url='https://github.com/trevorhowarth16/otto_diff',
    author='Trevor Howarth',
    author_email='thowarth95@gmail.com',
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy >= 1.19',
        'scipy >= 1.5'
    ],
    python_requires=', '.join(
        [
            '>=3.0',
        ]
    ),
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
