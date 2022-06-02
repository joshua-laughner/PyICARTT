from setuptools import setup

setup(
    name='ICARTT',
    version='0.1.0',
    packages=['icartt'],
    url='',
    license='',
    author='Joshua Laughner',
    author_email='jlaugh@caltech.edu',
    description='',
    entry_points={'console_scripts': [
            'icartt2nc=icartt.icartt2nc:main'
        ]}, 
)
