from setuptools import setup, Extension
module = Extension('libpython_fixer', sources=['libpython_fixer.c'])
setup(name='libpython_fixer', version='1.0', description='Fix libpython', ext_modules=[module])
