from setuptools import setup, Extension

cpp_args = ['-std=c++11']

ext_modules = [
    Extension(
    'log_marginal_prob',
        ['bhclust/log_marginal_prob.cpp'],
        include_dirs=['pybind11/include', 'eigen'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(name='bhclust',
      version='1.0',
      description='Bayesian Hierarchical Clustering',
      author='Muxin Diao, Chenyang Wang',
      license='MIT',
      packages=['bhclust'],
	  ext_modules=ext_modules,
	  zip_safe=False)