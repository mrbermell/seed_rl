from setuptools import setup, find_packages

setup(name='seed_rl',
      version="0.1-goteboy",
      include_package_data=True,
      install_requires=[
          'tensorflow',
          'pytest',
      ],
      packages=find_packages()
)
