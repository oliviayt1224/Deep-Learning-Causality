from setuptools import setup, find_packages

setup(
    name="dpc",
    version="1.0.0",
    author="Yuting Li",
    author_email="yutingli.olivia@gmail.com",
    description="This package is used to analyze tracks between two points.",
    packages=find_packages(),
    install_requires=['argparse', 'pytest'],
    entry_points={'console_scripts': ['cwp=Functions.command_cwp:process_cwp',
                                      'fun2=Functions.command:main'],
                  }
)
