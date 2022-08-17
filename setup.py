from setuptools import setup, find_packages

setup(
    name="DLcausality",
    version="1.0.0",
    author="Yuting Li",
    author_email="yutingli.olivia@gmail.com",
    description="This package is used to analyze tracks between two points.",
    packages=find_packages(),
    install_requires=['argparse', 'pytest'],
    entry_points={'console_scripts': ['cwp=DLcausality.commands.command_cwp:process_cwp',
                                      'twp=DLcausality.commands.command_twp:process_twp',
                                      'clm=DLcausality.commands.command_clm:process_clm',
                                      'tlm=DLcausality.commands.command_tlm:process_tlm'],
                  }
)
