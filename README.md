## Package "DLcausality"
***
This is a package called ***"DLcausality"*** designed for providing both linear and nonlinear methods of detecting Granger causality. The following sections will introduce the usage of this package.

### Dependencies
***
The required Python dependencies are these following package: ***"argparse"***, ***"pytest"***, ***"tensorflow"***, ***"statsmodels"***, ***"pandas"*** and ***"numpy"***. Users do not need to install them one by one as it has been defined in the file ***"setup.py"***. Users can directly install the package ***"DLcausality"***.

### Installation
***
Before starting to install the package, users should first download it from Github. Users should turn on the terminal under the directory where they would like to save the files and input the following command:

```bash
git clone https://github.com/oliviayt1224/Deep-Learning-Causality
```
After downloading the files to the local side, users can install the package into their local environments by using the codes below:
```bash
pip install .
```
### Command Introduction
***
There are four different commands implemented in this package: “cwp”, “clm”, “twp” and “tlm”, each of them refers to one of the synthetic data distribution described in the paper and will be introduced individually below:

- ### Command "cwp"
This command enables users to investigate the causality for coupled wiener processes. There are five parameters that users can specify their values, and users can find more information of the equations and definitions in the paper. A brief introduction of parameters is summarized in the table below:

| Parameter | Definition | Type |Range|Defalut Value|
| :-----:| :----: | :----: |:----: | :----: |
| T | time length | float |(0, +∞)|1|
| N | time steps | int |(0, +∞)|300|
| alpha | coefficient | float |[0,1]|0.5|
| lag | time lag | int |(0, +∞), smaller than N|5|
| num_exp | number of experiments | int |(0, +∞)|100|

Users can execute the command by following the syntax:
```bash
cwp --T <time length> --N <time steps> --alpha <coefficient> --lag <time lag> --num_exp <number of experiments>
```
More simply, users can use this command with all defaults values as inputs by doing:
```bash
cwp
```
- ### Command "clm"
This command enables users to investigate the causality for coupled logistic maps. There are five parameters that users can specify their values, and users can find more information of the equations and definitions in the paper. A brief introduction of parameters is summarized in the table below:

| Parameter | Definition | Type |Range|Defalut Value|
| :-----:| :----: | :----: |:----: | :----: |
| T | time length | float |(0, +∞)|1|
| N | time steps | int |(0, +∞)|1000|
| alpha | coefficient | float |[0,1]|0.4|
| epsilon | coefficient | float |[0,1]|0.9|
| num_exp | number of experiments | int |(0, +∞)|100|

Users can execute the command by following the syntax:
```bash
clm --T <time length> --N <time steps> --alpha <coefficient> --epsilon <coefficient> --num_exp <number of experiments>
```
More simply, users can use this command with all defaults values as inputs by doing:
```bash
clm
```
- ### Command "twp"
This command enables users to investigate the causality for ternary wiener processes. There are seven parameters that users can specify their values, and users can find more information of the equations and definitions in the paper. A brief introduction of parameters is summarized in the table below:

| Parameter | Definition | Type |Range|Defalut Value|
| :-----:| :----: | :----: |:----: | :----: |
| T | time length | float |(0, +∞)|1|
| N | time steps | int |(0, +∞)|300|
| alpha | coefficient | float |[0,1]|0.5|
| phi | coefficient | float |[0,1]|0.5|
| beta | coefficient | float |[0,1]|0.5|
| lag | time lag | int |(0, +∞), smaller than N|5|
| num_exp | number of experiments | int |(0, +∞)|100|

Users can execute the command by following the syntax:
```bash
twp --T <time length> --N <time steps> --alpha <coefficient> --phi <coefficient> --beta <coefficient> --lag <time lag> --num_exp <number of experiments>
```
More simply, users can use this command with all defaults values as inputs by doing:
```bash
twp
```

- ### Command "tlm"
This command enables users to investigate the causality for ternary logistic maps. There are five parameters that users can specify their values, and users can find more information of the equations and definitions in the paper. A brief introduction of parameters is summarized in the table below:

| Parameter | Definition | Type |Range|Defalut Value|
| :-----:| :----: | :----: |:----: | :----: |
| T | time length | float |(0, +∞)|1|
| N | time steps | int |(0, +∞)|700|
| alpha | coefficient | float |[0,1]|0.4|
| epsilon | coefficient | float |[0,1]|0.9|
| num_exp | number of experiments | int |(0, +∞)|100|

Users can execute the command by following the syntax:
```bash
tlm --T <time length> --N <time steps> --alpha <coefficient> --epsilon <coefficient> --num_exp <number of experiments>
```
More simply, users can use this command with all defaults values as inputs by doing:
```bash
tlm
```
### Unit Tests
***
There is a file called ***"test_TE.py"*** which includes multiple unit tests for functions in this package. Users can run through all the unit tests by simply entering the code below in the terminal.
```bash
pytest -v
```
