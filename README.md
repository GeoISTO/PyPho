# PyPho

This repository provides a Python package dedicated to photogrammetric design called **PyPho**.  
Notebooks for testing and using pypho online are provided in a twin repository: [PyPho_notebooks](https://github.com/GeoISTO/PyPho_notebooks)

## Development ##

Current version: 0.0.2  
Main repository for PyPho is at: [GitHub/PyPho](https://github.com/GeoISTO/PyPho)

### Dependencies ###

**PyPho** primarilly depends on the incredible [pyvista](https://pyvista.org/) package for the interactive scenes and 3D visualisation.  
**Pyvista** in turns relies on [trame](https://kitware.github.io/trame/guide/) and interfaces [vtk](https://vtk.org) for this.  
Note that **trame** comes with a main package, but requires companion packages to be installed as well to support the various backends.

Some computational aspects such as rotations are imported from [scipy](https://scipy.org)  
and dataframes manipulations rely on [pandas](https://pandas.pydata.org/).  
Note that [numpy](https://numpy.org/) is also used, but should be already installed as a dependence of pandas.

Please refer to the [pyproject.toml](./pyproject.toml) file for a complete list of requirements and versions.  
Check the ```project.dependencies``` variable.

### Environment and Install ###

It is recommended to create an environment to separate contexts when using python packages.  
You can either create a [venv](https://docs.python.org/3/library/venv.html) or use a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) env.

#### Installation
See below for creating your environment if needed: [Python Virtual Env](#python-virtual-env)

For installing **PyPho** simply run:
```
python -m pip install pypho
```

If you want additional dependencies installed, you can add ```[option]```.  
Here are the available options:
* for notebook support (recommended for pypho):
    ```
    python -m pip install pypho[nb]
    ```

#### Python Virtual Env

* Make sure the python version installed complies with the specifications in the [toml file](pyproject.toml)
* Go to where you'd like to create the env: ```cd where_you_have_your_envs```
* Create the environment: ```python -m venv .\pypho```
* Activate it by running the activate script ```.\pypho\Scripts\activate```
* then install pypho with pip: [see installation](#installation)

#### Conda Env

* Start your conda command line.
* Create and environment based on the desired python version: ```conda create -c conda-forge -n pypho python=3.11```
* Activate it: ```conda activate pypho```
* then install pypho with pip: [see installation](#installation)

See the conda documentation for a detailed view of environments: [conda envs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)  

