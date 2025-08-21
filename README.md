# PyPho

This repository provides a Python package dedicated to photogrammetric design called **PyPho**.  
Notebooks for testing and using pypho online are provided in a twin repository: [PyPho_notebooks](https://github.com/GeoISTO/PyPho_notebooks)

## Development ##

Current version: 0.0.1  
Main repository for PyPho is at: [GitHub/PyPho](https://github.com/GeoISTO/PyPho)

### Dependencies ###

**PyPho** primarilly depends on the incredible [pyvista](https://pyvista.org/) package for the interactive scenes and 3D visualisation.  
**Pyvista** in turns relies on [trame](https://kitware.github.io/trame/guide/) and interfaces [vtk](https://vtk.org) for this.  
Note that **trame** comes with a main package, but requires companion packages to be installed as well to support the various backends.

Some computational aspects such as rotations are imported from [scipy](https://scipy.org)  
and dataframes manipulations rely on [pandas](https://pandas.pydata.org/).  
Note that [numpy](https://numpy.org/) is also used, but should be already installed as a dependence of pandas.

### Environment ###

Please refer to the [pyproject.toml](./pyproject.toml) file for a complete list of requirements and versions.  
Check the ```project.dependencies``` variable.

It is recommended to create an environment to separate contexts when using python packages.  
You can either create 

Environment for PyPho is described in the requirements.txt file and can be used as follows:
```pip install -r requirements.txt```


