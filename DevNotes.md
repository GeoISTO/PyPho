# Development notes for PyPho

Some of these notes are general comments and notions of Python development.
Others are specific to implementation choices of PyPho.
Nonetheless, it should serve developers.

## Packaging

The project is described in [pyproject.toml](./pyproject.toml) file.  
The project package is hosted on [PiPy/pypho](https://pypi.org/project/pypho/)

Usefull ressources that served as references for development:
* [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) 
* [RealPython tutorial](https://realpython.com/pypi-publish-python-package/#version-your-package) 

The following tools are used in PyPho:
* [pip](https://packaging.python.org/en/latest/key_projects/#pip) for managing installation
* [build](https://packaging.python.org/en/latest/key_projects/#build) as a front end for building
* [setuptools](https://setuptools.pypa.io/en/latest/index.html) as a backend for building the package
* [bumpver](https://pypi.org/project/bumpver/) for versionning incrementation
* [twine](https://packaging.python.org/en/latest/key_projects/#twine) for publishing to PiPy

Sequence/List of actions that can be done:
* Get the packaging tools (if needed): ```python -m pip install build twine bumpver```
* Install/Update PyPho locally: ```python -m pip install -e .``` (to be run in PyPho's root directory with the desired environment acitvated)
* Increment the version ```bumpver update --minor``` or ```bumpver update --major``` or ```bumpver update --patch```, to make a test, add the ```-v --dry``` flag
* Build the package: ```python -m build``` -> they should be added in [./dist](./dist)
* Check the created dist: ```twine check dist/*```
