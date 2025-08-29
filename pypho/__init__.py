"""Photogrammetric Design

This module proposes classes for designing a photogrammetric campaign.
There are definitions of **Camera**, **Lenses**, and **Pictures** (aka CameraShot),
as well as basic Objects to be modelled and visualisation tools
based on `pyvista <https://docs.pyvista.org/>`_.

Modules:

* :py:mod:`~pypho.camera`: defines camera registry, manipulation, and how to take pictures
* :py:mod:`~pypho.target`: defines different kinds of typical object to be digitised
* :py:mod:`~pypho.graphics`: to visualise configuration of the photography campaing

"""
__version__ = "0.0.4"