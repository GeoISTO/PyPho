.. PyPho documentation master file, created by
   sphinx-quickstart on Wed Aug 27 17:15:58 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPho documentation
===================

**PyPho** is a python package for photogrammetric design.  
It is meant to provide interactive envrionment for visualising the effects of internal camera settings and layout on a pohotgrammetric realisation.

Photogrammetry is a great technique for reconstructing a 3D object from a series of pictures.
While available tools and modern photographic gear make this technique affordable and easy to implement,
it might appear difficult to new users to grasp the effect of the various camera settings on the expected results.
**PyPho** provides a way to interactively explore these effects in a 3D visualisation envrionment
that can be easily embedded into jupyter notebooks.

**PyPho** is also meant for more advanced users whow would be interested in testing and visualising their acquisition design in more complex environments.
In some instance, you might not have the opportunity to repeat the photography shootage
and might want to make sure to achieve the expected resolution and characteristics of the 3D result.
In such instance, it is best to compute and visualise everything beforehand, and **PyPho** is here to help.

.. note::

   This documentation is under development.

Graphical Abstract
____________________

.. pyvista-plot::
   :caption: This is PyPho's default scene.
   :include-source: False

   >>> import pypho
   >>> from pypho import graphics
   >>> obj, cam, viewer = pypho.graphics.get_default_scene(show= False)
   >>> viewer.set_active_scalars("resolution")
   >>> viewer.show()

Installation
___________________

**PyPho** can be simply installed from PyPI:

.. code-block:: console
   
    (pypho) $ pip install pypho

Main features
___________________

**PyPho** provides three main classes:

#. :class:`pypho.camera.Camera` 
#. :class:`pypho.target.TargetObject` 
#. :class:`pypho.graphics.Viewer3D` 

Table of Contents
___________________

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api