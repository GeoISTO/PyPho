"""target.py

This module provides objects that represents the target of the photographies.
They heavily rely on pyvista for the geometrical definition of objects.

More complex examples of objects and textures are also provided:
- get_pebble_dataset: creates and return a dataset that represents 
a small pebble-shaped rock sample.
- get_rock_texture: returns a simple pyvista texture that somewhat looks
like rock. 
"""

import os
import numpy as np
import pyvista as pv
import scipy.spatial.transform as spt

#----------------------------------------------------------------------------
# Target objects
class TargetObject(pv.PolyData):
    """A base class for the different kinds of targets
    
    This is basically a helper class that interfaces a Pyvista object.
    This object inherits from pyvista.PolyData
    """
    target_kinds = [
        "Sample",
        "Outdoor",
        "Indoor"
    ]
    
    def __hash__(self):
        """making Target objects hashable"""
        return id(self)
    
    def __init__(self,
                 pv_object:pv.PolyData,
                 translation = None,
                 rotation = None,
                 deep_copy = False,
                 kind = "Sample"
                 ):
        """Constructor of a TargetObject directly from a pyvista PolyData
        
        The given object will be triangulated if it is not all triangles.
        
        Parameters:
        - pv_object: a Pyvista PolyData that represents the target object
        - deep_copy: if False (default) this object will just be a reference to the given one,
        if True, an actual duplicate will be made.
        - translation: a translation to apply to the object to place its center to a given location
        This is given as a list of 3 coordinates [dx,dy,dz]. If None (default), the object is left as is.
        - rotation: a rotation to be applied to the object as defined as around its center.
        This is given as a list of up to 3 rotation angles (as defined by a scipy.Rotation from euler angles in ZYX order),
        if a single scalar values is given it will be used a a rotation around Z.
        - kind: specifies which kind of object this is among:
         * Sample (default): a smaller 3D object viewed from all around
         * Outdoor: an outcrop or object typically viewed inplace 
         and from only some of its faces (eg., from face or top)
         * Indoor: an outcrop or object viewed from its inside,
         typically a room or a cave.
        """
        assert(pv_object is not None), "A pyvista.PolyData object must be given to initialise this."
        super().__init__(pv_object, deep= deep_copy)
        
        # the object should be all triangle, else triangulate it
        if not self.is_all_triangles:
            self.triangulate(inplace= True)
            
        # apply rotation and translation
        if rotation is not None:
            self.rotate(rotation)
        if translation is not None:
            self.translate(translation)
            
        # save the kind
        self.kind = kind
    
    def translate(self, translation, return_object= False):
        """Applies a translation to the object
        
        This is mainly to bypass the inplace argument of pyvista.
        This method is applied inplace whereas default pyvista method is not."""
        super().translate(translation, inplace= True)
        if return_object: return self
        
    def rotate(self, rotation_angles, return_object= False):
        """Apply a rotation to the object
        
        Parameters:
        - rotation: a list of three angles in degrees around Z Y and X axes"""
        rotation = np.array([rotation_angles]).ravel()
        rot = spt.Rotation.from_euler("ZYX"[:len(rotation)],  
                        rotation,
                        degrees= True
                    )
        self.points = rot.apply(self.points - self.center) + self.center
        self.compute_normals(inplace=True)
        if return_object: return self
        
        
    def bounding_box(self):
        """returns a bounding box object"""
        min = self.points.min(axis=0)
        max = self.points.max(axis=0)
        bounds = np.concatenate([[min_i, max_i] for min_i, max_i in zip(min,max)])
        return pv.Box(bounds, quads=False)
            
class Sphere(TargetObject):
    """A spherical target object"""
    
    def __init__(self, radius= 1.0,
                 translation = None,
                 rotation = None,
                 nsub= 4):
        """Builds the sphere target object
        
        Parameters:
        - translation: a translation to be applied to move the sample.
        By default (if None) the sphere is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject), by default (if None)
        the cylinder is set upright with a direction of [0,0,1]
        - radius: the radius of the sphere in meters, the center is initialized at [0,0,radius]
        """
        pv_object = pv.Icosphere(center = [0,0,radius], radius = radius, nsub= nsub)
        pv_object.texture_map_to_sphere(prevent_seam=False, inplace=True)
        super().__init__(pv_object = pv_object,
                         translation= translation,
                         rotation = rotation,
                         kind= "Sample")
        
class DrillCore(TargetObject):
    """A target object that represents a drill core
    
    This is a cylindrical shape typical of drill cores from field drilling campaing
    or sampling from rocks"""
    
    def __init__(self, 
                 radius = 0.0525,
                 translation = None,
                 rotation = None,
                 height = 0.2,
                 nsub= 2
                 ):
        """Creates a drill core sample
        
        Parameters:
        - translation: a translation to be applied to move the sample.
        By default (if None) the drill core is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject), by default (if None)
        the cylinder is set upright with a direction of [0,0,1]"""
        core = pv.Cylinder( 
                   center = [0,0, height/2],
                   direction= [0,0,1],
                   radius = radius,
                   height = height,
                   capping= True
                )
        core = core.triangulate().subdivide(nsub, "loop").texture_map_to_sphere()
        
        core.compute_normals(split_vertices=True, inplace=True)
        super().__init__(pv_object= core,
                         translation= translation,
                         rotation = rotation,
                         kind= "Sample")
        
class CubeSample(TargetObject):
    """A target object which shape is a parallelepipedon"""
    
    def __init__(self, 
                 side_lengths = [0.25, 0.3, 0.1],
                 translation = None,
                 rotation = None,
                 nsub= 3
                 ):
        """Creates a Cuboid rock sample
        
        Parameters:
        - side_lengths: the lengths of the sides (XYZ) in meters
        - translation: a translation to be applied to move the sample.
        By default (if None) the sample is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject)"""
        sample = pv.Cube(
            center = [0.,0., side_lengths[2]/2],
            x_length= side_lengths[0],
            y_length= side_lengths[1],
            z_length= side_lengths[2],
            clean= False
        ).triangulate().subdivide(nsub, "linear")
        #sample.compute_normals(split_vertices=True, inplace=True)
        
        super().__init__(pv_object= sample,
                         translation= translation,
                         rotation = rotation,
                         kind= "Sample")

class Outcrop(TargetObject):
    """A TargetObject that represents an outcrop"""
    
    def __init__(self,
                 width = 3,
                 height = 2,
                 translation = None,
                 rotation = None,
                 nsub= 5
                 ):
        """Creates a planar outcrop
        
        Parameters:
        - width: the width of the outcrop in meters
        - height: the height of the outcrop in meters
        - translation: a translation to be applied to move the sample.
        By default (if None) the sample is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject),
        by default, the outcrop is facing West (to -X)"""
        outcrop = pv.Rectangle([
            [0, +width/2, 0],
            [0, -width/2, 0],
            [0, -width/2, height]
        ]).triangulate().subdivide(nsub, "linear")
        outcrop.texture_map_to_plane(inplace=True)
        super().__init__(pv_object= outcrop,
                         translation= translation,
                         rotation = rotation,
                         kind= "Outdoor")

class Ground(TargetObject):
    """A TargetObject that represents an ground surface"""
    
    def __init__(self,
                 side_lengths = [30,10],
                 translation = None,
                 rotation = None,
                 nsub= 5
                 ):
        """Creates a planar ground surface
        
        Parameters:
        - side_lengths: a list of the two lengths in the initial X and Y directions in meters
        - translation: a translation to be applied to move the sample.
        By default (if None) the sample is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject),
        by default, the surface is facing up (to Z)"""
        x,y = 0.5*np.array(side_lengths)
        surface = pv.Rectangle([
            [-x, -y, 0],
            [+x, -y, 0],
            [+x, +y, 0]
        ]).triangulate().subdivide(nsub, "linear")
        surface.texture_map_to_plane(inplace=True)
        super().__init__(pv_object= surface,
                         translation= translation,
                         rotation = rotation,
                         kind= "Outdoor")
        
class Room(TargetObject):
    """A TargetObject to represent a cubic room"""
    
    def __init__(self, 
                 side_lengths = [12,10,2],
                 translation = None,
                 rotation = None,
                 nsub= 5
                 ):
        """Creates a Cuboid Room
        
        Parameters:
        - side_lengths: the lengths of the sides (XYZ) in meters
        - translation: a translation to be applied to move the sample.
        By default (if None) the sample is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject)"""
        room = pv.Cube(
            center = [0.,0., side_lengths[2]/2],
            x_length= side_lengths[0],
            y_length= side_lengths[1],
            z_length= side_lengths[2],
            clean= False
        ).triangulate().subdivide(nsub, "linear")
        room.compute_normals(split_vertices= True,
                             flip_normals=True,
                             inplace=True)
        super().__init__(pv_object= room,
                         translation= translation,
                         rotation = rotation,
                         kind= "Indoor")
        
class Well(TargetObject):
    """A TargetObject to represent a well section"""
    
    def __init__(self, 
                 radius= 3,
                 height = 1.35,
                 translation = None,
                 rotation = None,
                 nsub= 3
                 ):
        """Creates a Cuboid Room
        
        Parameters:
        - radius: the radius of the cylinder in meters
        - height: the height of the cylinder in meters
        - translation: a translation to be applied to move the sample.
        By default (if None) the sample is created so that its base is set on the XY plane
        at Z = 0, centered on Y=0, X=0
        - rotation: rotation angles in degrees (cf. TargetObject)"""
        well = pv.Cylinder( 
                   center = [0,0,height/2],
                   direction= [0,0,1],
                   radius = radius,
                   height = height,
                   capping= False
        ).triangulate().subdivide(nsub)
        well.compute_normals(split_vertices= True,
                             flip_normals=True,
                             inplace=True)
        super().__init__(pv_object= well,
                         translation= translation,
                         rotation = rotation,
                         kind= "Indoor")


#-----------------------------------------------------------------------------
# examples
def get_pebble_dataset(s= 1.0, nsub= 3):
    """Creates and access the Pebble dataset.
    
    This dataset represents a small pebble-shaped rock sample.
    
    Parameters:
    - s: dimensions are [s, 0.8*s, 0.35*s]
    - nsub: number of subdivisions to be applied to refine the object
    
    Returns:
    - a Pyvista.PolyData with initialized texture mapping.
    See get_rock_texture for loading the texture.
    """
    cube = pv.Cube(
                                center = [0, 0, 0],
                                x_length = 1 * s,
                                y_length = 0.8 * s,
                                z_length = 0.35 * s
                            ).triangulate().subdivide(2, "loop")
    cube = cube.subdivide(nsub)
    cube.compute_normals(inplace= True)
    
    z = cube.points[:,2]
    h = z.max() - z.min()
    target_object = TargetObject(
                            cube,
                            translation = [0,0, h/2],
                            rotation = [0,0]
                        )
    return target_object 

def get_rock_texture():
    """Access an example rock texture
    
    This is a Pyvista.Texture that was created from original pictures
    from O-ZNS observatory Beauce Limestones."""
    base_folder = os.path.dirname(__file__)
    file = os.path.join(base_folder, "rock_texture.vti")
    return pv.read_texture(file)
