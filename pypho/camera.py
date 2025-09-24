"""camera.py

Defines Camera and shots manipulations.

This is the main module of the photogrammetric design.
The typical use case would be to
- (optional) define and add the parameters of you camera and lens in the
registry. Note that this is required if it is not already known.
- get a Camera object from the registry (CameraRegistry.get_camera)
- set the configuration of the Camera for both its internal parameters
(e.g., focal length, aperture) and external parameters (location, orientation)
- take a (camera) shot to create a picture

CameraShot (ie., pictures) and Camera both derive from CameraBase, 
which centralises base definitions, such as geometrical aspects.
In essence, the difference between a Camera and a CameraSHot is that
the parameters of a Camera can be modified via different update functions,
while the CameraShot parameters are fixed and defined by the configuration of
the Camera taking the shot when it was taken.

CameraBase provides geometrical objects that can help visualise the Camera and
CameraShot. Those are only computed on demand and will be automatically 
updated when modifying the camera parameters.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import scipy.optimize as optim
from scipy import special
import pyvista as pv

from difflib import get_close_matches

from .target import TargetObject, Ground, Outcrop
    
#-----------------------------------------------------------
# Camera Registry

class CameraRegistry:
    """Stores a registry of Cameras.
    
    This class is a singleton, as there could be only one registry.
    The different cameras definitions are stored in camera_base DataFrame.
    Associated lenses are stored in lense_base DataFrame.
    Cameras are referenced by a unique id (uid) and a human-readable name.

    Methods:
    - CameraRegistry.register_camera: to register a new camera
    - CameraRegistry.register_lens: to register a new lens 
    (this can be avoided and simple parameters of the lens can be passed when needed)
    - CameraRegistry.camera_base: stores and gives access to the camera definitions
    - CameraRegistry.lens_base: stores and gives access to the lense definitions
    - CameraRegistry.get_camera: to create an instance of a given camera (and lens)

    The information stored are:
    - uid: a unique identifier
    - name: a human-readable name
    - sensor_width: the width of the sensor in mm
    - sensor_height: the height of the sensor in mm
    - nb_pixel_width: the number of pixels in width direction
    - nb_pixel_height: the number of pixels in height direction
    Other information are computed:
    - pixel_pitch: the pitch (i.e., distance from one pixel center to the next)
    of the pixels on the sensor in mm
    - nb_pixel: the definition of the sensor, i.e., the number of pixels
    - MPix: the definition in mega pixels
    A new instance of Camera can then be created by the getter get_camera()
    """
    camera_base = pd.DataFrame(
        {"name": pd.Series([], dtype= str),
         "sensor_width": pd.Series([], dtype= str),
         "sensor_height": pd.Series([], dtype= str),
         "nb_pixel_width": pd.Series([], dtype= int), 
         "nb_pixel_height": pd.Series([], dtype= int),
         "pixel_pitch": pd.Series([], dtype= float),
         "nb_pixel": pd.Series([], dtype= int),
         "MPix": pd.Series([], dtype= float)
        },
        index = pd.CategoricalIndex([],name="uid")
    )
    
    lens_base = pd.DataFrame(
        {"name": pd.Series([], dtype= str),
         "focal_min": pd.Series([], dtype= float),
         "focal_max": pd.Series([], dtype= float),
         "focal_fixed": pd.Series([], dtype= bool),
         "min_focus_distance": pd.Series([], dtype= float)
        },
        index = pd.CategoricalIndex([],name="uid")
    )
        
    def __new__(cls):
        """Method to access (and create if needed) the only allowed instance of this class.
        
        Returns:
        - an instance of CameraRegistry"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(CameraRegistry, cls).__new__(cls)
        return cls.instance
    
    def __init__(cls):
        """Generates and access the camera registry"""
        print("Init registry")
        print(CameraRegistry.camera_base)
        
    @classmethod
    def register_camera(cls, uid, name,
                   sensor_width, sensor_height,
                   nb_pixel_width, nb_pixel_height):
        """ Adds a new camera to the registry

        Parameters:
        - uid: a unique identifier
        - name: a human-readable name
        - sensor_width: the width of the sensor in mm
        - sensor_height: the height of the sensor in mm
        - nb_pixel_width: the number of pixels in width direction
        - nb_pixel_height: the number of pixels in height direction
        """
        cls.camera_base.loc[uid] = {
            "name":name, 
            "sensor_width":sensor_width, "sensor_height":sensor_height,
            "nb_pixel_width":nb_pixel_width, "nb_pixel_height":nb_pixel_height,
            "pixel_pitch": sensor_width/nb_pixel_width,
            "nb_pixel": nb_pixel_width * nb_pixel_height,
            "MPix": round(nb_pixel_width * nb_pixel_height / 1e6, 1)
        }
    
    @classmethod
    def register_lens(cls, uid, name,
                   min_focus_distance,
                   focal_min= None, focal_max= None,
                   focal= None):
        """ Adds a new lens to the registry

        Parameters:
        - uid: a unique identifier
        - name: a human-readable name
        - focal: the focal in mm, if fixed focal, else, use focal_min focal_max
        - focal_min: the minimum focal in mm, must be None if focal is given, and
        must be given if focal_max is not None
        - focal_max: the maximum focal in mm, must be None if focal is given, and
        must be given if focal_min is not None
        - min_focus_distance: the minimal distance of focus in m
        """
        if focal is None:
            assert focal_min is not None and focal_max is not None, "give either focal or focal_min and focal_max"
        else:
            assert focal_min is None and focal_max is None, "give either focal or focal_min and focal_max"
            
        cls.lens_base.loc[uid] = {
            "name":name,
            "focal_min": focal if focal is not None else focal_min,
            "focal_max": focal if focal is not None else focal_max,
            "focal_fixed": focal is not None,
            "min_focus_distance":min_focus_distance
        }
        
    @classmethod
    def get_camera(cls, cam_id, lens_id= None, **kargs):
        """Creates a Camera after the characteristics form the database
        
        Parameters:
        - cam_id: the unique id of the selected camera
        - lens_id: the unique id of the selected lens if set, None otherwise
        - kargs: optional arguments, see Camera
        
        Returns:
        - a Camera object if uid is in the database, None otherwise
        """
        if cam_id not in cls.camera_base.index:
            close_matches = get_close_matches(cam_id, CameraRegistry.camera_base.index)
            assert(cam_id in cls.camera_base.index), f"the cam id ({cam_id:}) can't be"+\
                  "found in the registry. "+\
                  "Possible candidates are:\n"+",".join(close_matches) if len(close_matches)>0 else ""
        if (lens_id is not None) and (lens_id not in cls.lens_base.index):
            close_matches = get_close_matches(lens_id, CameraRegistry.lens_base.index)
            assert(lens_id in cls.lens_base.index), f"the lens id ({lens_id:}) can't be"+\
                  "found in the registry. "+\
                  "Possible candidates are:\n"+",".join(close_matches) if len(close_matches)>0 else ""
        if lens_id is None:
            assert "min_focus_distance" in kargs and (
                "focal" in kargs or (
                    "focal_min" in kargs and "focal_max" in kargs)
                ), "make sure the focal and min_focus_distance are specified."
            lens_id = None
        return Camera(cam_id, lens_id, **kargs)

# default cameras
# after https://www.dxomark.com/Cameras/Sony/A7RIV---Specifications
CameraRegistry.register_camera(
    uid="A7RIV", name= "Sony Alpha 7RIV",
    sensor_width= 35.7, sensor_height= 23.8,
    nb_pixel_width= 9600, nb_pixel_height= 6376
)

CameraRegistry.register_lens(
    uid= "FE35",
    name= "Sony FE 35 mm f/1.4 Zeiss Distagon T*",
    focal= 35,
    min_focus_distance= 0.3 
)

def get_default_camera():
    """Returns the first camera+lens in the database"""
    cam = CameraRegistry.get_camera(
                                cam_id= CameraRegistry.camera_base.index[0],
                                lens_id= CameraRegistry.lens_base.index[0]
                            )
    return cam

#----------------------------------------------
class CameraBase(object):
    """Defines base behaviour for Camera and shots.
    
    A Camera and CameraShot have a lot in common because all the 
    camera information and orientation parameters have to be recorded 
    when the shot is taken, so these parameters and corresponding methods
    are supported by this base class.
    
    This class holds methods for creating and getting representations of the camera or shot.
    Representation objects will be created if not yet instanciated,
    although they can only be updated by a proper Camera and not a CameraShot.
    """
    
    def __init__(self):
        """Initialises a CameraBase"""

    def _compute_confusion_circle(self):
        """Computes the confusion circle diameter
        
        - confusion_circle: the  maximal diameter of a disk on the sensor that would
        appear as single point, i.e., considered as focussed.
                
        Depends on the confusion_method: either metairie (1.5*pixel_pitch, after christophe metairie [http://www.cmp-color.fr/pdc.html])
        or bouillot (2*pixel_pitch, after Bouillot, p 289)
        
        Returns: confusion circle diameter
        """
        confusion_factor = 1.5 if self.confusion_method == "metairie" else 2.0
        return confusion_factor * self.pixel_pitch
    
    @classmethod
    def get_aperture_number_list(cls, max_N= 25, step = 1, rounding= 1):
        """Returns a list of aperture numbers N as f/N corresponding to aperture stops
        
        :param max_N: the maximum aperture number to get (included), default is 25
        :type max_N: float, optional
        :param step: step between the stops, by default is full stops (ie. 1), but 1/2 or 1/3 can be useful
        :type step: float, optional
        :returns: list of aperture stops
        """
        max_N += 0.1 # to include last
        n_max = np.log(max_N)/np.log(np.sqrt(2))+1
        return [round(float(np.sqrt(2)**(i-1)),rounding) for i in np.arange(1, n_max, step)]
    
    @classmethod
    def _compute_magnification_from_camera_parameters(cls, f, Z):
        """Computes the magnification

        :param f: the focal in mm
        :type f: float
        :param Z: the focus distance in m
        :type Z: float
        :return: f/Z
        :rtype: float
        """        
        return f / Z / 1000
        
    @classmethod
    def _compute_image_dimension_from_camera_parameters(cls, magnification, width_or_height):
        """Computes the image width or height

        :param magnification: magniication factor f/Z, dimensionless
        :type magnification: float
        :param width_or_height: sensor dimension in mm
        :type width_or_height: float
        :return: w / m, in meters
        :rtype: float
        """        
        return width_or_height / magnification / 1000
    
    def compute_image_size(self, Z, direction= "width"):
        """Computes the image size when focusing at a given distance Z
        
        :returns: the image width in meters if direction is "width" else the image height
        """
        m = Camera._compute_magnification_from_camera_parameters(self.focal, Z)
        if direction == "width":
            return self._compute_image_dimension_from_camera_parameters(m, self.sensor_width) 
        else:
            return self._compute_image_dimension_from_camera_parameters(m, self.sensor_height) 
            
    
    @classmethod
    def _compute_sampling_distance_from_camera_parameters(cls, z, focal, sensor_pixel_pitch):
        """Abstract method to compute Sampling Distance
        
        This is a generic function that can be called withoud creating a camera
        This computation only depends on the focal, distance (z along the
        view direction) and the sensor pixel_pitch 
        
        Parameters:
        - z: distance along the view direction in meters
        - focal: the focal in millimeters
        - sensor_pixel_pitch: the size of sensor's pixels in mm
        
        Returns:
        - a numpy.ma.array with the ground sampling distance (ie., image pixel size projected in the real life) in millimeters
        """
        return np.ma.array(
            1000 * z / focal * sensor_pixel_pitch,
            mask= z < 0,
            fill_value= np.nan
        )
    
    @classmethod
    def _compute_resolution_from_camera_parameters(cls, z, focal, sensor_pixel_pitch):
        """Abstract method to compute pixel resolution in image
        
        This doesn't depend on a camera definition as it is abstract.
        This computation only depends on the focal, distance (z along the
        view direction) and the sensor pixel_pitch 
        
        Parameters:
        - z: distance along the view direction in meters
        - focal: the focal in millimeters
        - sensor_pixel_pitch: the size of sensor's pixels in mm
        
        Returns:
        - the image pixel resolution pixels per millimeter
        """
        return 1/cls._compute_sampling_distance_from_camera_parameters(z, focal, sensor_pixel_pitch)
    
    def _compute_sampling_distance(self, z):
        """Computes the Sampling Distance at a given  focus distance
        
        Parameters:
        - z: the distance to the object along the view direction in meters
        (see self.compute_camera_coordinates)
        
        Returns:
        - ground sampling distance in millimeters
        """
        return CameraBase._compute_sampling_distance_from_camera_parameters(z, self.focal, self.pixel_pitch)
    
    def _compute_resolution(self, z):
        """Computes the image pixel resolution in image
        
        Parameters:
        - z: the distance to the object along the view direction in meters
        (see self.compute_camera_coordinates)
        
        Returns:
        - the image pixel resolution pixels per millimeter
        """
        return CameraBase._compute_resolution_from_camera_parameters(z, self.focal, self.pixel_pitch)
        
    def _compute_z_from_resolution(self, resolution):
        """Computes the target distance required for having a given resolution.
        
        :param resolution: in pixels per millimeter
        :type resolution: float
        :returns: the desired distance in meters 
        """
        return self.focal / resolution / self.pixel_pitch / 1000
        
    @classmethod
    def _compute_hyperfocal_from_camera_parameters(cls,
            focal:float, N:float, confusion_circle_diameter:float
        ):
        """Computes the hyperfocal distance from given parameters

        This is by definition the distance at which the focus should be set 
        to obtain the deepest sharpness zone, i.e., the smallest distance to focus on
        to obtain the far focus plan at infinity. This parameter only depends on 
        the focal, the aperture, and a confusion size. The computation is based
        on the size of the confusion circle (the largest diameter
        of a circle on the sensor that would be considered as a focussed point and 
        not a blur).
        :param focal: the focal distance in millimters
        :type focal: float
        :param N: the aperture number, as focal/N
        :type N: float
        :param confusion_circle_diameter: the diameter of the confusion circle in millimeters
        :type confusion_circle_diameter: float
        :returns: the hyperfocal distance in meters
        """
        return (focal**2 / N / confusion_circle_diameter + focal) / 1000 
    
    def _compute_hyperfocal(self):
        """Computes and sets the hyperfocal distance"""
        self.H = CameraBase._compute_hyperfocal_from_camera_parameters(
            focal= self.focal, N= self.N, confusion_circle_diameter= self.confusion_circle_diameter
        )
        return self.H
    
    def _compute_sharpness_zone(self, Z= None, N= None):
        """Computes the sharpness zone
        
        The sharpness zone is the zone where the blur due to defocus is limited enough to be smaller than the
        confusion circle.
        Note: this is not taking the diffraction into account
        
        if N is given, H is computed else the self.N is used
        if Z is not given, self.Z is used
        :returns: the distance in meters to the first and last sharp plans
        """
        Z = Z if Z is not None else self.Z
        if N is not None:
            H = CameraBase._compute_hyperfocal_from_camera_parameters(
                focal= self.focal,
                N= N,
                confusion_circle_diameter= self.confusion_circle_diameter
            )
        else: 
            H = self.H
        Z_f = Z - self.focal/1000
        H_f = H - self.focal/1000
        z1 = Z * H_f / (H_f + Z_f)
        z2 = Z * H_f / (H_f- Z_f) if Z_f < H_f else np.inf
        return z1, z2
    
    def _compute_depth_of_field(self):
        """Computes the depth of field
        
        The depth of field is the dimension of the sharpness zone along the camera view axis.
        Note: this is not taking the diffraction into account
        
        :returns: the depth of field in meters
        """
        return self.Z2 - self.Z1
    
    def _compute_depth_of_image(self):
        """Computes the image spread in micrometers
        
        This is the spread of the sharpness zone on the image side after Conrad 2006
        """
        if self.Z2 is np.inf:
            return np.inf
        z1_mm = self.Z1 * 1000
        z2_mm = self.Z2 * 1000
        self.depth_of_image = (z2_mm - z1_mm) * self.focal**2 / (z2_mm - self.focal) / (z1_mm - self.focal)
        return self.depth_of_image * 1000
       
    def _compute_defocus_diameter(self, zx, Z= None, N= None):
        """Computes the diameter of the image disk due to defocus

        :param zx: the distance of the object in meters
        :type zx: float
        :param N: the aperture to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera aperture is used. 
        :type N: float, optional
        :param Z: the focus distance to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera Z is used. 
        :type Z: float, optional
        :returns: a numpy.ma.array with the defocus diameter in micrometers
        """
        zx_mm = zx * 1000 
        z_mm = 1000 * (self.Z if Z is None else Z)
        N = self.N if N is None else N
        defocus_diameter = np.ma.array(
            self.focal**2 / N * abs( z_mm - zx_mm) / zx_mm / (z_mm - self.focal),
            mask= zx_mm <= 0,
            fill_value= np.nan
        )
        return defocus_diameter*1000
    
    def _compute_diffraction_x(self, q, zx):
        """Compute the x parameter in diffraction equations"""
        f = 1000 * self.focal
        m = np.ma.array(
            f / zx / 1e6,
            mask= zx <= 0,
            fill_value= np.nan
        )
        k = 2 * np.pi / self.wavelength_nm * 1000
        a = f / 2 / self.N
        q_corrected = q / (1 + m)
        sin_th = q_corrected / np.ma.sqrt(q_corrected**2 + f**2)
        return k*a*sin_th
        
    def _compute_diffraction_intensity(self, q, zx):
        """Compute the intensity of the diffraction pattern normalised by its intensity at the center

        :param q: distance to the center of the pattern in micrometers
        :type q: float
        :param zx: object distance in meters
        :type zx: float
        :return: the relative intensity
        :rtype: float
        """    
        x = self._compute_diffraction_x(q, zx)
        return (2 * special.jv(1,x)/x)**2
        
    def _compute_diffraction_intensity_within_distance(self, q, zx):
        x = self._compute_diffraction_x(q, zx)
        return (1 - special.jv(0, x)**2 - special.jv(1,x)**2)
        
    def _compute_diffraction_diameter(self, zx, N= None, wavelength_nm= None):
        """Computes the Airy diameter of diffraction partern
        
        :param zx: the distance of the object in meters
        :type zx: float
        :param N: the aperture to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera aperture is used. 
        :type N: float, optional
        :returns: the diameter of the first black circle in the diffraction patern in micrometers
        """
        zx_mm = zx * 1000
        wavelength_nm = wavelength_nm if wavelength_nm is not None else self.wavelength_nm
        lambda_mm = wavelength_nm / 1e6
        N = self.N if N is None else N
        return np.ma.array(
            2.44 * 1000 * lambda_mm * N * (1 + self.focal / zx_mm),
            mask= zx <= 0,
            fill_value= np.nan
        )
    
    def _compute_combined_blur_spot_diameter(self, zx, Z= None, N= None, wavelength_nm= None, effective_disk_ratio= None):
        """Computes the diameter of the blur spot as a result of both defocus and diffraction
        
        Both effects are combined as sqrt(d1**2 + d2**2) as suggested by Hansma 1996 and Conrad 2006.
        
        :param zx: the distance of the object in meters
        :type zx: float
        :param Z: the focus distance to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera Z is used. 
        :type Z: float, optional
        :param N: the aperture to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera aperture is used. 
        :type N: float, optional
        :type float:
        :returns: the diameter of the first black circle in the diffraction patern in micrometers
        
        """
        defocus_diameter = self._compute_defocus_diameter(zx= zx, Z=Z, N= N)
        effective_disk_ratio = effective_disk_ratio if effective_disk_ratio is not None else self.effective_diffraction_disk_ratio
        diffraction_diameter = effective_disk_ratio * self._compute_diffraction_diameter(zx= zx, N= N, wavelength_nm= wavelength_nm)
        sum_sqr = defocus_diameter**2 + diffraction_diameter**2
        return np.ma.sqrt(sum_sqr)

    def _compute_maximum_sharp_aperture(self, Z= None):
        """Computes the maximum aperture number before diffraction removes the entire sharpness zone
        
        Be carefull, not to be confused with the aperture giving the maximum sharp zone.
        :param Z: the focus distance to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera Z is used. 
        :type Z: float, optional
        :param wavelength_nm: the wavelength of the light in nanometers (default to 546 nm, yellow-green light)
        :type float:
        :param effective_disk_ratio: if 1, this returns the diameter of the Airy disk,
        i.e., the diameter of the first dark ridge in the diffraction partern. In practice,
        the light intensity becomes very low even before reaching this distance, that's why we suggest using only 
        a fraction of this diameter. Default is 0.6 as it concentates more than 84% of the light.
        :type float:
        """
        lambda_mm = self.wavelength_nm / 1e6
        z_mm = 1000 *(self.Z if Z is None else Z)
        return self.confusion_circle_diameter / 2.44 / lambda_mm / (1 + self.focal/z_mm) / self.effective_diffraction_disk_ratio
        
    def _compute_sharpness_zone_including_diffraction(self, Z= None, N= None):
        """Computes the sharpness zone while taking diffraction into account
        
        :param Z: the focus distance to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera Z is used. 
        :type Z: float, optional
        :param N: the aperture to use in the computation. This is optional and only useful for computing
        blur spot without affecting the camera settings. By default (when None) the camera aperture is used. 
        :type N: float, optional
        :returns: a tuple with the distances to first and last sharp plan in meters.
        Note that if the aperture number is above the maximum sharp aperture, then the diffraction becomes visible
        even on the focus plan and there is no sharpness zone. (None, None) is returned then.
        """
        max_N = self._compute_maximum_sharp_aperture(Z=Z)
        N = self.N if N is None else N
        if N >= max_N:
            return None, None
        optim_function = lambda zx: (1000*self.confusion_circle_diameter - self._compute_combined_blur_spot_diameter(zx, Z=Z, N=N))**2
        
        Z = self.Z if Z is None else Z
        Z1, Z2 = self._compute_sharpness_zone(Z=Z, N= N)
        z1_combined = optim.minimize_scalar(optim_function, bounds= (Z1, Z)).x
        z2_combined = optim.minimize_scalar(optim_function, bounds= (Z, Z2 if Z2<np.inf else 1e6)).x
        return z1_combined, z2_combined

    def _compute_depth_of_field_with_diffraction(self, Z= None, N= None):
        """Computes the depth of field while taking diffraction into account
        """
        Z1d, Z2d = self._compute_sharpness_zone_including_diffraction(Z= Z, N= N) if (N is not None) or (Z is not None) else (self.Z1d, self.Z2d)
        return Z2d - Z1d if Z1d is not None else 0

    def compute_optimal_aperture(self, Z= None, max_N= 25, step= 1/3, rounding= 2):
        """Computes the optimal aperture considering diffraction effect
        
        :param use_closest_stop: if False, the exact optimal value is used, if True it is
        "rounded" to the biggest f_stop below the optimal value
        :type use_closest_stop: bool, default True
        :returns: the optimal aperture number N
        """
        
        N_list = self.get_aperture_number_list(max_N= max_N, step= step, rounding= rounding)
        max_i = np.argmax([self._compute_depth_of_field_with_diffraction(Z=Z, N=N_i) for N_i in N_list])
        return N_list[max_i]
    
    def optimize_sharpness(self, 
                           target_points= None,
                            mode= "middle",
                            use_diffraction= False,
                            optimize_location= False,
                            optimize_N= False,
                            inside_weight= 0.2,
                            outside_weight= 10,
                            shift_weight= 0.1,
                            Z_weight= 0.05,
                            shift_lim_vs_Z= 2,
                            optim_method= optim.shgo,
                            update= True,
                            return_result= True
                        ):
        """Optimizes the focus distance and location to have the target_points as sharp as possible
        
        :param mode: is either front back or middle
        -front: the sharpness zone is set to the front of the point set
        -back: the sharpness zone is set to the back of the point set
        -middle (or anything that is not front or back): the sharpness zone is set
        to a balanced position.
        :type mode: str, optional, default to middle
        :param use_diffraction: if True, the sharpness zone considers the diffraction effect.
        This is requiring more optimisation (aperture and diffraction limits), takes longer run, and may fail.
        Consider using first set as False to get closer to a solution and try again in case of trouble.
        :type use_diffraction: bool, optional, default to False
        :param optimize_location: if False (default) the location is fixed, otherwise the position of the camera is optimized.
        :type optimize_location: bool optional, default is False
        :param inside_weight: the weight to be given to the contribution of points inside the inner zone.
        This zone depends on the mode parameter.
        :type inside_weight: float, default to 0.2
        :param outside_weight: the weight to be given to the contribution of points outside the inner zone.
        This zone depends on the mode parameter. Outside points are the once in front of the sharpness zone if the mode is front,
        or behind the sharpness zone if it is back, or both otherwise.
        :type outside_weight: float, default to 10
        :param shift_weight: the weight to be given to the contribution the camera shift if optimize_location is set.
        This is to ensure the camera is at the closest possible location.
        :type shift_weight: float, default to 0.1
        :param Z_weight: the weight to be given to the contribution the focus distance.
        This is only used in front and back modes to make sure the sharpness zone is close to the front or back respectively.
        :type Z_weight: float, default to 0.2
        :param target_points: a list of points, if None, the facing points are used
        :type target_points: list of 3D coordinates, optionnal
        :param shift_lim_vs_Z: setting the limit of search for Z as shift_lim_vs_Z * self.Z
        :type shift_lim_vs_Z: float
        :param update: if True the camera is updated with the optimized parameters.
        Else they are returned as a dict {Z, camera_z_shift} Z being the new focus distance,
        camera_z_shift being the distance to move backward.
        
        :returns: the optimisation result if return_result is True. 
        """
        # getting the distances
        if target_points is None:
            assert(self.target_object is not None), "please attach an object"
            facing = self.target_object.point_data["facing_b"]
            target_points = self.target_object.points[facing]
        target_zx = self._compute_camera_coordinates(target_points)[:,0]
        min_zx, max_zx = np.nanmin(target_zx), np.nanmax(target_zx)
        
        # setting weights and bounds
        w_in, w_out = np.array([inside_weight, outside_weight]) * 150 / len(target_zx) 
        lim = shift_lim_vs_Z * self.Z
        bounds = (
            (self.min_focus_distance, max_zx + lim),
            (-min_zx + self.min_focus_distance, lim)
        )
        
        # todo
        if mode == "front":
            res = optim_method(
                lambda x: np.sum(
                    self._optimize_sharpness_cost(
                        target_zx, 0, x[0], w_in= w_in, w_out= w_out,
                                    mode= mode,
                                    use_diffraction=use_diffraction
                                    )) - Z_weight * x[0]**2, ## minimize Z as well
                bounds= [(min_zx, max_zx )]
            )
            
        elif mode == "back":
            res = optim_method(
                lambda x: np.sum(
                    self._optimize_sharpness_cost(
                        target_zx, 0, x[0], w_in= w_in, w_out= w_out,
                                    mode= mode,
                                    use_diffraction=use_diffraction,
                                    optimize_N= optimize_N
                                    )) + Z_weight * x[0]**2, ## minimize Z as well
                bounds= [(min_zx, max_zx )]
            )
        else:
            if optimize_location:
                res = optim_method(
                    lambda x: np.sum(
                        self._optimize_sharpness_cost(
                            target_zx, x[1], x[0], w_in= w_in, w_out= w_out,
                            mode= mode,
                            use_diffraction=use_diffraction,
                                    optimize_N= optimize_N
                        )
                    ) + shift_weight * (min_zx + x[1])**2,
                    bounds= bounds
                )
            else:
                res = optim_method(
                    lambda x: np.sum(
                        self._optimize_sharpness_cost(
                            target_zx, 0, x[0], w_in= w_in, w_out= w_out,
                            mode= mode,
                            use_diffraction=use_diffraction,
                                    optimize_N= optimize_N
                        )) ,
                    bounds= [(min_zx, max_zx )]
                )
            
        if res.success:
            camera_z_shift = res.x[1] if len(res.x) > 1 else 0
            if update:
                self.move( camera_z_shift, "backward")
                self.update_focus(Z= res.x[0], optimize_N= use_diffraction and optimize_N)
        if return_result:
            return res
        
    def _optimize_sharpness_get_limits(self, Z, optimize_N= True, use_diffraction= True):
        """Computes sharpness zone limits
        
        :returns: Z1 & Z2 if use_diffraction is False, Z1d Z2d otherwise.
        if optimize_N & use_diffraction, the aperture is optimised.
        """
        N = self.compute_optimal_aperture(Z = Z) if optimize_N and use_diffraction else self.N
        if use_diffraction:
            return self._compute_sharpness_zone_including_diffraction(Z=Z, N=N)
        else:
            return self._compute_sharpness_zone(Z, N if optimize_N else None)
        
    def _optimize_sharpness_inner_zone(self, zx, Z1d, Z2d, mode= "middle"):
        """Computes zones for sharpness optimisation
        
        :returns: an array like zx with True for the inner zone False otherwise.
        """
        if mode == "front":
            return zx >= Z1d
        elif mode == "back":
            return zx <= Z2d
        else:
            return np.logical_and(zx >= Z1d, zx <= Z2d)
        
    def _optimize_sharpness_cost(self,
                                 zx, cam_shift, Z,
                                 optimize_N= True,
                                 mode = "middle",
                                 use_diffraction= True,
                                 w_in= 0.1, w_out=1
                                ):
        """Cost function for the optimisation of sharpness"""
        Z1d, Z2d = self._optimize_sharpness_get_limits(Z, optimize_N= optimize_N, use_diffraction= use_diffraction)
        
        zx = zx + cam_shift
        zi = self._optimize_sharpness_inner_zone(zx, Z1d, Z2d, mode)
        
        x = zx - Z
        scaled = np.where(x<0, x/Z1d, x/Z2d)
        w = np.where(zi, w_in, w_out)
            
        return np.multiply(w, scaled**2)
    
        
    def _compute_depth_reconstruction_error(self, z, baseline= None, image_std= 1/3):
        """Computes the depth reconstruction error standard deviation sigma_z.
        
        This implements the equation from Wenzel et al. (2013),
        which derives the so called normal case from Kraus (2007),
        i.e., considering two pictures taken with a simple translation :math: `B`
        parallel to the horizontal axis of the camera (hence, orthogonal
        to the aiming direction).
        
        :math: `\frac{p z^2}{f B}\sigma_d`
        
        :param baseline: the translation along the baseline between the two camera shots
        in meters. If not given, the default one is used (1 m)
        :type baseline: float, optional
        :param image_precision: the measurement precision in image space,
        i.e., as a ratio of sensor pixel size (no dimension),
        default is 1/3 as suggested by Wenzel et al 2013.
        :type image_precision: float
        """
        baseline = baseline if baseline is not None else self.default_baseline
        return np.ma.array(
            self.pixel_pitch / self.focal * (z**2) / baseline * image_std,
            mask= z<0,
            fill_value= np.nan
        )
    
    @classmethod
    def _compute_depth_precision_vs_resolution(cls,
            resolution, b_z, image_precision= 1/3
        ):
        """Computes depth precision with respect to image resolution

        This is obtained by :math: `\frac{1}{r B/z} \sigma_d`
        :param resolution: the image resolution in pixels per meter
        :type resolution: float
        :param b_z: the ratio between the baseline distance (i.e.,
        between the two cameras in normal case) and the focus distance
        :type b_z: float
        :param image_precision: the measurement precision of the image
        :type image_precision: float
        """
        return image_precision / resolution / b_z
    
    def _compute_plan_geometry(self, distance):
        """Computes the geometry of the sensor/image at a given distance
        
        Parameters:
        - distance: the distance from the focal point to the plan
        
        Returns:
        - an array containing the XYZ coordinates of the four corners of the plan in clockwise order looking
        in the view direction
        """
        distance = distance if distance != np.inf else 1e6 # to handle focusing beyond H
        center = self.location + self.dir_vector * distance
        mag = distance / self.focal
        half_width_vector = 0.5 * mag * self.sensor_width * self.w_vector
        half_height_vector = 0.5 * mag * self.sensor_height * self.h_vector
        corners = [ center + i * half_width_vector + j * half_height_vector
                for i in [-1,1]
                for j in [-1,1]
                ]
        return np.array(corners)[[0,2,3,1]]
    
    def _compute_camera_coordinates(self, points):
        """Computes the coordinates of points with respect to camera definition
        
        Parameters:
        - points: the points for which the coordinates must be computed
        
        Returns:
        - an array of the same shape as points with the coordinates in columns:
         - z : in the view direction
         - x : in the width direction of the sensor
         - y : in the height direction of the sensor
        """
        points = np.array(points).reshape((-1,3))
        centered = points - self.location
                
        z = np.dot(centered,self.dir_vector)
        y = np.dot(centered,self.h_vector)
        x = np.dot(centered,self.w_vector)
        return np.array([z,x,y]).T
    
    def _compute_deviation_angles(self, camera_coords):
        """Computes the deviation angles from camera coordinates
        """
        z,x,y = camera_coords.T
        angle_w = np.rad2deg(np.arctan2(x,z))
        angle_h = np.rad2deg(np.arctan2(y,z))
        return np.array([angle_w,angle_h]).T
    
    def _compute_pixel_coordinates(self, camera_coords):
        """Computes the pixel coordinate corresponding to each point
        """        
        z,x,y = camera_coords.T
        pix_x = x/z*self.focal/self.pixel_pitch
        pix_y = y/z*self.focal/self.pixel_pitch
        return np.array([pix_x,pix_y]).T
    
    def _compute_facing(self, points, normals):
        """Tells if the point is facing towards the camera
        """        
        points = np.array(points).reshape((-1,3))
        normals = np.array(normals).reshape((-1,3))
        v = points - self.location
        return np.einsum("ij,ij->i", normals,v) <= 0
    
    def _compute_is_in_view(self, angles):
        """Computes if a point is inside the view frame (visible)"""
        return np.logical_and(
            np.abs(angles[:,0]) < self.fov_w/2,
            np.abs(angles[:,1]) < self.fov_h/2
        )
        
    def _compute_visible(self, in_view, facing):
        """Computes if a point is visible (ie., in view and facing towards the camera)"""
        return np.logical_and(in_view, facing)
    
    def _compute_is_sharp(self, blur_spot_diameter):
        """Computes if a given blur spot is considered as sharp
        
        :param blur_spot_diameter: the blur spot diameter in micrometers
        :type blur_spot_diameter: float
        """
        return blur_spot_diameter <= self.confusion_circle_diameter * 1000
    
    def compute_sharpness(self, zx):
        """Computes if points at a given distance is sharp
        
        :param zx: the distance in meters
        :type zx: float
        :returns: an array with True where the point is sharp, False otherwise
        :type: np.array(bool)"""
        dt = self._compute_combined_blur_spot_diameter(zx)
        return self._compute_is_sharp(dt)
    
    def _compute_visible_and_sharp(self, visible, sharp):
        """Computes if a point is visible and sharp """
        return np.logical_and(visible, sharp)
    
    def _create_location_object(self):
        """Creates a representation of the camera/shot location"""
        self.location_object = pv.wrap(self.location)
        
    def get_location_object(self):
        """Returns the geometry of the focal point location as a Sphere
        
        If the object is not initialised yet, it will be created.
        """
        if "location_object" not in self.__dict__ or self.location_object is None:
            self._create_location_object()
        return self.location_object
    
    def _create_orientation_object(self):
        """Creates a representation of the camera/shot orientation"""
        self.orientation_objects = {}
        
        self.orientation_objects["front"] = pv.Arrow(
            start= self.location,
            direction= self.dir_vector,
            scale= float(self.Z) /100 * 5
            )
        
        self.orientation_objects["left"] = pv.Arrow(
            start= self.location,
            direction= self.w_vector,
            scale= float(self.Z) /100 * 5
            )
        
        self.orientation_objects["up"] = pv.Arrow(
            start= self.location,
            direction= self.h_vector,
            scale= float(self.Z) /100 * 5
            )
        
    def get_orientation_object(self):
        """Returns the geometry of the camera direction as arrows
        
        If the object is not initialised yet, it will be created.
        """
        if "orientation_objects" not in self.__dict__ or self.orientation_objects is None:
            self._create_orientation_object()
        return self.orientation_objects
    
    def _create_sensor_object(self, mirror):
        "Creates a representation of the sensor"
        corners = self._get_sensor_geometry(mirror= mirror)
        faces = [[0,2,1],[0,3,2]]
        self.sensor_object = pv.PolyData(corners, faces= [i for face_i in faces for i in [len(face_i)]+face_i])
        
    def _get_sensor_geometry(self, mirror= True):
        """Returns the sensor geometry
        
        An array containing the XYZ coordinates of the four corners of the sensor in clockwise order looking
        in the view direction.
        
        Parameters:
        - mirror (bool): if True (default) the sensor is projected in the view direction
        instead of its natural place behind the focal point.
        """
        return self._compute_plan_geometry((self.focal if mirror else -self.focal)/1000)
    
    def get_sensor_object(self, mirror= True):
        """Returns the sensor as a 3D object
        
        If the object is not initialised yet, it will be created.
        
        Parameters:
        - mirror (bool): if True (default) the sensor is projected in the view direction
        instead of its natural place behind the focal point.
        """
        if "sensor_object" not in self.__dict__ \
        or self.sensor_object is None:
            self._create_sensor_object(mirror= mirror)
        return self.sensor_object
    
    def _create_camera_object(self):
        "Creates a representation of the camera"
        corners = np.concatenate((
            self._get_sensor_geometry(),
            [self.location]
        ))
        faces = [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]
        self.camera_object = pv.PolyData(corners, faces= [i for face_i in faces for i in [len(face_i)]+face_i])
        
    def get_camera_object(self):
        """Returns the camera geometry
        
        If the object is not initialised yet, it will be created.
        This is a pyramid with the sensor at its base (projected in the view direction)
        and the focal point at its tip.
        """
        if "camera_object" not in self.__dict__ \
        or self.camera_object is None:
            self._create_camera_object()
        return self.camera_object
    
    def _create_focus_plan_object(self):
        "Creates a representation of the focus plan and the view frame"
        corners = self._compute_plan_geometry(self.Z)
        pyramid_corners = np.concatenate((
            corners,
            [self.location]
        ))
        faces = [[0,2,1],[0,3,2]]
        self.focus_plan_object = pv.PolyData(corners, faces= [i for face_i in faces for i in [len(face_i)]+face_i])
        pyramid_faces = [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]
        self.view_frame_object = pv.PolyData(pyramid_corners, faces= [i for face_i in pyramid_faces for i in [len(face_i)]+face_i])
        
    def get_focus_plan_object(self):
        """Returns the focus plan as a 3D object
        
        If the object is not initialised yet, it will be created.
        """
        if "focus_plan_object" not in self.__dict__ \
        or self.focus_plan_object is None:
            self._create_focus_plan_object()
        return self.focus_plan_object
    
    def get_view_frame_object(self):
        """Returns the view frame as a 3D object
        
        If the object is not initialised yet, it will be created.
        """
        if "view_frame_object" not in self.__dict__ \
        or self.view_frame_object is None:
            self._create_focus_plan_object()
        return self.view_frame_object
    
    def _create_sharpness_or_diffraction_object(self, z1, z2):
        """ generic creation of object for diffraction or sharpness"""
        near_plan = self._compute_plan_geometry(z1)
        far_plan  = self._compute_plan_geometry(z2)
        corners = np.concatenate((near_plan, far_plan))
        faces = [[3,0,1],[3,1,2],[0,4,1],[1,4,5],
                [4,7,5],[5,7,6],[7,3,2],[7,2,6],
                [2,1,5],[2,5,6],[0,3,4],[4,3,7],
                ]
        return pv.PolyData(corners, faces= [i for face_i in faces for i in [len(face_i)]+face_i])
        
    def _create_sharpness_object(self):
        "Creates a representation of the sharpness zone and edges"
        self.sharpness_object = self._create_sharpness_or_diffraction_object(self.Z1, self.Z2)
        self.sharpness_object_edges = self.sharpness_object.extract_feature_edges()
            
    def _create_diffraction_object(self):
        "Creates a representation of the sharpness zone and edges"
        
        if self.Z1d is not None and self.Z2d is not None:
            self.diffraction_object = self._create_sharpness_or_diffraction_object(self.Z1d, self.Z2d)
            self.diffraction_object_edges = self.diffraction_object.extract_feature_edges()
        else:
            del self.diffraction_object
            self.diffraction_object = None
            del self.diffraction_object_edges
            self.diffraction_object_edges = None
                
    def get_sharpness_object(self):
        """Returns the sharpness zone object
        
        If the object is not initialised yet, it will be created.
        This is a polygon going from the plan of first sharp object to the last sharp plan.
        """
        if "sharpness_object" not in self.__dict__ \
        or self.sharpness_object is None:
            self._create_sharpness_object()
        return self.sharpness_object

    def get_sharpness_object_edges(self):
        """Edges of the sharpness object
        
        If the object is not initialised yet, it will be created.
        """
        if "sharpness_object_edges" not in self.__dict__ \
        or self.sharpness_object_edges is None:
            self._create_sharpness_object()
        return self.sharpness_object_edges
    
    def get_diffraction_object(self):
        """Returns the diffraction zone object
        
        If the object is not initialised yet, it will be created.
        This is a polygon going from the plan of first sharp object to the last sharp plan.
        """
        if "diffraction_object" not in self.__dict__ \
        or self.diffraction_object is None:
            self._create_diffraction_object()
        return self.diffraction_object

    def get_diffraction_object_edges(self):
        """Edges of the diffraction object
        
        If the object is not initialised yet, it will be created.
        """
        if "diffraction_object_edges" not in self.__dict__ \
        or self.diffraction_object_edges is None:
            self._create_diffraction_object()
        return self.diffraction_object_edges
    
    def _create_visibility_object(self, zmax= None):
        """Creates a representation of the visible zone up to a distance
        
        Parameters:
        - zmax: the visibility zone extends up to this distance.
        If None (default), the target object is used instead to determine zmax,
        if no target_object is attached, zmax is twice Z.
        Note: this parameter is stored as self.zmax to facilitate update
        when not computed from target object."""
        if zmax is None:
            if self.target_object is None:
                zmax = 2*self.Z
            else:
                obj = self.target_object
        
            if obj.kind =="Sample":
                facing = self._compute_facing(obj.points, obj.point_normals)
                facing_points = obj.points[facing]
                assert(len(facing_points) > 0), "at least a point should be visible"
                    
            else:
                # Non Sample-like objects are considered
                # completely facing the camera
                facing_points = obj.points
            cam_coords = self._compute_camera_coordinates(facing_points)
            z_min, z_max = np.nanmin(cam_coords[:,0]), np.nanmax(cam_coords[:,0])
            zmax =  1.005 * z_max + 0.05*(z_max-z_min)

        self.zmax = zmax
        corners = np.concatenate((
                    self._compute_plan_geometry(zmax),
                    [self.location]
                ))
        faces = [[0,2,1],[0,3,2],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]
        self.visibility_object = pv.PolyData(corners, faces= [i for face_i in faces for i in [len(face_i)]+face_i])
        self.visibility_object.flip_faces()
        
    def get_visibility_object(self):
        """gets the visibility zone object
        
        If the object is not initialised yet, it will be created.
        """
        if "visibility_object" not in self.__dict__ \
        or self.visibility_object is None:
            self._create_visibility_object()
        return self.visibility_object
    
    def _create_visible_part_object(self):
        """Creates a representation of the visibile part of the target object"""
        if self.target_object is None:
                return
        
        self.visible_part_object = self.target_object.clip_surface(
            self.get_visibility_object()
        )
        return self.visible_part_object
    
    def get_visible_part_object(self):
        """gets the visible part object
        
        If the object is not initialised yet, it will be created.
        """
        if "visible_part_object" not in self.__dict__ \
        or self.visible_part_object is None:
            self._create_visible_part_object()
        return self.visible_part_object
        
    def __str__(self):
        """Shows camera definition"""
        report = "Camera settings:\n"
        for key in self.__dict__:
            if key == "units":
                continue
            if key == "pixel_size" or key == "confusion_circle":
                report += "  {}: {}{}\n".format(key, round(self.__dict__[key] * 1000, 3), " micron")
                continue
            report += "  {}: {}{}\n".format(key, self.__dict__[key], " "+self.units[key] if key in self.units.keys() else "")
        return report
    
class Camera(CameraBase):
    
    """Defines a camera for taking a Picture.
    
    This class defines a Camera, i.e., a physical apparatus made of
    a sensor and mounted with a Lens. A Camera has various parameters
    that can be modified, e.g., its position, orientation, aperture.
    
    This class also gathers functionnalities for computing 
    the configuration of a CameraShot, which means that several CameraShots
    could result from a same Camera.
    They are listed under self.shots
    
    This class also creates and stores visualisation objects.
    They are create on demand only when getter functions are used,
    but will then be updated each time an update function modifies the camera.
    
    # About views:
    Cameras have a view array to count which points of an objects are viewed in each shot.
    This is structured as a dictionnary associating a numpy array to each attached object.
    Each line corresponds to a give shot (referenced by its interger id)
    each column is for a point of the attached object.
    """
    
    def __init__(self, cam_id=None, lens_id= None, name= None,
                 **kargs):
        """Creates a Camera
        
        Parameters:
        - cam_id: the unique id of the camera referenced in the CameraRegistry.
        Note that if this is not given, the minimal following parameters must be passed:
        'sensor_width', 'sensor_height', 'nb_pixel_width', 'nb_pixel_height'
        - lens_id: the unique id of the lens as defined in the CameraRegistry, 
        if None then focal and min_focus_distance must be passed as keyword arguments
        - name: the nick name of the camera if given, if None (default) the uid is used instead
        - kargs: optional keyword arguments
         * orientation: landscape (default) or portrait
         * confusion_circle_method: see compute_confusion_circle
         * focal: optional, if lens_id is not set, the focal length in mm
         * min_focus_distance: optional, if lens_id is not set, the minimal focus distance in m
        :param wavelength: the wavelength of the light in nanometers (default to 546 nm, yellow-green light)
        :type float:
        :param effective_diffraction_disk_ratio: if 1, this returns the diameter of the Airy disk,
        i.e., the diameter of the first dark ridge in the diffraction partern. In practice,
        the light intensity becomes very low even before reaching this distance, that's why we suggest using only 
        a fraction of this diameter. Default is 0.6 as it concentates more than 84% of the light.
        """
        self._setup_defaults_and_methods(**kargs)
        
        # camera & lens identification
        self._setup_camera_id(cam_id, name= name, **kargs)
        self.setup_lens(lens_id, **kargs)
        
        # focus
        self.update_focus( Z = self.min_focus_distance, optimize_N= True )
        
        # camera view
        self.update_view(
            location= [0,0,0],
            yaw=0,
            pitch=0,
            roll=0,
            orientation="landscape"
        )
        
        # init views array
        self.views = {}
        self.sharp_views = {}
                
    def _setup_defaults_and_methods(self, **kargs):
        """Initialises the default methods"""
        self.shots = []
        self.target_object = None
        
        self.orientation = "landscape"
        self.location = np.array([0,0,0])
        self.yaw, self.pitch, self.roll = (0,0,0)
        
        # camera orientation
        self.rotation = np.identity(3)
        self.dir_vector, self.w_vector, self.h_vector = np.identity(3)

        # parameters related to sharpness zone
        self.confusion_method = kargs.get("confusion_circle_method", "bouillot")
        self.wavelength_nm = kargs.get("wavelength_nm", 546)
        self.effective_diffraction_disk_ratio = kargs.get("effective_diffraction_disk_ratio", 0.6)
        self.default_baseline = kargs.get("baseline", 1)
        
        self.units = {
            "sensor_width": "mm",
            "sensor_height": "mm",
            "pixel_pitch": "mm",
            "confusion_circle_diameter": "mm",
            "focal": "mm",
            "min_focus_distance": "m",
            "focal_min": "mm",
            "focal_max": "mm",
            "H": "m",
            "Z1": "m",
            "Z2": "m",
            "fov_w": "°",
            "fov_h": "°",
            "depth_of_field": "m",
            "image_width": "m",
            "image_height": "m"
        }
        
    def _setup_camera_id(self, cam_id, name = None, **kargs):
        """Initialises the camera definition
        
        Parameters:
        - cam_id: the unique id of the camera referenced in the CameraRegistry
        Note that if this is not given, the minimal following parameters must be passed:
        'sensor_width', 'sensor_height', 'nb_pixel_width', 'nb_pixel_height'
        - name: the nick name of the camera if given, if None (default) the uid is used instead
        """
        self.cam_id = cam_id
        if cam_id is None:
            if 'sensor_width' not in kargs or\
               'sensor_height' not in kargs or\
               'nb_pixel_width' not in kargs or\
               'nb_pixel_height' not in kargs:
                raise Exception("When creating a custom Camera, you must at least give parameters: 'sensor_width', 'sensor_height', 'nb_pixel_width', 'nb_pixel_height'")
            self.camera_characteristics = {key:kargs[key] for key in kargs if key in CameraRegistry.camera_base.columns}
            self.camera_characteristics["pixel_pitch"] = kargs["sensor_width"]/kargs["nb_pixel_width"]
            self.camera_type = ""
        else:
            self.camera_characteristics = CameraRegistry.camera_base.loc[cam_id]
            self.camera_type = self.camera_characteristics["name"]
        self.name = cam_id if name is None else name
        
        # sensor definition
        self._setup_sensor()
        
    def _setup_sensor(self):
        """Initialises the sensor definition
        
        Defined parameters are:
        - self.sensor_width 
        - self.sensor_height
        - self.pitch 
        - self.confusion_circle 
        """
        self.sensor_width = self.camera_characteristics["sensor_width"]
        self.sensor_height = self.camera_characteristics["sensor_height"]
        self.pixel_pitch = self.camera_characteristics["pixel_pitch"]
        self.confusion_circle_diameter = self._compute_confusion_circle()
        
    def setup_lens(self, lens_id = None, **kargs):
        """Initialises the lens parameters
        
        Parmeters:
        - lens_id: the unique id of the lens as defined in the CameraRegistry, 
        if None then focal and min_focus_distance must be passed as keyword arguments
        - kargs: optional keyword arguments
         * focal: optional, if lens_id is not set, the focal length in mm
         alternatively, focal_min and focal_max can be given for zoom
         * min_focus_distance: optional, if lens_id is not set, the minimal focus distance in m
        """
        self.lens_id = lens_id
        self.N = 2.8 # initialise aperture number before computing focal updates
        if lens_id is not None:
            self.lens_characteristics = CameraRegistry.lens_base.loc[lens_id]
            self.focal_min = self.lens_characteristics["focal_min"]
            self.focal_max = self.lens_characteristics["focal_max"]
            self.fixed_focal = self.focal_min == self.focal_max
            self.min_focus_distance = self.lens_characteristics["min_focus_distance"]
            self.Z = self.min_focus_distance # must initialise Z before update focal
            self.update_focal(self.focal_min)
        else:
            self.min_focus_distance = kargs.get("min_focus_distance", 0.3)
            self.Z = self.min_focus_distance # must initialise Z before update focal
            if "focal" in kargs: 
                self.fixed_focal = True
                self.focal_min = kargs["focal"] 
                self.focal_max = kargs["focal"] 
                self.update_focal(kargs["focal"])
            elif "focal_min" in kargs and "focal_max" in kargs:
                self.fixed_focal = False
                self.focal_min = kargs["focal_min"]
                self.focal_max = kargs["focal_max"]
                self.update_focal(kargs["focal_min"])
            else:
                raise Exception("When creating a custom Camera, you must at least give a focal parameter.")
                
    def update_focal(self, focal, force=False):
        """Updates the focal length of the camera
        
        Parameters:
        - focal: focal length in mm
        - force: if False (default) the focal can only be within the focal range
        (self.focal_min and self.focal_max). If True, this check is bypassed,
        but it means the lens is in a state that goes beyond its capabilities.
        This is only allowed for making it easier to inspect the effect of focal length.
                
        This is updating the hyperfocal and update_field_of_view().
        """
        if not force:
            assert focal <= self.focal_max and focal >= self.focal_min, f"focal({focal:}) must be within [{self.focal_min:},{self.focal_max:}. ]"+\
                "Use force=True to bypass this test"
        self.focal = focal
        self._update_field_of_view()
        self._update_hyperfocal()
        self._update_magnification()
        self._update_object_view_properties()
        
        # update visualisation objects
        self._update_orientation_object()
        self._update_sensor_object()
        self._update_camera_object()
        self._update_sharpness_object()
        self._update_focus_plan_object()
    
    def _update_field_of_view(self):
        """Updates the Field of View parameters
        
        self.fov_w: angle of view in the width direction of the sensor
        self.fov_h: angle of view in the height direction of the sensor
        """
        self.fov_w = 2 * np.rad2deg(np.arctan(0.5 * self.sensor_width / self.focal))
        self.fov_h = 2 * np.rad2deg(np.arctan(0.5 * self.sensor_height / self.focal))

    def _update_hyperfocal(self):
        """ Updates the hyperfocal
        
        This computes self.H, the hyperfocal.
        This is by definition the distance at which the focus should be set 
        to obtain the deepest sharpness zone. This parameter only depends on 
        the focal, the aperture, and a confusion size. The computation is based
        on the size of the confusion circle (the largest diameter
        of a circle on the sensor that would be considered as a focussed point and 
        not a blur).
        """
        self._compute_hyperfocal()
        self._update_sharpness_zone()  
              
    def update_reference_light_wavelength(self, wavelength_nm):
        """Updates the wavelength of the light used as a reference for sharpness computation"""
        self.wavelength_nm = wavelength_nm
        self._update_sharpness_zone()
        
        # update visualisation objects
        self._update_sharpness_object()
        
    def update_effective_diffraction_disk_ratio(self, effective_diffraction_disk_ratio):
        """Updates the wavelength of the light used as a reference for sharpness computation"""
        self.effective_diffraction_disk_ratio = effective_diffraction_disk_ratio
        self._update_sharpness_zone()
        
        # update visualisation objects
        self._update_sharpness_object()
        
    def _update_sharpness_zone(self):
        """Updates the sharpness zone
        
        This is defined by the minimal/ maximal distance (Z1 and Z2) at which an object would be sharp.
        """
        self.Z1, self.Z2 = self._compute_sharpness_zone()
        self.depth_of_field = self._compute_depth_of_field()
        self._compute_depth_of_image()
        self.Z1d, self.Z2d = self._compute_sharpness_zone_including_diffraction()
        self.depth_of_field_diffraction_included = self._compute_depth_of_field_with_diffraction()
        
    def _update_target_point(self):
        """Updates the target pointed by the camera"""
        self.target_point = self.location + self.Z * self.dir_vector
        
    def update_focus(self, Z, optimize_N= False):
        """Updates or initializes the focus definition of the camera
        
        This is setting up:
        - self.Z: the distance from focal point to target focus point
        - the sharpness zone by calling update_sharpness_zone()
        
        Arguments:
        - Z (float): the distance from focal point to target focus point.
         If None (default) the minimal focus distance is set (self.min_focus_distance) 
        - optimize_N(bool): if True, the aperture is optimize to have the largest epth of field
        """
        assert (Z is None or isinstance(Z, float) or isinstance(Z,int)), "Z must be None, int or float"
        self.Z = Z if Z is not None else self.min_focus_distance
        if optimize_N: self.update_aperture()
        self._update_target_point()
        self._update_sharpness_zone()
        self._update_magnification()
        self._update_object_view_properties()
        
        # update visualisation objects
        self._update_sharpness_object()
        self._update_focus_plan_object()
    
    def update_aperture(self, N= None):
        """Updates the aperture of the camera lens
        
        Parameters:
        - N: number of aperture (i.e., ratio between focal and aperture diameter, aperture = f/N)
        If None, by default, the optimal aperture is computed considering diffraction effect
        """
        self.N = N if N is not None else self.compute_optimal_aperture()
        self._update_hyperfocal()
        self._update_sharpness_object()
        self._update_object_view_properties()
        
    def _update_magnification(self):
        """Updates the magnification factor (Z/f)"""
        self.magnification = CameraBase._compute_magnification_from_camera_parameters(self.focal, self.Z)
        self.image_width = CameraBase._compute_image_dimension_from_camera_parameters(self.magnification, self.sensor_width)
        self.image_height = CameraBase._compute_image_dimension_from_camera_parameters(self.magnification, self.sensor_height)
    
    def update_view(self, 
            location= None,
            yaw= None,
            pitch= None,
            roll= None,
            orientation= None
        ):
        """Initialises or updates the camera view (location and orientation)
        
        Parameters:
        - location: 3d location of the focal point (x,y,z). If None (default),
        the location of the camera is not changed.
        - yaw: angle of rotation around Z axis in degrees, from X towards Y.
        If None (default), the yaw of the camera is not changed.
        - pitch: angle of rotation around the rotated Y axis, Z towards X.
        If None (default), the pitch of the camera is not changed.
        - roll: angle of rotation around rotated X axis, Z towards Y.
        If None (default), the roll of the camera is not changed.
        - orientation: either "landscape" or "portrait". If None (default),
        the orientation is left unchanged.
        """
        # update location if needed
        if location is not None:
            self.location = np.array(location).astype(float)
        
        # update rotation if needed
        update_rotation = False
        if orientation is not None:
            self.orientation = orientation
            update_rotation = True
        if yaw is not None:
            self.yaw = yaw
            update_rotation = True
        if pitch is not None:
            self.pitch = pitch
            update_rotation = True
        if roll is not None:
            self.roll = roll
            update_rotation = True
        
        # update camera orientation
        if update_rotation:
            effective_roll = self.roll + (90 if self.orientation == "portrait" else 0)
            self.rotation = spt.Rotation.from_euler("ZYX",  
                    [self.yaw, self.pitch, -effective_roll],
                    degrees= True
                )
            
        self._update_orientation_vectors()
        self._update_object_view_properties()
        # update visualisation objects
        self._update_all_objects()
    
    def _update_orientation_vectors(self):
        """Updates the orientation vectors"""
        self.dir_vector, self.w_vector, self.h_vector = self.rotation.apply(np.identity(3))
        self._update_target_point()
        
    def _update_euler_angles(self):
        """Updates the yaw pitch roll angles from rotation"""
        self.yaw, self.pitch, self.roll = np.array([1,1,-1]) * self.rotation.as_euler("ZYX", degrees=True)
        if self.orientation == "portrait":
            self.roll -= 90
        
    def _update_rotation_from_vectors(self):
        """Updates the internal rotation based on updates direction vectors"""
        
        matrix = np.array([self.dir_vector, self.w_vector, self.h_vector])
        self.rotation = spt.Rotation.from_matrix(matrix.T)
        self._update_euler_angles()
        self._update_target_point()
        
    def move(self, translation, direction= "forward"):
        """Move the camera by a given translation
        
        Parameters:
        - translation: either a single value giving the amount of displacement
        (in which case direction must be given), or a list or array with 
        the 3 components of the displacement [dX,dY,dZ] in the 3D space
        - direction: the direction of the displacement relative 
        to the current orientation of the camera: "left", "right", "up", "down",
        "backard", or "forward" (default).
        """
        translation = np.array([translation]).flatten()
        if len(translation) == 1:
            translation = translation[0]
            if direction == "left":
                self.location += translation * self.w_vector
            elif direction == "right":
                self.location -= translation * self.w_vector
            elif direction == "down":
                self.location -= translation * self.h_vector
            elif direction == "up":
                self.location += translation * self.h_vector
            elif direction == "backward":
                self.location -= translation * self.dir_vector
            elif direction == "forward":
                self.location += translation * self.dir_vector
            else:
                assert(True),'direction must be one of "left", "right", "up", "down",\
                    "backard", or "forward"'
        else:
            assert(len(translation)==3),"translation must be a 3D vector"
            self.location += np.array(translation)
        
        self._update_target_point()
        self._update_object_view_properties()
        # update visualisation objects
        self._update_all_objects()
    
    def move_to(self, coord, mode= "absolute", target= None, nitmax=10, **kargs):
        """Moves the camera
        
        :param mode: tells how the camera should be displaced.
        - absolute: puts the camera at the absolute location given in coord
        - distance: puts the camera at the distance of the focus point given by coord
        :type mode: str, optional, default is "absolute"
        """
        if mode == "absolute":
            coord = np.array([coord]).ravel()
            assert(len(coord)==3),"coordinates must be a 3D vector"
            self.location = np.array(coord)
            self._update_target_point()
            self._update_object_view_properties()
            # update visualisation objects
            self._update_all_objects()
        elif mode == "distance":
            assert(isinstance(coord, float) or isinstance(coord, int)), "distance must be a scalar value"
            
            # optimise location until convergence (the points visible may change after update)
            for i in range(nitmax):
                self.focus_on(target= target, **kargs)
                translation = self.Z - coord
                self.move(translation, "forward")
        elif mode == "resolution":
            z = self._compute_z_from_resolution(coord)
            self.move_to(z, "distance")
        else:
            raise Exception("Mode not supported: "+mode)

    def turn(self, angle, direction= "left"):
        """Turns the camera
        
        Parameters:
        - angle: the angle of rotation in degrees
        - direction: one of "left", "right", "up", "down", "roll_left", "roll_right".
        Default is "left". 
        """
        if direction == "left":
            new_rotation = spt.Rotation.from_rotvec(angle * self.h_vector, degrees=True)
        elif direction == "right":
            new_rotation = spt.Rotation.from_rotvec(-angle * self.h_vector, degrees=True)
        if direction == "roll_left":
            new_rotation = spt.Rotation.from_rotvec(-angle * self.dir_vector, degrees=True)
        elif direction == "roll_right":
            new_rotation = spt.Rotation.from_rotvec(+angle * self.dir_vector, degrees=True)
        elif direction == "up":
            new_rotation = spt.Rotation.from_rotvec(-angle * self.w_vector, degrees=True)
        elif direction == "down":
            new_rotation = spt.Rotation.from_rotvec(angle * self.w_vector, degrees=True)
        
        self.rotation = new_rotation * self.rotation 
        self._update_euler_angles()
        self._update_orientation_vectors()
        self._update_target_point()
        self._update_object_view_properties()
        
        # update visualisation objects
        self._update_all_objects()
    
    def orbit(self, angle, around= None, axis=[0,0,1]):
        """Rotate the camera around a center
        
        Parameters:
        - angle: rotation angle in degrees
        - around: either the point around which the rotation is performed
        or an object around whose center to turn.
        If None (default) the current target_point is used.
        - axis: rotation axis (default, vertical axis)
        """
        axis = axis / np.linalg.norm(axis)
        r = spt.Rotation.from_rotvec(angle * axis, degrees= True)
        
        around = around if around is not None else self.target_point
        center = around.center if hasattr(around, "center") else around
        new_location = r.apply(self.location - center) + center
        self.dir_vector = r.apply(self.location + self.dir_vector - center ) + center - new_location
        self.w_vector = r.apply(self.location + self.w_vector - center ) + center - new_location
        self.h_vector = r.apply(self.location + self.h_vector - center ) + center - new_location
        self.location = new_location
        
        # rotation
        self._update_rotation_from_vectors()
        self._update_target_point()
        self._update_object_view_properties()
        
        # update visualisation objects
        self._update_all_objects()
        
    def _update_location_object(self):
        """Updates the object representing camera location for 3D view.
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "location_object" in self.__dict__ and self.location_object is not None:
            self.location_object.points = self.location
    
    def _update_orientation_object(self):
        """Updates the object representing camera orientation for 3D view.
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "orientation_objects" in self.__dict__ and self.orientation_objects is not None:
                
            self.orientation_objects["front"].points = pv.Arrow(
                start= self.location,
                direction= self.dir_vector,
                scale= float(self.Z) /100 * 5
                ).points
            
            self.orientation_objects["left"].points = pv.Arrow(
                start= self.location,
                direction= self.w_vector,
                scale= float(self.Z) /100 * 5
                ).points
            
            self.orientation_objects["up"].points = pv.Arrow(
                start= self.location,
                direction= self.h_vector,
                scale= float(self.Z) /100 * 5
                ).points
            
    def _update_sensor_object(self):
        """Updates the object representing the sensor for 3D view
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "sensor_object" in self.__dict__ \
        and self.sensor_object is not None:
            corners = self._get_sensor_geometry()
            self.sensor_object.points = corners
                
    def _update_camera_object(self):
        """Updates the object representing the camera for 3D view.
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "camera_object" in self.__dict__ \
        and self.camera_object is not None:    
            corners = np.concatenate((
                self._get_sensor_geometry(),
                [self.location]
            ))
            self.camera_object.points = corners
        
    def _update_focus_plan_object(self):
        """Updates the object representing the focus plan for 3D view
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "focus_plan_object" in self.__dict__ \
        and self.focus_plan_object is not None:   
            corners = self._compute_plan_geometry(self.Z)
            pyramid_corners = np.concatenate((
                corners,
                [self.location]
            ))
            self.focus_plan_object.points = corners
            self.view_frame_object.points = pyramid_corners
                
    def _update_sharpness_object(self):
        """Updates the object representing the sharpness zone in 3D
        
        This is only performed if the representation object exists,
        otherwise it will be created on demand and already up to date.
        """
        if "sharpness_object" in self.__dict__ \
        and self.sharpness_object is not None:
            # sharpness object
            near_plan = self._compute_plan_geometry(self.Z1)
            far_plan  = self._compute_plan_geometry(self.Z2)
            corners = np.concatenate((near_plan, far_plan))
            self.sharpness_object.points = corners
            self.sharpness_object_edges.points = self.sharpness_object.extract_feature_edges().points
        
        if "diffraction_object" in self.__dict__ \
        and self.diffraction_object is not None:
            # diffraction object
            if self.Z1d is not None and self.Z2d is not None:
                near_plan_diff = self._compute_plan_geometry(self.Z1d)
                far_plan_diff  = self._compute_plan_geometry(self.Z2d)
                corners_diff = np.concatenate((near_plan_diff, far_plan_diff))
                self.diffraction_object.points = corners_diff
                self.diffraction_object_edges.points = self.diffraction_object.extract_feature_edges().points
            else:
                self.diffraction_object = None
                self.diffraction_object_edges = None
                
    def _update_visibility_object(self):
        """Updates the visibility object"""
        
        if "visibility_object" in self.__dict__ \
        and self.visibility_object is not None:
            
            if self.target_object is not None:
                obj = self.target_object
                if obj.kind == "Sample":
                    facing = self._compute_facing(obj.points, obj.point_normals)
                    facing_points = obj.points[facing]
                else:
                    # Non Sample-like objects are considered
                    # completely facing the camera
                    facing_points = obj.points
                cam_coords = self._compute_camera_coordinates(facing_points)
                z_min, z_max = np.nanmin(cam_coords[:,0]), np.nanmax(cam_coords[:,0])
                zmax =  z_max + 0.05*(z_max-z_min)
            else:
                zmax = self.zmax

            corners = np.concatenate((
                        self._compute_plan_geometry(zmax),
                        [self.location]
                    ))
            self.visibility_object.points = corners
        
    def _update_visible_part_object(self):
        """Updates the visible_part_object"""
        
        if "visible_part_object" in self.__dict__ \
        and self.visible_part_object is not None:
            
            if self.target_object is not None:
                new_visible_part_object = self.target_object.clip_surfaces(
                    self.get_visibility_object()
                )
                
                self.visible_part_object.points = new_visible_part_object.points
                self.visible_part_object.faces = new_visible_part_object.faces
                
                self.visible_part_object.face_normals = new_visible_part_object.face_normals
                self.visible_part_object.point_normals = new_visible_part_object.point_normals
                
    def _update_all_objects(self):
        """Updates all representation objects (if they exist)"""
        self._update_location_object()
        self._update_orientation_object()
        self._update_sensor_object()
        self._update_camera_object()
        self._update_sharpness_object()
        self._update_focus_plan_object()
        self._update_visibility_object()
        
    def attach_target(self, target, aim = True, focus= True, **kargs):
        """Attaches a given target object to the camera
        
        Parameters:
        - target: a TargetObject that will be aimed at by the camera
        - aim: if True (default) the camera will be moved to aim at the target
        - focus: if True (default) the camera will focus on the attach object (see `focus_on`)
        - kargs: keyword arguments for the focus_on and/or aim_at method if focus and/or aim are True.
        """
        #assert (isinstance(target, TargetObject)), "Only Target Objects can be attached to cameras, see TargetObject constructor.\n"+\
        #    "Here the given type is: "+str(type(target))
        self.target_object = target
        if aim:
            self.aim_at(target= target, **kargs)
        if focus:
            self.focus_on( target= target, **kargs)
            
        # init fields
        if "nb_views" not in target.point_data.keys():
            target.point_data.set_array(0., "nb_views")
        
        # init views array
        self.views[target] = np.empty((0, target.n_points), dtype= bool)
        self.sharp_views[target] = np.empty((0, target.n_points), dtype= bool)
    
    def init_location(self, target_object, dist= None, **kargs):
        """Initialize the location of the camera with respect to the given object.
        
        Depending on the kind of object, the position is set:
        - for rock samples, at a distance to the left, ie., in the lower x values
        - for outdoor ground, above the maximum point a a given distance
        - for outdoor outcrop, at a distance in the normal direction
        - for indoor, at the center
        """
        if target_object.kind == "Sample":
            dist = self.min_focus_distance if dist is None else dist
            x_min = np.nanmin(target_object.points[:,0])
            self.move_to([x_min - dist, *target_object.center[1:]])
            self.update_focus(dist)
        elif target_object.kind == "Indoor":
            self.update_view(
                location= target_object.center,
                yaw= 0,
                pitch= 0,
                roll= 0
            )
            if dist is not None:
                self.focus_on(target_object, **kargs)
                self.move_to(dist, "distance", target= target_object)
                self.update_focus(dist)
        elif isinstance(target_object, Ground) or target_object.kind == "Ground":
            dist = 10 if dist is None else dist
            z_max = np.nanmax(target_object.points[:,2])
            self.update_view(
                location= [*target_object.center[:2], z_max + dist],
                yaw= 0,
                pitch= 90,
                roll= 0
            )
            #self.focus_on(target_object, **kargs)
            #self.update_focus(dist)
            self.move_to(dist, "distance", target= target_object, **kargs)
        elif isinstance(target_object, Outcrop) or target_object.kind == "Outcrop":
            dist = 1 if dist is None else dist
            n = target_object.point_data["Normals"].mean(axis=0)
            n /= np.linalg.norm(n)
            u = np.dot(target_object.points - target_object.center, n)
            loc = target_object.center + (np.nanmax(u) + dist) * n
            self.move_to(loc)
            self.update_focus(dist)
        else:
            raise Exception("Not supported: "+ target_object.kind)
            
    
    def aim_at(self, target= None, z_axis= [0,0,1], **kargs):
        """Aims the camera at the attached target 
        
        Parameters:
        - target: if None (default) this will aim at the center of the attached target object,
        otherwise this can be another target object or a given point
        - keyword arguments:
        * rotate: if True (default) the camera will be rotated to aim at the target.
        if False, the camera will be translated instead.
        Rotation is performed around the z_axis then in the vertical plan.
        If the object is directly above or below, the camera is rotated around the w_vector instead.
        If both rotate and translate are True an exception is raised
        * translate: if True (default is False) the camera will be translated to aim at the target
        if False, the camera will be rotated instead.
        If both rotate and translate are True an exception is raised
        """
        assert(target is not None or self.target_object is not None), "Warning: can't aim at target because None is attached. "+\
            "Please specify a target"
        assert(not (("rotate" in kargs and kargs["rotate"]) 
                and ("translate" in kargs and kargs["translate"]))), "when aiming "+\
            "at a target, you can not specify both rotate=True and translate=True. "+\
            "please use one or the other option."
        
        if not ("rotate" in kargs  or "translate" in kargs):
            kargs= {"rotate":True}
            
        if target is None:
            target = self.target_object.center
        elif isinstance(target, pv.PolyData):
            target = target.center
        else:
            assert(isinstance(target,list) or isinstance(target, np.ndarray)), "target must be given as a list, np.ndarray, of pyvista object"
            target = np.array(target)
            
        v = target - self.location
        
        if (("rotate" in kargs and kargs["rotate"]) or
            ("translate" in kargs and not kargs["translate"])):
            
            u = self.dir_vector
            h = self.h_vector
            w = self.w_vector
            
            v /= np.linalg.norm(v)   
            z = np.array(z_axis, dtype= float)
            z /= np.linalg.norm(z)
            y = np.cross(z, v)
            norm_y = np.linalg.norm(y)
            if norm_y == 0:
                y = w
            y /= np.linalg.norm(y)
            x = np.cross(y,z)
            x /= np.linalg.norm(x)

            # projection onto the plan
            u_z = u - np.dot(u,z) * z
            u_z /= np.linalg.norm(u_z)

            # rotation around Z is only applied if v is not aligned with Z
            if norm_y != 0:
                u_x = np.dot(u_z, x)
                u_y = np.dot(u_z, y)
                alpha = - np.arctan2(u_y,u_x)

                r = spt.Rotation.from_rotvec(z * alpha)
                u = r.apply(u)
                w = r.apply(w)
                h = r.apply(h)

            # rotation around y
            z_v = np.cross(v,y)
            u_v = np.dot(v, u)
            u_z = np.dot(z_v, u)
            beta = np.arctan2(u_z, u_v)

            r = spt.Rotation.from_rotvec(y * beta)
            u= r.apply(u)
            w = r.apply(w)
            h = r.apply(h)

            self.dir_vector = u
            self.h_vector = h
            self.w_vector = w
            
            self._update_rotation_from_vectors()
            self._update_euler_angles()
            self._update_target_point()
            self._update_object_view_properties()
            
        if (("translate" in kargs and kargs["translate"]) or
            ("rotate" in kargs and not kargs["rotate"])):
            dir_component = np.dot(v, self.dir_vector)
            self.location += v - dir_component*self.dir_vector
            self._update_target_point()
            self._update_object_view_properties()
            
        # update visualisation objects
        self._update_all_objects()
    
    def _compute_overlap_ratio_from_B_Z(self, B_Z_ratio, direction= "width"):
        """Convert B/Z to overlap ratio"""
        if direction == "width":
            return 1 - B_Z_ratio * self.focal / self.sensor_width
        elif direction == "height":
            return 1 - B_Z_ratio * self.focal / self.sensor_height
        else:
            raise Exception("Direction not handled: "+direction)
        
    def _compute_B_Z_from_overlap_ratio(self, overlap_ratio,  direction= "width"):
        """Convert overlap ratio to B/Z"""
        if direction == "width":
            return (1 - overlap_ratio) / self.focal * self.sensor_width
        elif direction == "height":
            return (1 - overlap_ratio) / self.focal * self.sensor_height
        else:
            raise Exception("Direction not handled: "+direction)
        
    
    def compute_overlap_ratio_in_normal_case(self, B, direction= "width"):
        """Computes the overlap between two pictures in normal case
        
        Normal case is when considering two pictures taken
        with a simple lateral shift.

        :param B: the length of the lateral shift in meters
        :type B: float
        :param direction: the direction of the shift, default is width
        alternatively height can be given
        :type B: str
        """
        if direction == "width":
            return (self.image_width - B)/self.image_width
        elif direction == "height":
            return (self.image_height - B)/self.image_height
        else:
            raise Exception("Direction not handled: "+direction)
        
    def compute_resolution(self, points):
        """Computes the image resolution at the given locations
        
        Parameters:
        - points: the points where the resolution should be evaluated
        
        Returns:
        - an array (or a single value if a single point) giving
        the resolution in termes of number of pixels per millimeter
        """
        cam_coords = self._compute_camera_coordinates(points)
        return self._compute_resolution(cam_coords[:,0])
    
    def compute_sampling_distance(self, points):
        """Computes the sampling distance at the given locations
        
        Parameters:
        - points: the points where should be evaluated
        
        Returns:
        - an array (or a single value if a single point) giving
        the sampling distance in millimeters
        """
        cam_coords = self._compute_camera_coordinates(points)
        return self._compute_sampling_distance(cam_coords[:,0])
    
    def compute_defocus_diameter(self, points):
        """Computes the diameter of the blur spot due to defocus at the given locations
        
        Parameters:
        - points: the points where should be evaluated
        
        Returns:
        - an array (or a single value if a single point) giving
        the defocus diameter in micrometers
        """
        cam_coords = self._compute_camera_coordinates(points)
        return self._compute_defocus_diameter(cam_coords[:,0])
    
    def compute_diffraction_diameter(self, points):
        """Computes the diameter of the blur spot due to diffraction at the given locations
        
        Parameters:
        - points: the points where should be evaluated
        
        Returns:
        - an array (or a single value if a single point) giving
        the diffraction diameter in micrometers
        """
        cam_coords = self._compute_camera_coordinates(points)
        return self._compute_diffraction_diameter(cam_coords[:,0])
    
    def compute_blur_spot_diameter(self, points):
        """Computes the total diameter of the blur spot at the given locations
        
        Parameters:
        - points: the points where should be evaluated
        
        Returns:
        - an array (or a single value if a single point) giving
        the blur diameter in micrometers
        """
        cam_coords = self._compute_camera_coordinates(points)
        return self._compute_combined_blur_spot_diameter(cam_coords[:,0])
    
    def compute_visible_and_sharp(self, points, normals):
        """Computes if points are visible and sharp"""
        cam_coords = self._compute_camera_coordinates(points)
        zx = cam_coords[:,0] # yes z is coord 0 and not 2
        sharp = self.compute_sharpness(zx)
        
        angles = self._compute_deviation_angles(cam_coords)
        in_view = self._compute_is_in_view(angles)
        facing = self._compute_facing(points, normals)
        visible = self._compute_visible(in_view, facing)
        
        return self._compute_visible_and_sharp(visible, sharp)
    
    def _update_object_view_properties(self, object= None, set_properties= True):
        """Computes the coordinates and properties of the given object in the view
        
        Parameters:
        - object: the target object for which to make the computation.
        If None(default) the attached target object will be used.
        - set_properties: if True (default) the computed properties are set to the
        point_data properties of the object.
        """
        object = self.target_object if object is None else object
        if object is None: 
            return
        
        properties = {}
        
        cam_coords = self._compute_camera_coordinates(object.points)
        properties["cam_coords"] = cam_coords
        
        angles = self._compute_deviation_angles(cam_coords)
        properties["angles"] = angles
        
        properties["pix"] = self._compute_pixel_coordinates(cam_coords)
        facing = self._compute_facing(object.points, object.point_normals)
        properties["facing"] = facing
        
        in_view = self._compute_is_in_view(angles)
        properties["in_view"] = in_view
        properties["visible"] = self._compute_visible(in_view, facing)
        
        properties["resolution"] = self.compute_resolution(object.points)
        properties["sampling_distance"] = self.compute_sampling_distance(object.points)
        
        properties["defocus_diameter"] = self._compute_defocus_diameter(cam_coords[:,0])
        properties["diffraction_diameter"] = self._compute_diffraction_diameter(cam_coords[:,0])
        properties["blur_spot_diameter"] = self._compute_combined_blur_spot_diameter(cam_coords[:,0])
        
        properties["depth_error_std"] = self._compute_depth_reconstruction_error(cam_coords[:,0])
        
        properties["sharp"] = self._compute_is_sharp(properties["blur_spot_diameter"])
        
        properties["visible_sharp"] = self._compute_visible_and_sharp(
            properties["visible"],
            properties["sharp"]
        )
        
        if set_properties:
            
            visible = properties["visible"]
            object.point_data.set_array(visible, "visible_b")
            object.point_data.set_array(visible.astype(float), "visible")
            
            object.point_data.set_array(cam_coords[:,1], "x_all")
            object.point_data.set_array(
                np.ma.array(cam_coords[:,1], mask= ~visible, fill_value= np.nan).filled(),
                "x"
            )
            object.point_data.set_array(cam_coords[:,2], "y_all")
            object.point_data.set_array(
                np.ma.array(cam_coords[:,2], mask= ~visible, fill_value= np.nan).filled(),
                "y"
            )
            # this is actually 0 for z
            object.point_data.set_array(cam_coords[:,0], "z_all")
            object.point_data.set_array(
                np.ma.array(cam_coords[:,0], mask= ~visible, fill_value= np.nan).filled(),
                "z"
            )
            object.point_data.set_array(angles[:,0], "angle_width_all")
            object.point_data.set_array(
                np.ma.array(angles[:,0], mask= ~visible, fill_value= np.nan).filled(),
                "angle_width"
            )
            object.point_data.set_array(angles[:,1], "angle_height_all")
            object.point_data.set_array(
                np.ma.array(angles[:,1], mask= ~visible, fill_value= np.nan).filled(),
                "angle_height"
            )
            
            pix = properties["pix"]
            object.point_data.set_array(pix[:,0], "pix_i_all")
            object.point_data.set_array(
                np.ma.array(pix[:,0], mask= ~visible, fill_value= np.nan).filled(),
                "pix_i"
            )
            object.point_data.set_array(pix[:,1], "pix_j_all")
            object.point_data.set_array(
                np.ma.array(pix[:,1], mask= ~visible, fill_value= np.nan).filled(),
                "pix_j"
            )
            
            object.point_data.set_array(facing, "facing_b")
            object.point_data.set_array(facing.astype(float), "facing")
            object.point_data.set_array(in_view, "in_view_b")
            object.point_data.set_array(in_view.astype(float), "in_view")
            
            for key in ["resolution", "sampling_distance",
                        "defocus_diameter", "diffraction_diameter",
                        "blur_spot_diameter", "depth_error_std"]:
                
                # setting the value everywhere with _all
                object.point_data.set_array(properties[key].filled(np.nan), key+"_all")
                # setting the value restricted to the viewed part
                object.point_data.set_array(
                    np.ma.array(properties[key], mask= ~visible).filled(np.nan), key
                )
            
            object.point_data.set_array(properties["sharp"], "sharp_b")
            object.point_data.set_array(properties["sharp"].astype(float), "sharp")
            object.point_data.set_array(properties["visible_sharp"], "visible_sharp_b")
            object.point_data.set_array(properties["visible_sharp"].astype(float), "visible_sharp")
        else:
            return properties
        
    def get_visible_points(self, points, normals=None):
        """Filters points according to visibility
        
        Parameters:
        - points: the points to be filtered (either list or np.array of dimension (n points, 3))
        - normals: optional, the normal of the points
        
        Returns:
        - visible points, with the same shape as entry points
        """
        visibility = self._compute_point_visibility(points=points, normals=normals)
        entry_points = np.array(points).reshape((-1,3))
        return entry_points[visibility]
    
    def _compute_point_visibility(self, points, normals=None):
        """Computes the visibility of points according
        
        Parameters:
        - points: the points to be filtered (either list or np.array of dimension 3 of (n points, 3))
        - normals: optional, the normal of the points
        """
        cam_coords = self._compute_camera_coordinates(points)
        angles = self._compute_deviation_angles(cam_coords)
        in_view = self._compute_is_in_view(angles)
        if normals is not None:
            facing = self._compute_facing(points, normals)
            visible = self._compute_visible(in_view, facing)
        else:
            visible = in_view
        return visible
        
    def focus_on(self, target= None, **kargs):
        """Updates the focus distance to aim at the target
        
        Parameters:
        - target: can be either a point, a list of points, or an object,
        if None (default) the self.target_object is used.
        If a point it should be given as a list of three coordinates.
        If an object, it will focus on the object's points,
        so when the default focus method is used (min) it will be the closest point.
        Note that, to focus on the center, one can pass obj.center.
        Note: use optimize_focus instead to better focus on a larger
        part on the object.
        - kargs:
        * visible: if True (default) only the visible points are used
        * method: function to apply to the z camera coordinates of the
        target points to pick the desired focus distance (default, np.min)
        * optimize_N: by default aperture is not changed but if this parameter is True
        it is optimised to account for diffraction.
        
        Returns:
        - True if the camera successfully focus on something, False otherwise
        (e.g., no points to focus on, consider aim_at() first)
        - points used to focus on, if return_points is in the kargs, optional
        """
        assert(not(target is None and self.target_object is None)), "Warning: can't aim at target because None is attached"
        
        visible = kargs.get("visible", True)
        if isinstance(target,list) or isinstance(target, np.ndarray) or isinstance(target, tuple):
            target_points = np.array(target).reshape((-1,3))
            target_points = self.get_visible_points(target_points) if visible else target_points
        else:
            target = self.target_object if target is None else target
            target_points = self.get_visible_points(target.points, target.point_normals)  if visible else target.points
            
        return_points = kargs.get("return_points", False)
        if len(target_points) == 0:           
            if return_points:
                return False, target_points
            else:
                return False
        cam_coords = self._compute_camera_coordinates(target_points)
        
        method = kargs.get("method", np.min)
        Z_focus = method(cam_coords[:,0])
        
        self.update_focus(Z_focus,
                          optimize_N= kargs.get("optimize_N", False)
                        )
        
        if return_points:
            return True, target_points
        else:
            return True
        
    def trigger(self, 
                add_to_viewer= False,
                update_viewer= False):
        """Take a shot with the camera
        
        By default the Viewer is not updated unless update_viewer is set to True.
        If add_to_viewer is True the shot will be added to the viewer.
        """
        
        self.shots += [CameraShot(self)]
        if add_to_viewer: self.viewer.add_shot(self.shots[-1], update= update_viewer)
        if update_viewer: self.viewer.update()
        return self.get_shot()
    
    def get_shot(self, index=None):
        """returns the shot with the given index or the last by default"""
        return self.shots[index if index is not None else -1]
    
    def copy(self):
        """Creates a duplicate of the current camera
        
        This is a deep copy, i.e., all sub objects are actually duplicated."""
        avoid_copy = ["viewer"]
        cam = Camera(self.cam_id, focal=self.focal)
        cam.__dict__ = deepcopy({key:cam.__dict__[key] for key in cam.__dict__ if key not in avoid_copy})
        
        return cam
        
class CameraShot(CameraBase):
    """Defines a given instance of picture taken with a Camera
    """
    def __init__(self, cam:Camera, init_visible_part_object= True):
        """Creates a camera shot as a copy of a Camera"""
        # parameters that shouldn't be copied
        # Note: careful! copying shots was causing dramatic combinatorial issues and delays
        avoid_copy = ["viewer",
                      "shots",
                      "location_object",
                      "orientation_objects",
                      "camera_object",
                      "sensor_object",
                      "focus_plan_object",
                      "view_frame_object",
                      "sharpness_object",
                      "sharpness_object_edges",
                      "diffraction_object",
                      "diffraction_object_edges",
                      "visibility_object",
                      "visible_part_object"
                      ]
        # copy camera parameters (in depth)
        self.__dict__ = deepcopy({key:cam.__dict__[key] for key in cam.__dict__ if key not in avoid_copy})
        
        # adapt id and name
        self.camera = cam
        self.shot_id = len(cam.shots)
        self.name = cam.name + "_{:05d}".format(self.shot_id)
                        
        # compute visible part in case the object moves later
        if init_visible_part_object: self.get_visible_part_object()
        
        # increment object view properties
        self._increment_view_count()
        

    def _increment_view_count(self):
        """Increments the view counter ob object"""
        if self.target_object is None:
            return
        
        obj = self.camera.target_object
        # if needed create the counter
        if "nb_views" not in obj.point_data.keys():
            obj.point_data.set_array(0., "nb_views")
        if "nb_sharp_views" not in obj.point_data.keys():
            obj.point_data.set_array(0., "nb_sharp_views")
        if "best_resolution" not in obj.point_data.keys():
            obj.point_data.set_array(obj.point_data["resolution"], "best_resolution")
        if "best_sampling" not in obj.point_data.keys():
            obj.point_data.set_array(obj.point_data["sampling_distance"], "best_sampling")
        if "best_blur_diameter" not in obj.point_data.keys():
            obj.point_data.set_array(obj.point_data["blur_spot_diameter"], "best_blur_diameter")
        if "best_defocus_diameter" not in obj.point_data.keys():
            obj.point_data.set_array(obj.point_data["defocus_diameter"], "best_defocus_diameter")
        
        # increment
        obj.point_data.update(
            dict(
                nb_views= np.where(
                    obj.point_data["visible_b"],
                    obj.point_data["nb_views"]+1,
                    obj.point_data["nb_views"]
                ),
                nb_sharp_views= np.where(
                    obj.point_data["visible_sharp_b"],
                    obj.point_data["nb_sharp_views"]+1,
                    obj.point_data["nb_sharp_views"]
                ),
                best_resolution= np.where(
                    obj.point_data["visible_b"],
                    np.fmax(
                        obj.point_data["best_resolution"],
                        obj.point_data["resolution"]
                    ),
                    obj.point_data["best_resolution"]
                ),
                best_sampling= np.where(
                    obj.point_data["visible_b"],
                    np.fmin(
                        obj.point_data["best_sampling"],
                        obj.point_data["sampling_distance"]
                    ),
                    obj.point_data["best_sampling"]
                ),
                best_blur_diameter= np.where(
                    obj.point_data["visible_b"],
                    np.fmin(
                        obj.point_data["best_blur_diameter"],
                        obj.point_data["blur_spot_diameter"]
                    ),
                    obj.point_data["best_blur_diameter"]
                ),
                best_defocus_diameter= np.where(
                    obj.point_data["visible_b"],
                    np.fmin(
                        obj.point_data["best_defocus_diameter"],
                        obj.point_data["defocus_diameter"]
                    ),
                    obj.point_data["best_defocus_diameter"]
                )
            )
        )
        
        # increment camera views
        self.camera.views[obj] = np.concatenate((
            self.camera.views[obj],
            [obj.point_data["visible_b"]]
        ))
        
        # increment camera sharp views
        self.camera.sharp_views[obj] = np.concatenate((
            self.camera.sharp_views[obj],
            [obj.point_data["visible_sharp_b"]]
        ))
        
