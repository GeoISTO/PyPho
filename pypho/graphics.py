"""graphics.py

This module offers helper objects to visualise photogrammetric configurations
in 3D view with pyvista and plot specialized grpahs such as ResolutionGraph.

Todo:
* attach color to each camera/shot and link to color of the visible part
* add annotations
"""

from .camera import Camera, CameraShot, get_default_camera

from .target import get_pebble_dataset

import numpy as np

# for view and geometrical transformation
import pyvista as pv
import scipy.spatial.transform as spt

import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib import colors, colormaps
from matplotlib import ticker
    
#----------------------------------------------------------------------------
# Graphs

param = {}
param["grid_linewidth"] = 0.5

## Resolution graphs
def plot_resolution_graph(
        cam,
        secondary_cam = [],
        use_log_res = True,
        plot_line_width = 1.8,
        ref_line_width = 1.2,
        second_yaxis= None,
        second_axis_use_log= False,
        z_min = None,
        z_max = None,
        min_z_f_ratio = None, 
        max_z_f_ratio = 500,
        z_ref = None,
        zf_ref = None,
        z_ref_ratio = None,
        use_z_f_ratio_on_x_axis= False
    ):
    """Plot a graph to illustrate the resolution of a camera vs. distance.
    
    Refer to ResolutionGraph for te detailed parameters.
    Parameters:
    - cam: a Camera object about which the graph is plotted
    - secondary_cam (list): (optional) possible additional camera
    to be added to the graph
    
    Returns: a ResolutionGraph
    """
    graph = ResolutionGraph(
        cam= cam, secondary_cam= secondary_cam,
        use_log_res= use_log_res, 
        plot_line_width= plot_line_width,
        ref_line_width= ref_line_width,
        second_yaxis= second_yaxis, second_axis_use_log= second_axis_use_log,
        z_min= z_min, z_max= z_max,
        min_z_f_ratio= min_z_f_ratio,
        max_z_f_ratio= max_z_f_ratio,
        z_ref= z_ref,
        zf_ref = zf_ref,
        z_ref_ratio= z_ref_ratio,
        use_z_f_ratio_on_x_axis= use_z_f_ratio_on_x_axis
    )
    graph.plot()
    return graph

class ResolutionGraph(object):
    """Creates a graph showing the resolution of a camera.
    
    The plot show the resolution and/or ground sampling distance with respect
     to the focus distance (Z).
    """
    
    def __init__(self,
            cam, secondary_cam = [],
            z_min = None,
            z_max = None,
            min_z_f_ratio = None,
            max_z_f_ratio = 500,
            use_log_res = True,
            second_yaxis= "SamplingDistance",
            second_axis_use_log= False,
            plot_line_width = 1.8,
            z_ref = None,
            zf_ref = None,
            z_ref_ratio = None,
            ref_line_width = 1.2,
            use_z_f_ratio_on_x_axis= False
        ):
        """Initializes the resolution graph.

        The graph show ground sampling distance and or resolution with respect to the distance.
        
        A reference distance to get a more accurate view of the values 
        at this distance. 
        :param cam: A camera that defines the focal and sampling_distance
        :type cam: :py:class: `Camera`
        :param secondary_cam: a list of additional :py:class: `Camera`.
        General usage would be to have only one or two more cameras, defaults to []
        :type secondary_cam: list, optional
        :param use_log_res: if True, the graph is drawn for the log of the resolution, defaults to True
        :type use_log_res: bool, optional
        :param second_yaxis: type of second axis plot: "SamplingDistance" or "ImageSize", defaults to "SamplingDistance"
        :type second_yaxis: str, optional
        :param second_axis_use_log: if True, the second axis is drawn for the log of the resolution, defaults to False
        :type second_axis_use_log: bool, optional
        :param plot_line_width: width of graph lines, defaults to 2.5
        :type plot_line_width: float, optional
        :param ref_line_width: _description_, defaults to 1.2
        :type ref_line_width: float, optional
        :param z_min: _description_, defaults to None
        :type z_min: _type_, optional
        :param z_max: _description_, defaults to None
        :type z_max: _type_, optional
        :param max_z_f_ratio: _description_, defaults to 500
        :type max_z_f_ratio: int, optional
        :param z_ref: _description_, defaults to None
        :type z_ref: _type_, optional
        :param z_ref_ratio: _description_, defaults to None
        :type z_ref_ratio: _type_, optional
        :param use_z_f_ratio_on_x_axis: if True, Z/f ratios are used in x axis instead of Z
        defaults to False
        :type use_z_f_ratio_on_x_axis: bool
        """
        
        # setting cameras
        self.cam = cam
        self.secondary_cam = secondary_cam
        
        # setting graph options
        self.use_log_res = use_log_res
        self.plot_line_width = plot_line_width
        self.ref_line_width = ref_line_width
        self.use_z_f_ratio_on_x_axis = use_z_f_ratio_on_x_axis
        
        # second axis
        if second_yaxis is not None:
            assert (second_yaxis in ["SamplingDistance", "ImageSize"]), "second axis option not implemented, try SamplingDistance or ImageSize"
        self.second_yaxis = second_yaxis
        self.second_axis_use_log = second_axis_use_log
        
        # setting z and z_reference
        self._init_z(z_min= z_min, z_max= z_max, min_z_f_ratio= min_z_f_ratio, max_z_f_ratio= max_z_f_ratio)
        self.set_z_ref(z_ref= z_ref, zf_ref= zf_ref, z_ref_ratio= z_ref_ratio)

    def _init_z(self, z_min= None, z_max= None, min_z_f_ratio = None, max_z_f_ratio= 500):
        """Initializes the distance grid :py:attr: `z`

        These distance values are given in meters (hence the 1000 factor when converting from focal distances)
        :param z_min: if given, this sets the starting point of distance values.
        If not set, then the minimal focus distance is taken from the primary camera
        :py:attr: `cam.min_focus_distance`
        :type z_min: float, optional
        :param z_max: maximum distance, if this is not set then :py:param: `max_z_f_ratio`
        must be given. 
        :type z_max: float, optional
        :param max_z_f_ratio: if :py:param: `z_max is not set, then the maximum distance
        will be computed as :py: `max_z_f_ratio * cam.focal/1000`
        :type max_z_f_ratio: float, optional
        """
        if self.use_z_f_ratio_on_x_axis:
            if min_z_f_ratio is not None:
                zf_min = self.zf_0 = min_z_f_ratio 
            elif z_min is not None:
                zf_min = self.zf_0 = z_min / self.cam.focal * 1000
            else:
                zf_min = self.cam.min_focus_distance / self.cam.focal * 1000
                self.zf_0 = 0
            self.zf = np.linspace(zf_min, max_z_f_ratio, 100)
            self.z = self.zf/1000*self.cam.focal
        else:
            z_max = z_max if z_max is not None else max_z_f_ratio * self.cam.focal/1000
            if z_min is None:
                z_min = self.cam.min_focus_distance
                self.z_0 = 0
            else:
                z_min = self.z_0 = z_min
            self.z = np.linspace(z_min, z_max, 100 )
    
    def set_z_ref(self,
            z_ref= None, zf_ref= None,
            z_ref_ratio= None
            ):
        """Initializes the reference distance (in meters).

        :param use_z_ref: tells if a reference distance should be shown, defaults to True
        :type use_z_ref: bool, optional
        :param z_ref: the reference distance. If not set then :py:param: `z_ref_ratio`
        must be given, defaults to None
        :type z_ref: float, optional
        :param z_ref_ratio: ratio of the distance range, defaults to 2/3
        :type z_ref_ratio: float, optional
        """
        if self.use_z_f_ratio_on_x_axis == False:
            if z_ref is None and z_ref_ratio is None:
                self.use_z_ref = False
            else:
                self.use_z_ref = True
                if z_ref is None:
                    zmin, zmax = self.z.nanmin(), self.z.nanmax()
                    self.z_ref = zmin + z_ref_ratio * (zmax - zmin)
                else:
                    self.z_ref = z_ref
        else:
            if zf_ref is None and z_ref is None and z_ref_ratio is None:
                self.use_z_ref = False
            else:
                self.use_z_ref = True
                if zf_ref is None:
                    if z_ref is None:
                        zfmin, zfmax = self.zf.nanmin(), self.zf.nanmax()
                        self.zf_ref = zfmin + z_ref_ratio * (zfmax - zfmin)
                        self.z_ref = zf_ref/1000*self.cam.focal
                    else:
                        self.zf_ref = z_ref/self.cam.focal*1000
                        self.z_ref = z_ref
                else:
                    self.zf_ref = zf_ref
                    self.z_ref = zf_ref/1000*self.cam.focal
            
    def plot(self):
        """Generate the plot.
        
        Figure and axes are accessible as `fig`and `ax`
        """
        if self.use_z_f_ratio_on_x_axis:
            self._plot_zf()
        else:
            self._plot_z()
    
    def _plot_z(self):
        
        # getting the sampling distance from the camera
        gsd = self.cam._compute_sampling_distance(self.z)
        res = 1/gsd
        
        # creating the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Resolution versus distance')

        # creating the absices axes
        self.ax.set_xlabel(r'$Z$ (m)', loc="right")
        ax_x2 = self.ax.secondary_xaxis("top", functions=(lambda z: z/self.cam.focal*1000, lambda z_f: z_f * self.cam.focal / 1000))
        ax_x2.set_xlabel(r"$Z/f$", loc="right")
        #ax_x2.xaxis.set_label_coords(1.25, y= 1.05)

        self.ax.set_xlim(self.z_0, self.z.nanmax())

        # setting the resolution axis
        ax_res = self.ax
        if self.use_log_res: ax_res.set_yscale("log")
        ax_res.set_ylabel('Resolution (pixels/mm)')
               
        # setting the secondary axis
        if self.second_yaxis in ["SamplingDistance", "ImageSize"]:
            yax2 = self.ax.twinx()
            if self.second_axis_use_log == True:
                yax2.set_yscale("log")
            if self.second_yaxis == "ImageSize":
                yax2_label = 'Image Size (m)'
                width = self.cam.compute_image_size(self.z)
                height = self.cam.compute_image_size(self.z, direction= "height")
                yax2.set_ylabel(yax2_label)
            elif self.second_yaxis == "SamplingDistance":
                yax2_label = 'Sampling Distance (mm)'
                gsd = self.cam._compute_sampling_distance(self.z)
                yax2.set_ylabel(yax2_label)

        # gridding
        self.ax.grid(which= "both", axis= "both", linewidth= param["grid_linewidth"])

        # plotting
        handles = []
        if len(self.secondary_cam) == 0:
            res_artist, = ax_res.plot(self.z, res,"k", label= "Resolution", linewidth= self.plot_line_width, zorder= 5)
            handles += [res_artist]
            
            if self.second_yaxis in ["SamplingDistance", "ImageSize"]:
                if self.second_yaxis == "ImageSize":
                    width_artist, = yax2.plot(self.z, width, linestyle= "-", label= "Image Width (m)", linewidth= 0.5*self.plot_line_width, zorder= 5)
                    height_artist, = yax2.plot(self.z, height, linestyle= "-", label= "Image Height (m)", linewidth= 0.5*self.plot_line_width, zorder= 5)
                    handles += [width_artist, height_artist]
                elif self.second_yaxis == "SamplingDistance":
                    gsd_artist, = yax2.plot(self.z, gsd, "k", linestyle= "-", label= "Sampling Distance (mm)", linewidth= 0.5*self.plot_line_width, zorder= 5)
                    handles += [gsd_artist]
        else:
            cam_name = self.cam.cam_id+" {:0.0f} mm".format(self.cam.focal)
            res_artist, = ax_res.plot(self.z, res, label= "Resolution "+cam_name, linewidth= self.plot_line_width, zorder= 5)
            handles += [res_artist]
            
            if self.second_yaxis in ["SamplingDistance", "ImageSize"]:
                if self.second_yaxis == "ImageSize":
                    width_artist, = yax2.plot(self.z, width,  linestyle= "-", label= "Image Width "+cam_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
                    height_artist, = yax2.plot(self.z, height,  linestyle= "-", label= "Image Height "+cam_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
                    handles += [width_artist, height_artist]
                elif self.second_yaxis == "SamplingDistance":
                    gsd_artist, = yax2.plot(self.z, gsd, linestyle= "-", label= "Sampling Distance "+cam_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
                    handles += [gsd_artist]
            ax_x2.set_xlabel(r"$Z/f$ ("+cam_name+")", loc="right")
        if self.second_yaxis in ["SamplingDistance", "ImageSize"]:
            yax2.set_ylim(bottom = 0, top = None)

        # adding a reference point
        if self.use_z_ref:
            res_ref = self.cam._compute_resolution(self.z_ref)
            # draw reference point
            ax_res.scatter(self.z_ref, res_ref, c="k", zorder= 10)
            ax_res.vlines(self.z_ref, ymin= 0, ymax= res_ref, colors="k", linestyles="dotted", zorder= 1)
            ax_res.hlines(res_ref, xmin= ax_res.get_xlim()[0], xmax= self.z_ref, colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
            
            if self.second_yaxis == "ImageSize":
                width_ref = self.cam.compute_image_size(self.z_ref, direction= "width")
                yax2.vlines(self.z_ref, ymin= 0, ymax= width_ref, colors="k", linestyles="dotted", zorder= 1)
                yax2.hlines(width_ref, xmin= self.z_ref, xmax= yax2.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
                yax2.scatter(self.z_ref, width_ref, c="k", zorder= 10)
                
                height_ref = self.cam.compute_image_size(self.z_ref, direction= "height")
                yax2.vlines(self.z_ref, ymin= 0, ymax= height_ref, colors="k", linestyles="dotted", zorder= 1)
                yax2.hlines(height_ref, xmin= self.z_ref, xmax= yax2.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
                yax2.scatter(self.z_ref, height_ref, c="k", zorder= 10)
                
            elif self.second_yaxis == "SamplingDistance":
                pix_size_ref = self.cam._compute_sampling_distance(self.z_ref)
                yax2.vlines(self.z_ref, ymin= 0, ymax= pix_size_ref, colors="k", linestyles="dotted", zorder= 1)
                yax2.hlines(pix_size_ref, xmin= self.z_ref, xmax= yax2.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
                yax2.scatter(self.z_ref, pix_size_ref, c="k", zorder= 10)
                
            # legend string
            cam_name = self.cam.cam_id+" {:0.0f} mm".format(self.cam.focal)
            ref_string = f"$Z$: {self.z_ref:0.1f} m\n\n"+\
                cam_name+"\n"+r"|"+\
                " $Z/f$: {:0.0f}\n".format(
                    self.z_ref/self.cam.focal*1000
                )+\
                r"|"+" Resolution: {:0.1f} p/mm\n".format(res_ref)
            
            if self.second_yaxis == "ImageSize":
                ref_string += r"|"+" Image width: {:.2f} m\n".format(width_ref)+\
                    r"|"+" Image height: {:.2f} m".format(height_ref)
            elif self.second_yaxis == "SamplingDistance":
                ref_string += r"|"+" Sampling Distance: {:.2f} mm".format(
                    pix_size_ref
                )

        if self.second_yaxis == "ImageSize" and len(self.secondary_cam):
            raise Exception("Image dimensions as second axis is only supported without additional camera.")
        else:
            # additional camera
            for i,cam2 in enumerate(self.secondary_cam):

                cami_name = cam2.cam_id+" {:0.0f} mm".format(cam2.focal)
                ax_x2_cam2 = self.ax.secondary_xaxis(1 + 0.15*(i+1), functions=(lambda z: z/cam2.focal*1000, lambda z_f: z_f * cam2.focal / 1000))
                ax_x2_cam2.set_xlabel(r"$Z/f$ ("+cami_name+")", loc="right")
                    
                z2 = np.linspace(max(self.z.nanmin(), cam2.min_focus_distance), self.z.nanmax(), 100 )
                pix_size2 = cam2._compute_sampling_distance(z2)
                res2 = 1/pix_size2

                res_artist_i, = ax_res.plot(z2, res2, label= "Resolution "+cami_name, linewidth= self.plot_line_width, zorder= 5)
                pix_size_artist_i, = yax2.plot(z2, pix_size2, linestyle= "-", label= "Sampling Distance "+cami_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
                handles += [res_artist_i, pix_size_artist_i]
                
                if self.use_z_ref: 
                    res_refi = cam2._compute_resolution(self.z_ref)
                    pix_size_refi = cam2._compute_sampling_distance(self.z_ref)
                    ref_string += "\n\n"+cami_name+"\n"+r"|"+" Z/f: {:0.0f}\n"+r"|"+" Resolution: {:0.1f} p/mm\n"+r"|"+" Sampling Distance: {:.2f} mm".format(
                        self.z_ref/cam2.focal*1000,
                        res_refi, pix_size_refi
                    )
                    ax_res.scatter(self.z_ref, res_refi, c="k", zorder= 10)
                    yax2.scatter(self.z_ref, pix_size_refi, c="k", zorder= 10)
                    ax_res.vlines(self.z_ref, ymin= 0, ymax= res_refi, colors="k", linestyles="dotted", zorder= 1)
                    yax2.vlines(self.z_ref, ymin= 0, ymax= pix_size_refi, colors="k", linestyles="dotted", zorder= 1)
                    ax_res.hlines(res_refi, xmin= ax_res.get_xlim()[0], xmax= self.z_ref, colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
                    yax2.hlines(pix_size_refi, xmin= self.z_ref, xmax= yax2.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)

            
        # legend
        ax_res.legend(handles= handles, loc= "upper left", bbox_to_anchor= (1.1,1) )
        if self.use_z_ref:
            self.ax.text(
                1.16, 0.7,
                ref_string,
                transform= self.ax.transAxes,
                verticalalignment= "top",
                bbox= dict(edgecolor="gray", facecolor="white") 
            )
            
    def _plot_zf(self):
        
        # getting the sampling distance from the camera
        gsd = self.cam._compute_sampling_distance(self.z)
        res = 1/gsd
        
        # creating the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Resolution versus distance')

        # creating the absices axes
        self.ax.set_xlabel(r'$Z/f$', loc="right")
        ax_x2 = self.ax.secondary_xaxis("top", functions=(lambda zf: zf*self.cam.focal/1000, lambda z: z /self.cam.focal * 1000))
        ax_x2.set_xlabel(r"$Z$ (m)", loc="right")

        self.ax.set_xlim(self.zf_0, self.zf.nanmax())

        # setting the resolution axis
        ax_res = self.ax
        if self.use_log_res: ax_res.set_yscale("log")
        ax_res.set_ylabel('Resolution (pixels/mm)')

        # setting the sampling_distance axis
        ax_pix_size = self.ax.twinx()
        ax_pix_size.set_ylabel('Sampling Distance (mm)')

        # gridding
        self.ax.grid(which= "both", axis= "both", linewidth= param["grid_linewidth"])

        # plotting
        if len(self.secondary_cam) == 0:
            res_artist, = ax_res.plot(self.zf, res,"k", label= "Resolution", linewidth= self.plot_line_width, zorder= 5)
            pix_size_artist, = ax_pix_size.plot(self.zf, gsd,"k", linestyle= "-", label= "Sampling Distance", linewidth= 0.5*self.plot_line_width, zorder= 5)
        else:
            cam_name = self.cam.cam_id+" {:0.0f} mm".format(self.cam.focal)
            res_artist, = ax_res.plot(self.zf, res,"k", label= "Resolution "+cam_name, linewidth= self.plot_line_width, zorder= 5)
            pix_size_artist, = ax_pix_size.plot(self.zf, gsd,"k", linestyle= "-", label= "Sampling Distance "+cam_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
            ax_x2.set_xlabel(r"$Z$ (m) ("+cam_name+")", loc="right")
        handles = [res_artist, pix_size_artist]
        ax_pix_size.set_ylim(bottom = 0, top = None)

        # adding a reference point
        if self.use_z_ref:
            res_ref = self.cam._compute_resolution(self.z_ref)
            pix_size_ref = self.cam._compute_sampling_distance(self.z_ref)

            # draw reference point
            ax_res.scatter(self.zf_ref, res_ref, c="k", zorder= 10)
            ax_pix_size.scatter(self.zf_ref, pix_size_ref, c="k", zorder= 10)
            ax_res.vlines(self.zf_ref, ymin= 0, ymax= res_ref, colors="k", linestyles="dotted", zorder= 1)
            ax_pix_size.vlines(self.zf_ref, ymin= 0, ymax= pix_size_ref, colors="k", linestyles="dotted", zorder= 1)
            ax_res.hlines(res_ref, xmin= ax_res.get_xlim()[0], xmax= self.zf_ref, colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
            ax_pix_size.hlines(pix_size_ref, xmin= self.zf_ref, xmax= ax_pix_size.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)

            # legend string
            cam_name = self.cam.cam_id+" {:0.0f} mm".format(self.cam.focal)
            ref_string = f"Z/f: {self.zf_ref:0.0f}\n\n"+cam_name+"\n"+r"|"+" Z (m): {:0.1f}\n"+r"|"+" Resolution: {:0.1f} p/mm\n"+r"|"+" Sampling Distance: {:.2f} mm".format(
                self.zf_ref*self.cam.focal/1000,
                res_ref, pix_size_ref
            )

        # additional camera
        for i,cam2 in enumerate(self.secondary_cam):

            cami_name = cam2.cam_id+" {:0.0f} mm".format(cam2.focal)
            ax_x2_cam2 = self.ax.secondary_xaxis(1 + 0.15*(i+1), functions=(lambda z_f: z_f*cam2.focal/1000, lambda z: z / cam2.focal * 1000))
            ax_x2_cam2.set_xlabel(r"$Z$ (m) ("+cami_name+")", loc="right")
                
            zf2 = np.linspace(max(self.zf.nanmin(), cam2.min_focus_distance/cam2.focal*1000), self.zf.nanmax(), 100 )
            z2 = zf2*cam2.focal/1000
            pix_size2 = cam2._compute_sampling_distance(z2)
            res2 = 1/pix_size2

            res_artist_i, = ax_res.plot(zf2, res2,"k", label= "Resolution "+cami_name, linewidth= self.plot_line_width, zorder= 5)
            pix_size_artist_i, = ax_pix_size.plot(zf2, pix_size2,"k", linestyle= "-", label= "Sampling Distance "+cami_name, linewidth= 0.5*self.plot_line_width, zorder= 5)
            handles += [res_artist_i, pix_size_artist_i]
            
            if self.use_z_ref: 
                z_refi = self.zf_ref*cam2.focal/1000
                res_refi = cam2._compute_resolution(z_refi)
                pix_size_refi = cam2._compute_sampling_distance(z_refi)
                ref_string += "\n\n"+cami_name+"\n"+r"|"+" Z (m): {:0.1f}\n"+r"|"+" Resolution: {:0.1f} p/mm\n"+r"|"+" Sampling Distance: {:.2f} mm".format(
                    z_refi,
                    res_refi, pix_size_refi
                )
                ax_res.scatter(self.zf_ref, res_refi, c="k", zorder= 10)
                ax_pix_size.scatter(self.zf_ref, pix_size_refi, c="k", zorder= 10)
                ax_res.vlines(self.zf_ref, ymin= 0, ymax= res_refi, colors="k", linestyles="dotted", zorder= 1)
                ax_pix_size.vlines(self.zf_ref, ymin= 0, ymax= pix_size_refi, colors="k", linestyles="dotted", zorder= 1)
                ax_res.hlines(res_refi, xmin= ax_res.get_xlim()[0], xmax= self.zf_ref, colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)
                ax_pix_size.hlines(pix_size_refi, xmin= self.zf_ref, xmax= ax_pix_size.get_xlim()[1], colors="k", linestyles="dotted", linewidth= self.ref_line_width, zorder= 2)

            
        # legend
        ax_res.legend(handles= handles, loc= "upper left", bbox_to_anchor= (1.1,1) )
        if self.use_z_ref:
            self.ax.text(
                1.16, 0.7,
                ref_string,
                transform= self.ax.transAxes,
                verticalalignment= "top",
                bbox= dict(edgecolor="gray", facecolor="white") 
            )

## Depth precision graphs
class DepthPrecisionGraph(object):
    """Depth precision graph for a given focus
    
    This graph shows the depth precision, 
    or in fact the depth reconstruction error, i.e., the uncertainty of the distance
    of a point by photogrammetry with a pair of pictures in normal case
    (i.e., with a simple lateral shift of a given distance B, aka. baseline).
    
    The graph can be shown for several several B/Z ratios or overlap ratios,
    as the depth precision is inversely propertionnal to this ratio.
    """
    def __init__(self,
                 cam: Camera,
                 Z: float= None,
                 B_Z_ratio = None,
                 overlap_ratio= None,
                 B= None,
                 xaxis= "Z",
                 show_second_xaxis= True,
                 second_yaxis= "Resolution",
                 z_min = None,
                 z_max = None,
                 z_padding_ratio= 0.1,
                 plot_line_width = 1.8,
                 set_title= True,
                 use_log= False,
                 plot_Z= True,
                 image_std= 1/3,
                 y_text= None,
                 ax= None
                ):
        """Creates the graph

        :param cam: the camera used for defining the graph
        :type cam: Camera
        :param Z: the focus distance (meters), if None, cam.Z is used
        :type Z: float, optional
        :param B_Z_ratio: the value to consider for the baseline (ie. lateral displacement of the camera)
        given as a constant B/Z ratio,
        if None default, :py:param: `overlap_ratio` will be used and converted,
        if also None, a constant baseline :py:param: `B` will be used if given,
        otherwise :py:param: `B_Z_ratio` is 1/5.
        Alternatively, a list of values can be given and several lines will be drawn.
        :type B_Z_ratio: float or list(float), optional
        :param overlap_ratio: alternative to :py:param: `B_Z_ratio` giving the overlap ratio between
        two pictures
        :type overlap_ratio: float or list(float), optional
        :param B: the value to consider for the baseline (ie. lateral displacement of the camera).
        It is only used if :py:param:`B_Z_ratio` and :py:param:`overlap_ratio` are None.
        If neither :py:param:`B_Z_ratio`, :py:param:`overlap_ratio`, nor :py:param:`B` are given, then B/Z is 1/5
        :type B: float or list(float), optional
        :param xaxis: tells what to use as x axis, default "Z", otherwise "Z/f"
        :type xaxis: string, optional
        :param show_second_xaxis: if True (default), the other option for xaxis is shown above the graph
        :type show_second_xaxis: bool, optional
        :param second_yaxis: tells what to show as a secondary axis,
        defaults to "Resolution" to show the picture resolution,
        otherwise, "SamplingDistance" to show the Sampling Distance,
        or None or "" to avoid using secondary Y axis.
        :type second_yaxis: string, optional
        :param z_min: _description_, defaults to None
        :type z_min: _type_, optional
        :param z_max: _description_, defaults to None
        :type z_max: _type_, optional
        :param image_std: standard deviation of the error on the sensor, in ratio of the pixel pitch, default 1/3 after Wenzel et al.
        :type image_std: float, optional
        :param plot_Z: if True (default) vertical lines are plotted to shown where Z, z1 and z2 fall
        :type plot_Z: bool, optional
        """        
        self.ax = ax
        self.cam = cam
        if Z is not None:
            cam.update_focus(Z)
        self.xaxis = xaxis
        self.show_second_xaxis = show_second_xaxis
        self.show_second_xaxis = show_second_xaxis 
        self.second_yaxis = second_yaxis
        self.use_log = use_log
        self.plot_Z = plot_Z
        self.image_std = image_std
        self.y_text = y_text
        
        self.set_title = set_title
        self.plot_line_width = plot_line_width
        
        # setting z and z_reference
        self._init_z(z_min= z_min, z_max= z_max, z_padding_ratio= z_padding_ratio)
        
        # at least one of B, B_Z_ratio or overlapratio must be given
        assert (not(B is None and B_Z_ratio is None and overlap_ratio is None)), "at least one of B, B_Z_ratio or overlapratio must be given"
        if B is not None:
            self.B = np.array([B]).ravel()
            self.B_Z_ratio = self.B / self.cam.Z 
            self.overlap_ratio = self.cam.compute_overlap_ratio_in_normal_case(self.B)
        elif B_Z_ratio is not None:
            self.B_Z_ratio = np.array([B_Z_ratio]).ravel()
            self.B = self.B_Z_ratio * self.cam.Z 
            self.overlap_ratio = self.cam.compute_overlap_ratio_in_normal_case(self.B)
        elif overlap_ratio is not None:
            self.overlap_ratio = np.array([overlap_ratio]).ravel()
            self.B_Z_ratio = self.cam._compute_B_Z_from_overlap_ratio(self.overlap_ratio)
            self.B = self.B_Z_ratio * self.cam.Z         

    def _init_z(self, z_min= None, z_max= None, z_padding_ratio= 0.1):
        """Initializes the distance grid :py:attr: `z`

        These distance values are given in meters (hence the 1000 factor when converting from focal distances)
        (this is different from Z the focus, its z the distance)
        :param z_min: if given, this sets the starting point of distance values.
        If not set, then the minimal focus distance is taken from the primary camera
        :py:attr: `cam.min_focus_distance`
        :type z_min: float, optional
        :param z_max: maximum distance, if this is not set then :py:param: `max_z_f_ratio`
        must be given. 
        :type z_max: float, optional
        :param z_padding_ratio: z goes from cam.Z1 to cam.Z2, but an extra margin is used given by this parameter (0.1 by default)
        :type z_padding_ratio: float, optional
        :param max_z_f_ratio: if :py:param: `z_max is not set, then the maximum distance
        will be computed as :py: `max_z_f_ratio * cam.focal/1000`
        :type max_z_f_ratio: float, optional
        """
        z1 = self.cam.Z1
        z2 = self.cam.Z2
        Z = self.cam.Z
    
        z_max = z_max if z_max is not None else z2 + z_padding_ratio * (z2 - Z)
        z_min = z_min if z_min is not None else z1 + z_padding_ratio * (z1 - Z)
        self.z = np.linspace(z_min, z_max, 100 )
        self.zf = self.z / self.cam.focal * 1000
        
    def rescale_yaxis(self):
                
        ticks = self.ax.get_yticks()
        scale_pow = -int(round(np.log10(ticks[-1]),0))
        if scale_pow == 2:
            scaling = 100
            unit = "cm"
        elif scale_pow == 3:
            scaling = 1000
            unit = "mm"
        elif scale_pow == 6:
            scaling = 1000000
            unit = "\mu m"
        else:
            scaling = 1
            unit = "m"
        self.ax.set_yticks(scaling * ticks)
        for line_i in self.ax.lines:
            line_i.set_ydata(
                line_i.get_ydata() * scaling
            )
        self.ax.set_ylabel('Depth reconstruction error $\sigma_z$ ('+unit+')')
        
            
    def plot(self):
        """Generates the depth precision plot.
        """
        
        # creating the figure
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
            if self.set_title: self.ax.set_title('Depth reconstruction error graph')
        else:
            self.fig = self.ax.get_figure()
        
        # setting the depth precision axis
        if self.use_log: self.ax.set_yscale("log")
                
        # creating the absices axes
        if self.xaxis == "Z":
            x_val = self.z
            self.ax.set_xlabel("$Z$ (m)")
            if self.show_second_xaxis:
                ax_x2 = self.ax.secondary_xaxis("top", functions=(
                    lambda z: z /self.cam.focal * 1000,
                    lambda zf: zf*self.cam.focal/1000)
                    )
                
                ax_x2.set_xlabel(r'$Z/f$', loc="right")
            self.ax.set_xlim(self.z.nanmin(), self.z.nanmax())
            
        else:
            x_val = self.zf
            self.ax.set_xlabel(r'$Z/f$', loc="right")
            if self.show_second_xaxis:
                ax_x2 = self.ax.secondary_xaxis("top", functions=(
                    lambda zf: zf*self.cam.focal/1000,
                    lambda z: z /self.cam.focal * 1000)
                    )
                ax_x2.set_xlabel(r"$Z$ (m)", loc="right")
            self.ax.set_xlim(self.zf.nanmin(), self.zf.nanmax())
            
        # compute depth precision
        for B, B_Z, tau in zip(self.B, self.B_Z_ratio, self.overlap_ratio):
                sigma_z = self.cam._compute_depth_reconstruction_error(self.z, baseline= B, image_std= self.image_std)
                label_base = "$B$: {:.2f} m".format(B)
                if 1/B_Z % 1 == 0:
                    label = label_base+", $B/Z: 1/{:d}$".format(int(1/B_Z))
                else:
                    label = label_base+", $B/Z: {:0.2f}$".format(B_Z)
                label += ", Overlap: {:2.0f} %".format(100 * tau)
                self.ax.plot(x_val, sigma_z, 
                             linestyle= "-", 
                             label= label, 
                             linewidth= self.plot_line_width, 
                             zorder= 5)
        
        self.rescale_yaxis()
                
        # setting the secondary axis
        if self.second_yaxis in ["Resolution", "SamplingDistance"]:
            yax2 = self.ax.twinx()
            if self.second_yaxis == "Resolution":
                label = 'Resolution (p/mm)'
                yax2.set_ylabel(label)
                # yax2.set_yscale("log")
                res = self.cam._compute_resolution(self.z)
                yax2.plot(x_val, res, 
                          color = "k", linestyle= "-",
                          label= label, linewidth= 0.5*self.plot_line_width, zorder= 1)
            else:
                label = 'Sampling Distance (mm)'
                yax2.set_ylabel(label)
                yax2.set_yscale("linear")
                pix = self.cam._compute_sampling_distance(self.z)
                yax2.plot(x_val, pix, 
                          color = "k", linestyle= "-",
                          label= label, linewidth= 0.5*self.plot_line_width, zorder= 1)
            
        # gridding
        self.ax.grid(which= "both", axis= "both", linewidth= param["grid_linewidth"])
        
        # vertical lines
        if self.plot_Z:
            ymin, ymax = self.ax.get_ylim()
            if self.use_log is False: self.ax.set_ylim((ymin, ymax))
            ytext = self.y_text if self.y_text else ymin + 0.015 *(ymax - ymin) if self.use_log is False else ymin + 0.0075* (ymax - ymin)
            zrange = self.cam.Z2 - self.cam.Z1
            self.ax.vlines(self.cam.Z, ymin, ymax, "k","-", linewidth= 0.5 * self.plot_line_width )
            self.ax.vlines(self.cam.Z1, ymin, ymax, "k","-", linewidth= 0.5 * self.plot_line_width )
            self.ax.vlines(self.cam.Z2, ymin, ymax, "k","-", linewidth= 0.5 * self.plot_line_width )
            self.ax.annotate("$Z$",(self.cam.Z, ytext), (self.cam.Z + 0.025 * zrange, ytext))
            self.ax.annotate("$Z_1$",(self.cam.Z1, ytext), (self.cam.Z1 + 0.025 * zrange, ytext))
            self.ax.annotate("$Z_2$",(self.cam.Z2,ytext), (self.cam.Z2+ 0.025 * zrange, ytext))
        
        # legend
        self.ax.legend(handles= self.ax.lines + yax2.lines,
                       loc= "upper left", bbox_to_anchor= [1.1, 1])
          
class DepthPrecisionGraphVSFocus(object):
    """Depth precision graph with respect to focus

    This graph shows the depth precision, i.e., the uncertainty of the distance
    of a point by photogrammetry with a pair of pictures in normal case
    (i.e., with a simple lateral shift of a given distance B, aka. baseline).
    
    The graph can be shown for several baseline distances B,
    or several B/Z ratios, as the depth precision is inversely propertionnal to this ratio.
    """
    def __init__(self,
                 cam: Camera,
                 B_Z_ratio = None,
                 B= None,
                 xaxis= "Z",
                 show_second_xaxis= True,
                 second_yaxis= "Resolution",
                 z_min = None,
                 z_max = None,
                 min_z_f_ratio= None,
                 max_z_f_ratio = 500,
                 plot_line_width = 1.8,
                 set_title= True,
                 use_log= True,
                 second_axis_use_log= None,
                 rescale_y= True,
                 ax= None
                ):
        """Creates the graph

        :param cam: the camera used for defining the graph
        :type cam: Camera
        :param B_Z_ratio: the value to consider for the baseline (ie. lateral displacement of the camera)
        given as a constant B/Z ratio,
        if None default, a constant baseline :py:param: `B` will be used if given,
        otherwise :py:param: `B_Z_ratio` is 1/5.
        Alternatively, a list of values can be given and several lines will be drawn.
        :type B_Z_ratio: float or list(float), optional
        :param B: the value to consider for the baseline (ie. lateral displacement of the camera).
        It is only sed if :py:param:`B_Z_ratio` is None.
        Alternatively, a list of values can be given and several lines will be drawn.
        If neither :py:param:`B_Z_ratio` nor :py:param:`B`is given, then B/Z is 1/5
        :type B: float or list(float), optional
        :param xaxis: tells what to use as x axis, default "Z", otherwise "Z/f"
        :type xaxis: string, optional
        :param show_second_xaxis: if True (default), the other option for xaxis is shown above the graph
        :type show_second_xaxis: bool, optional
        :param second_yaxis: tells what to show as a secondary axis,
        defaults to "Resolution" to show the picture resolution,
        otherwise, "SamplingDistance" to show the Sampling Distance,
        or None or "" to avoid using secondary Y axis.
        :type second_yaxis: string, optional
        :param z_min: _description_, defaults to None
        :type z_min: _type_, optional
        :param z_max: _description_, defaults to None
        :type z_max: _type_, optional
        :param max_z_f_ratio: _description_, defaults to None
        :type max_z_f_ratio: int, optional
        :param max_z_f_ratio: _description_, defaults to 500
        :type max_z_f_ratio: int, optional
        """        
        self.ax = ax
        self.cam = cam
        self.xaxis = xaxis
        self.show_second_xaxis = show_second_xaxis
        self.show_second_xaxis = show_second_xaxis 
        self.second_yaxis = second_yaxis
        self.use_log = use_log
        self.rescale_y = rescale_y
        self.second_axis_use_log = second_axis_use_log
        
        self.set_title = set_title
        self.plot_line_width = plot_line_width
        
        if B_Z_ratio is not None:
            self.B_Z_ratio = np.array(B_Z_ratio)
            self.B = None
        elif B is not None:
            self.B = np.array(B)
            self.B_Z_ratio = None
        else:
            self.B_Z_ratio = 1/5
            self.B = self.B_Z_ratio * self.cam.Z
        
        # setting z and z_reference
        self._init_z(z_min= z_min, z_max= z_max, min_z_f_ratio= min_z_f_ratio, max_z_f_ratio= max_z_f_ratio)

    def _init_z(self, z_min= None, z_max= None, min_z_f_ratio = None, max_z_f_ratio= 500):
        """Initializes the distance grid :py:attr: `z`

        These distance values are given in meters (hence the 1000 factor when converting from focal distances)
        :param z_min: if given, this sets the starting point of distance values.
        If not set, then the minimal focus distance is taken from the primary camera
        :py:attr: `cam.min_focus_distance`
        :type z_min: float, optional
        :param z_max: maximum distance, if this is not set then :py:param: `max_z_f_ratio`
        must be given. 
        :type z_max: float, optional
        :param min_z_f_ratio: if the x axis is Z/F, this is used as a strating value if given
        :type min_z_f_ratio: float, optional
        :param max_z_f_ratio: if :py:param: `z_max is not set, then the maximum distance
        will be computed as :py: `max_z_f_ratio * cam.focal/1000`
        :type max_z_f_ratio: float, optional
        """
        if self.xaxis == "Z/f":
            if min_z_f_ratio is not None:
                zf_min = self.zf_0 = min_z_f_ratio 
            elif z_min is not None:
                zf_min = self.zf_0 = z_min / self.cam.focal * 1000
            else:
                zf_min = self.cam.min_focus_distance / self.cam.focal * 1000
                self.zf_0 = 0
            self.zf = np.linspace(zf_min, max_z_f_ratio, 100)
            self.z = self.zf / 1000 * self.cam.focal
        else:
            z_max = z_max if z_max is not None else max_z_f_ratio * self.cam.focal/1000
            if z_min is None:
                z_min = self.cam.min_focus_distance
                self.z_0 = 0
            else:
                z_min = self.z_0 = z_min
            self.z = np.linspace(z_min, z_max, 100 )
            self.zf = self.z / self.cam.focal * 1000
            
    def rescale_yaxis(self):
                
        ticks = self.ax.get_yticks()
        max_y = max([line_i.get_ydata()[-1] for line_i in self.ax.lines])
        
        scale_pow = 1 - int(round(np.log10(max_y),0))
        if scale_pow == 2:
            scaling = 100
            unit = "cm"
        elif scale_pow == 3:
            scaling = 1000
            unit = "mm"
        elif scale_pow == 6:
            scaling = 1000000
            unit = "\mu m"
        else:
            scaling = 1
            unit = "m"
            
        self.ax.set_ylim(np.array(self.ax.get_ylim()) * scaling)
        new_ticks = scaling * ticks
        self.ax.set_yticks(new_ticks[1:-1])
        for line_i in self.ax.lines:
            line_i.set_ydata(
                line_i.get_ydata() * scaling
            )
        self.ax.set_ylabel('Depth reconstruction error $\sigma_z$ ('+unit+')')
        
        
            
    def plot(self):
        """Generates the depth precision plot.
        """
        
        # creating the figure
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
            if self.set_title: self.ax.set_title('Depth precision graph')
        else:
            self.fig = self.ax.get_figure()
        
        # setting the depth precision axis
        if self.use_log: self.ax.set_yscale("log")
        
        label_base = ""
        
        # creating the absices axes
        if self.xaxis == "Z":
            x_val = self.z
            self.ax.set_xlabel("$Z$ (m)")
            if self.show_second_xaxis:
                ax_x2 = self.ax.secondary_xaxis("top", functions=(
                    lambda z: z /self.cam.focal * 1000,
                    lambda zf: zf*self.cam.focal/1000)
                    )
                
                ax_x2.set_xlabel(r'$Z/f$', loc="right")
            self.ax.set_xlim(self.z_0, self.z.nanmax())
            
        else:
            x_val = self.z_f
            self.ax.set_xlabel(r'$Z/f$', loc="right")
            if self.show_second_xaxis:
                ax_x2 = self.ax.secondary_xaxis("top", functions=(
                    lambda zf: zf*self.cam.focal/1000,
                    lambda z: z /self.cam.focal * 1000)
                    )
                ax_x2.set_xlabel(r"$Z$ (m)", loc="right")
            self.ax.set_xlim(self.zf_0, self.zf.nanmax())
            
        # compute depth precision
        if self.B_Z_ratio is not None:
            if isinstance(self.B_Z_ratio, float):
                sigma_z = [self.cam._compute_depth_reconstruction_error(z_i, z_i * self.B_Z_ratio) for z_i in self.z]
                if 1/self.B_Z_ratio % 1 == 0:
                    label = label_base+"$B/Z: 1/{:d}$".format(int(1/self.B_Z_ratio))
                else:
                    label = label_base+"$B/Z: {:0.2f}$".format(self.B_Z_ratio)
                    
                overlap = np.round(100 * self.cam._compute_overlap_ratio_from_B_Z(self.B_Z_ratio),0)
                label += " | Overlap: {:2.0f} %".format(overlap)
                self.ax.plot(x_val, sigma_z, 
                             color = None if len(self.secondary_cam)>0 else "k", 
                             linestyle= "-", 
                             label= label, 
                             linewidth= self.plot_line_width, 
                             zorder= 5)
            else:
                for b_z_i in self.B_Z_ratio:
                    sigma_z = [self.cam._compute_depth_reconstruction_error(z_i, z_i * b_z_i) for z_i in self.z]
                    if 1/b_z_i % 1 == 0:
                        label = label_base+"$B/Z: 1/{:d}$".format(int(1/b_z_i))
                    else:
                        label = label_base+"$B/Z: {:0.2f}$".format(b_z_i)
                    overlap = np.round(100 * self.cam._compute_overlap_ratio_from_B_Z(b_z_i),0)
                    label += " | Overlap: {:2.0f} %".format(overlap)
                    self.ax.plot(x_val, sigma_z,
                             linestyle= "-", 
                             label= label,
                             linewidth= self.plot_line_width,
                             zorder= 5)
        else:
            if isinstance(self.B, float):
                sigma_z = self.cam._compute_depth_reconstruction_error(self.z, self.B)
                if self.B < 0.1:
                    label = label_base+"$B: {:.2f}$ mm".format(self.B * 1000)
                else:
                    label = label_base+"$B: {:.2f}$ m".format(self.B)
                self.ax.plot(x_val, sigma_z,
                             color = None if len(self.secondary_cam)>0 else "k", 
                             linestyle= "-", 
                             label= label, 
                             linewidth= self.plot_line_width, 
                             zorder= 5)
            else:
                for b_i in self.B:
                    sigma_z = self.cam._compute_depth_reconstruction_error(self.z, b_i)
                    if b_i < 0.1:
                        label = label_base+"$B: {:.2f}$ mm".format(b_i * 1000)
                    else:
                        label = label_base+"$B: {:.2f}$ m".format(b_i)
                    self.ax.plot(x_val, sigma_z, 
                             linestyle= "-", 
                             label= label,
                             linewidth= self.plot_line_width,
                             zorder= 5)
                    
        
        self.ax.set_ylabel('Depth reconstruction error $\sigma_z$ (m)')
        if self.rescale_y:
            self.rescale_yaxis()
                
        # setting the secondary axis
        if self.second_yaxis in ["Resolution", "SamplingDistance", "ImageWidth"]:
            yax2 = self.ax.twinx()
            if self.second_yaxis == "Resolution":
                label = label_base+'Resolution (p/mm)'
                yax2.set_ylabel(label)
                if self.second_axis_use_log is None or self.second_axis_use_log == True:
                    yax2.set_yscale("log")
                res = self.cam._compute_resolution(self.z)
                yax2.plot(x_val, res,
                          label= label, linewidth= 0.5*self.plot_line_width, zorder= 1)
            elif self.second_yaxis == "ImageWidth":
                label = 'Image Size (m)'
                yax2.set_ylabel(label)
                if self.second_axis_use_log is not None and self.second_axis_use_log == True:
                    yax2.set_yscale("log")
                width = self.cam.compute_image_size(self.z)
                yax2.plot(x_val, width, 
                          label= "Image Width (m)", linewidth= 0.5*self.plot_line_width, zorder= 1)
                
                height = self.cam.compute_image_size(self.z, direction= "height")
                yax2.plot(x_val, height, 
                          label= "Image Height (m)", linewidth= 0.5*self.plot_line_width, zorder= 1)
            elif self.second_yaxis == "SamplingDistance":
                label = label_base+'Sampling Distance (mm)'
                yax2.set_ylabel(label)
                if self.second_axis_use_log is not None and self.second_axis_use_log == True:
                    yax2.set_yscale("log")
                pix = self.cam._compute_sampling_distance(self.z)
                yax2.plot(x_val, pix, 
                          label= label, linewidth= 0.5*self.plot_line_width, zorder= 1)

        # gridding
        self.ax.grid(which= "both", axis= "both", linewidth= param["grid_linewidth"])
        
        # legend
        self.ax.legend(handles= self.ax.lines + yax2.lines,
                       loc= "upper left", bbox_to_anchor= [1.1, 1])
      
# depth of filed versus aperture
def plot_depth_of_field_vs_aperture(cam, 
            N= Camera.get_aperture_number_list(9),
            wavelength= None, effective_diffraction_disk_ratio= None, ax= None
        ):
    
    if wavelength is not None:
        cam.update_reference_light_wavelength(wavelength)
    if effective_diffraction_disk_ratio is not None:
        cam.update_effective_diffraction_disk_ratio(effective_diffraction_disk_ratio)
    
    ax = ax if ax is not None else plt.gca()
    
    N_array = np.linspace(1,N[-1], 200)
    dof = []
    for N_i in N_array:
        cam.update_aperture(N_i)
        dof += [cam._compute_depth_of_field_with_diffraction()]
        
    # scale
    scaling = 1
    if max(dof) < 0.5:
        scaling = 100
        unit = "cm"
        if max(dof) < 0.01:
            scaling = 1000
            unit = "mm"
            
    ax.plot(N_array, scaling * np.array(dof), "k")
            
    dof = []
    for N_i in N:
        cam.update_aperture(N_i)
        dof_i = cam._compute_depth_of_field_with_diffraction()
        dof += [scaling * dof_i]
        
    ax.scatter(N, dof, c="k")

    dx, dy = 0.8, max(dof)/70
    for i, N_i in enumerate(N):
        ax.annotate(f"{N_i:.0f}" if N_i%1==0 else f"{N_i:.1f}", (N_i, dof[i]),
                    xytext= (N_i - (0.5 if N_i%1==0 else 1)*dx \
                        + (1 if i>0 and dof[i]<dof[i-1] else 0),
                        dof[i] + dy))
    
    ax.set_title("Depth of field with respect to Aperture")
    ax.set_xlabel("Aperture (N)")
    ax.set_ylabel(f"Depth of field ({unit})")
    
    return ax.get_figure()
    
# Sharpness Zone vs Aperture
def plot_sharpness_zone_vs_aperture(cam, N_0=1, N_max= 11, Z_list= None,
                                    effective_part_list= [1,0.6, 1/2],
                                    wavelength= 546):
    """Draws a graph of sharpness zone

    :param cam: a camera
    :type cam: Camera
    :param N_0: minimal aperture, defaults to 1
    :type N_0: int, optional
    :param N_max: maximal aperture, defaults to 11
    :type N_max: int, optional
    :param Z_list: focus distance (in meter), a subplot is created fo each,
    defaults to None and if None uses the camera.Z
    :type Z_list: _type_, optional
    :param effective_part_list: effective part of Airy disk for diffraction, defaults are [1,0.6, 1/2]
    :type effective_part_list: _type_, optional
    :param wavelength: the wavelength of the light color in nm, default 546 nm
    :type wavelength: float
    :return: a SharpnessZoneVsApertureGraph
    :rtype: SharpnessZoneVsApertureGraph
    """
    graph = SharpnessZoneVsApertureGraph(cam=cam, N_0= N_0, N_max= N_max,
                    Z_list= Z_list, effective_part_list= effective_part_list,
                    wavelength= wavelength)
    graph.plot()
    return graph

class SharpnessZoneVsApertureGraph(object):
    """Draws a graph of sharpness zone"""  
    
    def __init__(self, cam, N_0=1, N_max= 11, Z_list= None,
                 effective_part_list= [1,0.6, 1/2], wavelength= 546):
        self.cam = cam
        self.N_0 = N_0
        self.N_max = N_max
        self.Z_list = Z_list if Z_list is not None else [cam.Z]
        self.effective_part_list = effective_part_list
        self.wavelength = wavelength
        
    def plot(self):
        """Actually plots the graph"""   
        self.cam.update_reference_light_wavelength(self.wavelength)
        
        n_plot = len(self.Z_list)
        self.fig, self.axes = plt.subplots(1, n_plot,sharey=True)
        if n_plot == 1:
            self.axes = [self.axes]
        self.fig.set_size_inches((12,8))

        self.axes[0].set_ylim(self.N_0, 1.1 * self.N_max)
        self.cam.update_focus(self.Z_list[0])
        N_list = np.array(self.cam.get_aperture_number_list())
        N_list = N_list[N_list <= self.N_max]
        self.axes[0].set_yticks(N_list)
        self.axes[0].set_ylabel("Aperture Number (f-stop)", fontsize=14)
        self.fig.suptitle("Sharpness zone with respect to Aperture", fontsize=14)

        for ax,Z in zip(self.axes, self.Z_list):
            self.cam.update_focus(Z)

            for effective_part in self.effective_part_list:
                self.cam.update_effective_diffraction_disk_ratio(effective_part)
                N_max_i = min(self.N_max, self.cam._compute_maximum_sharp_aperture())
                N= np.linspace(self.N_0, N_max_i, 400, endpoint=False)
                N = np.append(N,[N_max_i])
                z1, z2, z1_t, z2_t = [],[],[],[]
                H = []
                for N_i in N:
                    self.cam.update_aperture(N_i)
                    z1 += [self.cam.Z1]
                    z2 += [self.cam.Z2]
                    H += [self.cam.H]
                    z1_t_i, z2_t_i = self.cam._compute_sharpness_zone_including_diffraction()
                    z1_t += [z1_t_i]
                    z2_t += [z2_t_i]
                z1_t[-1] = Z
                z2_t[-1] = Z
                ax.plot(np.array(z1_t) *(100 if Z<2 else 1), N, color="k", linestyle= "-", label="Z1 (diffraction)", linewidth= 1/effective_part)
                ax.plot(np.array(z2_t) *(100 if Z<2 else 1), N, color="k", linestyle= "-", label="Z2 (diffraction)", linewidth= 1/effective_part)

            ax.plot(np.array(z1) *(100 if Z<2 else 1), N, color="gray", linestyle= "-", label="Z1")
            ax.plot(np.array(z2) *(100 if Z<2 else 1), N, color="gray", linestyle= "-", label="Z2")
            if min(H) < max(z2) and max(H) > min(z1):
                h_select = np.array(H) < max(z2)
                H = np.array(H)[h_select]
                N = np.array(N)[h_select]
                ax.plot(np.array(H) *(100 if Z<2 else 1), N, color="red", linestyle= "-", label="H", zorder=0)
            
            ax.vlines(Z*(100 if Z<2 else 1), self.N_0, self.N_max, color="k", linewidth = 0.3)
            
            if Z < 2 :
                ax.set_xlabel("Z (cm)")
                ax.set_title("Focus at {} cm".format(Z*100))
            else:
                ax.set_xlabel("Z (m)")
                ax.set_title(f"Focus at {Z:} m")
            
        self.fig.tight_layout()

def plot_diffraction_effect(cam, Z= None, N= None, wavelength= None, effective_part= None, plot_spread = 1,
     ax= None):
    
    if effective_part is not None:
        cam.update_effective_diffraction_disk_ratio(effective_part)
    if wavelength is not None:
        cam.update_reference_light_wavelength(wavelength)
    
    if Z is not None:
        cam.update_focus(Z)
    if N is not None:
        cam.update_aperture(N)
        
    zmin = max(cam.min_focus_distance/2, cam.Z1 - plot_spread* cam.depth_of_field)
    zmax = min(cam.Z2 + plot_spread* cam.depth_of_field, cam.Z + (cam.Z - zmin))
    zx = np.concatenate((np.linspace(zmin, cam.Z, 100), np.linspace( cam.Z, zmax, 100)))

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.set_title(f"Diffraction effect for N={cam.N:}\n(effective diffraction diameter ratio= {cam.effective_diffraction_disk_ratio})")
    ax.set_xlim(xmin= zmin, xmax = zmax)
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("Blur diameter on sensor (micrometers)")

    d_defocus = cam._compute_defocus_diameter(zx) 
    defocus_artist, = ax.plot(zx, d_defocus, color="gray", label= "defocus diameter", zorder= 10)
    d_combined = cam._compute_combined_blur_spot_diameter(zx)
    combined_artist, = ax.plot(zx, d_combined, color= "k", label= "combined effect diameter", zorder= 10)
    d_diffraction = cam._compute_diffraction_diameter(zx) * cam.effective_diffraction_disk_ratio
    diffraction_artist, = ax.plot(zx, d_diffraction, color="red", linewidth=0.65, label= "diffraction diameter", zorder=12)

    c = cam.confusion_circle_diameter * 1000
    ymin, ymax = 0, min(d_defocus.nanmax(), 10 * c)
    ax.set_ylim(ymin,ymax)
    ax.vlines(cam.Z1, ymin= ymin, ymax= ymax, color="gray")
    ax.vlines(cam.Z2, ymin= ymin, ymax= ymax, color="gray")
    confusion_artist = ax.hlines(y= c, xmin= zmin, xmax = zmax , color="gray",linewidth= 0.5,
                                 label="confusion circle diameter")
    plt.legend(handles = [defocus_artist, diffraction_artist, combined_artist, confusion_artist ],
               bbox_to_anchor= [1,1], loc= "upper left")

    z1_t, z2_t = cam._compute_sharpness_zone_including_diffraction()
    if z1_t is not None:
        ax.vlines(z1_t, ymin= ymin, ymax= ymax, color="k")
        ax.vlines(z2_t, ymin= ymin, ymax= ymax, color="k")
    
    return fig
        
#---------------------------------------------------------------------------
# Diffraction pattern

def plot_diffraction_pattern(cam, object_distance, ratio= None, ax= None):
    """Plots the diffraction pattern
    
    :param cam: a Camera
    :type cam: Camera
    :param object_distance: the distance to the object in meters
    :type object_distance: float
    """
    ax = ax if ax is not None else plt.gca()
    
    airy_d_half = 0.5* cam._compute_diffraction_diameter(object_distance)

    bound = 1.25 * airy_d_half
    grid_pix = np.linspace(-bound,bound, 150)
    x,y = np.meshgrid(grid_pix,grid_pix)
    diff_artist = ax.imshow(cam._compute_diffraction_intensity(np.sqrt(x**2+y**2), object_distance), extent=(-bound,bound,-bound,bound))
    
    ratio = ratio if ratio is not None else [cam.effective_diffraction_disk_ratio]
    for ratio_i in ratio:
        ax.add_patch(plt.Circle((0,0), ratio_i*airy_d_half, fill=False))
    ax.add_patch(plt.Circle((0,0), 1*airy_d_half, fill=False))

    pixel_grid_line_width = 0.4

    d = 2 * bound
    p = cam.pixel_pitch * 1000
    for line_i in range(int(1 + (d - p)//(2*p))):
        line_p_i = (0.5 + line_i) * p
        ax.hlines(line_p_i, -bound,bound, color= "grey", linewidth= pixel_grid_line_width)
        ax.hlines(-line_p_i, -bound,bound, color= "grey", linewidth= pixel_grid_line_width)
        ax.vlines(line_p_i, -bound,bound, color= "grey", linewidth= pixel_grid_line_width)
        ax.vlines(-line_p_i, -bound,bound, color= "grey", linewidth= pixel_grid_line_width)

    ratio_perc = 100 * cam.effective_diffraction_disk_ratio
    ax.set_xlabel(f"Distance to pattern center (micrometers)\nInner disk is {ratio_perc:.0f} % of Airy disk")
    plt.colorbar(diff_artist)
    return ax.get_figure()

def plot_diffraction_intensity_graph(cam, ax= None):
    ax = ax if ax is not None else plt.gca()
    
    zx = cam.Z
    airy_radius = 0.5 * cam._compute_diffraction_diameter(zx)
    q = np.linspace(0, 2*airy_radius)
    
    # show Airy diameter
    ax.vlines(airy_radius, ymin=0, ymax=1, color= "k", label= "Airy radius")
    effective_ratio = cam.effective_diffraction_disk_ratio 
    effective_relative_cumul = cam._compute_diffraction_intensity_within_distance(effective_ratio * airy_radius, zx) / cam._compute_diffraction_intensity_within_distance(airy_radius, zx)
    effective_relative_cumul_perc = 100 * effective_relative_cumul
    ax.vlines(effective_ratio * airy_radius, ymin=0, 
              ymax= effective_relative_cumul,
              color= "gray", label= f"Effective radius: {effective_ratio}\nCumulated light: {effective_relative_cumul_perc:.0f} %")
    
    # show intesity
    ax.plot(q, cam._compute_diffraction_intensity(q, zx), label="Relative intensity")
    
    # show cumulative intensity
    relative_cumul = cam._compute_diffraction_intensity_within_distance(q, zx) / cam._compute_diffraction_intensity_within_distance(airy_radius, zx)
    ax.plot(q, relative_cumul, label="Cumulative Intensity")
    
    plt.legend(loc="center right")
    
    ax.set_xlim(0, 2*airy_radius)
    ax.set_ylim(0,1.05)
    return ax.get_figure()
    
#----------------------------------------------------------------------------
def plot_view_matrix(cam, obj = None, show_sum= True, height_ratio= 0.1, ax= None, return_fig= False):
    if obj == None:
        if len(cam.views) == 0: print("Nothing to show, no views.")
        else:
            obj = list(cam.views.keys())[0]
       
    if show_sum:
        fig, ax = plt.subplots(2, 1, sharex= True, 
                       layout="compressed",
                       height_ratios=[1, height_ratio]
                       )
        ax_matrix = ax[0]
        
        nb_views = cam.views[obj].sum(axis=0)
        ax[1].bar(range(len(nb_views)), nb_views)
        ax[1].spines["top"].set_visible(False) 
        ax[1].spines["right"].set_visible(False) 
    else:
        ax_matrix = plt.gca() if ax is None else ax
    ax_matrix.imshow(cam.views[obj])
    if return_fig: return fig, ax

#----------------------------------------------------------------------------

def plot_sharpness_optimisation_cost(cam, target_points, res= None, cam_shift=None, 
                            inside_weight= 0.2,
                            outside_weight= 10,
                            shift_weight= 0.1,
                            Z_weight= 0.05,
                            apply_shift= False,
                            n= 50,
                            **kargs
                        ):
    
    if cam_shift is None:
        cam_shift = res.x[1] if res is not None and len(res.x)>1 else 0
    if not apply_shift:
        cam_shift = 0
    target_zx = cam._compute_camera_coordinates(target_points)[:,0]
    target_zx_init = target_zx
    target_zx = target_zx_init + cam_shift
    
    Z = np.linspace(target_zx.nanmin(), target_zx.nanmax(), n)
    total_cost = [np.sum(cam._optimize_sharpness_cost(target_zx_init, cam_shift, Z_i, w_in=inside_weight, w_out=outside_weight)) for Z_i in Z]
    plt.plot(Z, total_cost)
    
    global_min = np.argmin(total_cost)
    plt.vlines(Z[global_min], 0, max(total_cost))
    
    if res is not None:
        cost = np.sum(cam._optimize_sharpness_cost(target_zx_init, cam_shift, res.x[0], w_in=inside_weight, w_out=outside_weight))
        plt.scatter(res.x[0], [cost], c= "r")
    return Z[global_min], total_cost[global_min]
    
def plot_sharpness_optimisation(cam, target_points, res, cam_shift= None, Z= None, optimize_N= False, mode = "middle", use_diffraction= False, 
                            inside_weight= 0.2,
                            outside_weight= 10,
                            shift_weight= 0.1,
                            Z_weight= 0.05,
                            apply_shift= False,
                            log= True,
                            **kargs
                            ):
    if cam_shift is None:
        cam_shift = res.x[1] if res is not None and len(res.x)>1 else 0
    if not apply_shift:
        cam_shift = 0
    if Z is None:
        Z = res.x[0]
    target_zx = cam._compute_camera_coordinates(target_points)[:,0]
    target_zx_init = target_zx
    target_zx = target_zx_init + cam_shift
    
    Z1d, Z2d = cam._optimize_sharpness_get_limits(Z, optimize_N= optimize_N, use_diffraction= use_diffraction)
    zx = np.concatenate((
        np.linspace(
            min(0.9*Z1d, target_zx.nanmin()),
            Z1d, endpoint=False),
        np.linspace(
            Z1d, Z2d, 100, endpoint= False
        ),
        np.linspace(
            Z2d,
            max(1.1*Z2d, target_zx.nanmax())
        )
    ))
    zi = cam._optimize_sharpness_inner_zone(zx, Z1d, Z2d, mode)
    target_zi = cam._optimize_sharpness_inner_zone(target_zx, Z1d, Z2d, mode)

    fig, ax = plt.subplots(2,1)
    ax[1].scatter( zx, zi)
    ax[1].scatter( target_zx, target_zi)

    ax[1].vlines([Z, Z1d, Z2d], 0, 6, "k")

    c = cam._optimize_sharpness_cost(zx - cam_shift, cam_shift, Z, w_in= inside_weight, w_out= outside_weight, optimize_N= optimize_N, use_diffraction= use_diffraction, mode= mode)
    ax[0].plot(zx, c, "k-")
    ax[0].scatter(target_zx, cam._optimize_sharpness_cost(target_zx_init, cam_shift, Z, w_in=inside_weight, w_out= outside_weight, optimize_N= optimize_N, use_diffraction= use_diffraction, mode= mode), c="k")

    if log: ax[0].set_yscale("log")
    ax[0].vlines([Z, Z1d, Z2d], 0, c.nanmax(), "k")
 
#----------------------------------------------------------------------------
# Viewer
class Viewer(object):
    
    def __init__(self, cameras= None, objects= None, shots= None, lights= None, **kargs):
        """Initialises the scene
        
        Parameters:
        - cameras: a single or a list of cameras or shots that have to be drawn in the scene
        - shots: single or a list of camera shots that might have been taken in the scene
        - objects: a single or a list of objects that have to be drawn in the scene
        - lights: a single or a list of lights that have to be drawn in the scene
        """
        self.cameras = np.array([cameras]).ravel().tolist() if cameras is not None else []
        self.shots = np.array([shots]).ravel().tolist() if shots is not None else []
        self.objects = np.array([objects]).ravel().tolist() if objects is not None else []
        self.lights = np.array([lights]).ravel().tolist() if lights is not None else []
        self.object_texture = {}
        
        # attach viewer to cameras
        for cam in self.cameras:
            cam.viewer = self
        
    def add_camera(self, camera:Camera, update = True):
        """Adds a camera and update the plot
        
        Parameters:
        - camera: a Camera object to be shown or a list of cameras
        - update: True (default) if the view should be updated immediatelly, False otherwise
        """
        new_cameras = np.array([camera]).ravel().tolist()
        self.cameras += new_cameras
        if update:
            self.plot_cameras(new_cameras)
            self.update()
        
        # attach viewer to cameras
        for cam in new_cameras:
            cam.viewer = self

    def add_shot(self, shot: CameraShot, update = True):
        """Adds a shot and update the plot
        
        Parameters:
        - shot: a CameraShot object to be shown or a list
        - update: True (default) if the view should be updated immediatelly, False otherwise
        """
        new_shots = np.array([shot]).ravel().tolist()
        self.shots += new_shots
        if update:
            self.plot_shots(new_shots)
            self.update()
            
    def add_object(self, object, texture= None, update = True):
        """Adds an object and update the plot
        
        Parameters:
        - object: a TargetObject object to be shown or a list
        - texture: if not None, a pyvista texture can be passed and will be textured on the object when plotted
        - update: True (default) if the view should be updated immediatelly, False otherwise
        """
        new_objects = np.array([object]).ravel().tolist()
        self.objects += new_objects
        for object in new_objects:
            self.object_texture[object] = texture
        if update:
            self.plot_objects(new_objects)
            self.update()
            
    def add_light(self, light, update = True):
        """Adds a light and update the plot
        
        Parameters:
        - light: a light object to be shown or a list
        - update: True (default) if the view should be updated immediatelly, False otherwise
        """
        new_lights = np.array([light]).ravel().tolist()
        self.lights += new_lights
        if update:
            self.plot_light(new_lights)
            self.update()
            
    def plot_cameras(self, cameras = None):
        """Plots the cameras"""
        for camera in (self.cameras if cameras is None else cameras):
            self.plot_camera(camera)
            
    def plot_objects(self, objects = None):
        """Plots the objects"""
        for obj in (self.objects if objects is None else objects):
            self.plot_object(obj)
            
    def plot_shots(self, shots = None):
        """Plots the shots"""
        for shot in (self.shots if shots is None else shots):
            self.plot_shot(shot)
            
    def plot_lights(self, lights = None):
        """Plots the lights"""
        for light in (self.lights if lights is None else lights):
            self.plot_light(light)
    
class Viewer3D(Viewer):
    """A viewer where objects can be visualized in 3D with pyvista"""
    
    def __init__(self, cameras= None, objects= None, shots= None, lights= None, plotter=None, **kargs):
        """Initialises the scene
        
        Parameters:
        - cameras: a list of cameras or shots that have to be drawn in the scene
        - shots: camera shots that might have been taken in the scene
        - objects: a list of objects that have to be drawn in the scene
        - lights: a list of lights that have to be drawn in the scene
        - plotter: a Pyvista Plotter to be used if given, if None (default) a new one is created
        - extra parameters (kargs):
        * plot_camera_location: if True, represents the camera location as a sphere located at the focal point (default False).
        * use_diffraction: if True the sharpness zone is shown including the effect of diffraction (default)
        """
        super().__init__(cameras= cameras, shots= shots, objects= objects, lights= lights, **kargs)
        
        # initialise the plotter
        self.plotter = pv.Plotter() if plotter is None else plotter
        self.actors = {}
        self.diffraction_actors = {}
        self.diffraction_edges_actors = {}
        
        # defaults
        self.set_view_parameters(**kargs)
        
    def set_view_parameters(self, **kargs):
        """Set the default parameters"""
        self.camera_param = {}
        self.camera_param["plot_camera_location"] = kargs.get("plot_camera_location", False)
        self.camera_param["camera_location_color"] =  kargs.get("camera_location_color", "grey")
        self.camera_param["camera_location_opacity"] =  kargs.get("camera_location_opacity", 1.0)
        self.camera_param["camera_location_point_size_factor"] =  kargs.get("camera_location_point_size_factor", 5)
        
        self.camera_param["plot_camera_orientation"] = kargs.get("plot_camera_orientation", False)
        self.camera_param["camera_front_direction_color"] =  kargs.get("camera_front_direction_color", "orange")
        self.camera_param["camera_left_direction_color"] =  kargs.get("camera_left_direction_color", "blue")
        self.camera_param["camera_up_direction_color"] =  kargs.get("camera_up_direction_color", "grey")
        
        self.camera_param["plot_camera_object"] = kargs.get("plot_camera_object", True)
        self.camera_param["camera_object_color"] = kargs.get("camera_object_color", "black")
        self.camera_param["camera_object_opacity"] = kargs.get("camera_object_opacity", 0.5)
        self.camera_param["camera_object_edges_width"] = kargs.get("camera_object_edges_width", 3.0)
        
        self.camera_param["plot_sensor_object"] = kargs.get("plot_sensor_object", True)
        self.camera_param["sensor_object_color"] = kargs.get("sensor_object_color", "red")
        self.camera_param["sensor_object_opacity"] = kargs.get("sensor_object_opacity", 1.0)
        
        self.camera_param["plot_focus_plan_object"] = kargs.get("plot_focus_plan_object", True)
        self.camera_param["focus_plan_object_color"] = kargs.get("focus_plan_object_color", "red")
        self.camera_param["focus_plan_object_opacity"] = kargs.get("focus_plan_object_opacity", 0.25)
    
        self.camera_param["plot_sharpness_object"] = kargs.get("plot_sharpness_object", True)
        self.camera_param["plot_sharpness_object_edges"] = kargs.get("plot_sharpness_object_edges", True)
        self.camera_param["sharpness_object_color"] = kargs.get("sharpness_object_color", "orange")
        self.camera_param["sharpness_object_opacity"] = kargs.get("sharpness_object_opacity", 0.1)
        self.camera_param["sharpness_object_edges_color"] = kargs.get("sharpness_object_edges_color", "black")
        self.camera_param["sharpness_object_edges_opacity"] = kargs.get("sharpness_object_edges_opacity", 1.0)
        self.camera_param["sharpness_object_edges_width"] = kargs.get("sharpness_object_edges_width", 2.0)
    
        self.camera_param["plot_diffraction_object"] = kargs.get("plot_diffraction_object", True)
        self.camera_param["plot_diffraction_object_edges"] = kargs.get("plot_diffraction_object_edges", True)
        self.camera_param["diffraction_object_color"] = kargs.get("diffraction_object_color", "green")
        self.camera_param["diffraction_object_opacity"] = kargs.get("diffraction_object_opacity", 0.1)
        self.camera_param["diffraction_object_edges_color"] = kargs.get("diffraction_object_edges_color", "black")
        self.camera_param["diffraction_object_edges_opacity"] = kargs.get("diffraction_object_edges_opacity", 1.0)
        self.camera_param["diffraction_object_edges_width"] = kargs.get("diffraction_object_edges_width", 3.0)
    
        self.camera_param["plot_view_frame_object"] = kargs.get("plot_view_frame_object", True)
        self.camera_param["view_frame_object_edges_color"] = kargs.get("view_frame_object_edges_color", "black")
        self.camera_param["view_frame_object_edges_opacity"] = kargs.get("view_frame_object_edges_opacity", 1.0)
        self.camera_param["view_frame_object_edges_width"] = kargs.get("view_frame_object_edges_width", 1.0)
    
        self.camera_param["plot_visibility_object"] = kargs.get("plot_visibility_object", False)
        self.camera_param["visibility_object_edges_color"] = kargs.get("visibility_object_edges_color", "black")
        self.camera_param["visibility_object_edges_opacity"] = kargs.get("visibility_object_edges_opacity", 1.0)
        self.camera_param["visibility_object_edges_width"] = kargs.get("visibility_object_edges_width", 2.0)
    
        self.camera_param["plot_visible_part"] = kargs.get("plot_visible_part", False)
        self.camera_param["visible_part_color"] = kargs.get("visible_part_color", "red")
        self.camera_param["visible_part_opacity"] = kargs.get("visible_part_opacity", 1.0)
        
        self.shot_param = {}
        self.shot_param["plot_camera_location"] = kargs.get("plot_shot_location", False)
        self.shot_param["camera_location_color"] =  kargs.get("shot_location_color", "grey")
        self.shot_param["camera_location_opacity"] =  kargs.get("shot_location_opacity", 1.0)
        self.shot_param["camera_location_point_size_factor"] =  kargs.get("shot_location_point_size_factor", 5)
        
        self.shot_param["plot_camera_object"] = kargs.get("plot_shot_object", True)
        self.shot_param["camera_object_color"] = kargs.get("shot_object_color", "black")
        self.shot_param["camera_object_opacity"] = kargs.get("shot_object_opacity", 0.75)
        self.shot_param["camera_object_edges_width"] = kargs.get("shot_object_edges_width", 2.0)
        
        self.shot_param["plot_sensor_object"] = kargs.get("plot_shot_sensor_object", False)
        self.shot_param["sensor_object_color"] = kargs.get("shot_sensor_object_color", "red")
        self.shot_param["sensor_object_opacity"] = kargs.get("shot_sensor_object_opacity", 1.0)
        
        self.shot_param["plot_focus_plan_object"] = kargs.get("plot_shot_focus_plan_object", False)
        self.shot_param["focus_plan_object_color"] = kargs.get("shot_focus_plan_object_color", "grey")
        self.shot_param["focus_plan_object_opacity"] = kargs.get("shot_focus_plan_object_opacity", 0.25)
    
        self.shot_param["plot_sharpness_object"] = kargs.get("plot_sharpness_object", False)
        self.shot_param["plot_sharpness_object_edges"] = kargs.get("plot_sharpness_object_edges", False)
        self.shot_param["sharpness_object_color"] = kargs.get("sharpness_object_color", "orange")
        self.shot_param["sharpness_object_opacity"] = kargs.get("sharpness_object_opacity", 0.1)
        self.shot_param["sharpness_object_edges_color"] = kargs.get("sharpness_object_edges_color", "black")
        self.shot_param["sharpness_object_edges_opacity"] = kargs.get("sharpness_object_edges_opacity", 1.0)
        self.shot_param["sharpness_object_edges_width"] = kargs.get("sharpness_object_edges_width", 2.0)
    
        self.shot_param["plot_diffraction_object"] = kargs.get("plot_diffraction_object", False)
        self.shot_param["plot_diffraction_object_edges"] = kargs.get("plot_diffraction_object_edges", False)
        self.shot_param["diffraction_object_color"] = kargs.get("diffraction_object_color", "green")
        self.shot_param["diffraction_object_opacity"] = kargs.get("diffraction_object_opacity", 0.1)
        self.shot_param["diffraction_object_edges_color"] = kargs.get("diffraction_object_edges_color", "black")
        self.shot_param["diffraction_object_edges_opacity"] = kargs.get("diffraction_object_edges_opacity", 1.0)
        self.shot_param["diffraction_object_edges_width"] = kargs.get("diffraction_object_edges_width", 3.0)
    
        self.shot_param["plot_view_frame_object"] = kargs.get("plot_shot_view_frame_object", False)
        self.shot_param["view_frame_object_edges_color"] = kargs.get("shot_view_frame_object_edges_color", "black")
        self.shot_param["view_frame_object_edges_opacity"] = kargs.get("shot_view_frame_object_edges_opacity", 1.0)
        self.shot_param["view_frame_object_edges_width"] = kargs.get("shot_view_frame_object_edges_width", 1.0)
        
        self.shot_param["plot_visibility_object"] = kargs.get("plot_visibility_object", False)
        self.shot_param["visibility_object_edges_color"] = kargs.get("visibility_object_edges_color", "black")
        self.shot_param["visibility_object_edges_opacity"] = kargs.get("visibility_object_edges_opacity", 1.0)
        self.shot_param["visibility_object_edges_width"] = kargs.get("visibility_object_edges_width", 2.0)
    
        self.shot_param["plot_visible_part"] = kargs.get("plot_visible_part", False)
        self.shot_param["visible_part_color"] = kargs.get("visible_part_color", "orange")
        self.shot_param["visible_part_opacity"] = kargs.get("visible_part_opacity", 0.8)
        
        self.target_param = {}
        self.target_param["target_object_line_width"] = kargs.get("target_object_line_width", 5.0)
        self.target_param["target_object_backface_color"] = kargs.get("target_object_backface_color", "pink")
        self.target_param["target_object_backface_opacity"] = kargs.get("target_object_backface_opacity", 0.8)
        self.target_param["target_object_texture"] = kargs.get("target_object_texture", None)
        self.target_param["target_object_opacity"] = kargs.get("target_object_opacity", 1.0)
        self.target_param["target_object_scalar"] = kargs.get("target_object_scalar", None)
        
        self.target_param["scalar_bar_args"] = kargs.get("scalar_bar_args", None)
        
    def show(self,
             show_cameras= True,
             show_shots= True,
             show_objects= True,
             show_lights= True,
             show_plotter= True,
             reset_plotter= True
             ):
        """Creates a visualization of the scene in 3D"""
        if reset_plotter:
            self.plotter = pv.Plotter()
        if show_cameras: self.plot_cameras()
        if show_objects: self.plot_objects()
        if show_shots: self.plot_shots()
        if show_lights: self.plot_lights()
        
        if show_plotter: self.plotter.show()
        
        self.grid = self.plotter.show_grid()
        self.plotter.add_axes(interactive= True)
    
    def _plot_camera(self, camera, param):
        """Generic function for ploting camera and shots"""
        if param["plot_camera_location"]:
            self.plotter.add_mesh(camera.get_location_object(),
                color = param["camera_location_color"], render_points_as_spheres= True,
                opacity= param["camera_location_opacity"],
                point_size= camera.focal * param["camera_location_point_size_factor"])
                # TODO :make it a param of viewer size
        
        if param["plot_camera_orientation"]:
            orientation_objects = camera.get_orientation_object()
            self.plotter.add_mesh(orientation_objects["front"],
                color = param["camera_front_direction_color"]
                )
            self.plotter.add_mesh(orientation_objects["left"],
                color = param["camera_left_direction_color"]
                )
            self.plotter.add_mesh(orientation_objects["up"],
                color = param["camera_up_direction_color"]
                )
            
        if param["plot_camera_object"]:
            self.plotter.add_mesh(camera.get_camera_object(),
                                  opacity= param["camera_object_opacity"],
                                  edge_color= param["camera_object_color"],
                                  show_edges= True,
                                  interpolate_before_map=True,
                                  line_width= param["camera_object_edges_width"])
        if param["plot_sensor_object"]:
            self.plotter.add_mesh(camera.get_sensor_object(),
                                  color = param["sensor_object_color"],
                                  opacity= param["sensor_object_opacity"])
        if param["plot_focus_plan_object"]:
            self.plotter.add_mesh(camera.get_focus_plan_object(),
                                  color = param["focus_plan_object_color"],
                                  opacity= param["focus_plan_object_opacity"])
        if param["plot_sharpness_object"]:
            self.plotter.add_mesh(camera.get_sharpness_object(),
                                  color = param["sharpness_object_color"],
                                  opacity= param["sharpness_object_opacity"])
        if param["plot_sharpness_object_edges"]:
            self.plotter.add_mesh(camera.get_sharpness_object_edges(),
                                  style="wireframe",
                                  color= param["sharpness_object_edges_color"],
                                  opacity= param["sharpness_object_edges_opacity"],
                                  line_width= param["sharpness_object_edges_width"])
            
        if param["plot_diffraction_object"]:
            diffraction_object = camera.get_diffraction_object()
            if diffraction_object is not None:
                self.diffraction_actors[camera] = self.plotter.add_mesh(diffraction_object,
                                    color = param["diffraction_object_color"],
                                    opacity= param["diffraction_object_opacity"])
        if param["plot_diffraction_object_edges"]:
            diffraction_object_edges = camera.get_diffraction_object_edges()
            if diffraction_object_edges is not None:
                self.diffraction_edges_actors[camera]  = self.plotter.add_mesh(diffraction_object_edges,
                                  style="wireframe",
                                  color= param["diffraction_object_edges_color"],
                                  opacity= param["diffraction_object_edges_opacity"],
                                  line_width= param["diffraction_object_edges_width"])
        if param["plot_view_frame_object"]:
            self.plotter.add_mesh(camera.get_view_frame_object(), style="wireframe",
                                  color= param["view_frame_object_edges_color"],
                                  opacity= param["view_frame_object_edges_opacity"],
                                  line_width= param["view_frame_object_edges_width"])
        if param["plot_visibility_object"]:
            self.plotter.add_mesh(camera.get_visibility_object(), style="wireframe",
                                  color= param["visibility_object_edges_color"],
                                  opacity= param["visibility_object_edges_opacity"],
                                  line_width= param["visibility_object_edges_width"])
        if param["plot_visible_part"]:
            visible_part = camera.get_visible_part_object()
            if len(visible_part.points) > 0 :
                self.plotter.add_mesh(visible_part,
                                  color= param["visible_part_color"],
                                  opacity= param["visible_part_opacity"],
                                  interpolate_before_map=True
                                  )
        
    def plot_camera(self, camera):
        """Plot the given camera"""
        self._plot_camera(camera, self.camera_param)
        
    def plot_shot(self, shot):
        """Plot the given camera"""
        self._plot_camera(shot, self.shot_param)
        
    def plot_object(self, object):
        """Plot the given object"""
        texture = self.object_texture[object] if object in self.object_texture else self.target_param["target_object_texture"] 
        if object.n_cells == 0:
            # plot as 
            print("plot points")
            print(object)
            object.plot(color="red", eye_dome_lighting= True)
            self.actors[object] = self.plotter.add_mesh(object)
        else:
            self.actors[object] = self.plotter.add_mesh(object,
                              split_sharp_edges= True,
                              line_width= self.target_param["target_object_line_width"],
                              texture= texture,
                              backface_params=dict(
                                  color= self.target_param["target_object_backface_color"],
                                  opacity= self.target_param["target_object_backface_opacity"]
                                  ),
                              opacity= self.target_param["target_object_opacity"],
                              scalars= self.target_param["target_object_scalar"],
                              scalar_bar_args= self.target_param["scalar_bar_args"]
                            )
    
    def remove_scalars(self, obj= None):
        """Remove the scalar field and colorbars
        
        Parameters:
        - obj: if None (default) remove it on all the objects, else only the given object
        """
        # get a list on objects, either given or from the Viewer objects
        if obj is not None:
            if isinstance(obj, list):
                if len(obj) == 0:
                    obj_list = self.objects
                else:
                    obj_list = obj
            else:
                obj_list = [obj]
        else:
            obj_list = self.objects
        obj_list = [obj_i for obj_i in obj_list if obj_i in self.actors]
        
        for obj_i in obj_list:
            obj_i.set_active_scalars(None)
            mapper = self.actors[obj_i].mapper
            mapper.scalar_visibility = False
            mapper.array_name = None
            mapper.scalar_range = 0, 1
            
        self.scalar_name = None
        self.scalar_cmap= "viridis"
        
        for key in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(key)
            
        self.update(update_active_scalars= False)
        
    def set_active_scalars(self, 
                           scalar_name= None,
                           obj= None,
                           cmap= "viridis",
                           vmin= None,
                           vmax= None,
                           remove_scalar_bar= True,
                           interactive= True,
                           vertical= False,
                           update_viewer= True,
                           update_cmap= False,
                           **kargs
                        ):
        """Activates the given property on a given object
        """
        self.scalar_name = scalar_name
        self.scalar_cmap= cmap
        self.scalar_interactive= interactive
        self.scalar_vertical= vertical
        self.scalar_kargs = kargs
        self.vmin, self.vmax = vmin, vmax
        
        # if scalar_name is None -> remove scalar field
        if scalar_name is None:
            self.remove_scalars(obj= obj)
            return
        
        # get a list on objects, either given or from the Viewer objects
        if obj is not None:
            if isinstance(obj, list):
                if len(obj) == 0:
                    obj_list = self.objects
                else:
                    obj_list = obj
            else:
                obj_list = [obj]
        else:
            obj_list = self.objects
                
        self.scalar_obj = obj_list
        
        # get global min max values
        if len(obj_list) > 0:
            if obj_list[0] in self.actors:
                scalar_values = obj_list[0].point_data[scalar_name]
                vmin, vmax = np.nanmin(scalar_values), np.nanmax(scalar_values)   
        if len(obj_list) > 1:
            for obj_i in obj_list[1:]:
                if obj_i in self.actors:
                    scalar_values = obj_i.point_data[scalar_name] 
                    vmin = np.nanmin(vmin, np.nanmin(scalar_values))
                    vmax = np.nanmax(vmax, np.nanmax(scalar_values))
        vmin = self.vmin if self.vmin is not None else vmin
        vmax = self.vmax if self.vmax is not None else vmax
        
        # update colormap if possible
        if update_cmap and hasattr(cmap, "update") and callable(getattr(cmap, "update")):
            cmap.update()
                
        # apply on all objects
        for obj_i in obj_list:
                if obj_i in self.actors:
                    obj_i.set_active_scalars(scalar_name)
                    mapper = self.actors[obj_i].mapper
                    mapper.scalar_visibility = True
                    mapper.array_name = scalar_name
                    mapper.scalar_range = vmin, vmax
                    mapper.lookup_table.cmap = cmap
                    
        # if needed, remove the scalar bars (easier than updating it)
        if remove_scalar_bar:
            for key in list(self.plotter.scalar_bars.keys()):
                self.plotter.remove_scalar_bar(key)
        # then add a new one
        if len(obj_list) > 0:
            if obj_list[0] in self.actors:
                self.plotter.add_scalar_bar(scalar_name, 
                                        mapper= self.actors[obj_list[0]].mapper,
                                        interactive= interactive,
                                        vertical= vertical,
                                        **kargs)
            self.plotter.update_scalar_bar_range((vmin,vmax),scalar_name)
            
        # update the plot
        if update_viewer: self.update(update_active_scalars= False)
          
    def update(self, update_active_scalars= True, update_grid= True):
        """updates the graphic
        
        This should be called whenever objects ar modified directly"""
        self.plotter.update()
        
        # handling diffraction zone object which might appear or disappear with N changes
        for cam_i in self.cameras:
            # remove diffrection object if it was destroyed
            if cam_i.diffraction_object is None and cam_i in self.diffraction_actors:
                self.plotter.remove_actor(self.diffraction_actors[cam_i])
            if cam_i.diffraction_object_edges is None and cam_i in self.diffraction_edges_actors:
                self.plotter.remove_actor(self.diffraction_edges_actors[cam_i])
            
            # add it if it needed again
            if cam_i.Z1d is not None and (\
                "diffraction_object" not in cam_i.__dict__ or\
                cam_i.diffraction_object is None\
            ):  
                if self.camera_param["plot_diffraction_object"]:
                    diffraction_object = cam_i.get_diffraction_object()
                    if diffraction_object is not None:
                        self.diffraction_actors[cam_i] = self.plotter.add_mesh(diffraction_object,
                                            color = self.camera_param["diffraction_object_color"],
                                            opacity= self.camera_param["diffraction_object_opacity"])
                if self.camera_param["plot_diffraction_object_edges"]:
                    diffraction_object_edges = cam_i.get_diffraction_object_edges()
                    if diffraction_object_edges is not None:
                        self.diffraction_edges_actors[cam_i]  = self.plotter.add_mesh(diffraction_object_edges,
                                        style="wireframe",
                                        color= self.camera_param["diffraction_object_edges_color"],
                                        opacity= self.camera_param["diffraction_object_edges_opacity"],
                                        line_width= self.camera_param["diffraction_object_edges_width"])
            
        if update_active_scalars:
            if "scalar_name" in self.__dict__:
                self.set_active_scalars(
                           scalar_name= self.scalar_name,
                           obj= self.scalar_obj,
                           cmap= self.scalar_cmap,
                           remove_scalar_bar= True,
                           interactive= self.scalar_interactive,
                           vertical= self.scalar_vertical,
                           update_viewer= False,
                           update_cmap= True,
                           **self.scalar_kargs
                        )
                
        if update_grid:
            self.update_grid()
                
    def update_grid(self, bounds= None):
        """Updates the grid
        
        bounds are (x_min, x_max, y_min, y_max, z_min, z_max)
        if not given they are inferred from the drawn objects.
        """
        if "grid" not in self.__dict__ or self.grid is None:
            return
        
        if bounds is None:
            if len(self.plotter.meshes) > 0:
                low_bound = self.plotter.meshes[0].points.min(axis=0)
                high_bound = self.plotter.meshes[0].points.max(axis=0)
                
                if len(self.plotter.meshes) > 1:
                    for mesh in self.plotter.meshes:
                        p = mesh.points
                        low_bound = np.minimum(low_bound, p.min(axis=0))
                        high_bound = np.maximum(high_bound, p.max(axis=0))
                        
            bounds = np.array(list(zip(low_bound, high_bound))).ravel() 
        self.grid.update_bounds(bounds)
        
# helper functions
class CustomColormap(colors.ListedColormap):
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
class ZoneColormap(CustomColormap):
    def __init__(self, 
                 threshold,
                 vmin,
                 vmax,
                 cmap_in = colormaps['Blues_r'],
                 cmap_out = colormaps['Oranges'],
                 name= "Zone"):
        
        self.cmap_in = cmap_in
        self.cmap_out = cmap_out
        self.threshold = threshold
        
        combined_colors = self.get_combined_colors(threshold, vmin, vmax)
        super().__init__(colors= combined_colors, name= name)
        
    def get_combined_colors(self, t, v0, v1):
        n_in = max(0,
                   min(256,
                       1 + round(255 * (t - v0) / (v1 - v0))
                    )
        )
        n_out = 256 - n_in
        
        out_cmap = self.cmap_out.resampled(n_out)
        in_cmap = self.cmap_in.resampled(n_in)

        if n_in == 0:
            combined_colors = out_cmap(np.linspace(0,1, n_out))
        elif n_out == 0:
            combined_colors = in_cmap(np.linspace(0,1, n_in))
        else:
            combined_colors = np.vstack((
                in_cmap(np.linspace(0,1, n_in)),
                out_cmap(np.linspace(0,1, n_out))
            ))
        
        return combined_colors
     
class BlurSpotDiameterColormap(ZoneColormap):
    
    def __init__(self,
                 cam,
                 obj,
                 vmin= None,
                 vmax= None,
                 field= "blur_spot_diameter",
                 name= "BlurSpotCmap"
                 ):
        self.cam= cam
        self.obj= obj
        self.field_name = field
        
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(
            cam.confusion_circle_diameter*1000,
            vmin if vmin is not None else np.nanmin(obj.point_data[self.field_name]),
            vmax if vmax is not None else np.nanmax(obj.point_data[self.field_name]),
            name= name
            )
        
    def update(self, cam=None, obj=None):
        cam = cam if cam is not None else self.cam
        obj = obj if obj is not None else self.obj
        
        vmin = self.vmin if self.vmin is not None else np.nanmin(obj.point_data[self.field_name])
        vmax = self.vmax if self.vmax is not None else np.nanmax(obj.point_data[self.field_name])
        super().__init__(
            cam.confusion_circle_diameter*1000,
            vmin=vmin, vmax= vmax,
            name= self.name
            )
        
            
        
# default Scene
def get_default_scene(obj_size= 1.0, dist= 1.0, nsub=3, show= True):
    """Creates a default scene with pebble dataset and default camera"""
    obj = get_pebble_dataset(obj_size, nsub=nsub)
    cam = get_default_camera()
    cam.move(-2 * obj_size)
    cam.attach_target(obj, translate=True, rotate=False)
    cam.move_to(dist, "distance")
    viewer = Viewer3D(objects= obj, cameras= cam)
    if show: viewer.show()
    
    viewer.plotter.camera.position = [-0.5,-3,0.25]
    viewer.plotter.camera.focal_point = [-0.5,0,0.25]
    viewer.plotter.camera.up = [0,0,1]
    viewer.plotter.parallel_projection = True
    return obj, cam, viewer
   
# 2D Viewer        
class Viewer2D(Viewer):
    """A viewer where objects can be drawn in 2D based on matplotlib
    """
    
    def __init__(self, cameras= None, shots= None, objects= None, lights= None):
        """Initialises the scene
        
        Parameters:
        - cameras: a list of cameras or shots that have to be drawn in the scene
        - shots: camera shots that might have been taken in the scene
        - objects: a list of objects that have to be drawn in the scene
        - lights: a list of lights that have to be drawn in the scene
        """
        super().__init__(cameras= cameras, shots= shots, objects= objects, lights= lights)
        
        self.update_area()
        
    def update_area(self, padding= 0.05):
        "updates the drawing zone"
        
        locations = np.array([cam.location[:2] for cam in self.cameras] +\
                    [shot.location[:2] for shot in self.shots] +\
                    [obj.bbox[:,:2] for obj in self.objects] +\
                    [shot.location[:2] for shot in self.lights]).reshape((-1,2))
        
        if len(locations) == 0:
            self.xlim = [0,1]
            self.ylim = [0,1]
        else:
            self.xlim = np.array([np.nanmin(locations[:,0]), np.nanmax(locations[:,0])])
            self.ylim = np.array([np.nanmin(locations[:,1]), np.nanmax(locations[:,1])])
            self.center = np.array([np.mean(self.xlim), np.mean(self.ylim)])
            
            if len(locations) == 1:
                self.xlim = np.array([self.xlim[0] - 1, self.xlim[1] + 1])
                self.ylim = np.array([self.ylim[0] - 1, self.ylim[1] + 1])
            else:
                self.xlim = self.center[0] + (1 + padding) * (np.array([self.xlim[0] , self.xlim[1]]) - self.center[0])
                self.ylim = self.center[1] + (1 + padding) * (np.array([self.ylim[0] , self.ylim[1]]) - self.center[1])
        
    def show(self, ax= None):
        """draw the graphics"""
        self.ax = plt.gca() if ax is None else ax
        
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect("equal")
        
        self._draw_cameras()
    
    def _draw_cameras(self):
        """adds the drawing of the cameras"""
        for camera in self.cameras:
            cam_marker = markers.MarkerStyle(marker='<')
            cam_marker._transform = cam_marker.get_transform().rotate_deg(camera.yaw)
            self.ax.scatter( *camera.location[:2],
                    marker=cam_marker,
                    s=60,
                    label="[{}] {}".format(camera.name, camera.camera_type)
            )
            
            # view line
            self._add_line(camera.location[:2], camera.Z, camera.yaw, self.ax)
            
            # field of view
            assert camera.roll % 90 == 0 , "can onl draw cameras in landscape or portrait without roll"
            assert camera.pitch == 0, "can't draw camera with a pitch"
            
            fov = camera.fov_w / 2 if camera.roll % 180 == 0 else camera.fov_h/2 
            half_fov = fov / 2
            
            fov_length = camera.Z / np.cos(np.deg2rad(half_fov))
            self._add_line(camera.location[:2], fov_length, camera.yaw - half_fov, self.ax)
            self._add_line(camera.location[:2], fov_length, camera.yaw + half_fov, self.ax)
            
            self.ax.legend()
            
    def _add_line(self, start, length, az_angle, ax, **kargs):
        vector = spt.Rotation.from_euler("Z",az_angle, degrees= True).apply(np.array([1,0,0]))
        end = start + length * vector[:2]
        ax.plot([start[0],end[0]],[start[1],end[1]],**kargs)
