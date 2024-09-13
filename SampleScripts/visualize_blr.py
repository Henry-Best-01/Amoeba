import numpy as np
import matplotlib.pyplot as plt
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps
from astropy import units as u
from astropy import constants as const


# agn params

smbh_mass_exp = 8.5
redshift_source = 0.7
inclination_angle = 35 # deg
smbh_spin = 0.6
name = "test agn"
max_disk_radius = 2000 # rg
accretion_disk_resolution = 200 # pixels / radial length


# blr params

inner_launch_radius = 700 # rg
inner_launch_angle = 45 # deg
inner_launch_radius = 1500 # rg
inner_launch_angle = 75 # deg

maximum_blr_height = 2000 # rg
characteristic_distance = 1000 # in rg, for outflowing velocity model
asymptotic_poloidal_velocity = 0.2 # units v / c
height_step = 50 # rg, defines thickness of each blr slab
radial_step = 10 # rg, defines resolution of each slab

blr_rest_frame_emission = 400 # nm



# create accretion disk and broad line region

agn_params_dict = create_maps(
    smbh_mass_exp,
    redshift_source,
    max_disk_radius,
    inclination_angle,
    accretion_disk_resolution,
    spin=smbh_spin,
    name=name
)

accretion_disk = AccretionDisk(agn_params_dict)


inner_streamline = Streamline(
    inner_launch_radius,
    inner_launch_angle,
    maximum_blr_height,
    characteristic_distance,
    asymptotic_poloidal_velocity,
    height_step=height_step,
)

outer_streamline = Streamline(
    outer_launch_radius,
    outer_launch_angle,
    maximum_blr_height,
    characteristic_distance,
    asymptotic_poloidal_velocity,
    height_step=height_step,
)

broad_line_region = BroadLineRegion(
    smbh_mass_exp,
    maximum_blr_height,
    blr_rest_frame_emission,
    redshift_source,
    radial_step=radial_step,
    height_step=height_step,
)




















    
