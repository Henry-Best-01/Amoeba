=====
Usage
=====

Amoeba is designed to be modular.
The most common class to use is the accretion disk object for reverberation mapping or microlensing.
However, generating everything for the class to work is not intuitive.
In order to bypass this issue, a convenience function is provided which serves as the practical interface between parameters and everything required for Amoeba to calculate accretion disk reprocessing.
The class and function are imported as::

    from amoeba.Classes.accretion_disk import AccretionDisk
    from amoeba.Util.util import create_maps

    param_dict_disk = create_maps(__your_parameters_here__)
    my_disk = AccretionDisk(**param_dict_disk)

You can input parameters of the active galactic nuclei's (AGN) accretion disk, such as the mass of the supermassive black hole (SMBH), inclination of the disk, eddington ratio, redshift of the source, etc..
The accretion disk object then has methods to compute everything from transfer functions to projected flux distributions.

To extend this beyond the accretion disk, you can build a broad line region object.
These require streamline objects to populate an internal set of 2-dimensional grids in R-Z space.
Typical use is as::

    from amoeba.Classes.blr import BroadLineRegion
    from amoeba.Classes.blr_streamline import Streamline

    inner_stream = Streamline(__your_parameters_here__)
    outer_stream = Streamline(__your_parameters_here__)

    my_blr = BroadLineRegion(__your_parameters_here__)
    my_blr.add_streamline_bounded_region(
        inner_streamline,
        outer_streamline,
    )

At this point, the BroadLineRegion object can now generate it's response and projection at any wavelength range.
Each BroadLineRegion is associated with a particular emission line, so multiple will be required to build up a spectrum.

It is easy to imagine that as you add more components to the model, it will be increasinngly difficult to keep track of each part.
To ease this, everything can be done within the AGN object, which acts as a container for all components.
The AGN object then has pipeline methods for both reprocessing through and projecting all components.
An AGN incorporating all of the above is prepared and used as::

    from amoeba.Classes.agn import Agn

    my_agn = Agn(__your_parameters_here__)

    my_agn.add_default_accretion_disk(**param_dict_disk)
    my_agn.add_blr(__your_parameters_for_blr_here__)

    streamline_dict = {
        "InnerStreamline": inner_stream,
        "OuterStreamline": outer_stream,
    }
    my_agn.add_streamline_bounded_region_to_blr(**streamline_dict)

    projection = my_agn.visualization_agn_pipeline(__your_parameters_here__)

    reprocessed_signals = my_agn.intrinsic_signal_propagation_pipeline(__your_perameters_here__)
    

Please check out the various Jupyter notebooks and example scripts provided for more usage and inspiration!

As always, feel free to contact me at hbest@gradcenter.cuny.edu for any questions!
    
    








