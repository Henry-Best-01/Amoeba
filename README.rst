======
Amoeba
======


.. image:: https://img.shields.io/pypi/v/amoeba-agn.svg
        :target: https://pypi.python.org/pypi/amoeba-agn

.. image:: https://readthedocs.org/projects/amoeba/badge/?version=latest
        :target: https://amoeba.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
        :target: https://www.sphinx-doc.org/en/master/usage/index.html

.. [![CI](https://github.com/Henry-Best-01/Amoeba/actions/workflows/CI.yml/badge.svg)](https://github.com/Henry-Best-01/Amoeba/actions/workflows/CI.yml)



Amoeba (an Agn Model of Optical Emissions Beyond steady-state Accretion disks) is a new, modular, and open source
quasar modeling code designed to model intrinsic and extrinsic variability within the context of wide field
optical surveys such as LSST. It has the capability to treat both emission and reverberation with general
relativistic corrections, including Doppler shifting, gravitational redshifting, and light bending around the
central black hole. We can simulate any inclination, where moderate to edge-on cases can significantly deviate
from the flat metric case. 

Accretion disk temperature profiles have been modelled as a thin disk profile[^3], an irradiated disk profile[^4],
or the disk+wind profile[^5]. We provide a flexible temperature profile which can include contributions from
lamppost heating and variable accretion flows, which smoothly converges to the thin disk temperature profile.
Beyond this temperature profile, Amoeba allows for any input (effective) temperature mapping to be used to create
arbitrary surface brightness / response maps. Transfer functions may be constructed from these response maps under
the assumed lamppost model.

We cannot upload magnification maps due to size limits. This code was written to use external microlensing magnification
maps, where many can be found on the [GERLUMPH](https://gerlumph.swin.edu.au) database[^1]. Alternatively, they may be
generated using the code found [here](https://github.com/weisluke/microlensing).

**Important** The function create_maps within Util.util is designed to generate all accretion disk maps required for
making a disk object. For most users, this will be the primary interface between your parameters and the amoeba code.
This function calls [Sim5](https://github.com/mbursa/sim5), a public geodesic ray-tracing code[^2], if available to
calculate impact positions of observed photons on the accretion disk. It creates a full dictionary of parameters
designed to be passed into the AccretionDisk or Agn object.

If you would like to run the accretion disk and microlensing examples notebook, you will be required to change file
paths and provide the disk file (creatable with Util.util.create_maps) and magnification map.

Broad Line Region (BLR) models are included. They may be tested using blr examples notebook. They are created by
defining streamlines, similar to the disk-wind model[^6]. Simple projections, line-of-sight velocity selected regions,
and transfer functions may be constructed now.

Further examples are provided in the example_scripts directory, with individual scripts aimed at visualizing certain
aspects of amoeba.

For those particularly interested in how amoeba works, please see the in-depth notebooks in the Notebooks directory.

Unit tests are included in the tests directory and our goal is to remain as close to 100% coverage throughout development!

Thank you for taking notice of this code! Any questions may be directed towards Henry Best via e-mail at
hbest@gradcenter.cuny.edu


* Free software: MIT license
* Documentation: https://amoeba.readthedocs.io.


Features and Uses
-----------------

Amoeba's primary function is to simulate flux distributions of AGN components at the smallest levels unresolvable by
modern telescopes and use that information to generate the expected variability which we can see in time-variable sources.
The accretion disk provides the natural source of the variable continuum which depends on both the parameters of the source
and the wavelength it is observed at. The accretion disk is believed to be in close proximity to the supermassive black hole,
leading to the desire of relativistic corrections. Amoeba handles this by running Sim5, where installation instructions
may be found [here](https://github.com/mbursa/sim5). The python interface must be installed for Amoeba to use it. 

Beyond the accretion disk, it is well known that type 1 AGN have braod emission lines. These may be simulated assuming a
rest frame emission and a distribution of particles, which may be given some bulk velocity and acceleration. Based on the
line-of-sight velocities, the rest frame emission becomes spectrally braodened and may contaminate the flux distributions
as well as the reverberating responses. The BLR module of Amoeba has the ability to simulate this.

However, the BLR also has the ability to contribute beyond emission lines through the diffuse continuum. This is believed to be
a series of free-bound interactions which release photons at energies above ionization frequencies. Most notibly is the
Balmer and Paschen continuua, where recombination of a free electron and proton falls into the n = 2 and 3 state, respectively.
The combination of an abundance of Hydrogen and strong ionization potential of the central region of the AGN makes this
an important effect to consider. The Diffuse continuum module of Amoeba has the ability to model this effect.

Highly inclined AGN are believed to be obscured by a dusty torus, which may be due to a multitude of effects. The most prominent
effect is the absorption, which may be modeled using Amoeba's torus module. In its current state, this module does not provide
the expected contribution to the infrared emission.

Amoeba was designed to join together variable sources with gravitational microlensing, a phenomena which occurs in strongly lensed
systems when a compact object modules the signal we observe at the image level. If there are multiple images of a source, microlensing
will occur in each image seperately. Amoeba simulates microlensing using the magnification map module, however this module
requires a representation of the 2-dimensional magnification map generated from an external code. Once this is generated, the
microlens' Einstein radius is calculated, angular diameter distances are computed, and flux distributions and magnification maps
are convolved to provide microlensing light curves of each source.

Some precomputed ray tracings may be found [here](https://drive.google.com/drive/folders/1vx8HUBXw6SaDq5uS4jQCyWdg13XfCRCv?usp=share_link)
which contains a zipped folder of .fits files for various inclination angles of Schwarzschild black holes.

More explicit use cases may be found at the [ReadTheDocs](https://amoeba.readthedocs.io/en/latest/?version=latest).


Citing (It's important for developers!)
---------------------------------------

If you use Amoeba in your work, please include the following ADS citation: https://ui.adsabs.harvard.edu/abs/2024arXiv241019630B/exportcitation

If you install and use the Sim5 ray tracing component for accretion disk modeling, please cite the authors according to https://github.com/mbursa/sim5





Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

This package supports Sim5 ray tracing: https://ascl.net/1811.011

[^1]: https://ui.adsabs.harvard.edu/abs/2014ApJS..211...16V/abstract
        
[^2]: https://ui.adsabs.harvard.edu/abs/2018ascl.soft11011B/abstract
        
[^3]: https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S/abstract
        
[^4]: https://ui.adsabs.harvard.edu/abs/2007MNRAS.380..669C
        
[^5]: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.2788S
        
[^6]: https://ui.adsabs.harvard.edu/abs/2017PASA...34...42Y
        

