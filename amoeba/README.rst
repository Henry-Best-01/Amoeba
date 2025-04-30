======
Amoeba
======


.. image:: https://img.shields.io/pypi/v/amoeba-agn.svg
        :target: https://pypi.python.org/pypi/amoeba-agn

.. image:: https://readthedocs.org/projects/amoeba/badge/?version=latest
        :target: https://amoeba.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Amoeba is a new, modular, and open source quasar modeling code designed to model intrinsic and extrinsic variability within the context of wide field optical surveys such as LSST. It has the capability to treat both emission and reverberation with general relativistic corrections, including Doppler shifting, gravitational redshifting, and light bending around the central black hole. We can simulate any inclination, where moderate to edge-on cases can significantly deviate from the flat metric case. 

Accretion disk temperature profiles have been modelled as a thin disk profile[^3], an irradiated disk profile[^4], or the disk+wind profile[^5]. We provide a flexible temperature profile which can include contributions from lamppost heating and variable accretion flows, which smoothly converges to the thin disk temperature profile. Beyond this temperature profile, Amoeba allows for any input (effective) temperature mapping to be used to create arbitrary surface brightness / response maps. Transfer functions may be constructed from these response maps under the assumed lamppost model.

We cannot upload magnification maps due to size limits. This code was written to use external microlensing magnification maps, where many can be found on the [GERLUMPH](https://gerlumph.swin.edu.au) database[^1]. Alternatively, they may be generated using the code found [here](https://github.com/weisluke/microlensing).

**Important** The function create_maps within Util.util is designed to generate all accretion disk maps required for making a disk object. For most users, this will be the primary interface between your parameters and the amoeba code. This function calls [Sim5](https://github.com/mbursa/sim5), a public geodesic ray-tracing code[^2], if available to calculate impact positions of observed photons on the accretion disk. It creates a full dictionary of parameters designed to be passed into the AccretionDisk or Agn object.

If you would like to run the accretion disk and microlensing examples notebook, you will be required to change file paths and provide the disk file (creatable with Util.util.create_maps) and magnification map.

Broad Line Region (BLR) models are included. They may be tested using blr examples notebook. They are created by defining streamlines, similar to the disk-wind model[^6]. Simple projections, line-of-sight velocity selected regions, and transfer functions may be constructed now.

Further examples are provided in the example_scripts directory, with individual scripts aimed at visualizing certain aspects of amoeba.

For those particularly interested in how amoeba works, please see the in-depth notebooks in the Notebooks directory.

Unit tests are included in the tests directory and our goal is to remain as close to 100% coverage throughout development!

Thank you for taking notice of this code! Any questions may be directed towards Henry Best via e-mail at hbest@gradcenter.cuny.edu


* Free software: MIT license
* Documentation: https://amoeba.readthedocs.io.


Features
--------


To use Sim5 for ray tracing, please follow Sim5 installation instructions [here](https://github.com/mbursa/sim5), including the instructions to install the python interface.

Microlensing simulations will require an external magnification map. In the example notebook please provide the directory to these maps.

Some precomputed ray tracings may be found [here](https://drive.google.com/drive/folders/1vx8HUBXw6SaDq5uS4jQCyWdg13XfCRCv?usp=share_link) which contains a zipped folder of .fits files for various inclination angles and black hole spins. Providing the path to one of these ray traces in the notebook will show you how to use these files with Amoeba. Beyond this zip file, a single file is included with Amoeba.


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

