======
Amoeba
======


.. image:: https://img.shields.io/pypi/v/amoeba.svg
        :target: https://pypi.python.org/pypi/amoeba

.. image:: https://img.shields.io/travis/Henry-Best-01/amoeba.svg
        :target: https://travis-ci.com/Henry-Best-01/amoeba

.. image:: https://readthedocs.org/projects/amoeba/badge/?version=latest
        :target: https://amoeba.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Amoeba is a new, modular, and open source quasar modeling code designed to model intrinsic and extrinsic variability within the context of wide field optical surveys such as LSST. It has the capability to treat both emission and reverberation with general relativistic corrections, including Doppler shifting, gravitational redshifting, and light bending around the central black hole. We can simulate any inclination, where moderate to edge-on cases can significantly deviate from the flat metric case. 

Accretion disk temperature profiles have been modelled as a thin disk profile[^3], an irradiated disk profile[^4], or the disk+wind profile[^5]. We provide a flexible temperature profile which can include contributions from lamppost heating and variable accretion flows, which smoothly converges to the thin disk temperature profile. Beyond this temperature profile, Amoeba allows for any input (effective) temperature mapping to be used to create arbitrary surface brightness / response maps. Transfer functions may be constructed from these response maps under the assumed lamppost model.

We cannot upload magnification maps due to size limits. This code was written to use external microlensing magnification maps, where many can be found on the [GERLUMPH](https://gerlumph.swin.edu.au) database[^1]. 

The function "CreateMaps" within QMF is designed to generate all accretion disk maps required for making a disk object. This function calls [Sim5](https://github.com/mbursa/sim5), a public geodesic ray-tracing code[^2], if available to calculate impact positions of observed photons on the accretion disk.

If you would like to run the AmoebaExamples.ipynb notebook, you will be required to change file paths and provide the disk file (creatable with QMF.CreateMaps) and magnification map.

Broad Line Region (BLR) models have been given some TLC and are added again! They may be tested using TestingBLR.ipynb. They are created by defining streamlines, similar to the disk-wind model[^6]. Simple projections, line-of-sight velocity slices, and scattering transfer functions may be constructed now.

To make this code even more accessible, sample scripts are included to show exactly the kinds of variability involved with these accretion disks. Any scripts prefaced "Write" will create a .json file (stored in SampleJsons), and values may be adjusted either through the writing script or the .json directly. 

Thank you for taking notice of my code! I would be happy to answer any questions. I can be contacted directly at hbest@gradcenter.cuny.edu


* Free software: MIT license
* Documentation: https://amoeba.readthedocs.io.


Features
--------


To use Sim5 for ray tracing, please follow Sim5 installation instructions [here](https://github.com/mbursa/sim5), including the instructions to install the python interface.

Microlensing simulations will require an external magnification map. In the example notebook please provide the directory to these maps.

Some precomputed ray tracings may be found [here](https://drive.google.com/drive/folders/1vx8HUBXw6SaDq5uS4jQCyWdg13XfCRCv?usp=share_link) which contains a zipped folder of .fits files for various inclination angles and black hole spins. Providing the path to one of these ray traces in the notebook will show you how to use these files with Amoeba. Beyond this zip file, a single file is included with Amoeba.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage



[^1]: https://ui.adsabs.harvard.edu/abs/2014ApJS..211...16V/abstract
[^2]: https://ui.adsabs.harvard.edu/abs/2018ascl.soft11011B/abstract
[^3]: https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S/abstract
[^4]: https://ui.adsabs.harvard.edu/abs/2007MNRAS.380..669C
[^5]: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.2788S
[^6]: https://ui.adsabs.harvard.edu/abs/2017PASA...34...42Y

