# Amoeba
Amoeba is a new, modular, and open source quasar modeling code designed to model intrinsic and extrinsic variability. It treats both emissions and reverberations together with corrections from Doppler shifting, relativistic corrections, and geodesic ray tracing around the central black hole. We can simulate any inclination angle, where moderate to edge-on cases can significantly deviate from the case without including these effects. We allow for a flexible temperature profile which can include contributions from lamp post heating and variable accretion flows, which smoothly converges to the thin disk temperature profile. Beyond this temperature profile, Amoeba allows for any input (effective) temperature mapping to be used to create arbitrary surface brightness / response maps. Transfer functions may be constructed from these response maps under the assumed lamp post model, which may be extended by combining multiple lampposts to build up any arbitrary driving source.

I cannot upload a magnification map sample due to size limits. My code was written to use microlensing magnification maps which are pre-calculated, where many can be found on the GERLUMPH database (Vernardos+ 2014). 

The function "CreateMaps" within QMF is designed to generate all accretion disk maps required for making a disk object. This function relies on Sim5, a public geodesic ray-tracing code (Bursa 18), to calculate impact positions of observed photons on the accretion disk. Only geometrically flat disks are supported, as the position of impact is calculated using Jacobi elliptics. 

The temperature profiles may take arguments for either the thin disk profile (Shakura+Sunyaev 1973), the irradiated disk profile (See Cackett+ 2013), and the disk+wind profile (See Sun+ 2019). These smoothly transition into eachother, where the irradiation term increases the maximum temp, and the wind term increases the asypmtotic slope of T(r).

If you would like to run the AmoebaExamples.ipynb notebook, you will be required to change file paths and provide the disk file (creatable with QMF.CreateMaps) and magnification map as well.

Broad Line Region (BLR) models have been given some TLC and are added again! They may be tested using TestingBLR.ipynb. They are created by defining streamlines, similar to the disk-wind model in Yong+ (2017). Simple projections, line-of-sight velocity slices, and scattering transfer functions may be constructed now.


Thank you for taking notice of my code! I would be happy to answer any questions. I can be contacted directly at hbest@gradcenter.cuny.edu

