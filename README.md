# PublicQuasarVariabilityModeling
This repository contains a few examples of my modeling of quasar variability, both intrinsically and extrinsically.

I cannot upload a magnification map sample due to size limits. My code was written to use microlensing magnification maps which are pre-calculated, where many can be found on the GERLUMPH database. 

The function "CreateMaps" within QMF is designed to generate all accretion disk maps required for making a disk object. This function relies on Sim5, a public geodesic ray-tracing code, to calculate impact positions of observed photons on the accretion disk. Only geometrically flat disks are supported, as calculating the colision with a point outside the midplane is non-trivial. However, this is a good estimation for the position where the photons originate from. The temperature profiles may take arguments for either the thin disk profile (Shakura Sunyaev 1973), the irradiated disk profile (See Cackett+ 2013), and the disk+wind profile (See Sun+ 2019). These smoothly transition into eachother, where the irradiation term increases the maximum temp, and the wind term increases the asypmtotic slope of T(r).

If you would like to run the Examples.ipynb notebook, you will be required to change file paths and provide the disk file (creatable with QMF.CreateMaps) and magnification map as well.

Broad Line Region (BLR) models have been removed for now while they receive some TLC. 


Thank you for taking notice of my code! I would be happy to answer any questions.
