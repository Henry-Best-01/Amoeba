This is a brief description of how to use the provided scripts

Depending on your desired output, you will need a different combination of these.


###########################################################################
################# For time variable light curve outputs ###################
###########################################################################

This includes
    -single light curves at a single wavelength
    -multiple self-consistent light curves
    -single light curves summed over multiple wavelengths
    -light curves considering multiple wavelengths and throughputs
    -snapshots of the time variable accretion disk at given timesteps


To produce any of these, you will use "WriteVariableDiskSimulationJson.py" as a
starting point. This script shows 4 examples of setting up various Json files
to be used as inputs. Their outputs are stored in local directory
"../SampleJsons/", which can be editted directly if so desired. If your science
use aligns with one of the samples listed at the top, you may proceed to
directly edit the corresponding Json.

Definitions of all keys and values are outlined in the next script, which is where
the simulation happens. This script, "VariableDiskSimulation.py" will perform all
steps between the input file and the output. There are four major steps in this,
labeled Step_1, Step_2, Step_3, Step_4. Step_1 defines the type of driving signal.
Step_2 defines your reprocessing model. Step_3 defines the wavelength(s + throughput)
to be simulated. Step_4 defines the output type (e.g. multiple light curves, a single
light curve, snapshots). Further details of inputs are outlined in the docstring.

Step_1 and Step_2 accept the "user" value. If Step_1 is set to "user", another
file must be provided as the signal. This file should be a Json with a single
key labeled "signal", with value of a list of daily cadence sample points. If
this signal does not cover the entirete of the signal required to generate the output,
it will be repeated until this condition is satisfied. This makes generating
static or repeating signals very simple. If Step_2 is defined as "user", a list of
temperatures must be given, starting at 0 R_g and extending as far as desired. Any
values not provided will be defined as zero.

If a throughput is used (e.g. to simulate a known optical filter), a seperate file
must be included to represent the wavelengths and their relative contributions to
the sum.

Once the script completes, it will ask you to provide a file name to save this output
in. The fits extension will be included by default (e.g. "FileOutput" will be saved
as "FileOutput.fits").

A few sample calls are provided below:
    - python VariableDiskSimulation.py ../SampleJsons/Variability_model_1.json
    - python VariableDiskSimulation.py ../SampleJsons/Variability_model_2.json
    - python VariableDiskSimulation.py ../SampleJsons/Variability_model_3.json
    - python VariableDiskSimulation.py ../SampleJsons/Variability_model_4.json
            ../SampleJsons/Sample_signal.json ../SampleJsons/Sample_throughput.json 

Each of these produces different variability models. Due to the forward modeling
nature of this code, Amoeba does not produce signals at negative times. This is
because when signals are generated, there is no prior knowledge to the maximum time
lag across the entire system. Taking a signal and applying it to both a 10^8 and
10^9 solar massed black hole would require very different burn in times, so keep
this in mind. Clipping the first section of the signal can be helpful.

To have a visualization of this signal, use the script "ShowDiskVariability.py".
This will look at the fits file generated and display it naturally. If it was
multiple light curves, each will be shown and labeled. If a series of screen shots
was created, you will be shown the overall light curve (e.g. summing all pixels
at each time stamp, the total light curve), then a visualization of the contours
evolving with time. Sometimes, the contribution of the innermost regions may
outweigh the contours as higher radii, so this doesn't always look great.




###########################################################################
######### To create an interpolation between a precomputed movie ##########
###########################################################################

In the previous section a movie is created with Variability_model_1.json.
To change the timestamps of interpolation, you may use the two step process:
    "WriteInterpolatingTimestamps.py"
    "InterpolateMovie.py"
The first step, "WriteInterpolatingTimestamps.py" sets up a Json containing
the new timestamps (units days). This is stored in the file
"../SampleJsons/New_timestamps.json", which may be edited directly. There is
one key where the value is a list of time stamps as floats. These are the new
time stamps which the resulting movie will be interpolated to.

The second step actually prepares the new series of snapshots. Running
"InterpolateMovie.py" with the original fits file and the Json containing
new timestamps will keep all original dimensions aside from the time axis.
For example, a file of data shape (1, 1000, 200, 200) interpolates with
timestamp file of shape (300) into a new file of shape (1, 300, 200, 200).
There is no protection from extrapolation, but it will be obvious when
looking at the variability of the new file. This file may have its variability
displayed exactly as the outputs to "VariableDiskSimulation.py", by calling
the script "ShowDiskVariability.py" with the saved file.



###########################################################################
################## To create a Sim5 ray-traced map ########################
###########################################################################

This script is only available in python (for now), and requires a successful
installation of Sim5 with python interface. The script "SimpleRayTrace.py"
performs this step. In the imports, the paths to sim5 and QuasarModelFunctions
must be changed to your local directories. For example, change "path_to_sim5"
to the absolute pathway where sim5 was installed.     

Inputs to the ray trace (as well as some parameters relating to the assumed
temperature profile) may be adjusted under "Define inputs for ray tracing".
You will be prompted for a file name to save this fits file to. Leave this
blank (no white space!) if you do not want to save the ray trace. The fits
extension will be appended automatically.

Information on how the ray tracing may be used personally is described in
the commented section.
A very short blurb describes how to open the fits file and use it.

The file path may be passed as a string into "VariableDiskSimulation" as
key "ray_trace_fits" within the "Variability_model___.json" to be
incorportated into the calculation if desired. The temperature profile will
be overwritten to what "Variability_model___.json" defined.





