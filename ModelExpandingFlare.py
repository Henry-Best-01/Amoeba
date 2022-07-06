'''
This script aims to use the "InsertFlare" function in QMF to model and show a
propagating flare of the accretion disk
'''

import QuasarModelFunctions as QMF
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.random import randint, seed


mass = 'log8.0'
incang = '45'

DIR = '/Users/henrybest/PythonStuff/DiskImages/DataSetband/'
Diskfile = 'ImageFile'+mass+'0.15'+incang+'gband.disk'
Diskfile2 = 'ImageFile'+mass+'0.15'+incang+'rband.disk'
Diskfile3 = 'ImageFile'+mass+'0.15'+incang+'iband.disk'
Diskfile4 = 'ImageFile'+mass+'0.15'+incang+'zband.disk'

disk1 = QMF.LoadDiskImage(DIR+Diskfile, 131)
disk2 = QMF.LoadDiskImage(DIR+Diskfile2, 131)
disk3 = QMF.LoadDiskImage(DIR+Diskfile3, 131)
disk4 = QMF.LoadDiskImage(DIR+Diskfile4, 131)


nflares = 150
b1 = QMF.InsertFlare(disk1, 0, int(incang), 50, returnlightcurve=True)
b2 = QMF.InsertFlare(disk2, 0, int(incang), 50, returnlightcurve=True)
b3 = QMF.InsertFlare(disk3, 0, int(incang), 50, returnlightcurve=True)
b4 = QMF.InsertFlare(disk4, 0, int(incang), 50, returnlightcurve=True)

out1 = b1.copy()
out2 = b2.copy()
out3 = b3.copy()
out4 = b4.copy()

for ii in range(nflares):

    amp = randint(50)
    decay = randint(20, high=100)

    next1 = QMF.InsertFlare(disk1, amp, int(incang),
                                decay, returnlightcurve=True)
    next2 = QMF.InsertFlare(disk2, amp, int(incang),
                                decay, returnlightcurve=True)
    next3 = QMF.InsertFlare(disk3, amp, int(incang),
                                decay, returnlightcurve=True)
    next4 = QMF.InsertFlare(disk4, amp, int(incang),
                                decay, returnlightcurve=True)
    
    startpoint = randint(len(out1)) # This shifts 'nextcurve' n spaces along the simulated light curve
    shifted1 = np.empty([len(next1)])
    shifted2 = np.empty([len(next2)])
    shifted3 = np.empty([len(next3)])
    shifted4 = np.empty([len(next4)])
    
    for jj in range(len(next1)):
        shifted1[jj] = next1[jj - startpoint]
        shifted2[jj] = next2[jj - startpoint]
        shifted3[jj] = next3[jj - startpoint]
        shifted4[jj] = next4[jj - startpoint]
        
    out1 += shifted1 - b1
    out2 += shifted2 - b2
    out3 += shifted3 - b3
    out4 += shifted4 - b4
    

    f1 = QMF.InsertFlare(disk1, 50, int(incang), 50, returnlightcurve=True)
    f2 = QMF.InsertFlare(disk2, 50, int(incang), 50, returnlightcurve=True)
    f3 = QMF.InsertFlare(disk3, 50, int(incang), 50, returnlightcurve=True)
    f4 = QMF.InsertFlare(disk4, 50, int(incang), 50, returnlightcurve=True)

xaxis = np.linspace(0, len(f1), len(f1))

fig, ax = plt.subplots(2, sharex = 'all')
ax[0].plot(xaxis, 4*out1/b1[0], label="g' band x4")
ax[0].plot(xaxis, 3*out2/b2[0], label="r' band x3")
ax[0].plot(xaxis, 2*out3/b3[0], label="i' band x2")
ax[0].plot(xaxis, out4/b4[0], label="z' band x1")

ax[1].plot(xaxis, out1/out2, c='purple', label="g' / r'")
ax[1].plot(xaxis, out1/out3, c='orange', label="g' / i'")
ax[1].plot(xaxis, out1/out4, c='cyan', label="g' / z'")

fig.suptitle("Variability light curve due to "+str(nflares)+' flares located at center at random amplitudes, decay times, and initial times')
fig.supxlabel("Time [arb.]")
fig.supylabel("Relative Brightness")
fig.legend()


'''

seed(10)

nflares = 200
baseline = QMF.InsertFlare(disk, 0, int(incang), 20, returnlightcurve=True) #Subtract away baseline to get just the flaring component
output = baseline.copy()
for ii in range(nflares):
    nextcurve = QMF.InsertFlare(disk, randint(50), int(incang),
                                randint(20, high=200), returnlightcurve=True)
    startpoint = randint(len(output)) # This shifts 'nextcurve' n spaces along the simulated light curve
    shiftedcurve = np.empty([len(nextcurve)])
    for jj in range(len(nextcurve)):
#        if jj < startpoint:
#            shiftedcurve[jj] = nextcurve[0]
#        else:
        shiftedcurve[jj] = nextcurve[jj - startpoint]
    output += shiftedcurve - baseline
    print('done with', (ii+1), '/', nflares)

xaxis = np.linspace(0, len(output), len(output))
fig, ax = plt.subplots()
plt.plot(xaxis, output/baseline)
ax.set_title("Variability light curve due to "+str(nflares)+' flares located at center at random amplitudes, decay times, and initial times')
ax.set_xlabel("Time [arb.]")
ax.set_ylabel("Relative Brightness")



#'''


'''

seed(5000)

amplitudes = randint(500, size=(10))
initpoints = randint(300, size=(10, 2))
decays = randint(50, size=(10))
offset = randint(150, size=(10))

print(amplitudes)
print(initpoints)
print(decays)
print(offset)

print(type(int(offset[0])))

baseline = QMF.InsertFlare(disk, 0, int(incang), 20, returnlightcurve=True)

lightcurve1 = QMF.InsertFlare(disk, amplitudes[0], int(incang), decays[0],
                             initialpoint = [initpoints[0, 0], initpoints[0, 1]],
                             returnlightcurve=True)
lightcurve2 = QMF.InsertFlare(disk, amplitudes[1], int(incang), decays[1],
                             initialpoint = [initpoints[1, 0], initpoints[1, 1]],
                             returnlightcurve=True)
lightcurve3 = QMF.InsertFlare(disk, amplitudes[2], int(incang), decays[2],
                             initialpoint = [initpoints[2, 0], initpoints[2, 1]],
                             returnlightcurve=True)
lightcurve4 = QMF.InsertFlare(disk, amplitudes[3], int(incang), decays[3],
                             initialpoint = [initpoints[3, 0], initpoints[3, 1]],
                             returnlightcurve=True)
lightcurve5 = QMF.InsertFlare(disk, amplitudes[4], int(incang), decays[4],
                             initialpoint = [initpoints[4, 0], initpoints[4, 1]],
                             returnlightcurve=True)
lightcurve6 = QMF.InsertFlare(disk, amplitudes[5], int(incang), decays[5],
                             initialpoint = [initpoints[5, 0], initpoints[5, 1]],
                             returnlightcurve=True)
lightcurve7 = QMF.InsertFlare(disk, amplitudes[6], int(incang), decays[6],
                             initialpoint = [initpoints[6, 0], initpoints[6, 1]],
                             returnlightcurve=True)
lightcurve8 = QMF.InsertFlare(disk, amplitudes[7], int(incang), decays[7],
                             initialpoint = [initpoints[7, 0], initpoints[7, 1]],
                             returnlightcurve=True)
lightcurve9 = QMF.InsertFlare(disk, amplitudes[8], int(incang), decays[8],
                             initialpoint = [initpoints[8, 0], initpoints[8, 1]],
                             returnlightcurve=True)
lightcurve10 = QMF.InsertFlare(disk, amplitudes[9], int(incang), decays[9],
                             initialpoint = [initpoints[9, 0], initpoints[9, 1]],
                             returnlightcurve=True)

lc1 = np.concatenate((lightcurve1[int(offset[0]):], lightcurve1[:int(offset[0])]))
lc2 = np.concatenate((lightcurve2[int(offset[1]):], lightcurve2[:int(offset[1])]))
lc3 = np.concatenate((lightcurve3[int(offset[2]):], lightcurve3[:int(offset[2])]))
lc4 = np.concatenate((lightcurve4[int(offset[3]):], lightcurve4[:int(offset[3])]))
lc5 = np.concatenate((lightcurve5[int(offset[4]):], lightcurve5[:int(offset[4])]))
lc6 = np.concatenate((lightcurve6[int(offset[5]):], lightcurve6[:int(offset[5])]))
lc7 = np.concatenate((lightcurve7[int(offset[6]):], lightcurve7[:int(offset[6])]))
lc8 = np.concatenate((lightcurve8[int(offset[7]):], lightcurve8[:int(offset[7])]))
lc9 = np.concatenate((lightcurve9[int(offset[8]):], lightcurve9[:int(offset[8])]))
lc10 = np.concatenate((lightcurve10[int(offset[9]):], lightcurve10[:int(offset[9])]))


lightcurve = (lc1+lc2+lc3+lc4+lc5+lc6+lc7+lc8+lc9+lc10)-(9*baseline)



xaxis = np.linspace(0, len(lightcurve1), len(lightcurve1))
plt.plot(xaxis, lightcurve)
plt.legend()


''
lightcurve = QMF.InsertFlare(disk, 200, int(incang), 50,
                             returnmovie=True)

nsnaps = 12
snapshots = np.linspace(0, nsnaps-1, nsnaps)*600/nsnaps
x = np.linspace(0, np.size(lightcurve, 0), (np.size(lightcurve, 0)))
y = np.linspace(0, np.size(lightcurve, 1), (np.size(lightcurve, 1)))
X, Y = np.meshgrid(x, y, indexing='ij')


fig, ax = plt.subplots(4, 3, sharex='all', sharey='all')
for ii in range(len(snapshots)):
    ax[ii//3, ii%3].contourf(X, Y, (lightcurve[:, :, int(snapshots[ii])]))
plt.colorbar()

#'''             

plt.show()



















