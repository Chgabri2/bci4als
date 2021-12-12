import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from scipy import signal

objectRep = open("C:\\Users\\asus\\OneDrive\\BSC_brain_math\\year_c\\Yearly\\BCI\\bci4als\\recordings\\adi\\9\\trials.pickle", "rb")
file = pickle.load(objectRep)
all_data = np.zeros([len(file), 1230, 13])


sampling_rate = 125

fig, ax = plt.subplots(2,1, figsize=(16,4), sharey=True)
eeg = file[0]['C3']
time = np.arange(len(file[0]['C3']))/ sampling_rate

ax[0].plot(time, eeg, lw=1)
ax[0].set_xlabel('Time (sec)'), ax[0].set_ylabel('Voltage ($\mu$Volts)')
ax[0].set_xticks(np.arange(0, 10, 0.5))

ax[1].plot(time, eeg, lw=1, color='k')
ax[1].set_xlim(4.25,4.5)
#ax[1].set_xlim(12,14.5)
ax[1].set_xticks(np.arange(4.3,4.5,0.1))
ax[1].set_xlabel('Time (sec)')

FourierCoeff = np.fft.fft(eeg)/eeg.size
DC = [np.abs(FourierCoeff[0])]
amp = np.concatenate((DC, 2*np.abs(FourierCoeff[1:])))


# compute frequencies vector until half the sampling rate
Nyquist = sampling_rate/2
print('Nyquist frequency = %2.4f Hz'%Nyquist)
Nsamples = int( math.floor(eeg.size/2) )
hz = np.linspace(0, Nyquist, num = Nsamples + 1 )
dhz = hz[1]
print('Spectral resolution = %2.4f Hz'%hz[1])

# Perform Welch's periodogram
segment = int( 4*sampling_rate )
myhann = signal.get_window('hann', segment)

# obtain the power (uV^2) spectrum with Hann window and 50% overlap
myparams = dict(fs = sampling_rate, nperseg = segment, window = myhann, noverlap = segment/2,
                scaling = 'spectrum', return_onesided = True)
freq, ps = signal.welch(x = eeg, **myparams)# units uV**2
ps = 2*ps

# obtain the power density/Hz (uV^2) spectrum with Hann window and 50% overlap
# to get back to simply power, divide by the segment lenght in seconds (four in our case)
myparams2 = dict(fs = sampling_rate, nperseg = segment, window = myhann, noverlap = segment/2,
                 scaling = 'density', return_onesided = True)
freq, psd = signal.welch(x = eeg, **myparams2)# units uV**2/Hz
psd = 2*psd

dfreq = freq[1]
print('Spectral resolution = %2.4f Hz'%dfreq)

# Plot the power spectrum

fig, ax = plt.subplots(1, 2, figsize=(16, 4))

ax[0].set_title("Amplitude spectrum (Fourier transform)")
ax[0].plot(hz,amp[:len(hz)], lw=1, color='k')#, use_line_collection = True)
ax[0].plot(freq, np.sqrt(ps/10), color='red', lw=2)
ax[0].set_ylabel('Amplitude ($\mu V$)')

ax[1].set_title("Power spectrum (Welch's periodogram)")
ax[1].plot(hz, np.power(amp[:len(hz)],2), color='k', lw =1)
ax[1].plot(freq, (ps/10), color='C0', lw=2)#, use_line_collection = True)
ax[1].set_ylabel('Power ($\mu V^2$)')

for myax in ax:
    myax.set_xlabel('Frequency (Hz)')
    myax.set_xlim(0,40)
    myticks = list(range(0,40,10))
    myax.set_xticks(myticks)
    myax.set_ylim(0,5)
