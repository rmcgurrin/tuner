#!/usr/bin/env python
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal

class BandPassFilter:

    def __init__(self,startFreq=50.0,stopFreq=1200.0,order=5,sampleRate=8000.0):

        nyquist = sampleRate/2.0
        low = startFreq/nyquist
        high = stopFreq/nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass')

        '''
        w, h = signal.freqz(b, a)
        plt.plot(nyquist*w/math.pi, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(startFreq, color='green') # cutoff frequency
        plt.axvline(stopFreq, color='green') # cutoff frequency
        plt.show()
        '''

        self.a = a
        self.b = b
        self.zf = np.zeros((max(len(a), len(b)) - 1,))

    def filter(self, x):
        y,z = signal.lfilter(self.b, self.a, x.astype(np.float),zi=self.zf)
        self.zf = z
        return y
class Tuner:

    def __init__(self, sampleRate=8000,startFreq=50,stopFreq=1200,fftLen=1024):

        self.sampleRate = sampleRate
        self.fftLen = fftLen
        self.bp = BandPassFilter(sampleRate=8000,startFreq=50,stopFreq=1200)

    def run(self, x):

        # bandpass filter the signal
        y = self.bp.filter(x)

        # estimate the power spectrum of the signal
        f,Pxx = signal.periodogram(y,self.sampleRate,window='hanning',nfft=self.fftLen,scaling='spectrum')

        # create the harmonic spectrum
        PxxH = self.combineHarmonics(Pxx)
        fundamental = np.argmax(PxxH)

        # estimate the peak frequency
        peakLoc = self.interpPeak(Pxx,fundamental)

        print('peak found at %f %f'%(peakLoc,peakLoc*self.sampleRate/self.fftLen))

        return f,Pxx

    def interpPeak(self, Pxx, index):

        y1 = Pxx[index-1]
        y2 = Pxx[index]
        y3 = Pxx[index+1]

        offset = (y3-y1)/(y1 + y2 + y3)

        peakLoc = index + offset

        return peakLoc

    def combineHarmonics(self, Pxx):

        temp = np.copy(Pxx)
        for level in (2,3):
            print(len(Pxx),level,len(Pxx)/level)
            for index in np.arange(0,len(Pxx)/level):
                temp[index] *= Pxx[index*level]
        return temp

def main():

    sampleRate = 8000.
    fftLen = 1024
    testFreq = 199.0
    t = Tuner(sampleRate=sampleRate,startFreq=50,stopFreq=1200,fftLen=fftLen)

    x1=np.cos(2.0*np.pi*(testFreq/sampleRate)*np.arange(0,sampleRate-1));
    x2=2.0*np.cos(2.0*np.pi*(testFreq*2./sampleRate)*np.arange(0,sampleRate-1));
    x3=3.0*np.cos(2.0*np.pi*(testFreq*3./sampleRate)*np.arange(0,sampleRate-1));
    x = x1+x2+x3
    f,Pxx = t.run(x)

    plt.plot(10*np.log10(Pxx))
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
