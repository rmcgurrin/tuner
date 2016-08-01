#!/usr/bin/env python
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal

class BandPassFilter:

    def __init__(self,startFreq=50.0,stopFreq=1200.0,order=5,sampleRate=8000.0):
        self.startFreq = startFreq
        self.stopFreq = stopFreq
        self.order = order
        self.sampleRate = sampleRate

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
        self.zf = signal.lfilter_zi(b,a)
        print(self.zf)

    def filter(self, x):
        y, zf = signal.lfilter(self.b, self.a, x/32768.0, zi=self.zf)
        print(x[0],y[0],zf)
        self.zf = zf
        return y


class Tuner:

    def __init__(self, sample_rate=8000):
        print("Hello World")
        bp = BandPassFilter()




def main():
    t = Tuner()

if __name__ == "__main__":
    main()
