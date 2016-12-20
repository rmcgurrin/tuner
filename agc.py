#!/usr/bin/env python
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import wave

class AGC:

    def __init__(self,rmsLevel=1.0,smoothCoeff=.01):

        self.rmsLevel = rmsLevel
        self.smoothCoeff = smoothCoeff
        self.avgCoeff = 1.0
        self.gainHist = 0.0
        self.powEstimateHist = 0.0

    def run(self, x):

        y = np.zeros(x.shape,dtype=x.dtype)
        self.gain = np.zeros(x.shape,dtype=x.dtype)
        self.diff = np.zeros(x.shape,dtype=x.dtype)

        for n in np.arange(0,len(y)):

            self.diff[n] = self.rmsLevel - self.powEstimateHist
            self.gain[n] = self.gainHist + self.smoothCoeff*self.diff[n]
            y[n] = x[n]*self.gain[n]
            self.gainHist = self.gain[n]
            self.powEstimateHist = y[n]**2

        '''
        plt.figure()
        plt.plot(self.gain)
        plt.title('Gain')
        plt.figure()
        plt.plot(self.diff)
        plt.title('Diff')
        '''


        return y


def main():
    parser = argparse.ArgumentParser(description='Test the Tuner Class')
    parser.add_argument('--file', help='wave file name')
    parser.add_argument('--dbgtime', help='timestamp used for debugging', type = float, default = 0. )
    parser.add_argument('--dbglen', help=' duration of time for debug', type = float, default = 0. )
    args = parser.parse_args()

    w = wave.open(args.file)
    sampleRate = w.getframerate()
    numSamples = w.getnframes()
    numChannels = w.getnchannels()
    sampleWidth = w.getsampwidth()

    x = w.readframes(numSamples)
    # TODO - handle bitwidth, assumes 16bits for now
    x = np.fromstring(x, dtype=np.int16)
    x = np.reshape(x, (numSamples, numChannels))
    x = x/32768.
    x=x[0:40000]

    #x = np.ones((10000,),dtype=np.float)*.1

    a=AGC()
    y = a.run(x)

    plt.figure()
    plt.title('Output')
    plt.plot(y,'r')
    plt.figure()
    plt.plot(x)
    plt.title('Input')
    plt.show()

if __name__ == "__main__":
    main()
