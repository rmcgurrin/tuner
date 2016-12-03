#!/usr/bin/env python
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import wave
import pyaudio

class BandPassFilter:

    def __init__(self,startFreq=50.0,stopFreq=1200.0,order=5,sampleRate=8000.0):

        nyquist = sampleRate/2.0
        low = startFreq/nyquist
        high = stopFreq/nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass')

        self.a = a
        self.b = b
        self.zf = np.zeros((max(len(a), len(b)) - 1,))

    def filter(self, x):
        y,z = signal.lfilter(self.b, self.a, x.astype(np.float),zi=self.zf)
        self.zf = z
        return y

class Tuner:

    def __init__(self, sampleRate=8000,startFreq=50,stopFreq=1200,fftLen=4096):

        self.sampleRate = sampleRate
        self.fftLen = fftLen
        self.bp = BandPassFilter(sampleRate=8000,startFreq=50,stopFreq=1200)
        self.peak = 0.0;
        self.avgCoeff = 1.
        self.blockSize = 1*fftLen

    def readStream(self):
        sampleType = pyaudio.paInt16
        channels = 2
        sampleRate = 22050

        # start Recording
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=sampleRate, input=True,
                        frames_per_buffer=self.blockSize)

        while True:
            data = stream.read(self.blockSize)
            data_d = np.fromstring(data, dtype=np.int16)
            data_d = np.reshape(data_d, (self.blockSize, 1))
            peak, power = self.processSamples(data_d[:,0])
            print(peak)

    def readWave(self, filename):

        w = wave.open(filename)
        self.sampleRate = w.getframerate()
        numSamples = w.getnframes()
        numChannels = w.getnchannels()
        sampleWidth = w.getsampwidth()

        d = w.readframes(numSamples)
        # TODO - handle bitwidth, assumes 16bits for now
        d = np.fromstring(d, dtype=np.int16)
        d = np.reshape(d, (numSamples, numChannels))
        d = d/32768

        sampleCount = 0
        sample = []
        ploc = []
        pwr =[]
        while sampleCount + self.blockSize < numSamples:
            x = d[sampleCount:sampleCount+self.blockSize-1,0]
            p, power = self.processSamples(x)
            ploc.append(p)
            pwr.append(power)
            sample.append(sampleCount/self.sampleRate)
            sampleCount += self.blockSize

            time = sampleCount/self.sampleRate
            if time < 19.8 and time > 19.6:
                print(time,p)
                plt.figure()
                plt.plot(x)

                plt.figure()
                plt.plot(self.f,10*np.log10(self.Pxx))

                plt.figure()
                plt.plot(self.f,10*np.log10(self.PxxH))

                plt.figure()
                pxx = np.fft.ifft(self.Pxx)
                t=np.arange(0,(self.fftLen/2)+1)/(self.sampleRate/2)
                plt.plot(t,pxx)

                plt.figure()
                X = np.abs(np.fft.rfft(x))
                plt.plot(X)
                plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(sample,ploc,'r-*')
        ax2 = ax1.twinx()
        ax2.plot(sample,pwr,'b-*')
        plt.show()


    def processSamples(self, x):

        # bandpass filter the signal
        y = self.bp.filter(x)

        # estimate the power spectrum of the signal
        self.f,self.Pxx = signal.periodogram(y,self.sampleRate,window='hanning',nfft=self.fftLen,scaling='spectrum')

        # create the harmonic spectrum
        self.PxxH = self.combineHarmonics(self.Pxx)
        fundamental = np.argmax(self.PxxH[0:len(self.PxxH)/3])

        # estimate the peak frequency
        peakLoc, power = self.interpPeak(self.Pxx,fundamental)
        power = 10*np.log10(power)
        if power < -50.0:
            peakLoc = 0
        else:
            peakLoc *= self.sampleRate/self.fftLen

        return peakLoc, power

    def interpPeak(self, Pxx, index):

        y1 = Pxx[index-1]
        y2 = Pxx[index]
        y3 = Pxx[index+1]

        offset = (y3-y1)/(y1 + y2 + y3)

        peakLoc = index + offset
        power = (y1 + y2 + y3)/3.0
        return peakLoc,power

    def combineHarmonics(self, Pxx):

        temp = np.copy(Pxx)
        for level in (2,3):
            N = np.int(len(Pxx)/level)
            for index in np.arange(0,N):
                temp[index] *= Pxx[index*level]
        return temp

    def testSignal(self, testFreq):
        x1=np.cos(2.0*np.pi*(testFreq/self.sampleRate)*np.arange(0,self.sampleRate-1));
        x2=1.0*np.cos(2.0*np.pi*(testFreq*2./self.sampleRate)*np.arange(0,self.sampleRate-1));
        x3=1.0*np.cos(2.0*np.pi*(testFreq*3./self.sampleRate)*np.arange(0,self.sampleRate-1));
        x = x1+x2+x3
        return x

def main():
    parser = argparse.ArgumentParser(description='Test the Tuner Class')
    parser.add_argument('--file', help='wave file name')
    parser.add_argument('--stream', action='store_true',help='get samples from mic')
    parser.add_argument('--test', action='store_true',help='run test waveform')
    args = parser.parse_args()

    t = Tuner()

    if args.file:
        t.readWave(args.file)
    elif args.stream:
        t.readStream()
    elif args.test:
        x = t.testSignal(199.)
        peak, power = t.processSamples(x)
        print(peak,power)

if __name__ == "__main__":
    main()
