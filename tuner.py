#!/usr/bin/env python
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import argparse
import wave
import pyaudio
from agc import AGC
from collections import OrderedDict
from filterpy.gh import GHFilter

noteFreqs=OrderedDict(
    [
        ('E4',329.63),
        ('B3',246.94),
        ('G3',196.00),
        ('D3',146.83),
        ('A2',110.00),
        ('E2',82.41)
    ]
)

class BandPassFilter:

    def __init__(self,startFreq=0.0,stopFreq=1200.0,order=3,sampleRate=44100.0):

        nyquist = sampleRate/2.0
        low = startFreq/nyquist
        high = stopFreq/nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass')
        #b, a = signal.butter(order, high)

        self.a = a
        self.b = b
        self.zf = np.zeros((max(len(a), len(b)) - 1,))

    def filter(self, x):
        y,z = signal.lfilter(self.b, self.a, x.astype(np.float),zi=self.zf)
        self.zf = z
        return y

class Tuner:

    def __init__(self, sampleRate=44100,startFreq=10,stopFreq=1200,fftLen=4096,
        dbgTimeStart=0.,dbgTimeLen=0.,blockSize=1024,note='E2'):

        self.sampleRate = sampleRate
        self.fftLen = fftLen
        self.peak = 0.0
        self.avgCoeff = 1.
        self.blockSize = blockSize

        self.dbgTimeStart = dbgTimeStart
        self.dbgTimeLen = dbgTimeLen

        self.agc = AGC()

        self.note = note

        self.power = 0;
        self.tracker = GHFilter(x=noteFreqs[note], dx=0, dt=1., g=.05, h=.005)

        self.trackLoc = []


    def readStream(self):
        sampleType = pyaudio.paInt16
        channels = 2
        self.sampleRate = 8000
        self.filt = BandPassFilter(sampleRate=self.sampleRate,stopFreq=noteFreqs[self.note]);

        # start streaming samples from the soundcard
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=self.sampleRate, input=True,
                        frames_per_buffer=self.blockSize)

        while True:
            data = stream.read(self.blockSize)
            data_d = np.fromstring(data, dtype=np.int16)
            data_d = np.reshape(data_d, (self.blockSize, 1))
            peak, power = self.processSamples_autocorr(data_d[:,0]/32768.)
            print(peak,power)

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

        self.filt = BandPassFilter(sampleRate=self.sampleRate,stopFreq=noteFreqs[self.note]*2.);

        while sampleCount + self.blockSize < numSamples:
            x = d[sampleCount:sampleCount+self.blockSize-1,0]
            #p, power = self.processSamples_hps(x)
            p, power = self.processSamples_autocorr(x)
            ploc.append(p)
            pwr.append(power)
            sample.append(sampleCount/self.sampleRate)
            sampleCount += self.blockSize

            time = sampleCount/self.sampleRate

            if time > self.dbgTimeStart and time < self.dbgTimeStart + self.dbgTimeLen:

                print(time,p)
                plt.figure()
                plt.plot(x)

                #plt.figure()
                #plt.plot(self.f,10*np.log10(self.Pxx))

                #plt.figure()
                #plt.plot(self.f,10*np.log10(self.PxxH))

                #plt.figure()
                #pxx = np.fft.ifft(self.Pxx)
                #t=np.arange(0,(self.fftLen/2)+1)/(self.sampleRate/2)
                #plt.plot(t,pxx)

                plt.figure()
                plt.plot(self.xx)
                plt.plot(self.fit_range, self.xx_fit)
                plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(sample,ploc,'r*')
        ax1.plot(sample,self.trackLoc,'r')
        ax2 = ax1.twinx()
        ax2.plot(sample,pwr,'b')
        temp = np.array(pwr)
        temp = np.append(temp,0)
        ax2.plot(sample,np.abs(np.diff(temp)),'g')

        plt.figure()
        cents = 1200*np.log2(np.array(self.trackLoc)/noteFreqs[self.note])
        plt.plot(sample,cents)

        plt.show()
        print('HI')

    def processSamples_hps(self, x):

        # bandpass filter the signal
        x = self.filt.filter(x)
        x = self.agc.run(x)

        power = np.sum(x**2)/len(x)
        power = 10.*np.log10(power)

        # estimate the power spectrum of the signal
        self.f,self.Pxx = signal.periodogram(x,self.sampleRate,window='hanning',nfft=self.fftLen,scaling='spectrum')

        # create the harmonic spectrum
        self.PxxH = self.harmonicProductSpectrum(self.Pxx)
        fundamental = np.argmax(self.PxxH[0:np.uint16(len(self.PxxH)/3)])

        # estimate the peak frequency
        peakLoc, peakPower = self.interpPeak(self.Pxx,fundamental)
        if power < -3. or power > 3.0:
            peakLoc = 0
        else:
            peakLoc *= self.sampleRate/self.fftLen

        return peakLoc, power

    def peak_fit(self,x,a,b,c):
        return a*x**2 + b*x +c

    def processSamples_autocorr(self, x):

        power = np.sum(x**2)/len(x)
        power = 10.*np.log10(power)

        # run the automatic gain control algorithm to normalize the signal level
        # this should improve detecting the fundamental frequency in the
        # autocorrelation sequence
        #x = self.agc.run(x)
        x = self.filt.filter(x)

        X = np.fft.fft(x,n=self.fftLen)
        freq = np.fft.fftfreq(len(X), 1/self.sampleRate)
        i = freq > 0

        #plt.figure()
        #plt.plot(freq[i],np.abs(X[i]))
        XX = X*X.conj()
        xx = np.fft.ifft(XX,n=self.fftLen).real
        self.xx = xx

        # determine the range to search in the autocorr sequence
        stringFreqLo = noteFreqs[self.note]*2**(-2/12)
        stringFreqHi = noteFreqs[self.note]*2**(2/12)
        hi = np.int(self.sampleRate/stringFreqLo)
        lo = np.int(self.sampleRate/stringFreqHi)
        tt = np.argmax(xx[lo:hi])
        tt += lo

        #fit a parabola to interpolate the peak
        xdata = np.arange(-10,10)
        popt,_ = curve_fit(self.peak_fit, xdata, xx[tt+xdata])
        self.fit_range = np.arange(-10,9,.01)
        self.xx_fit = self.peak_fit(self.fit_range,popt[0],popt[1],popt[2])
        fit_peak_loc = np.argmax(self.xx_fit)
        peakLoc = self.sampleRate/(self.fit_range[fit_peak_loc]+tt)


        if fit_peak_loc+1 >= len(self.xx_fit):
            peakLoc = 0
        elif self.xx_fit[fit_peak_loc] < self.xx_fit[fit_peak_loc+1]:
            peakLoc = 0

        if power < -50.:
            peakLoc = 0

        #only use parts of the waveform where its not rising too quickly
        if np.abs(power - self.power) > 5:
            peakLoc = 0

        self.power = power

        if peakLoc == 0:
            self.tracker.update(self.tracker.x)
            self.tracker.dx = 0
        else:
            self.tracker.update(z=peakLoc)

        self.trackLoc.append(self.tracker.x)


        return peakLoc, power

    def interpPeak(self, Pxx, index):

        y1 = Pxx[index-1]
        y2 = Pxx[index]
        y3 = Pxx[index+1]

        offset = (y3-y1)/(y1 + y2 + y3)
        peakLoc = index + offset
        power = (y1 + y2 + y3)/3.0
        return peakLoc,power

    def harmonicProductSpectrum(self, Pxx):

        temp = np.copy(Pxx)
        for level in (2,3,4,5):
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
    parser.add_argument('--dbgtime', help='timestamp used for debugging', type = float, default = 0. )
    parser.add_argument('--dbglen', help=' duration of time for debug', type = float, default = 0. )
    parser.add_argument('--note', help=' note pitch', default = 'E4' )
    args = parser.parse_args()

    t = Tuner(dbgTimeStart = args.dbgtime, dbgTimeLen = args.dbglen, note=args.note)

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
