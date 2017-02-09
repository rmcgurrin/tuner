#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
import argparse
from collections import OrderedDict
from scipy import signal
from scipy.signal import find_peaks_cwt
from tuner import BandPassFilter

'Frequencies in xScientific pitch notation'
noteFreqs=OrderedDict(
    [
        ('E2',  82.41),
        ('F2',  87.31),
        ('F2S', 92.50),
        ('G2',  98.00),
        ('G2B', 103.83),
        ('A2',  110.00),
        ('A2S', 116.54),
        ('B2',  123.47),
        ('C3',  130.81),
        ('C3S', 138.59),
        ('D3',  146.83),
        ('D3S', 155.56),
        ('E3',  164.81),
        ('F3',  174.61),
        ('F3S', 185.00),
        ('G3',  196.00),
        ('G3S', 207.65),
        ('A3',  220.00),
        ('A3S', 233.88),
        ('B3',  246.94),
        ('C4',  261.63),
        ('C4S', 277.18),
        ('D4',  293.66),
        ('D4S', 311.13),
        ('E4',  329.63),
        ('F4',  349.23),
        ('F4S', 369.99),
        ('G4',  392.00)
    ]
)

class NoteFinder:

    def __init__(self,note,sampleRate):

        self.minFreq = 50
        self.maxFreq = 1200
        self.binsPerOctave = 12
        self.sampleRate = sampleRate
        self.note = note

        self.N,self.freqs = self.getN()
        self.numBins = len(self.N)

        self.noteBins={}
        self.noteHarmonics={}

        for note in noteFreqs.keys():
            bins = self.getHarmonicBins(noteFreqs[note])
            self.noteBins[note] = bins
            self.noteHarmonics[note] = self.bins2freq(bins)
        print(self.noteBins)

    def matchHarmonics(self,peaks):

        res = None
        if len(peaks) < 3:
            return res
        p = peaks[0:3]

        d_min = 99.
        note_min = 'E2'
        notes = noteFreqs.keys()
        for note in notes:
            d = np.linalg.norm(p - np.array(self.noteBins[note]))
            if d < d_min:
                d_min = d
                res = note

        if d_min < 2.0:
            return res
        else:
            return None

    def getN(self):

        Q= 1/(2**(1/self.binsPerOctave)-1)
        numFreqs = np.ceil(self.binsPerOctave*np.log2(self.maxFreq/self.minFreq))
        numFreqs = np.int(numFreqs)
        freqs = self.minFreq*2**(np.arange(0,numFreqs)/self.binsPerOctave)
        N = np.zeros((numFreqs,),np.int)
        for k in np.arange(0,numFreqs):
            N[k] = np.round(Q*self.sampleRate/(self.minFreq*2**(k/self.binsPerOctave)))
            N[k] = np.int(N[k])

        return N,freqs

    def getHarmonicBins(self, freq_in):

        bins = []
        freq = freq_in
        while True:
            bin = np.int(self.binsPerOctave*np.log2(freq/self.minFreq))
            if bin >= self.numBins:
                break
            bins.append(bin)
            if len(bins) >= 3:
                break
            freq += freq_in
        return(bins)

    def bins2freq(self, bins):
        freqs = []
        for bin in bins:
            f = self.minFreq*2**(bin/self.binsPerOctave)
            freqs.append(f)
        return freqs

    def slowQ(self, x):
        Q= 1/(2**(1/self.binsPerOctave)-1)

        numFreqs = np.ceil(self.binsPerOctave*np.log2(self.maxFreq/self.minFreq))
        numFreqs = np.int(numFreqs)

        freqs = self.minFreq*2**(np.arange(0,numFreqs)/self.binsPerOctave)

        cq = np.zeros(freqs.shape,freqs.dtype)
        for k in np.arange(0,numFreqs):
            N = np.round(Q*self.sampleRate/(self.minFreq*2**(k/self.binsPerOctave)))
            N = np.int(N)
            basis = np.exp( -2*np.pi*1j*Q*np.arange(0,N)/N)

            temp = np.zeros((N,),dtype=np.float)
            xLen = min(N,len(x))
            temp[0:xLen] = x[0:xLen]*np.hamming(xLen)
            cq[k]= abs(temp.dot( basis) / N)

        return cq,freqs

    def find_peaks(self, x,width,thresh_dB,snr_dB):

        xMax = 20*np.log10(np.max(x));
        thresh = np.maximum(xMax-30,thresh_dB)

        thresh = 10**(thresh/20)
        snr = 10**(snr_dB/20)
        peaks=[]
        vals=[]
        for i in range(width,len(x)-width):
            if x[i] < thresh:
                continue
            if x[i] > x[i-1] and x[i] > x[i+1]:
                if x[i] > x[i-width] and x[i] > x[i+width]:
                    if x[i] > snr*(x[i-width]+x[i+width])/2:
                        peaks.append(i)
                        vals.append(x[i])
        ind = np.argsort(vals)

        return peaks

    def findNote(self, x):
        self.cq, self.freqs = self.slowQ(x)
        self.peaks = self.find_peaks(self.cq,3,-80,6)
        test = self.matchHarmonics(self.peaks)
        return(test)



def main():

    parser = argparse.ArgumentParser(description='Test the Tuner Class')
    parser.add_argument('--note', help='guitar note, <E4,B3,G3,D3,A2,E2', default = 'E4' )
    parser.add_argument('--file', help='wave file name')
    parser.add_argument('--dbgtime', help='timestamp used for debugging', type = float, default = 0. )
    parser.add_argument('--dbglen', help=' duration of time for debug', type = float, default = 0. )
    args = parser.parse_args()

    w = wave.open(args.file)
    sampleRate = w.getframerate()
    numSamples = w.getnframes()
    numChannels = w.getnchannels()
    sampleWidth = w.getsampwidth()

    d = w.readframes(numSamples)
    # TODO - handle bitwidth, assumes 16bits for now
    d = np.fromstring(d, dtype=np.int16)
    d = np.reshape(d, (numSamples,))
    d = d/32768.

    sampleCount = 0
    blockSize = sampleRate

    nf = NoteFinder(args.note,sampleRate)

    filt = BandPassFilter(sampleRate=sampleRate,startFreq=noteFreqs[args.note]*.9,stopFreq=noteFreqs[args.note]*2.5);

    while sampleCount + blockSize < numSamples:
        time = sampleCount/sampleRate
        x = d[sampleCount:sampleCount+blockSize]
        x = filt.filter(x)

        test = nf.findNote(x)
        print(time, test)

        if time > args.dbgtime and time < args.dbgtime + args.dbglen:
            plt.figure()
            plt.plot(20*np.log10(abs(nf.cq)),'b*-')
            plt.show()

        sampleCount += blockSize/4

if __name__ == "__main__":
    main()
