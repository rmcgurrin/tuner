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

'''
    open    1st fret    2nd fret   3rd fret     4th fret
6th string  E       F           F♯/G♭      G            G♯/A♭
5th string	A       A♯/B♭	    B	       C	        C♯/D♭
4th string	D	    D♯/E♭	    E	       F	        F♯/G♭
3rd string	G	    G♯/A♭	    A	       A♯/B♭	    B
2nd string	B	    C	        C♯/D♭	   D	        D♯/E♭
1st string	E	    F	        F♯/G♭	   G	        G♯/A♭

String	Frequency	Scientific pitch notation
1 (E)	329.63 Hz	E4
2 (B)	246.94 Hz	B3
3 (G)	196.00 Hz	G3
4 (D)	146.83 Hz	D3
5 (A)	110.00 Hz	A2
6 (E)	82.41 Hz	E2
'''

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

noteBins={
    'E4':[],
    'B3':[],
    'G3':[],
    'D3':[],
    'A2':[],
    'E2':[]
}

noteHarmonics={
    'E4':[],
    'B3':[],
    'G3':[],
    'D3':[],
    'A2':[],
    'E2':[]
}

def matchHarmonics(peaks):

    res = None
    if len(peaks) < 3:
        return res
    p = peaks[0:3]

    d_min = 99.
    note_min = 'E2'
    notes = noteFreqs.keys()
    for note in notes:
        d = np.linalg.norm(p - np.array(noteBins[note]))
        if d < d_min:
            d_min = d
            res = note

    if d_min < 2.0:
        return res
    else:
        return None

def getN(minFreq, maxFreq, bins, fs):

    Q= 1/(2**(1/bins)-1)
    numFreqs = np.ceil(bins*np.log2(maxFreq/minFreq))
    numFreqs = np.int(numFreqs)
    freqs = minFreq*2**(np.arange(0,numFreqs)/bins)
    N = np.zeros((numFreqs,),np.int)
    for k in np.arange(0,numFreqs):
        N[k] = np.round(Q*fs/(minFreq*2**(k/bins)))
        N[k] = np.int(N[k])

    return N,freqs

def getHarmonicBins( freq_in, minFreq, binsPerOctave, numBins):

    bins = []
    freq = freq_in
    while True:
        bin = np.int(binsPerOctave*np.log2(freq/minFreq))
        if bin >= numBins:
            break
        bins.append(bin)
        if len(bins) >= 3:
            break
        freq += freq_in
    return(bins)

def bins2freq(bins,minFreq,binsPerOctave):
    freqs = []
    for bin in bins:
        f = minFreq*2**(bin/binsPerOctave)
        freqs.append(f)
    return freqs

def slowQ(x, minFreq, maxFreq, bins, fs):
    Q= 1/(2**(1/bins)-1)

    numFreqs = np.ceil(bins*np.log2(maxFreq/minFreq))
    numFreqs = np.int(numFreqs)

    freqs = minFreq*2**(np.arange(0,numFreqs)/bins)

    cq = np.zeros(freqs.shape,freqs.dtype)
    for k in np.arange(0,numFreqs):
        N = np.round(Q*fs/(minFreq*2**(k/bins)))
        N = np.int(N)
        basis = np.exp( -2*np.pi*1j*Q*np.arange(0,N)/N)

        temp = np.zeros((N,),dtype=np.float)
        xLen = min(N,len(x))
        temp[0:xLen] = x[0:xLen]*np.hamming(xLen)
        cq[k]= abs(temp.dot( basis) / N)

    return cq,freqs

def find_peaks(x,width,thresh_dB,snr_dB):


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

    minFreq = 50
    maxFreq = 1200
    binsPerOctave = 12

    N,freqs = getN(minFreq, maxFreq, binsPerOctave, sampleRate)

    for note in noteFreqs.keys():
        bins = getHarmonicBins(noteFreqs[note], minFreq, binsPerOctave, len(N))
        noteBins[note] = bins
        noteHarmonics[note] = bins2freq(bins, minFreq, binsPerOctave)

    sampleCount = 0
    blockSize = sampleRate

    filt = BandPassFilter(sampleRate=sampleRate,startFreq=noteFreqs[args.note]*.9,stopFreq=noteFreqs[args.note]*2.5);

    while sampleCount + blockSize < numSamples:
        time = sampleCount/sampleRate
        print(time)
        x = d[sampleCount:sampleCount+blockSize]
        x = filt.filter(x)
        cq, freqs = slowQ(x, minFreq, maxFreq, binsPerOctave, sampleRate)
        peaks = find_peaks(cq,3,-80,6)
        test = matchHarmonics(peaks)
        print(test)

        if time > args.dbgtime and time < args.dbgtime + args.dbglen:
            plt.figure()
            plt.plot(20*np.log10(abs(cq)),'b*-')
            plt.show()

        sampleCount += blockSize/4

if __name__ == "__main__":
    main()
