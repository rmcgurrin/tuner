#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
import argparse
from collections import OrderedDict
from scipy import signal

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
        ('E4',329.63),
        ('B3',246.94),
        ('G3',196.00),
        ('D3',146.83),
        ('A2',110.00),
        ('E2',82.41)
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


    b, a = signal.butter(7, noteFreqs[args.note]/(sampleRate/2))
    d = signal.lfilter(b, a, d)
    #d=d[0:-1:16]
    plt.figure()
    plt.plot(d)
    plt.show()
    return

    sampleRate /= 16.
    numSamples /= 16

    minFreq = noteFreqs[args.note]*.5
    maxFreq = noteFreqs[args.note]*2
    binsPerOctave = 1200/5

    N,freqs = getN(minFreq, maxFreq, binsPerOctave, sampleRate)
    print(N)
    print(freqs)

    '''
    for note in noteFreqs.keys():
        bins = getHarmonicBins(noteFreqs[note], minFreq, binsPerOctave, len(N))
        noteBins[note] = bins
        noteHarmonics[note] = bins2freq(bins, minFreq, binsPerOctave)
        #print(note,bins,'\n')
        #print(note,noteHarmonics[note],'\n')
    '''


    sampleCount = 0
    #blockSize = np.max(N)
    blockSize = sampleRate
    print('max:',blockSize)

    while sampleCount + blockSize < numSamples:
        x = d[sampleCount:sampleCount+blockSize]
        cq, freqs = slowQ(x, minFreq, maxFreq, binsPerOctave, sampleRate)
        time = sampleCount/sampleRate

        '''
        power = {}
        #find the note powers
        for note in noteFreqs.keys():
            power[note] = 0
            cnt = 0
            for bin in noteBins[note]:
                power[note] += abs(cq[bin]**2)
                cnt += 1
                if cnt == 3:
                    break
            power[note] =  10*np.log10(power[note])
        '''

        if time > args.dbgtime and time < args.dbgtime + args.dbglen:
            #print(power)
            plt.figure()
            plt.plot(freqs,20*np.log10(abs(cq)),'b*-')
            plt.show()

        maxbin = np.argmax(cq[0:(np.int(len(cq)*.75))])
        print(len(cq),np.int(len(cq)))
        peakFreq = minFreq*2**(maxbin/binsPerOctave)
        print('Peak Freq',peakFreq)
        sampleCount += blockSize/4

if __name__ == "__main__":
    main()
