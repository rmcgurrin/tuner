#!/usr/bin/env python
import pyaudio
import wave
import numpy as np
import time
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

#set up the plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(0,CHUNK)
line1, = ax1.plot(x, np.zeros((CHUNK,)))
#fig.canvas.draw()

print("recording...")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #data = stream.read(CHUNK)
    print(time.time())
    line1.set_ydata(np.ones((CHUNK,)))
    fig.canvas.draw()

print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
