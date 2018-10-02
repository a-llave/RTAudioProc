

"""
Description: Real time overlap-add blcok processing example

Author: Adrien Llave - CentraleSupelec
Date: 22/05/2018

Version: 1.0

Date    | Auth. | Vers. | Comments
18/05/22  ALl     1.0     Initialization

"""

import pyaudio
import time
import numpy as np
import __init__ as rt
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.clock()
    npdata_in = rt.decode(in_data, CHANNELS)                                            # INPUT DECODING
    BlockProc.input(npdata_in)
    frame_inp = BlockProc.get_frame()

    toto = np.fft.fft(frame_inp[:, 0], )
    threshold = -15
    ratio = 3
    tutu = 20*np.log10(np.abs(toto))
    tutu[tutu < threshold] = ratio * tutu[tutu < threshold] + threshold * (1 - ratio)
    titi = np.power(10, tutu / 20)
    toto = titi * np.exp(1j * np.angle(toto))
    BlockProc.toto = toto
    tata = np.real(np.fft.ifft(toto, ))
    frame_out = np.repeat(tata[:, np.newaxis], CHANNELS, axis=1)

    BlockProc.set_frame(frame_out)
    npdata_out = BlockProc.output()
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 2
RATE = 44100
samp_per_buffer = 220
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

BlockProc = rt.BlockProc(nb_buffsamp=samp_per_buffer, nb_channels=CHANNELS)

p = pyaudio.PyAudio()

# LOWPASS FILTER
# LPFilter = rt.Butterworth(samp_per_buffer,
#                           RATE,
#                           nb_channels=2,
#                           type_s='low',
#                           cut_freq=5000.,
#                           order_n=5)

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                # input_device_index=16,
                # output_device_index=16,
                frames_per_buffer=samp_per_buffer,
                stream_callback=callback)

plt.ion()
fig = plt.figure()
#
ax = fig.add_subplot(1, 1, 1)
line1, = ax.semilogx(BlockProc.frame1_m[:, 0], 'r')
plt.axis([0, samp_per_buffer, -60, 20])

stream.start_stream()
while stream.is_active():
    # line1.set_ydata(BlockProc.frame1_m[:, 0])
    # line1.set_ydata(BlockProc.window_m[:, 0])
    line1.set_ydata(20*np.log10(np.abs(BlockProc.toto)))
    fig.canvas.draw()
    time.sleep(0.2)

stream.stop_stream()
stream.close()

p.terminate()
