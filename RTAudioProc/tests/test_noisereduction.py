

"""
Description: Real time overlap-add blcok processing example

Author: Adrien Llave - CentraleSupelec
Date: 22/05/2018

Version: 1.0

Date    | Auth. | Comments
18/05/22  ALl     Initialization

"""

import pyaudio
import time
import numpy as np
import src.pkg.RTAudioProc as rt
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.clock()
    npdata_in = rt.decode(in_data, CHANNELS)                                            # INPUT DECODING
    npdata_in = np.repeat(npdata_in[:, 0][:, np.newaxis], 2, axis=1)
    BlockProc.input(npdata_in)
    frame_inp = BlockProc.get_frame()
    frame_out = ns.process(frame_inp)
    BlockProc.set_frame(frame_out)
    npdata_out = BlockProc.output()
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 2
RATE = 16000
samp_per_buffer = 160
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

BlockProc = rt.BlockProc(nb_buffsamp=samp_per_buffer, nb_channels=CHANNELS)

ns = rt.NoiseReduction(nb_buffsamp=samp_per_buffer*2, nb_channels=CHANNELS, threshold_f=-15, ratio_f=3)
p = pyaudio.PyAudio()

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
ax = fig.add_subplot(1, 1, 1)
line1, = ax.semilogx(ns.sig_out[:, 0], 'r')
line2, = ax.semilogx(ns.sig_inp[:, 0], 'g')
line3, = ax.semilogx(ns.threshold_f[:, 0], 'b')
plt.axis([0, samp_per_buffer, -60, 20])

ii = 0

stream.start_stream()
while stream.is_active():
    # line1.set_ydata(20 * np.log10(np.abs(np.fft.fft(ns.sig_out[:, 0]))))
    line1.set_ydata(20 * np.log10(np.abs(np.fft.fft(ns.sig_inp[:, 0]))))
    line2.set_ydata(-ns.grconv_m[:, 0])
    line3.set_ydata(ns.threshold_f[:, 0])
    fig.canvas.draw()

    ii += 1
    if ii > 20:
        ns.bypass = not ns.bypass
        ii = 0
        print('BYPASS: ' + str(ns.bypass))

    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
