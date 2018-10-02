

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

    npdata_in = rt.decode(in_data, CHANNELS)                                            # INPUT DECODING
    npdata_in = np.repeat(npdata_in[:, 0][:, np.newaxis], 2, axis=1)
    frame_inp = BlockProc.input2frame(npdata_in)
    fft_inp = np.fft.fft(frame_inp, nb_frmesamp, axis=0)[0:nb_frmesamp_2, :]
    # start_time = time.clock()
    fft_out = ns.processfft(fft_inp)
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_frmesamp_2 - 1, :]))), axis=0)
    frame_out = np.real(np.fft.ifft(fft_out, axis=0))
    npdata_out = BlockProc.frame2output(frame_out)
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 2
RATE = 16000
nb_buffsamp = 32
nb_frmesamp = 128
nb_frmesamp_2 = int(nb_frmesamp/2)+1
print('number sample per buffer:', nb_buffsamp)
print('frame duration (ms):', nb_buffsamp/RATE*1000)

BlockProc = rt.BlockProc2(nb_buffsamp=nb_buffsamp, nb_fft=nb_frmesamp, nb_channels=CHANNELS)

ns = rt.NoiseSoustraction(nb_buffsamp=nb_frmesamp, nb_channels=CHANNELS, samp_freq=RATE, threshold_f=6)
p = pyaudio.PyAudio()

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                # input_device_index=16,
                # output_device_index=16,
                frames_per_buffer=nb_buffsamp,
                stream_callback=callback)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1, = ax.semilogx(ns.magdb_inp[:, 0], 'r')
line2, = ax.semilogx(ns.noisedb_prev[:, 0], 'g')
line3, = ax.semilogx(20 * np.log10(np.abs(np.fft.fft(ns.magdb_out[:, 0]))), 'b')
line4, = ax.semilogx(ns.gain_redu_prev[:, 0], 'black')
plt.axis([0, nb_frmesamp_2, -60, 20])

ii = 0

stream.start_stream()
while stream.is_active():
    line1.set_ydata(ns.magdb_inp[:, 0])
    line2.set_ydata(ns.noisedb_prev[:, 0])
    line3.set_ydata(ns.magdb_out[:, 0])
    line4.set_ydata(20 * np.log10(ns.gain_redu_prev[:, 0]))
    fig.canvas.draw()
    #
    ii += 1
    if ii > 30:
        ns.bypass = not ns.bypass
        ii = 0
        print('BYPASS: ' + str(ns.bypass))

    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
