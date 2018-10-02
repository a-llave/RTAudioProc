
"""
Description:    Real time dynamic range compression (DRC) (stereo)
                Based on Digital Dynamic Range Compressor Design - A Tutorial and Analysis, Giannoulis et al. 2012

Author: Adrien Llave - CentraleSupelec
Date: 20/06/2018

Version: 1.0

Date    | Auth. | Version | Comments
18/06/20  ALl       1.0     Initialization

"""

import pyaudio
import time
import numpy as np
import matplotlib.pyplot as plt
import src.pkg.RTAudioProc as rt


def callback(in_data, frame_count, time_info, status):

    # ------------------------------------------------------------
    # ----------------- INPUT DECODING
    npdata_in = rt.decode(in_data, CHANNELS)
    BlockProc.input(npdata_in)
    frame_inp = BlockProc.get_frame()
    fft_inp = np.fft.fft(frame_inp, axis=0)[0:nb_frmesamp_2, :]
    # start_time = time.clock()
    fft_inp = ns.processfft(fft_inp)
    # ------------------------------------------------------------
    # ----------------- COMPRESSION
    fft_out = compressor.processfft(fft_inp)
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    # ------------------------------------------------------------
    # ----------------- OUTPUT ENCODING
    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_frmesamp_2 - 1, :]))), axis=0)
    frame_out = np.real(np.fft.ifft(fft_out, axis=0))
    BlockProc.set_frame(frame_out)
    npdata_out = BlockProc.output()
    data = rt.encode(npdata_out)

    return data, pyaudio.paContinue


# ----------------------------------------------------------------

CHANNELS = 2
RATE = 16000
nb_buffsamp = 64
nb_frmesamp = nb_buffsamp * 2
nb_frmesamp_2 = int(nb_frmesamp/2)+1

thrsh = -30
ratio = 10
time_attack = 0.005
time_release = 2
kneeWidth = 3

p = pyaudio.PyAudio()


BlockProc = rt.BlockProc(nb_buffsamp=nb_buffsamp, nb_channels=CHANNELS)

ns = rt.NoiseSoustraction(nb_buffsamp=nb_frmesamp, nb_channels=CHANNELS, samp_freq=RATE, threshold_f=6)

compressor = rt.CompressorFFT(nb_buffsamp=nb_frmesamp, nb_channels=CHANNELS, samp_freq=RATE,
                              threshold_f=-30, ratio_f=3, knee_width=10,
                              time_attack=0.005, time_release=0.05,
                              bypass=False, verbose=False)

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

stream.start_stream()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1, = ax.semilogx(compressor.mag_inp[:, 0], 'b')
line2, = ax.semilogx(compressor.mag_out[:, 0], 'r')
line3, = ax.semilogx(compressor.gr2_plot[:, 0], 'black')
plt.axis([0, nb_frmesamp_2, -60, 20])

# ax2 = fig.add_subplot(2, 1, 2)
# line21, = ax.semilogx(ns.magdb_inp[:, 0], 'r')
# line22, = ax.semilogx(ns.noisedb_prev[:, 0], 'g')
# line23, = ax.semilogx(20 * np.log10(np.abs(np.fft.fft(ns.magdb_out[:, 0]))), 'b')
# line24, = ax.semilogx(ns.gain_redu_prev[:, 0], 'black')

ii = 0

while stream.is_active():
    time.sleep(0.1)
    line1.set_ydata(compressor.mag_inp[:, 0])
    line2.set_ydata(compressor.mag_out[:, 0])
    line3.set_ydata(compressor.gr2_plot[:, 0])

    # line21.set_ydata(ns.magdb_inp[:, 0])
    # line22.set_ydata(ns.noisedb_prev[:, 0])
    # line23.set_ydata(ns.magdb_out[:, 0])
    # line24.set_ydata(20 * np.log10(ns.gain_redu_prev[:, 0]))
    fig.canvas.draw()
    ii += 1
    if ii > 30:
        ns.bypass = not ns.bypass
        compressor.bypass = not compressor.bypass
        ii = 0
        print('BYPASS: ' + str(ns.bypass))

stream.stop_stream()
stream.close()

p.terminate()

