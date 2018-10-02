
"""
Description:    Real time dynamic range compression (DRC) (stereo)
                Based on Digital Dynamic Range Compressor Design - A Tutorial and Analysis, Giannoulis et al. 2012

Author: Adrien Llave - CentraleSupelec
Date: 23/02/2018

Version: 1.0

Date    | Auth. | Version | Comments
23/02/12  ALl       1.0     Initialization

"""

import pyaudio
import time
import numpy as np
import matplotlib.pyplot as plt
import src.pkg.RTAudioProc as rt


CHANNELS = 2
RATE = 48000
samp_per_buffer = 10

comp_enable = 1

thrsh = -30
ratio = 10
time_attack = 0.005
time_release = 2
kneeWidth = 3

p = pyaudio.PyAudio()


def callback(in_data, frame_count, time_info, status):

    # ------------------------------------------------------------
    # ----------------- INPUT DECODING
    npdata_in = rt.decode(in_data, CHANNELS)

    # ------------------------------------------------------------
    # ----------------- COMPRESSION
    npdata_out = compressor.process(npdata_in)
    # ------------------------------------------------------------
    # ----------------- OUTPUT ENCODING
    data = rt.encode(npdata_out)

    return data, pyaudio.paContinue

# ----------------------------------------------------------------


compressor = rt.Compressor(samp_per_buffer, CHANNELS, thrsh, ratio, time_attack, time_release, kneeWidth)

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

stream.start_stream()

gain_red_db_plt = np.mean(compressor.gain_db, axis=0)

plt.ion()
fig = plt.figure()
#
ax = fig.add_subplot(2, 2, 1)
line1, = ax.plot(gain_red_db_plt, 'ro')
plt.axis([0, 1, -10, 0])
#
# ax2 = fig.add_subplot(222)
# plt.axis([0, samp_per_buffer, -1, 1])
# line2, = ax2.plot(npdata_in[:, 1], 'b-')
#
# ax3 = fig.add_subplot(223)
# plt.axis('equal')
# plt.axis([-60, 0, -60, 0])
# ax3.grid(color='black', linestyle='-', linewidth=1)
# line3, = ax3.plot(npdata_in[:, 1], npdata_out[:, 1], 'ro')
# line4, = ax3.plot(np.arange(-60, 0.0, 1), np.arange(-60, 0.0, 1), '-b')

while stream.is_active():
    time.sleep(0.1)
    gain_red_db_plt = np.mean(compressor.gain_db, axis=0)
    # mean_datainp = np.mean(20 * np.log10(np.absolute(npdata_in[:, :] / np.power(2, 16))), axis=0)
    # mean_dataout = np.mean(20 * np.log10(np.absolute(npdata_out[:, :] / np.power(2, 16))), axis=0)
    line1.set_ydata(gain_red_db_plt)
    # line2.set_ydata(npdata_in[:, 0] / np.power(2, 16))
    # line3.set_xdata(mean_datainp)
    # line3.set_ydata(mean_dataout)
    fig.canvas.draw()

stream.stop_stream()
stream.close()

p.terminate()

