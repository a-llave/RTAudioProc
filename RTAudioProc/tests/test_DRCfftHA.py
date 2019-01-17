
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
import RTAudioProc as rt
import utils_llave as u

def callback(in_data, frame_count, time_info, status):

    # ------------------------------------------------------------
    # ----------------- INPUT DECODING
    npdata_inp = rt.decode(in_data, CHANNELS)
    npdata_inp = npdata_inp * 0.1
    npdata_inp = npdata_inp * u.db2mag(100)
    frame_inp = BlockProc.input2frame(npdata_inp)
    fft_inp = np.fft.fft(frame_inp, nb_fft, axis=0)[0:nb_frq, :]
    # start_time = time.clock()
    fft_inp = ns.processfft(fft_inp)
    # ------------------------------------------------------------
    # ----------------- COMPRESSION
    fft_out = compressor.processfft(fft_inp)
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    # ------------------------------------------------------------
    # ----------------- OUTPUT ENCODING
    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_frq - 1, :]))), axis=0)
    frame_out = np.real(np.fft.ifft(fft_out, axis=0))
    npdata_out = BlockProc.frame2output(frame_out)
    npdata_out = npdata_out * u.db2mag(-100)
    data = rt.encode(npdata_out)

    return data, pyaudio.paContinue


# ----------------------------------------------------------------

CHANNELS = 2
RATE = 16000
nb_buffsamp = 32
nb_fft = 128
nb_frq = int(nb_fft/2)+1
p = pyaudio.PyAudio()

BlockProc = rt.BlockProc2(nb_buffsamp=nb_buffsamp, nb_fft=nb_fft,
                          nb_channels_inp=CHANNELS, nb_channels_out=CHANNELS)

ns = rt.NoiseSoustraction(nb_fft=nb_fft, nb_channels=CHANNELS, samp_freq=RATE, threshold_f=6)

compressor = rt.CompressorFftHa(nb_fft=nb_fft, nb_channels=CHANNELS, samp_freq=RATE,
                 fq_ctr_v=[250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000],
                 Laud_v=[0, 0, 10, 20, 30, 40, 50, 50, 60],
                 Lpain_v=[100, 100, 100, 100, 100, 100, 100, 100, 100],
                 thr_ratio_v=[2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3],
                 knee_width=1, time_attack=0.005, time_release=0.05,
                 wet_f=1, bypass=False, verbose=False)

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
line1, = ax.semilogx(compressor.freq_v, compressor.mag_inp[:, 0], 'b')
line2, = ax.semilogx(compressor.freq_v, compressor.mag_out[:, 0], 'r')
line3, = ax.semilogx(compressor.freq_v, compressor.gr_plot[:, 0], 'black')
line4, = ax.semilogx(compressor.freq_v, compressor.gr2_plot[:, 0], 'g')
line5, = ax.semilogx(compressor.fc_v, compressor.thrsh[:, 0])
plt.grid(True)
plt.axis([compressor.freq_v[0], compressor.freq_v[len(compressor.freq_v)-1], 0, 100])

while stream.is_active():
    time.sleep(0.1)
    line1.set_ydata(compressor.mag_inp[:, 0])
    line2.set_ydata(compressor.mag_out[:, 0])
    line3.set_ydata(compressor.gr_plot[:, 0])
    line4.set_ydata(compressor.gr2_plot[:, 0])
    fig.canvas.draw()

stream.stop_stream()
stream.close()

p.terminate()

