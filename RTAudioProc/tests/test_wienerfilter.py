

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
import src.pkg.RTAudioProc as rt
import matplotlib.pyplot as plt
import src.pkg.utils as u


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.clock()
    npdata_in = rt.decode(in_data, nb_channels)                                            # INPUT DECODING
    frame_inp = BlockProc.input2frame(npdata_in)

    fft_inp = np.fft.fft(frame_inp, nb_fft, axis=0)[0:nb_fft2, :]
    fft_out = wienerFilter.processfft(fft_inp)

    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_fft2 - 1, :]))), axis=0)
    frame_out = np.real(np.fft.ifft(fft_out, axis=0))

    npdata_out = BlockProc.frame2output(frame_out)
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


nb_channels = 2
RATE = 16000
nb_buffsamp = 32
nb_fft = 128
nb_fft2 = int(nb_fft/2) + 1
print('number sample per buffer:', nb_buffsamp)
print('frame duration (ms):', nb_buffsamp/RATE*1000)

BlockProc = rt.BlockProc2(nb_buffsamp=nb_buffsamp, nb_fft=nb_fft, nb_channels=nb_channels)
wienerFilter = rt.WienerFilter(nb_buffsamp=nb_fft, nb_channels=nb_channels, samp_freq=RATE)

p = pyaudio.PyAudio()

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=nb_channels,
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
line1, = ax.semilogx(wienerFilter.magdb_inp[:, 0], 'r')
line2, = ax.semilogx(wienerFilter.noisedb_prev[:, 0], 'g')
line3, = ax.semilogx(wienerFilter.magdb_out[:, 0], 'b')
line4, = ax.semilogx(wienerFilter.gain_redu_prev[:, 0], 'black')
plt.axis([1, nb_fft2, -60, 20])

ii = 0

stream.start_stream()
while stream.is_active():
    line1.set_ydata(wienerFilter.magdb_inp[:, 0])
    line2.set_ydata(wienerFilter.noisedb_prev[:, 0])
    line3.set_ydata(wienerFilter.magdb_out[:, 0])
    line4.set_ydata(u.mag2db(wienerFilter.gain_redu_prev[:, 0]))
    fig.canvas.draw()

    ii += 1
    if ii > 30:
        wienerFilter.bypass = not wienerFilter.bypass
        ii = 0
        print('BYPASS: ' + str(wienerFilter.bypass))

    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
