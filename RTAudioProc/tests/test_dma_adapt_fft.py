

"""
Description: Real time adaptive first order DMA beamformer in FREQ domain

Author: Adrien Llave - CentraleSupelec
Date: 19/07/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/07/19  ALl     1.0       Initialization

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
    frame_inp = blockproc.input2frame(npdata_in)
    fft_inp = np.fft.fft(frame_inp, nb_fft, axis=0)[0:nb_frq, :]
    fft_out = dma.processfft(fft_inp)
    # ---- PREPARE OUTPUT ----
    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_frq - 1, :]))), axis=0)
    frame_out = np.real(np.fft.ifft(fft_out, nb_fft, axis=0))
    npdata_out = blockproc.frame2output(frame_out)
    npdata_out = np.repeat(npdata_out, 2, axis=1)
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))

    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 2
RATE = 16000
samp_per_buffer = 32
nb_fft = 128
nb_frq = int(nb_fft/2) + 1
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

p = pyaudio.PyAudio()

blockproc = rt.BlockProc2(nb_buffsamp=samp_per_buffer, nb_fft=nb_fft,
                          nb_channels_inp=CHANNELS, nb_channels_out=1)

mic_dist = 0.068
dma = rt.DmaAdaptiveFFT(samp_freq=RATE,
                        nb_fft=nb_fft,
                        mic_dist=mic_dist,
                        verbose=False)

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
line1, = ax.plot(dma.null_angle_v, 'ro')
plt.axis([0, 128, 90, 180])

plot_rate = 10
ii = 0

stream.start_stream()
while stream.is_active():
    time.sleep(1/plot_rate)
    ii += 1
    if ii > 30:
        dma.bypass = not dma.bypass
        ii = 0
    line1.set_ydata(dma.null_angle_v)
    fig.canvas.draw()

stream.stop_stream()
stream.close()

p.terminate()
