

"""
Description: Real time double (left-right) adaptive first order DMA beamformer

Author: Adrien Llave - CentraleSupelec
Date: 16/05/2018

Version: 1.0

Date    | Auth. | Comments
18/05/16  ALl     Initialization

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
    # data_l = np.concatenate((npdata_in[:, 2][:, np.newaxis], npdata_in[:, 3][:, np.newaxis]), axis=1)
    # data_r = np.concatenate((npdata_in[:, 5][:, np.newaxis], npdata_in[:, 6][:, np.newaxis]), axis=1)
    data_l = np.concatenate((npdata_in[:, 0][:, np.newaxis], npdata_in[:, 1][:, np.newaxis]), axis=1)
    data_beam_l = dma_L.process(data_l)
    data_beam_r = dma_R.process(data_l)
    npdata_out = npdata_in
    npdata_out[:, 0] = data_beam_l[:, 0]
    npdata_out[:, 1] = data_beam_r[:, 0]
    npdata_out[:, 0:2] = LPFilter.process(npdata_out[:, 0:2])
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))

    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 2
RATE = 16000
samp_per_buffer = 64
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

p = pyaudio.PyAudio()

mic_dist = 0.068

dma_L = rt.DmaAdaptive(samp_freq=RATE,
                       nb_buffsamp=samp_per_buffer,
                       mic_dist=mic_dist,
                       verbose=True)
dma_R = rt.BeamformerDMA(samp_freq=RATE,
                         nb_buffsamp=samp_per_buffer,
                         mic_dist=mic_dist)

# LOWPASS FILTER
LPFilter = rt.Butterworth(samp_per_buffer,
                          RATE,
                          nb_channels=2,
                          type_s='low',
                          cut_freq=5000.,
                          order_n=5)

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


# plt.ion()
# fig = plt.figure()
# #
# ax = fig.add_subplot(1, 1, 1)
# line1, = ax.plot(np.zeros(100,), 'ro')
# plt.axis([0, 100, 90, 180])

stream.start_stream()
while stream.is_active():
    time.sleep(0.2)
    # line1.set_ydata(dma_L.null_angle_v)
    # fig.canvas.draw()

stream.stop_stream()
stream.close()

p.terminate()
