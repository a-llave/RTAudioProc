
"""
Description: Real time convolution (stereo) with a church IR

Author: Adrien Llave - CentraleSupelec
Date: 12/02/2018

Version: 1.0

Date    | Auth. | Comments
18/02/12  ALl     Initialization

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
    sig_inp = np.concatenate((npdata_in[:, 2][:, np.newaxis], npdata_in[:, 5][:, np.newaxis]), axis=1)
    data_beam = dma_inter.process(sig_inp)
    npdata_out = npdata_in
    npdata_out[:, 0:2] = data_beam
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))

    return data, pyaudio.paContinue

# ----------------------------------------------------------------


CHANNELS = 8
RATE = 48000
samp_per_buffer = 20
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

p = pyaudio.PyAudio()

dma_inter = rt.DmaInteraural(samp_freq=RATE,
                             nb_buffsamp=samp_per_buffer,
                             mic_dist=0.14,
                             freq_cut=800.,
                             nullangle_v=180.)

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                input_device_index=16,
                output_device_index=16,
                frames_per_buffer=samp_per_buffer,
                stream_callback=callback)

stream.start_stream()
while stream.is_active():
    dma_inter.bypass = not dma_inter.bypass
    print(dma_inter.bypass)
    time.sleep(5)


stream.stop_stream()
stream.close()

p.terminate()

