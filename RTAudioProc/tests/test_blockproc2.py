

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


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.clock()
    npdata_in = rt.decode(in_data, nb_channels)                                            # INPUT DECODING
    frame_inp = BlockProc.input2frame(npdata_in)

    fft_inp = np.fft.fft(frame_inp[:, 0], nb_fft)
    frame_out = np.real(np.fft.ifft(fft_inp, ))
    frame_out = np.repeat(frame_out[:, np.newaxis], nb_channels, axis=1)

    npdata_out = BlockProc.frame2output(frame_out)
    data = rt.encode(npdata_out)                                                        # OUTPUT ENCODING
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


nb_channels = 2
RATE = 16000
nb_buffersamp = 64
nb_fft = 256
print('number sample per buffer:', nb_buffersamp)
print('frame duration (ms):', nb_buffersamp/RATE*1000)

BlockProc = rt.BlockProc2(nb_buffsamp=nb_buffersamp, nb_fft=nb_fft, nb_channels=nb_channels)

p = pyaudio.PyAudio()

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=nb_channels,
                rate=RATE,
                input=True,
                output=True,
                # input_device_index=16,
                # output_device_index=16,
                frames_per_buffer=nb_buffersamp,
                stream_callback=callback)

stream.start_stream()
while stream.is_active():
    time.sleep(1)

stream.stop_stream()
stream.close()

p.terminate()
