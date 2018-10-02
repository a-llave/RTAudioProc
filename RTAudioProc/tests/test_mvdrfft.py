
"""
Description:    Real time MVDR in FFT paradigm

Author: Adrien Llave - CentraleSupelec
Date: 22/06/2018

Version: 1.0

Date    | Auth. | Version | Comments
18/06/22  ALl       1.0     Initialization

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
    fft_inp = np.fft.fft(npdata_in, axis=0)[0:nb_fft2, :]
    # start_time = time.clock()
    # ------------------------------------------------------------
    # ----------------- MVDR
    fft_out = mvdr.processfft(fft_inp)
    # print("--- %.2f ms ---" % ((time.clock() - start_time) * 1000.0))
    # ------------------------------------------------------------
    # ----------------- OUTPUT ENCODING
    fft_out = np.concatenate((fft_out, np.conjugate(np.flipud(fft_out[1:nb_fft2 - 1, :]))), axis=0)
    npdata_out = np.real(np.fft.ifft(fft_out, axis=0))
    data = rt.encode(npdata_out)

    return data, pyaudio.paContinue


# ----------------------------------------------------------------

CHANNELS = 2
RATE = 16000
nb_frmesamp = 128
nb_fft = nb_frmesamp * 2
nb_fft2 = int(nb_fft/2)+1

p = pyaudio.PyAudio()

mvdr = rt.BeamformerMVDR(nb_fft=nb_fft, nb_mic=CHANNELS, samp_freq=RATE,
                         verbose=False, bypass=False)

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                # input_device_index=16,
                # output_device_index=16,
                frames_per_buffer=nb_fft,
                stream_callback=callback)

stream.start_stream()

ii = 0

while stream.is_active():
    time.sleep(0.1)

    # ii += 1
    # if ii > 30:
    #     ii = 0
    #     print('BYPASS: ' + str(mvdr.bypass))

stream.stop_stream()
stream.close()

p.terminate()

