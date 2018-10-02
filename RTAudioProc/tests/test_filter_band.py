import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import src.pkg.RTAudioProc as rt
import time


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # INPUT DECODING
    npdata_in = rt.decode(in_data, nb_channels)
    # FILTER
    npdata_out = myFilter.process(npdata_in)
    # OUTPUT ENCODING
    data = rt.encode(npdata_out)
    return data, pyaudio.paContinue


# ----------------------------------------------------------------
nb_channels = 2
samp_freq = 48000
nb_buffsamp = 50
print('Buffer duration: ', nb_buffsamp/samp_freq*1000, 'ms')

p = pyaudio.PyAudio()

myFilter = rt.Butterworth(nb_buffsamp=nb_buffsamp,
                          samp_freq=samp_freq,
                          nb_channels=nb_channels,
                          type_s='band',
                          cut_freq=np.array([500, 1000]),
                          order_n=2)

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=nb_channels,
                rate=samp_freq,
                input=True,
                output=True,
                # input_device_index=16,
                # output_device_index=16,
                frames_per_buffer=nb_buffsamp,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    myFilter.bypass = not myFilter.bypass
    time.sleep(3)

stream.stop_stream()
stream.close()

p.terminate()


