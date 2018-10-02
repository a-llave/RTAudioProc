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
    if not myFilter.bypass:
        npdata_out = np.sum(npdata_out, axis=2)
    # OUTPUT ENCODING
    data = rt.encode(npdata_out)
    return data, pyaudio.paContinue


# ----------------------------------------------------------------
nb_channels = 2
samp_freq = 48000
nb_buffsamp = 50
print('Buffer duration: ', nb_buffsamp/samp_freq*1000, 'ms')

p = pyaudio.PyAudio()

myFilter = rt.FilterBank(nb_buffsamp=nb_buffsamp,
                         samp_freq=samp_freq,
                         nb_channels=nb_channels,
                         # frq_band_v=np.array([100, 500, 2000]))
                         frq_band_v=np.array([100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000]))

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
    print(myFilter.bypass)
    time.sleep(3)

stream.stop_stream()
stream.close()

p.terminate()


