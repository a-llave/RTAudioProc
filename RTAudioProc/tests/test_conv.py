
"""
Description: Real time convolution (stereo) with a church IR

Author: Adrien Llave - CentraleSupelec
Date: 12/02/2018

Version: 1.0

Date    | Auth. | Version | Comments
18/02/12  ALl       1.0     Initialization

"""

import pyaudio
import time
import numpy as np
import wave
import src.pkg.RTAudioProc as rt

CHANNELS = 2
RATE = 48000
samp_per_buffer = 300
len_ir = int(8192*4 + 1 - samp_per_buffer)
overlap_samp = len_ir - 1

p = pyaudio.PyAudio()


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):

    npdata_in = rt.decode(in_data, CHANNELS)            # INPUT DECODING
    npdata_out = convolution.process(npdata_in)         # CONVOLUTION
    data = rt.encode(npdata_out)                        # OUTPUT ENCODING

    return data, pyaudio.paContinue

# ----------------------------------------------------------------


# LOAD IR
wf = wave.open("../resources/IR_48k.wav", 'r')
IR = wf.readframes(wf.getnframes())         # READ
IR_m = rt.decode(IR, wf.getnchannels())     # DECODING
IR_m = IR_m[0:len_ir, :]                    # RESIZE
IR_m = np.divide(IR_m, 2.0*np.amax(IR_m))   # ATTENUATION

# CALL AUDIO FX
convolution = rt.ConvolutionIR(samp_per_buffer, IR_m, bypass=False)

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
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()

