
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
import wave
import src.pkg.RTAudioProc as rt
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.time()
    npdata_in = rt.decode(in_data, CHANNELS)                    # INPUT DECODING

    sig_inp = npdata_in[:, 2:]
    npdata_out = beam.process(sig_inp)                        # BEAMFORMING
    npdata_out = np.repeat(npdata_out, CHANNELS, axis=1)
    data = rt.encode(npdata_out)                                # OUTPUT ENCODING

    # print("--- %.2f ms ---" % ((time.time() - start_time)*1000))
    return data, pyaudio.paContinue

# ----------------------------------------------------------------


# ---- PLOT
disp_adaptive_filter_b = True
disp_beampattern_b = False

# ---- GENERAL
CHANNELS = 8
RATE = 48000
samp_per_buffer = 200
print('number sample per buffer:', samp_per_buffer)
print('frame duration: %.2f ms' % (samp_per_buffer/RATE*1000))
NORMALIZE = True
DIRECTION = 1

p = pyaudio.PyAudio()

beam = rt.BeamformerDAS(nb_buffsamp=samp_per_buffer)
beam.adaptive_b = True
# LOAD IR
wf = wave.open('../../resources\ir_mannequin_cafet\IR_48k_1.wav', 'r')
IR = wf.readframes(wf.getnframes())         # READ
IR_m = rt.decode(IR, wf.getnchannels())     # DECODING
IR_m = np.divide(IR_m, np.amax(IR_m))      # ATTENUATION

beam.update_ir(IR_m, normalize=True, mic_id=5)

# big_mat = np.zeros((IR_m.shape[0], IR_m.shape[1], 8))
# for idx in range(1, 8):
#     wf = wave.open('..\\resources\ir_linearmicarray\IR_48k_'+str(idx)+'.wav', 'r')
#     IR = wf.readframes(wf.getnframes())                     # READ
#     tmp_m = rt.decode(IR, wf.getnchannels())                # DECODING
#     big_mat[:, :, idx] = np.divide(tmp_m, np.amax(tmp_m))   # ATTENUATION
#
# coords_m = np.zeros((8, 3))
# coords_m[:, 0] = np.ones((8,))
# coords_m[:, 1] = np.array([0, 45, 90, 135, 22.5, 67.5, 112.5, 157.5])
#
# beam.load_data_for_beampattern(ir_by_dir=big_mat, coords_m=coords_m)

# STREAM
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input_device_index=16,
                output_device_index=16,
                input=True,
                output=True,
                frames_per_buffer=samp_per_buffer,
                stream_callback=callback)

stream.start_stream()

# ----------------------- Plot buffer and buffer - 1 -----------------------------------------
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
# line1, = ax.plot(np.arange(0, samp_per_buffer), beam.buff_2)
# line2, = ax.plot(np.arange(samp_per_buffer, 2*samp_per_buffer), beam.buff_1)
#
# line3, = ax.plot(np.arange(0, samp_per_buffer), beam.overlap_1+0.02)
# line4, = ax.plot(np.arange(samp_per_buffer, 2*samp_per_buffer), beam.overlap+0.02)
# plt.axis([0, 2*samp_per_buffer, -0.04, 0.04])
#
# ax2 = fig.add_subplot(2, 1, 2)
# line5, = ax2.plot(np.arange(0, beam.nb_ftsamp), beam.sigout_1)
# line6, = ax2.plot(np.arange(samp_per_buffer, beam.nb_ftsamp+samp_per_buffer), beam.sigout)
# plt.axis([0, beam.nb_ftsamp+samp_per_buffer, -0.04, 0.04])

# --------------------- Plot IR inverse filter -------------------------------------------
if disp_adaptive_filter_b:
    plt.ion()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(2, 1, 1)
    line21, = ax2.plot(np.real(np.fft.ifft(beam.Wf[0, :])), label='L Front')
    line22, = ax2.plot(np.real(np.fft.ifft(beam.Wf[1, :])), label='L Middle')
    line23, = ax2.plot(np.real(np.fft.ifft(beam.Wf[2, :])), label='L Rear')
    legend2 = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.axis([0, beam.nb_ftsamp, -0.3, 0.3])
    ax3 = fig2.add_subplot(2, 1, 2)
    line31, = ax3.plot(np.real(np.fft.ifft(beam.Wf[3, :])), label='R Front')
    line32, = ax3.plot(np.real(np.fft.ifft(beam.Wf[4, :])), label='R Middle')
    line33, = ax3.plot(np.real(np.fft.ifft(beam.Wf[5, :])), label='R Rear')
    legend3 = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.axis([0, beam.nb_ftsamp, -0.3, 0.3])

    fig2.canvas.draw()

# # ---------------------- Plot Beampattern ------------------------------------------
if disp_beampattern_b:
    plt.ion()
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1, projection='polar')
    # trash, linec1, trosh = ax3.stem(coords_m[:, 1], beam.pattern[50, :])
    linec1, = ax3.plot(np.deg2rad(coords_m[:, 1]), beam.pattern[50, :], 'ro')
    ax3.set_rmax(20)


while stream.is_active():

    if disp_adaptive_filter_b:
        line21.set_ydata(np.real(np.fft.ifft(beam.Wf[0, :])))
        line22.set_ydata(np.real(np.fft.ifft(beam.Wf[1, :])))
        line23.set_ydata(np.real(np.fft.ifft(beam.Wf[2, :])))
        line31.set_ydata(np.real(np.fft.ifft(beam.Wf[3, :])))
        line32.set_ydata(np.real(np.fft.ifft(beam.Wf[4, :])))
        line33.set_ydata(np.real(np.fft.ifft(beam.Wf[5, :])))
        fig2.canvas.draw()

    # line1.set_ydata(beam.buff_2)
    # line2.set_ydata(beam.buff_1)
    # line3.set_ydata(beam.overlap_1+0.02)
    # line4.set_ydata(beam.overlap+0.02)
    # line5.set_ydata(beam.sigout_1)
    # line6.set_ydata(beam.sigout)
    # fig.canvas.draw()

    # --- Plot Beampattern
    if disp_beampattern_b:
        beam.compute_beampattern()
        linec1.set_ydata(np.linalg.norm(beam.pattern, 2, axis=0))
        fig3.canvas.draw()

    time.sleep(0.5)


stream.stop_stream()
stream.close()

p.terminate()

