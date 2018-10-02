"""
Description: Real time Optimal Directivity Index Beamformer test

Author: Adrien Llave - CentraleSupelec
Date: 30/08/2018

Version: 1.2

Date    | Auth. | Vers.  |  Comments
18/08/24  ALl     1.0       Initialization
18/08/29  ALl     1.1       It works now
18/08/30  ALl     1.2       It really works today

"""

import pyaudio
import time
import numpy as np
import RTAudioProc as rt
import binauralbox as bb
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
def callback(in_data, frame_count, time_info, status):
    # start_time = time.clock()
    # ---- GENERATE WHITE NOISE ----
    # npdata_in = np.random.normal(0, 0.01, samp_per_buffer)[:, np.newaxis]
    npdata_in = rt.decode(in_data, CHANNELS)[:, 0][:, np.newaxis]
    # ---- BINAURALIZATION ----
    tmp_bck = bino_bck.process(npdata_in)
    tmp_frt = bino_frt.process(npdata_in)
    tmp_itc = bino_itc.process(npdata_in)
    npdata_in_bino = np.concatenate((tmp_bck, tmp_frt, tmp_itc), axis=1)
    # ---- GO PROCESS ----
    frame_inp = blockproc.input2frame(npdata_in_bino)
    fft_inp = np.fft.fft(frame_inp, nb_fft, axis=0)[0:nb_frq, :]
    fft_out = beam.processfft(fft_inp)
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
RATE = 44100
samp_per_buffer = 128
nb_fft = 512
nb_frq = int(nb_fft/2) + 1
print('number sample per buffer:', samp_per_buffer)
print('frame duration (ms):', samp_per_buffer/RATE*1000)

p = pyaudio.PyAudio()

nb_ch_inp = 6
blockproc = rt.BlockProc2(nb_buffsamp=samp_per_buffer, nb_fft=nb_fft,
                          nb_channels_inp=nb_ch_inp, nb_channels_out=1)

# LOAD HRTF
path_hrtf = 'D:\MATLAB\database_HA_HRTF\oreinos2013\HATS_BTE_hrirDatabase'
hl_bck = bb.HrtfData(fs_f=RATE)
hr_bck = bb.HrtfData(fs_f=RATE)
hl_frt = bb.HrtfData(fs_f=RATE)
hr_frt = bb.HrtfData(fs_f=RATE)
hl_itc = bb.HrtfData(fs_f=RATE)
hr_itc = bb.HrtfData(fs_f=RATE)
filename_bck = path_hrtf + '/HATS' + str(RATE) + '_BTEback_azcorrect.mat'
filename_frt = path_hrtf + '/HATS' + str(RATE) + '_BTEfront_azcorrect.mat'
filename_itc = path_hrtf + '/HATS' + str(RATE) + '_BnK_azcorrect.mat'
hl_bck.import_hrir_from_matfile(filename_s=filename_bck, ear_side='left')
hr_bck.import_hrir_from_matfile(filename_s=filename_bck, ear_side='right')
hl_frt.import_hrir_from_matfile(filename_s=filename_frt, ear_side='left')
hr_frt.import_hrir_from_matfile(filename_s=filename_frt, ear_side='right')
hl_itc.import_hrir_from_matfile(filename_s=filename_itc, ear_side='left')
hr_itc.import_hrir_from_matfile(filename_s=filename_itc, ear_side='right')

# DEFINE BINAURALIZER
bino_bck = rt.RTBinauralizer(hl_bck, hr_bck, nb_bufsamp=samp_per_buffer)
bino_frt = rt.RTBinauralizer(hl_frt, hr_frt, nb_bufsamp=samp_per_buffer)
bino_itc = rt.RTBinauralizer(hl_itc, hr_itc, nb_bufsamp=samp_per_buffer)
grid_targ = bb.Grid(norm_s='spherical_1', coords_m=np.array([1, 0, 0])[:, np.newaxis].T)
bino_bck.update_positions(grid_targ)
bino_frt.update_positions(grid_targ)
bino_itc.update_positions(grid_targ)

# DEFINE BEAMFORMER
# steer_v = np.array([1., 45., 0.])[:, np.newaxis].T
# steer_v = np.array([1., -45., 0.])[:, np.newaxis].T
steer_v = np.array([1., 0., 0.])[:, np.newaxis].T
beam = rt.SuperBeamformer(nb_buffsamp=samp_per_buffer, nb_fft=nb_fft)
beam.update_mic_directivity(hl_bck, hr_bck, hl_frt, hr_frt, hl_itc, hr_itc)
beam.update_optimal_filter(steer_v)
beam.get_beampattern(True)

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

ii = 0

while True:
    ii = np.mod(ii+5, 360)
    print([1, ii, 0])
    grid_targ = bb.Grid(norm_s='spherical_1', coords_m=np.array([1., ii, 0.])[:, np.newaxis].T)
    bino_bck.update_positions(grid_targ)
    bino_frt.update_positions(grid_targ)
    bino_itc.update_positions(grid_targ)
    time.sleep(0.1)

p.terminate()
