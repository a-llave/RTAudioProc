"""
Description: class and functions needed for the real time audio process

Author: Adrien Llave - CentraleSupélec
Date: 29/08/2018

Version: 11.0

Date    | Auth. | Vers.  |  Comments
18/03/28  ALl     1.0       Initialization
18/03/30  ALl     2.0       Bug fix in Butterworth class, remove initial click due to bad conditioning
18/04/01  ALl     3.0       Minor bug fix:  - DMA compensation gain
                                            - DMA hybrid Lowpass filter cutoff frequency high to optimize WNG
18/05/22  ALl     4.0       Add processing: - FilterBank
                                            - DMA adaptive
                                            - Multiband expander
                                            - Overlap-add block processing
18/06/18  ALl     5.0       DMA adaptive:   - bug fix
                                            - apply LP and HP filter before cross corr estimation
18/06/19  ALl     6.0       - AudioCallback: Add management of the end ('end_b' flag attribute, fill by zeros last buffer)
                            - Add processing: NoiseSoustraction
18/06/21  ALl     7.0       Add processing: - CompressorFFT
                                            - OverlapSave
18/07/02  ALl     8.0       Add processing: - BeamformerMVDR in order to replace BeamformerDAS which is wrong
18/07/10  ALl     8.1       Remove class:   - BeamformerDAS (old version of MVDR)
                                            - Noise reduction (multi-band expander)
18/07/17  ALl     8.2       BeamformerMVDR : bug fix, complex cast some variables
18/07/19  ALl     9.0       Add processing: DMA adaptive in FREQ domain
                                            bug fix: prevent division by 0 in coeff estimation
18/08/24  ALl    10.0       Add processing: SuperBeamformer (Optimal DI Beamformer)
18/08/29  ALl    11.0       - Dependency issue fixing between RTAudioProc and binauralbox
                                - remove binauralbox dependency to RTAudioProc in order to make RTAudioProc dependant to bb
                                - move RTBinauralizer and RTBinauralizerFFT from bb to rt
                                TODO: change 'bb' to 'rt' when using those classes
                            - Add security freq2time in RTBinauralizer

"""

import numpy as np
import scipy.signal as spsig
import src.pkg.utils as u
import src.pkg.binauralbox as bb
import copy
import matplotlib.pyplot as plt
import time

# ======================================================================================================
# =================== FUNCTIONS ========================================================================
# ======================================================================================================


def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with
    shape (chunk_size, channels)

    Samples are interleaved, so for a stereo stream with left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
    is ordered as [L0, R0, L1, R1, ...]
    """
    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    result = np.fromstring(in_data, dtype=np.int16)
    chunk_length = int(len(result) / channels)
    assert chunk_length == int(chunk_length)
    result = np.reshape(result, (chunk_length, channels))
    result = result.astype(np.float32)
    result = result / np.power(2.0, 16)
    return result


def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio

    Signal should be a numpy array with shape (chunk_size, channels)
    """
    signal = signal * np.power(2.0, 16)
    signal = signal.astype(np.int16)
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype(np.int16).tostring()
    return out_data


def voice_activity_detector(signal, threshold=0.0):
    nb_xzero = 0
    nb_sample = len(signal)
    vad_b = False
    for nn in range(1, nb_sample):
        if not np.sign(signal[nn]) == np.sign(signal[nn-1]):
            nb_xzero += 1
    xzero_rate = nb_xzero / nb_sample
    if xzero_rate < threshold:
        vad_b = True
    return vad_b

# ======================================================================================================
# =================== CLASSES ==========================================================================
# ======================================================================================================


class AudioCallback:
    def __init__(self, sig_m, nb_buffsamp):
        self.sig_m = sig_m
        self.nb_samples = self.sig_m.shape[0]
        self.nb_channels = self.sig_m.shape[1]
        self.nb_buffsamp = nb_buffsamp
        self.marker_in = 0
        self.end_b = False
        return

    def readframes(self):
        if self.marker_in+self.nb_buffsamp <= self.nb_samples:
            buffer_m = self.sig_m[self.marker_in:self.marker_in+self.nb_buffsamp, :]
        elif self.marker_in+self.nb_buffsamp > self.nb_samples and self.marker_in < self.nb_samples:
            buffer_m = np.concatenate((self.sig_m[self.marker_in:, :], np.zeros((self.marker_in+self.nb_buffsamp-self.nb_samples, self.nb_channels))))
        else:
            buffer_m = np.zeros((self.nb_buffsamp, self.nb_channels))
            self.end_b = True

        self.marker_in += self.nb_buffsamp
        return buffer_m


# ========================= DYNAMIC RANGE COMPRESSOR =========================
class Compressor:
    """
    ================================
    Real time dynamic range compressor
    ================================

    """
    def __init__(self, nb_bufsamp,
                 nb_channels,
                 thrsh=0,
                 ratio=1,
                 time_attack=0.005,
                 time_release=0.05,
                 knee_width=1,
                 fs_f=48000,
                 bypass=False,
                 verbose=False):
        """
        Constructor
        """

        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.gain_db = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.env_prev = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.thrsh = thrsh
        self.ratio = ratio
        self.coeff_attack = np.exp(-1/(time_attack*fs_f))
        self.coeff_release = np.exp(-1/(time_release*fs_f))
        self.kneeWidth = knee_width
        self.bypass = bypass
        self.verbose = verbose

    def process(self, npdata_in):
        """
        Process dynamic range compression
        """
        if not self.bypass:
            # GAIN COMPUTER
            inp_gr = 20 * np.log10(np.absolute(npdata_in) + np.finfo(float).eps)
            out_gr = np.zeros(npdata_in.shape)
            out_env = np.zeros(npdata_in.shape)
            mask_lin = 2 * (inp_gr - self.thrsh) < - self.kneeWidth
            mask_kne = 2 * np.absolute(inp_gr - self.thrsh) <= self.kneeWidth
            mask_cmp = 2 * (inp_gr - self.thrsh) > self.kneeWidth
            out_gr[mask_lin] = inp_gr[mask_lin]
            out_gr[mask_kne] = inp_gr[mask_kne] \
                               + (1 / self.ratio - 1) * (inp_gr[mask_kne] - self.thrsh + self.kneeWidth / 2) ** 2 \
                               / (2 * self.kneeWidth)
            out_gr[mask_cmp] = self.thrsh + (inp_gr[mask_cmp] - self.thrsh) / self.ratio
            # PEAK DETECTOR
            for cc in range(0, self.nb_channels):
                inp_env = inp_gr - out_gr
                if inp_env[0, cc] > self.env_prev[self.nb_bufsamp-1, cc]:
                    out_env[0, cc] = self.coeff_attack * self.env_prev[self.nb_bufsamp-1, cc] + (1 - self.coeff_attack) * inp_env[0, cc]
                else:
                    out_env[0, cc] = self.coeff_release * self.env_prev[self.nb_bufsamp-1, cc]

                for nn in range(1, self.nb_bufsamp):
                    if inp_env[nn, cc] > out_env[nn - 1, cc]:
                        out_env[nn, cc] = self.coeff_attack * out_env[nn - 1, cc] + (1 - self.coeff_attack) * inp_env[nn, cc]
                    else:
                        out_env[nn, cc] = self.coeff_release * out_env[nn - 1, cc]

            # BACK UP
            self.env_prev = out_env
            # GAIN REDUCTION
            self.gain_db = -out_env
            npdata_out = npdata_in * np.power(10, self.gain_db / 20)
            if np.min(self.gain_db) < -1 and self.verbose:
                gr_v = np.mean(self.gain_db, axis=0)
                print('GAIN REDUCTION:')
                for ch, gr in enumerate(gr_v):
                    print('CHANNEL %i: %.0f' % (ch, gr))
        else:
            npdata_out = npdata_in

        return npdata_out


class CompressorFFT:
    """
    ================================
    Real time dynamic range compressor
    ================================
    Ref.:   - Digital Dynamic Range Compressor Design - A Tutorial and Analysis, Giannoulis et al. 2012
            - Principles of Digital Dynamic-Range Compression, James M. Kates 2005
    """
    def __init__(self, nb_buffsamp, nb_channels, samp_freq=48000,
                 fq_ctr_v=[250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000],
                 thrsh_v=[-10, -10, -15, -20, -30, -40, -40, -50, -50],
                 ratio_v=[3, 3, 3, 3, 3, 3, 3, 3, 3],
                 knee_width=1, time_attack=0.005, time_release=0.05,

                 bypass=False, verbose=False):
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.nb_buffsamp = nb_buffsamp
        self.nb_buffsamp2 = int(nb_buffsamp/2+1)
        self.nb_channels = nb_channels
        self.samp_freq = samp_freq
        # FREQ SCALE
        self.fc_v = fq_ctr_v
        self.nb_band = len(self.fc_v)
        self.flim_v = np.zeros((self.nb_band + 1))
        for ii in range(self.nb_band - 1):
            self.flim_v[ii + 1] = np.round(np.sqrt(self.fc_v[ii] * self.fc_v[ii + 1]))
            self.flim_v[len(self.flim_v) - 1] = self.samp_freq / 2
        self.freq_v = np.linspace(0, self.samp_freq / 2, self.nb_buffsamp2)
        self.mask = [0] * self.nb_band
        for band in range(self.nb_band):
            mask1 = self.freq_v >= self.flim_v[band]
            mask2 = self.freq_v < self.flim_v[band + 1]
            self.mask[band] = mask1 * mask2
        # PARAMETERS
        self.thrsh = np.repeat(np.array(thrsh_v)[:, np.newaxis], self.nb_channels, axis=1)
        self.ratio = np.repeat(np.array(ratio_v)[:, np.newaxis], self.nb_channels, axis=1)
        self.kneeWidth = knee_width
        self.time_attack = time_attack
        self.time_release = time_release
        # SMOOTH GR IN TIME
        self.coefA_f = np.exp(-1 / (self.time_attack * self.samp_freq))
        self.coefR_f = np.exp(-1 / (self.time_release * self.samp_freq))
        self.gr_prev_m = np.zeros((self.nb_band, self.nb_channels))
        # SMOOTH GR IN FREQ
        N = self.nb_buffsamp / 10.
        self.win_v = np.hanning(N) / (N/2)
        # FOR PLOT
        self.mag_inp = np.zeros((self.nb_buffsamp2, self.nb_channels))
        self.mag_out = np.zeros((self.nb_buffsamp2, self.nb_channels))
        self.gr_plot = np.zeros((self.nb_buffsamp2, self.nb_channels))
        self.gr2_plot = np.zeros((self.nb_buffsamp2, self.nb_channels))

    def processfft(self, fft_inp):
        magdb_inp = u.mag2db(fft_inp)
        if not self.bypass:
            rms_band = np.zeros((self.nb_band, self.nb_channels))
            for band in range(self.nb_band):
                mag_band = magdb_inp[self.mask[band]]
                rms_band[band, :] = np.mean(mag_band, axis=0)
            # GAIN COMPUTER
            gc_inp = rms_band
            gc_out = np.zeros(gc_inp.shape)
            mask_lin = 2 * (gc_inp - self.thrsh) < - self.kneeWidth
            mask_kne = 2 * np.absolute(gc_inp - self.thrsh) <= self.kneeWidth
            mask_cmp = 2 * (gc_inp - self.thrsh) > self.kneeWidth
            gc_out[mask_lin] = gc_inp[mask_lin]
            gc_out[mask_kne] = gc_inp[mask_kne] \
                               + (1 / self.ratio[mask_kne] - 1) * (gc_inp[mask_kne] - self.thrsh[mask_kne] + self.kneeWidth / 2) ** 2 \
                               / (2 * self.kneeWidth)
            gc_out[mask_cmp] = self.thrsh[mask_cmp] + (gc_inp[mask_cmp] - self.thrsh[mask_cmp]) / self.ratio[mask_cmp]
            gr_m = gc_inp - gc_out
            # SMOOTH GR IN TIME
            mask_atk_b = gr_m >= self.gr_prev_m
            mask_rls_b = gr_m < self.gr_prev_m
            gr_m[mask_atk_b] = self.coefA_f * gr_m[mask_atk_b] + (1 - self.coefA_f) * self.gr_prev_m[mask_atk_b]
            gr_m[mask_rls_b] = self.coefR_f * gr_m[mask_rls_b] + (1 - self.coefR_f) * self.gr_prev_m[mask_rls_b]
            self.gr_prev_m = gr_m
            # INTERPOLATION GR
            tmp = np.zeros(fft_inp.shape)
            for band in range(self.nb_band):
                tmp[self.mask[band], :] = gr_m[band, :]
            gr_m = tmp
            self.gr_plot = -gr_m

            # ---- SMOOTH GR IN FREQ
            toto = np.zeros((self.nb_buffsamp2 + len(self.win_v) - 1, self.nb_channels))
            for cc in range(0, self.nb_channels):
                # WEIGHTED MEAN FILTERING (HANN)
                toto[:, cc] = np.convolve(gr_m[:, cc], self.win_v)
            idx = int(len(self.win_v) / 2)
            gr_m = toto[idx:idx + self.nb_buffsamp2, :]
            self.gr2_plot = -gr_m
            # GAIN REDUCTION TO FILTER
            gain_redu = u.db2mag(-gr_m)
            gain_redu = u.min_phase_spectrum(gain_redu)
            # APPLY GAIN REDUCTION
            fft_out = fft_inp * gain_redu
        else:
            fft_out = fft_inp

        # FOR PLOT
        self.mag_inp = magdb_inp
        self.mag_out = u.mag2db(fft_out)
        return fft_out


# ========================= CONVOLUTION =========================
class Convolution:
    def __init__(self, nb_bufsamp, nb_channels, bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.data_overlap = np.zeros((self.nb_bufsamp-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1, sig2):
        """
        Process the convolution between sig 1 and sig2
        """
        if not self.bypass:
            # SHAPE
            sig1_shape = sig1.shape
            sig2_shape = sig2.shape
            # ASSERTION NCHANNELS
            nsample_ft = sig1_shape[0] + sig2_shape[0] - 1
            # FFT PARTY
            sig1_ft = np.fft.fft(sig1, nsample_ft, 0)
            sig2_ft = np.fft.fft(sig2, nsample_ft, 0)
            sig_out_ft = np.multiply(sig1_ft, sig2_ft)
            # COME BACK TO TIME DOMAIN
            sig_out = np.real(np.fft.ifft(sig_out_ft, nsample_ft, 0))
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:sig1_shape[0], :]
            overlap_tmp = sig_out[sig1_shape[0]:sig_out.shape[0], :]
            # OVERLAP ADD
            if sig1_shape[0] > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((sig1_shape[0] - self.data_overlap.shape[0], self.data_overlap.shape[1]))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:sig1_shape[0]]
                overlap_prev = self.data_overlap[sig1_shape[0]:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((sig1_shape[0], self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output


class ConvolutionIR:
    def __init__(self, nb_bufsamp, ir_m=np.concatenate((np.array([1.0])[:, np.newaxis], np.zeros((127, 1))),
                                                       axis=0), bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = ir_m.shape[1]
        self.IR_m = ir_m
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)
        self.data_overlap = np.zeros((self.IR_m.shape[0]-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1):
        """
        Process the convolution between sig 1 and IR
        """
        if not self.bypass:
            # CHECK
            if len(sig1.shape) == 1:
                sig1 = sig1[:, np.newaxis]

            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                      'columns i.e. the same number of channels'
            # CONVOLUTION
            sig_out = np.zeros((self.nsample_ft, self.nb_channels))
            for ch in range(self.nb_channels):
                sig_out[:, ch] = np.convolve(sig1[:, ch], self.IR_m[:, ch])
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:self.nb_bufsamp, :]
            overlap_tmp = sig_out[self.nb_bufsamp:self.nsample_ft, :]
            # OVERLAP ADD
            if self.nb_bufsamp > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((self.nb_bufsamp - self.data_overlap.shape[0], self.nb_channels))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:self.nb_bufsamp]
                overlap_prev = self.data_overlap[self.nb_bufsamp:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((self.nb_bufsamp, self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE FOR NEXT FRAME
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output

    def update_ir(self, ir_m):
        """
        Update impulse response
        """
        # TODO: remove discontinuity due to data_overlap size actualization when M < N or P < L
        old_size_ir_n = self.IR_m.shape[0]
        old_nb_ch_n = self.IR_m.shape[1]
        self.IR_m = ir_m
        self.nb_channels = self.IR_m.shape[1]
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)

        # OVERLAP VECTOR SIZE ACTUALIZATION
        # RESIZE NB CHANNELS
        if self.IR_m.shape[1] == old_nb_ch_n:  # M = N
            # print('M = N')
            pass
        elif self.IR_m.shape[1] > old_nb_ch_n:  # M > N
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((old_size_ir_n-1, self.nb_channels-old_nb_ch_n))), axis=1)
        elif self.IR_m.shape[1] < old_size_ir_n:  # M < N
            self.data_overlap = self.data_overlap[:, 0:self.nb_channels]
        else:
            print('PROBLEM IN UPDATE IR')

        # RESIZE IR SIZE
        if self.IR_m.shape[0] == old_size_ir_n:  # P = L
            # print('P = L')
            pass
        elif self.IR_m.shape[0] > old_size_ir_n:  # P > L
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((self.IR_m.shape[0]-old_size_ir_n, self.nb_channels))), axis=0)
        elif self.IR_m.shape[0] < old_size_ir_n:  # P < L
            self.data_overlap = self.data_overlap[0:self.IR_m.shape[0] - 1, :]
        else:
            print('PROBLEM IN UPDATE IR')

        return


# ========================= FILTER AND FILTER BANKS =========================
class Butterworth:
    def __init__(self, nb_buffsamp, samp_freq=48000, nb_channels=1, type_s='low', cut_freq=100., order_n=1, bypass=False):
        """
        Constructor
        """
        # GENERAL
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        self.samp_freq = samp_freq
        self.nb_channels = nb_channels
        self.xfade_b = False
        # FILTER
        self.type_s = type_s
        self.cut_freq = cut_freq
        self.order_n = order_n
        wn = 2 * self.cut_freq / self.samp_freq
        self.filt_b, self.filt_a = spsig.butter(self.order_n, wn, self.type_s, analog=False)
        self.filt_b_old = self.filt_b
        self.filt_a_old = self.filt_a
        self.previous_info = spsig.lfiltic(self.filt_b, self.filt_a, np.zeros(len(self.filt_b)-1,))
        self.previous_info = np.repeat(self.previous_info[:, np.newaxis], self.nb_channels, axis=1)
        self.previous_info_old = self.previous_info
        # FADE I/O
        self.fadeinp_v = np.repeat(np.sin(np.linspace(0, np.pi/2, self.nb_buffsamp))[:, np.newaxis]**2,
                                   self.nb_channels, axis=1)
        self.fadeout_v = np.repeat(np.cos(np.linspace(0, np.pi/2, self.nb_buffsamp))[:, np.newaxis]**2,
                                   self.nb_channels, axis=1)

    def process(self, sig1):
        """
        Process the filtering on sig1
        :param sig1: matrix [nb_samples X nb_channels]
        :return output: matrix [nb_samples X nb_channels]
        """
        if not self.bypass:
            # CHECK
            if len(sig1.shape) == 1:
                sig1 = sig1[:, np.newaxis]

            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                      'columns i.e. the same number of channels'
            assert sig1.shape[0] == self.nb_buffsamp, 'signal input nb. rows and the nb. sample per buffer ' \
                                                      'must be the same'

            output, self.previous_info = spsig.lfilter(self.filt_b,
                                                       self.filt_a,
                                                       sig1,
                                                       axis=0,
                                                       zi=self.previous_info)
            # XFADE WHEN CHANGE FILTER PARAM
            if self.xfade_b:
                xoutput, _ = spsig.lfilter(self.filt_b_old,
                                           self.filt_a_old,
                                           sig1,
                                           axis=0,
                                           zi=self.previous_info_old)

                output = self.fadeout_v * xoutput + self.fadeinp_v * output
                self.xfade_b = False
        else:
            output = sig1

        return output

    def update_filter(self, type_s='low', cut_freq=100, order_n=1):
        """
        Update filter
        :param type_s: 'low', 'high', 'band'. Default: 'low'
        :param cut_freq: cut-off frequency. Default: 100 Hz
        :param order_n: filter order. Default: 1
        :return:
        """

        self.type_s = type_s
        if self.type_s is 'band':
            assert len(cut_freq) == 2, 'You must specify low and high cut-off frequency for a band-pass'
        self.cut_freq = cut_freq
        self.order_n = order_n
        wn = 2 * self.cut_freq / self.samp_freq
        old_len_prev = self.previous_info.shape[0]
        self.filt_a_old = self.filt_a
        self.filt_b_old = self.filt_b
        self.previous_info_old = self.previous_info
        self.filt_b, self.filt_a = spsig.butter(self.order_n, wn, self.type_s, analog=False)
        new_len_prev = max(len(self.filt_a), len(self.filt_b)) - 1
        if old_len_prev > new_len_prev:
            self.previous_info = self.previous_info[0:new_len_prev, :]
            self.xfade_b = True
        elif old_len_prev == new_len_prev:
            pass
        elif old_len_prev < new_len_prev:
            self.previous_info = np.concatenate((self.previous_info,
                                                 np.zeros((new_len_prev - old_len_prev,
                                                          self.nb_channels))),
                                                axis=0)
        else:
            print('UNEXPECTED')

        return


class FilterBank:
    def __init__(self, nb_buffsamp, samp_freq, nb_channels, frq_band_v, bypass=False):
        # GENERAL
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        self.samp_freq = samp_freq
        self.nb_channels = nb_channels
        # FILTER BANK
        self.frq_band_v = frq_band_v
        self.nb_band = self.frq_band_v.shape[0]
        self.frq_cut_v = np.zeros((self.nb_band-1,))
        for id_band in range(0, self.nb_band-1):
            self.frq_cut_v[id_band] = np.sqrt(self.frq_band_v[id_band] * self.frq_band_v[id_band+1])
        self.Filters = []
        # FIRST LOW PASS
        self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                        samp_freq=samp_freq,
                                        nb_channels=nb_channels,
                                        type_s='low',
                                        cut_freq=self.frq_cut_v[0],
                                        order_n=2)
                            )
        # BAND PASS
        for id_band in range(1, self.nb_band-1):
            self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                            samp_freq=self.samp_freq,
                                            nb_channels=self.nb_channels,
                                            type_s='band',
                                            cut_freq=np.array([self.frq_cut_v[id_band-1], self.frq_cut_v[id_band]]),
                                            order_n=2)
                                )
        # LAST HIGH PASS
        self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                        samp_freq=self.samp_freq,
                                        nb_channels=self.nb_channels,
                                        type_s='high',
                                        cut_freq=self.frq_cut_v[self.nb_band-2],
                                        order_n=2)
                            )

    def process(self, sig1):
        """
        Process the filtering on sig1
        :param sig1: matrix [nb_samples X nb_channels]
        :return output: matrix [nb_samples X nb_channels x nb_band]
        """
        if not self.bypass:
            sig_out = np.zeros((self.nb_buffsamp, self.nb_channels, self.nb_band))
            for id_band in range(0, self.nb_band):
                sig_out[:, :, id_band] = self.Filters[id_band].process(sig1)
        else:
            sig_out = sig1
        return sig_out


# ========================= BEAMFORMING =========================
class BeamformerMVDR:
    def __init__(self, nb_fft, nb_mic=4, samp_freq=48000,
                 eps_f=1e-8, cov_timecste_f=0.5,
                 fq_ctr_v=[150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000],
                 adaptive_b=True, verbose=False, bypass=False):
        # TODO: CHECK IF fq_ctr_v IS SORTED
        # ---- MODE
        self.bypass = bypass
        self.verbose = verbose
        self.adaptive_b = adaptive_b
        # ---- PARAMETERS
        self.nb_mic = nb_mic
        self.nb_fft = nb_fft
        self.nb_freq = int(self.nb_fft/2)+1
        self.samp_freq = samp_freq
        self.eps_f = eps_f
        self.cov_timecste_f = cov_timecste_f
        self.cov_coef_f = 1 / (1 + self.samp_freq * self.cov_timecste_f / self.nb_fft)
        # ---- INIT VAR
        self.IR_m = np.zeros((self.nb_fft, self.nb_mic))
        self.IR_m[0, :] = 1.
        self.steer_m = np.ones((self.nb_freq, self.nb_mic), dtype=complex)
        self.cov_m = np.repeat(np.eye(self.nb_mic, dtype=complex)[:, :, np.newaxis], repeats=self.nb_freq, axis=2)
        self.cov_inv_m = np.repeat(np.eye(self.nb_mic, dtype=complex)[:, :, np.newaxis], repeats=self.nb_freq, axis=2)
        self.filt_m = np.zeros((self.nb_freq, self.nb_mic), dtype=complex)

    def processfft(self, fft_inp):
        if not self.bypass:
            self.compute_cov_mat(fft_inp)
            self.update_filter()
            # FILTERING
            sig_filt_m = np.zeros(fft_inp.shape, dtype=complex)
            for mm in range(self.nb_mic):
                sig_filt_m[:, mm] = fft_inp[:, mm] * self.filt_m[:, mm]
            # SUMMATION
            fft_out = np.sum(sig_filt_m, axis=1)[:, np.newaxis]
        else:
            fft_out = fft_inp[:, 0][:, np.newaxis]
        return fft_out

    def update_filter(self):
        """

        :return:
        """
        for freq in range(self.nb_freq):
            steer_v = self.steer_m[freq, :][np.newaxis, :].T
            self.cov_inv_m[:, :, freq] = np.linalg.inv(np.squeeze(self.cov_m[:, :, freq])
                                                       + self.eps_f * np.eye(self.nb_mic, dtype=complex))
            cov_inv_m = np.squeeze(self.cov_inv_m[:, :, freq])
            denom = np.matmul(np.conjugate(steer_v.T), np.matmul(cov_inv_m, steer_v))
            self.filt_m[freq, :] = np.squeeze(np.matmul(cov_inv_m, steer_v) / denom)
        self.filt_m = np.conjugate(self.filt_m)

        return

    def compute_cov_mat(self, fft_inp):
        """"""
        for freq in range(self.nb_freq):
            sig_tmp_v = fft_inp[freq, :][:, np.newaxis]
            self.cov_m[:, :, freq] = self.cov_coef_f * np.matmul(sig_tmp_v, np.conjugate(sig_tmp_v.T)) \
                                     + (1 - self.cov_coef_f) * np.squeeze(self.cov_m[:, :, freq])
        return

    def update_ir(self, ir_m, normalize=True, mic_id=4):
        """
        Update impulse response
        """
        self.IR_m = ir_m
        self.steer_m = np.fft.fft(self.IR_m, self.nb_fft, 0)[0:self.nb_freq]

        # NORMALIZE
        if normalize:
            self.steer_m = self.steer_m / np.repeat(self.steer_m[:, mic_id][:, np.newaxis], self.nb_mic, axis=1)

        return


class BeamformerDMA:
    """
    Order 1 DMA
    """
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=180., mic_dist=0.013,
                 freq_cutlp=100., freq_cuthp=100., bypass=False):
        """

        :param samp_freq:
        :param nb_buffsamp:
        :param nullangle_v:
        :param mic_dist:
        :param bypass:
        """
        # GENERAL
        self.samp_freq = samp_freq
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.coeff_v = np.cos(np.deg2rad(self.nullangle_v)) / (np.cos(np.deg2rad(self.nullangle_v)) - 1)
        # FILTER
        self.freq_cutlp = freq_cutlp
        self.freq_cuthp = freq_cuthp
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='low',
                                    cut_freq=self.freq_cutlp,
                                    order_n=1
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='high',
                                    cut_freq=self.freq_cuthp,
                                    order_n=3
                                    )
        # COMPENSATION GAIN
        self.velocity = 343.
        self.mic_dist = mic_dist
        # I DONT REMEMBER THE ORIGIN OF THE COMMENTED FORMULA, REPLACE BY THE SECOND ONE
        # self.gain_dipole = np.sqrt(1 + (self.velocity / (self.mic_dist * 6 * self.freq_cutlp)) ** 2)
        self.gain_dipole = np.sqrt(1 + (1000 / self.freq_cutlp) ** 2) \
                           / (2 * np.abs(np.sin(self.mic_dist * np.pi * 1000 / self.velocity)))

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_dif = sig_inp[:, 0] - sig_inp[:, 1]  # DIFF
            sig_out = self.coeff_v * sig_inp[:, 0] \
                      + (1 - self.coeff_v) * self.LPFilter.process(sig_dif)[:, 0] * self.gain_dipole
            sig_out = self.HPFilter.process(sig_out)  # HIGHPASS FILTER
        else:
            sig_out = sig_inp[:, 0]

        return sig_out

    def define_nullangle(self, nullangle_v=180.):
        """

        :param nullangle_v: angle of the destructive constraint
        :return:
        """
        self.nullangle_v = nullangle_v
        self.coeff_v = np.cos(np.deg2rad(self.nullangle_v)) / (np.cos(np.deg2rad(self.nullangle_v)) - 1)
        return


class BeamformerDMA2:
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=np.array([180., 90.]),
                 mic_dist=0.013, freq_cutlp=100., freq_cuthp=80., bypass=False):
        """
        Constructor
        :param samp_freq: sampling frequency
        :param nb_buffsamp: buffer number of sample
        :param nullangle_v: angles of the destructive constrains
        (cardio: (180,90) ; hypercardio: (144,72) ; supercardio: (153,106) ; quadrupole: (135,45)
        :param mic_dist: distance between microphones
        :param bypass: Bypass
        """
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.freq_cutlp = freq_cutlp
        self.freq_cuthp = freq_cuthp
        self.sub_dma_1 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp)
        self.sub_dma_1.HPFilter.bypass = True
        self.sub_dma_2 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp)
        self.sub_dma_2.HPFilter.bypass = True
        self.sub_dma_3 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[1], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp, freq_cuthp=self.freq_cuthp)

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_subout_1 = self.sub_dma_1.process(sig_inp[:, 0:2])
            sig_subout_2 = self.sub_dma_2.process(sig_inp[:, 1:3])
            sig_tmp = np.concatenate((sig_subout_1[:, np.newaxis], sig_subout_2[:, np.newaxis]), axis=1)
            sig_out = self.sub_dma_3.process(sig_tmp)
        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out

    def define_nullangle(self, nullangle_v):
        """

        :param nullangle_v: angle of the destructive constraint
        :return:
        """
        self.nullangle_v = nullangle_v
        self.sub_dma_1.define_nullangle(nullangle_v=nullangle_v[0])
        self.sub_dma_2.define_nullangle(nullangle_v=nullangle_v[0])
        self.sub_dma_3.define_nullangle(nullangle_v=nullangle_v[1])
        return


class BeamformerDMA15:
    """

    """
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=np.array([180., 180., 90.]),
                 mic_dist=0.013, freq_cut=800., freq_cutlp_dma2=300., bypass=False):
        """

        :param samp_freq: <float>
        :param nb_buffsamp: <int>
        :param nullangle_v: <3x1 vector>
        :param mic_dist: <float>
        :param freq_cut: <float>
        :param freq_cutlp_dma2: <float>
        :param bypass: <bool>
        """
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.sub_dma_1 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=2*self.mic_dist)
        self.freq_cutlp_dma2 = freq_cutlp_dma2
        self.sub_dma_2 = BeamformerDMA2(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                        nullangle_v=self.nullangle_v[1:3], mic_dist=self.mic_dist, freq_cutlp=self.freq_cutlp_dma2)
        # CROSSOVER FILTER
        self.velocity = 343.
        self.freq_cut = freq_cut
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='low',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='high',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_inp_1 = np.concatenate((sig_inp[:, 0][:, np.newaxis], sig_inp[:, 2][:, np.newaxis]), axis=1)
            sig_subout_1 = self.sub_dma_1.process(sig_inp_1)
            sig_subout_1_lp = self.LPFilter.process(sig_subout_1)
            sig_subout_2 = self.sub_dma_2.process(sig_inp)
            sig_subout_2_hp = self.HPFilter.process(sig_subout_2)
            sig_out = sig_subout_1_lp + sig_subout_2_hp

        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out


class DmaInteraural:
    """
    Reference: Dieudonné and Francart 2018
    """
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=180., mic_dist=0.14, freq_cut=800., bypass=False):
        """

        :param samp_freq:
        :param nb_buffsamp:
        :param nullangle_v:
        :param mic_dist:
        :param freq_cut:
        :param bypass:
        """
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.dma_l = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                   nullangle_v=self.nullangle_v, mic_dist=self.mic_dist)
        self.dma_r = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                   nullangle_v=self.nullangle_v, mic_dist=self.mic_dist)
        # CROSSOVER FILTER
        self.freq_cut = freq_cut
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=2,
                                    type_s='low',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=2,
                                    type_s='high',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:

            sig_inp_l = sig_inp
            sig_inp_r = np.fliplr(sig_inp)

            sig_l = self.dma_l.process(sig_inp_l)
            sig_r = self.dma_r.process(sig_inp_r)
            sig_dma = np.concatenate((sig_l, sig_r), axis=1)

            sig_lp = self.LPFilter.process(sig_dma)
            sig_hp = self.HPFilter.process(sig_inp)

            sig_out = sig_lp + sig_hp

        else:
            sig_out = sig_inp

        return sig_out


class DmaAdaptive:
    """
    Reference: Elko 1995, Luo 2002
    Description: adaptive first order DMA beamformer
    """
    def __init__(self, samp_freq, nb_buffsamp, mic_dist=0.015,
                 freq_cutlp=100., freq_cuthp=100., bypass=False, verbose=False):
        # CONSTANT
        self.celerity = 343.
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        self.mic_dist = mic_dist
        self.delay_n = np.maximum(int(round(self.mic_dist/self.celerity*self.samp_freq)), 1)
        self.delay_line_m = np.zeros((self.delay_n, 2))
        # ESTIMATION
        self.smooth_coeff_f = 0.1  # 0.5
        self.xcorr_frt_back = 0.
        self.power_bck = 0.
        self.null_angle_v = np.zeros((int(self.samp_freq/self.nb_buffsamp),))
        self.coeff_f = 0

        # LOWPASS FILTER
        self.freq_cutlp = freq_cutlp
        self.LPFilter = Butterworth(self.nb_buffsamp,
                                    self.samp_freq,
                                    nb_channels=2,
                                    type_s='low',
                                    cut_freq=self.freq_cutlp,
                                    order_n=1)
        # HIGHPASS
        self.freq_cuthp = freq_cuthp
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=2,
                                    type_s='high',
                                    cut_freq=self.freq_cuthp,
                                    order_n=3
                                    )
        # COMPENSATION GAIN
        self.gain_f = np.sqrt(1 + (1000/self.freq_cutlp)**2)\
                      / (2 * np.abs(np.sin(2 * self.mic_dist * np.pi * 1000 / self.celerity)))

    def process(self, sig_inp):
        if self.verbose:
            print('------------------------------------------------------')

        sig_del = np.concatenate((self.delay_line_m, sig_inp[0:self.nb_buffsamp-self.delay_n]), axis=0)
        self.delay_line_m = sig_inp[self.nb_buffsamp - self.delay_n:, :]
        if not self.bypass:
            # BUILD FRONT/BACK CARDIOID
            sig_frt = sig_inp[:, 0] - sig_del[:, 1]
            sig_bck = sig_inp[:, 1] - sig_del[:, 0]
            sig_crd = np.concatenate((sig_frt[:, np.newaxis], sig_bck[:, np.newaxis]), axis=1)
            sig_crd = self.LPFilter.process(sig_crd)
            sig_crd = self.HPFilter.process(sig_crd)
            # ADAPTIVE COEFF ESTIMATION
            coeff_f = self.estimate_coeff(sig_crd)
            sig_out = sig_crd[:, 0] + coeff_f * sig_crd[:, 1]
            sig_out = sig_out[:, np.newaxis]
            sig_out = sig_out * self.gain_f

        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out

    def estimate_coeff(self, sig_crd):
        # xcorr_frt_bck = self.smooth_coeff_f * np.dot(sig_crd[:, 0], sig_crd[:, 1]) / self.nb_buffsamp \
        #                 + (1-self.smooth_coeff_f) * self.xcorr_frt_back
        # power_bck = self.smooth_coeff_f * np.dot(sig_crd[:, 1], sig_crd[:, 1]) / self.nb_buffsamp \
        #                 + (1 - self.smooth_coeff_f) * self.power_bck

        xcorr_frt_bck, power_bck = np.cov(sig_crd, rowvar=False)[1, :]
        xcorr_frt_bck = self.smooth_coeff_f * xcorr_frt_bck + (1-self.smooth_coeff_f) * self.xcorr_frt_back
        power_bck = self.smooth_coeff_f * power_bck + (1 - self.smooth_coeff_f) * self.power_bck

        self.xcorr_frt_back = xcorr_frt_bck
        self.power_bck = power_bck
        coeff_f = - xcorr_frt_bck / power_bck
        coeff_f = np.clip(coeff_f, -1., 1.)
        # self.coeff_f = self.smooth_coeff_f * coeff_f + (1-self.smooth_coeff_f) * self.coeff_f
        null_angle = np.rad2deg(np.arccos((coeff_f + 1) / (coeff_f - 1)))
        if self.verbose:
            print('xcorr frt bck:   %.2f x10e-6' % (xcorr_frt_bck*1000000))
            print('power back:      %.2f x10e-6' % (power_bck*1000000))
            print('coeff:           %.2f' % coeff_f)
            print('null angle:      %.0f°' % null_angle)

        self.null_angle_v[0:len(self.null_angle_v)-1] = self.null_angle_v[1:len(self.null_angle_v)]
        self.null_angle_v[len(self.null_angle_v)-1] = null_angle
        return coeff_f


class DmaAdaptiveFFT:
    """
    Reference: Elko 1995, Luo 2002, Chen 2013
    Description: adaptive first order DMA beamformer working in FREQ domain
    """
    def __init__(self, samp_freq, nb_fft, mic_dist=0.015,
                 freq_cutlp=100., freq_cuthp=80., bypass=False, verbose=False):
        # CONSTANT
        self.celerity = 343.
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.mic_dist = mic_dist
        #
        self.samp_freq = samp_freq
        self.nb_fft = nb_fft
        self.nb_frq = int(nb_fft/2) + 1
        # CROSS CORR ESTIMATION
        self.smooth_coeff_f = 0.5
        self.xcorr_frt_back = 0.
        self.power_bck = 0.
        self.null_angle_v = np.zeros((int(self.samp_freq / self.nb_fft),))
        #
        freq_v = np.linspace(0, samp_freq/2, self.nb_frq)
        self.del_m = np.repeat(np.exp(-1j * 2 * np.pi * freq_v * self.mic_dist / self.celerity)[:, np.newaxis],
                               2, axis=1)
        # Low Pass -- High Pass
        self.freq_cutlp = freq_cutlp
        self.freq_cuthp = freq_cuthp
        LP_m = 1 / (1 + 1j * freq_v / self.freq_cutlp)
        HP_m = ((1j * freq_v / self.freq_cuthp) / (1 + 1j * freq_v / self.freq_cuthp)) ** 3
        gain_f = np.sqrt(1 + (1000 / self.freq_cutlp) ** 2) \
                 / (2 * np.abs(np.sin(2 * self.mic_dist * np.pi * 1000 / self.celerity)))
        self.comp_filter_m = LP_m * HP_m * gain_f
        # WINDOW IR IN TIME DOMAIN
        comp_fft = np.concatenate((self.comp_filter_m, np.conjugate(np.flipud(self.comp_filter_m[1:self.nb_frq - 1]))))
        comp_ir = np.real(np.fft.ifft(comp_fft, nb_fft))
        win_size = int(nb_fft/6)
        window_v = np.concatenate((np.ones(nb_fft-win_size-int(nb_fft/2)),
                                   np.hanning(2*win_size)[win_size:],
                                   np.zeros((int(nb_fft/2)))),
                                  axis=0)
        comp_ir = comp_ir * window_v
        comp_fft = np.fft.fft(comp_ir)
        self.comp_filter_m = comp_fft[0:self.nb_frq]

    def processfft(self, fft_inp):
        if not self.bypass:
            fft_del = fft_inp * self.del_m
            fft_frt = fft_inp[:, 0] - fft_del[:, 1]
            fft_bck = fft_inp[:, 1] - fft_del[:, 0]
            fft_crd = np.concatenate((fft_frt[:, np.newaxis], fft_bck[:, np.newaxis]), axis=1)
            coeff_f = self.estimate_coeff(fft_crd)
            fft_out = fft_crd[:, 0] + coeff_f * fft_crd[:, 1]
            fft_out = fft_out * self.comp_filter_m
            fft_out = fft_out[:, np.newaxis]
        else:
            fft_out = fft_inp[:, 0][:, np.newaxis]

        return fft_out

    def estimate_coeff(self, fft_crd):
        xcorr_frt_bck, power_bck = np.cov(np.abs(fft_crd), rowvar=False)[1, :]
        xcorr_frt_bck = self.smooth_coeff_f * xcorr_frt_bck + (1 - self.smooth_coeff_f) * self.xcorr_frt_back
        power_bck = self.smooth_coeff_f * power_bck + (1 - self.smooth_coeff_f) * self.power_bck
        self.xcorr_frt_back = xcorr_frt_bck
        self.power_bck = power_bck
        coeff_f = - xcorr_frt_bck / (power_bck + 10**-10)
        coeff_f = np.clip(coeff_f, -1., 1.)

        null_angle = np.rad2deg(np.arccos((coeff_f + 1) / (coeff_f - 1)))
        if self.verbose:
            print('xcorr frt bck:   %.2f x10e-6' % (xcorr_frt_bck * 1000000))
            print('power back:      %.2f x10e-6' % (power_bck * 1000000))
            print('coeff:           %.2f' % coeff_f)
            print('null angle:      %.0f°' % null_angle)

        self.null_angle_v[0:len(self.null_angle_v) - 1] = self.null_angle_v[1:len(self.null_angle_v)]
        self.null_angle_v[len(self.null_angle_v) - 1] = null_angle

        return coeff_f


class SuperBeamformer:
    """
    Optimal Directivity Index (DI) beamformer
    c.f. Stadler and Rabinowitz 1993, Harder PhD dissertation 2015
    """
    def __init__(self, nb_buffsamp, nb_fft):
        self.nb_buffsamp = nb_buffsamp
        self.mic = None
        self.nb_mic = 0
        self.weight_v = 0
        self.nb_freq = int(nb_fft/2+1)
        self.grid = None
        self.nb_dir = 0
        self.dir_targ_v = np.array([1, 0, 0])
        self.w_m = 0
        self.DI_v = 0

    def update_mic_directivity(self, *args):
        """

        :param args: expect microphone directivity in format HrtfData object
        :return:
        """
        # CHECK INPUT DATA
        self.nb_mic = len(args)
        self.mic = [0] * self.nb_mic
        grids_tmp = [0] * self.nb_mic
        for ii, arg in enumerate(args):
            assert bb.is_hrtfdata(arg), 'INPUT ARGUMENT MUST BE HRTF DATA OBJECT'
            self.mic[ii] = copy.deepcopy(arg)
            self.mic[ii].time2freq(self.nb_freq)
            grids_tmp[ii] = arg.get_grid()
            if ii > 0:
                if grids_tmp[ii].norm_s == grids_tmp[0].norm_s:
                    grids_tmp[ii].convert_coordinates(grids_tmp[0].norm_s)
                assert grids_tmp[ii].coords_m.shape == grids_tmp[0].coords_m.shape, \
                    'GRID MUST HAVE THE SAME NUMBER OF POSITIONS'
                assert np.sum(np.sum(grids_tmp[ii].coords_m == grids_tmp[0].coords_m)), 'GRID NUMBER '+ii+' IS NOT EQUAL TO GRID 0'
        # GET GRID WEIGHTING
        self.grid = grids_tmp[0]
        self.nb_dir = self.grid.coords_m.shape[0]
        self.weight_v = self.grid.get_spherical_weighting_harder(self.grid)
        return

    def update_optimal_filter(self, dir_targ_v):
        """

        :param dir_targ_v: vector [1x3] float in format [radius (m), azim (deg), elev (deg)]
        :return:
        """
        # PREPARE
        self.dir_targ_v = dir_targ_v
        (trash, idx_targ, trash2) = self.grid.find_closest_point(self.dir_targ_v, norm_s='spherical_1')
        Szz = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        Szz_inv = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        self.w_m = np.zeros((self.nb_mic, self.nb_freq), dtype=complex)
        self.DI_v = np.zeros((self.nb_freq,))
        eps_f = 10**-8
        beampattern_m = np.zeros((self.nb_dir, self.nb_freq), dtype=complex)
        # COMPUTE COMPENSATION FILTERS
        for ff in range(self.nb_freq):
            # BUILD BIG MATRIX
            for mm, mic in enumerate(self.mic):
                if mm == 0:
                    MAT_m = mic.data_m[:, ff][:, np.newaxis].T * self.weight_v
                    steer_v = mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]
                else:
                    MAT_m = np.concatenate((MAT_m, mic.data_m[:, ff][:, np.newaxis].T * self.weight_v), axis=0)
                    steer_v = np.concatenate((steer_v, mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]), axis=0)

            # COVARIANCE MATRIX AND INVERSION
            Szz[:, :, ff] = MAT_m @ MAT_m.conjugate().T
            Szz_inv[:, :, ff] = np.linalg.inv(Szz[:, :, ff] + eps_f * np.eye(self.nb_mic))
            # COMPUTE COMPENSATION FILTERS
            denom = steer_v.conjugate().T @ Szz_inv[:, :, ff] @ steer_v
            self.w_m[:, ff] = steer_v.conjugate().T @ Szz_inv[:, :, ff] / denom

        # plt.semilogx(np.linspace(0, 44100/2, self.nb_freq), u.mag2db(np.abs(self.w_m.T)))
        # plt.show()
        # FILTER LOW AND HIGH PASS FOR SAFETY
        filter_v = np.concatenate((np.hanning(6)[0:3, np.newaxis].T,
                                   np.ones((1, self.nb_freq-3-50), dtype=complex),
                                   np.hanning(100)[50:, np.newaxis].T),
                                  axis=1)
        self.w_m = self.w_m * np.repeat(filter_v, self.nb_mic, axis=0)
        # FLIP IMPULSE RESPONSE
        ir_m = np.fft.ifft(np.concatenate((self.w_m.T, np.flipud(self.w_m[:, 1:self.nb_freq-1].conjugate().T)),  axis=0),
                           axis=0)
        ir_m = np.concatenate((ir_m[int(ir_m.shape[0] / 2):, :], ir_m[0:int(ir_m.shape[0] / 2), :]), axis=0)
        self.w_m = np.fft.fft(ir_m, axis=0)[0:self.nb_freq, :].T

        return

    def get_beampattern(self, plot_b=True):
        # PREPARE
        mag_lim = -40
        (trash, idx_targ, trash2) = self.grid.find_closest_point(self.dir_targ_v, norm_s='spherical_1')
        Szz = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        self.DI_v = np.zeros((self.nb_freq,))
        beampattern_m = np.zeros((self.nb_dir, self.nb_freq), dtype=complex)
        Grid_hp_S = copy.deepcopy(self.grid)
        mask_hp = self.grid.coords_m[:, 2] == 0
        Grid_hp_S.coords_m = self.grid.coords_m[mask_hp, :]
        idx_order = np.argsort(Grid_hp_S.coords_m[:, 1])
        Grid_hp_S.coords_m = Grid_hp_S.coords_m[idx_order, :]
        Grid_hp_S.coords_m = np.concatenate((Grid_hp_S.coords_m, Grid_hp_S.coords_m[0, :][np.newaxis, :]), axis=0)

        # PREPARE PLOT
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        line1, = ax.plot(Grid_hp_S.coords_m[:, 0], Grid_hp_S.coords_m[:, 1])
        plt.axis([mag_lim, -mag_lim, mag_lim, -mag_lim])

        # COMPUTE COMPENSATION FILTERS
        for ff in range(self.nb_freq):
            # BUILD BIG MATRIX
            for mm, mic in enumerate(self.mic):
                if mm == 0:
                    MAT_m = mic.data_m[:, ff][:, np.newaxis].T * self.weight_v
                    steer_v = mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]
                else:
                    MAT_m = np.concatenate((MAT_m, mic.data_m[:, ff][:, np.newaxis].T * self.weight_v), axis=0)
                    steer_v = np.concatenate((steer_v, mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]), axis=0)

            # COVARIANCE MATRIX AND INVERSION
            Szz[:, :, ff] = MAT_m @ MAT_m.conjugate().T
            # COMPUTE DIRECTIVITY INDEX
            denom_DI = self.w_m[:, ff][:, np.newaxis].T @ Szz[:, :, ff] @ np.conj(self.w_m[:, ff][:, np.newaxis]) / self.nb_dir
            self.DI_v[ff] = np.real(self.w_m[:, ff][:, np.newaxis].T
                                    @ steer_v
                                    @ steer_v.conjugate().T
                                    @ np.conj(self.w_m[:, ff][:, np.newaxis])
                                    / denom_DI)

            # COMPUTE BEAMPATTERN
            for dd in range(self.nb_dir):
                beampattern_m[dd, ff] = np.dot(self.w_m[:, ff], MAT_m[:, dd] / self.weight_v[dd])
            dataplot = u.mag2db(np.abs(beampattern_m[:, ff]))
            dataplot[dataplot < mag_lim] = mag_lim

            dataplot_hp = dataplot[mask_hp]
            dataplot_hp = dataplot_hp[idx_order]
            dataplot_hp = np.concatenate((dataplot_hp[:, np.newaxis],
                                          dataplot_hp[0][np.newaxis, np.newaxis]),
                                         axis=0)[:, 0]
            Grid_hp_S.coords_m[:, 0] = dataplot_hp - mag_lim
            Grid_hp_cart_S = copy.deepcopy(Grid_hp_S)
            Grid_hp_cart_S.convert_coordinates('cartesian')

            if plot_b:
                line1.set_xdata(Grid_hp_cart_S.coords_m[:, 0])
                line1.set_ydata(Grid_hp_cart_S.coords_m[:, 1])
                plt.title('Beampattern at ' + str(int(self.mic[0].xaxis_v[ff])) + ' Hz')
                fig.canvas.draw()
                time.sleep(0.1)

        return

    def processfft(self, fft_inp):
        """

        :param fft_inp: matrix [nb_freq x nb_mic] of complex spectra from input microphone
        :return: fft_out: vector [nb_freq x 1] of complex spectrum beamformer output
        """
        # FILTERING
        sig_filt_m = np.zeros(fft_inp.shape, dtype=complex)
        for mm in range(self.nb_mic):
            sig_filt_m[:, mm] = fft_inp[:, mm] * self.w_m[mm, :]
        # SUMMATION
        fft_out = np.sum(sig_filt_m, axis=1)[:, np.newaxis]

        return fft_out


# ========================= BUFFER MANAGEMENT =========================
class BlockProc:
    def __init__(self, nb_buffsamp, nb_channels, window_s='hann'):
        # PARAMETERS
        self.buffer_count = 0
        self.nb_buffsamp = nb_buffsamp
        self.nb_framesamp = self.nb_buffsamp * 2
        self.nb_channels = nb_channels
        # LINES
        self.frame1_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.frame2_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.buf_out = np.zeros((self.nb_buffsamp, self.nb_channels))
        # WINDOW
        self.window_s = window_s
        self.window_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.update_window(self.window_s)

        self.toto = 0

    def input(self, buf_inp):
        self.buffer_count = np.mod(self.buffer_count+1, 4)

        if np.mod(self.buffer_count, 2):
            self.frame1_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] = buf_inp
            self.frame2_m[0:self.nb_buffsamp, :] = buf_inp
        else:
            self.frame1_m[0:self.nb_buffsamp, :] = buf_inp
            self.frame2_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] = buf_inp

        return

    def get_frame(self):
        if np.mod(self.buffer_count, 2):
            frame_m = self.frame1_m
        else:
            frame_m = self.frame2_m

        return frame_m

    def set_frame(self, frame_m):
        if np.mod(self.buffer_count, 2):
            self.frame1_m = frame_m * self.window_m
        else:
            self.frame2_m = frame_m * self.window_m

        return

    def output(self):
        if not np.mod(self.buffer_count, 2):
            buf_out = self.frame1_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] + \
                      self.frame2_m[0:self.nb_buffsamp, :]
        else:
            buf_out = self.frame1_m[0:self.nb_buffsamp, :] + \
                      self.frame2_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :]

        return buf_out

    def update_window(self, window_s='hann'):
        """

        :param window_s: 'blackman' cf. Kates 2005 // 'hann' ensure perfect overlap
        :return:
        """
        if window_s is 'blackman':
            self.window_m = np.repeat(np.blackman(self.nb_framesamp)[:, np.newaxis], self.nb_channels, axis=1)
        elif window_s is 'hann':
            self.window_m = np.repeat(np.hanning(self.nb_framesamp)[:, np.newaxis], self.nb_channels, axis=1)

        return


class OverlapSave:
    def __init__(self, nb_bufsamp, nb_channels, nb_datain, bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.nb_overlap = nb_datain - nb_bufsamp
        self.data_overlap = np.zeros((self.nb_overlap, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig_inp):
        """"""
        # SPLIT OUTPUT AND OVERLAP TMP
        sig_out = sig_inp[0:self.nb_bufsamp, :]
        sig_2nextframe = sig_inp[self.nb_bufsamp:, :]
        # OVERLAP ADD
        if self.nb_bufsamp > self.data_overlap.shape[0]:  # N > L-1
            overlap_out = np.concatenate((self.data_overlap, np.zeros((self.nb_bufsamp - self.data_overlap.shape[0], self.nb_channels))), axis=0)
            overlap_2nextframe = np.zeros(self.data_overlap.shape)
        else:  # L-1 >= N
            overlap_out = self.data_overlap[0:self.nb_bufsamp]
            overlap_2nextframe = self.data_overlap[self.nb_bufsamp:]
            overlap_2nextframe = np.concatenate((overlap_2nextframe, np.zeros((self.nb_bufsamp, self.nb_channels))), axis=0)

        sig_out = sig_out + overlap_out
        # OVERLAP UPDATE
        self.data_overlap = sig_2nextframe + overlap_2nextframe

        return sig_out


class BlockProc2:
    """
    Ref: Kates 2005, Principles of Digital Dynamic-Range Compression
    """
    def __init__(self, nb_buffsamp, nb_fft, nb_channels_inp, nb_channels_out, window_s='hann'):
        # PARAMETERS
        self.nb_buffsamp = nb_buffsamp
        self.nb_framesamp = self.nb_buffsamp * 2
        self.nb_fft = nb_fft
        self.overlap_size = self.nb_fft - self.nb_buffsamp
        # self.overlap_size = 2 * self.nb_fft - 1 - self.nb_buffsamp
        self.nb_channels_inp = nb_channels_inp
        self.nb_channels_out = nb_channels_out
        # LINES
        self.frame_inp = np.zeros((self.nb_framesamp, self.nb_channels_inp))
        self.overlap_m = np.zeros((self.overlap_size, self.nb_channels_out))
        # WINDOW
        self.window_s = window_s
        self.window_m = np.zeros((self.nb_framesamp, self.nb_channels_inp))
        self.update_window(self.window_s)

    def input2frame(self, buffer_inp):
        self.frame_inp = np.concatenate((self.frame_inp[int(self.nb_framesamp/2):, :], buffer_inp), axis=0)
        frame_win = self.frame_inp * self.window_m
        return frame_win

    def frame2output(self, frame_out):
        # PREPARE BUFFER OUT
        frame2buffer = frame_out[0:self.nb_buffsamp, :]
        overlap2buffer = self.overlap_m[0:self.nb_buffsamp, :]
        buffer_out = frame2buffer + overlap2buffer

        # UPDATE OVERLAP SAVE ARRAY
        self.overlap_m = np.concatenate((self.overlap_m[self.nb_buffsamp:, :], np.zeros((self.nb_buffsamp, self.nb_channels_out))), axis=0)
        self.overlap_m += frame_out[self.nb_buffsamp:, :]
        return buffer_out

    def update_window(self, window_s='hann'):
        """

        :param window_s: 'blackman' cf. Kates 2005 // 'hann' ensure perfect overlap
        :return:
        """
        if window_s is 'blackman':
            self.window_m = np.repeat(np.blackman(self.nb_framesamp)[:, np.newaxis], self.nb_channels_inp, axis=1)
        elif window_s is 'hann':
            self.window_m = np.repeat(np.hanning(self.nb_framesamp)[:, np.newaxis], self.nb_channels_inp, axis=1)

        return


# ========================= SINGLE CANAL SPEECH ENHANCEMENT =========================

class NoiseSoustraction:
    def __init__(self, nb_buffsamp, nb_channels, samp_freq, threshold_f=6, gr_timecste_f=0.5, bypass=False, verbose=False):
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.nb_buffsamp = nb_buffsamp
        self.nb_buffsamp_2 = int(self.nb_buffsamp/2)+1
        self.nb_channels = nb_channels
        self.samp_freq = samp_freq
        # PARAMETERS
        self.threshold_f = threshold_f
        # NOISE MAG ESTIMATION
        self.noisedb_prev = np.zeros((self.nb_buffsamp_2, nb_channels))
        self.db_atk_f = 3.
        self.db_rls_f = 10.
        # SMOOTH GR IN FREQ
        N = self.nb_buffsamp / 10.
        self.win_v = np.hanning(N) / (N/2)
        # SMOOTH GR IN TIME
        self.gr_timecste_f = gr_timecste_f
        self.gr_coef_f = 1 / (1 + self.samp_freq * self.gr_timecste_f / self.nb_buffsamp)
        self.gain_redu_prev = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        # FOR PLOT
        self.magdb_inp = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        self.magdb_out = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        # SPECIAL
        self.first_flag = True

    def processfft(self, fft_inp):
        magdb_inp = u.mag2db(fft_inp)
        if not self.bypass:
            # ---- GET MAGNITUDE
            mag_inp = np.abs(fft_inp)

            # ---- NOISE MAGNITUDE ESTIMATION
            noise_coef = np.zeros(magdb_inp.shape)
            if not self.first_flag:
                mask_atk_b = magdb_inp - self.noisedb_prev > 0
                mask_rls_b = magdb_inp - self.noisedb_prev < 0
                noise_coef[mask_atk_b] = np.exp(-np.abs(magdb_inp[mask_atk_b] - self.noisedb_prev[mask_atk_b])
                                                / self.db_atk_f)
                noise_coef[mask_rls_b] = np.exp(-np.abs(magdb_inp[mask_rls_b] - self.noisedb_prev[mask_rls_b])
                                                / self.db_rls_f)
                noisedb = noise_coef * magdb_inp + (1 - noise_coef) * self.noisedb_prev
            else:
                noisedb = magdb_inp
                self.first_flag = False

            # BACKUP NOISE MAG
            self.noisedb_prev = noisedb

            noisedb = noisedb + self.threshold_f
            noisemag = u.db2mag(noisedb)

            # ---- COMPUTE GR
            gain_redu = 10 ** -6 * np.ones(mag_inp.shape)
            mask = mag_inp ** 2 > noisemag ** 2
            gain_redu[mask] = np.sqrt(1 - (noisemag[mask] ** 2) / (mag_inp[mask] ** 2))
            # ---- SMOOTH GR IN FREQ
            toto = np.zeros((self.nb_buffsamp_2 + len(self.win_v) - 1, self.nb_channels))
            for cc in range(self.nb_channels):
                # MEDIAN FILTERING
                gain_redu[:, cc] = spsig.medfilt(gain_redu[:, cc], 7)
                # WEIGHTED MEAN FILTERING (HANN)
                toto[:, cc] = np.convolve(gain_redu[:, cc], self.win_v)
            idx = int(len(self.win_v) / 2)
            gain_redu = toto[idx:idx + self.nb_buffsamp_2, :]
            # ---- SMOOTH GR IN TIME
            gain_redu = self.gr_coef_f * gain_redu + (1 - self.gr_coef_f) * self.gain_redu_prev
            self.gain_redu_prev = gain_redu
            gain_redu = u.min_phase_spectrum(gain_redu)
            # ---- APPLY GR
            fft_out = fft_inp * gain_redu

        else:
            fft_out = fft_inp

        # FOR PLOT
        magdb_out = u.mag2db(fft_out)
        self.magdb_inp = magdb_inp
        self.magdb_out = magdb_out

        return fft_out


class WienerFilter:
    def __init__(self, nb_buffsamp, nb_channels, samp_freq, threshold_f=6, gr_timecste_f=0.5, bypass=False, verbose=False):
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.nb_buffsamp = nb_buffsamp
        self.nb_buffsamp_2 = int(self.nb_buffsamp/2)+1
        self.nb_channels = nb_channels
        self.samp_freq = samp_freq
        # PARAMETERS
        self.threshold_f = threshold_f
        # NOISE MAG ESTIMATION
        self.noisedb_prev = np.zeros((self.nb_buffsamp_2, nb_channels))
        self.db_atk_f = 3.
        self.db_rls_f = 10.
        # SMOOTH GR IN FREQ
        N = self.nb_buffsamp / 10.
        self.win_v = np.hanning(N) / (N/2)
        # SMOOTH GR IN TIME
        self.gr_timecste_f = gr_timecste_f
        self.gr_coef_f = 1 / (1 + self.samp_freq * self.gr_timecste_f / self.nb_buffsamp)
        self.gain_redu_prev = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        # FOR PLOT
        self.magdb_inp = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        self.magdb_out = np.zeros((self.nb_buffsamp_2, self.nb_channels))
        # SPECIAL
        self.first_flag = True

    def processfft(self, fft_inp):
        magdb_inp = u.mag2db(fft_inp)
        if not self.bypass:
            # ---- GET MAGNITUDE
            mag_inp = np.abs(fft_inp)

            # ---- NOISE MAGNITUDE ESTIMATION
            noise_coef = np.zeros(magdb_inp.shape)
            if not self.first_flag:
                mask_atk_b = magdb_inp - self.noisedb_prev > 0
                mask_rls_b = magdb_inp - self.noisedb_prev < 0
                noise_coef[mask_atk_b] = np.exp(-np.abs(magdb_inp[mask_atk_b] - self.noisedb_prev[mask_atk_b])
                                                / self.db_atk_f)
                noise_coef[mask_rls_b] = np.exp(-np.abs(magdb_inp[mask_rls_b] - self.noisedb_prev[mask_rls_b])
                                                / self.db_rls_f)
                noisedb = noise_coef * magdb_inp + (1 - noise_coef) * self.noisedb_prev
            else:
                noisedb = magdb_inp
                self.first_flag = False

            # BACKUP NOISE MAG
            self.noisedb_prev = noisedb

            noisedb = noisedb + self.threshold_f
            noisemag = u.db2mag(noisedb)

            # ---- COMPUTE GR
            gain_redu = np.maximum(mag_inp**2 - noisemag**2, np.zeros(mag_inp.shape)) / (mag_inp**2 + 10**-6 * np.ones(mag_inp.shape))
            # ---- SMOOTH GR IN TIME
            gain_redu = self.gr_coef_f * gain_redu + (1 - self.gr_coef_f) * self.gain_redu_prev
            self.gain_redu_prev = gain_redu
            gain_redu = u.min_phase_spectrum(gain_redu)
            # ---- APPLY GR
            fft_out = fft_inp * gain_redu

        else:
            fft_out = fft_inp

        # FOR PLOT
        magdb_out = u.mag2db(fft_out)
        self.magdb_inp = magdb_inp
        self.magdb_out = magdb_out

        return fft_out


# ========================= BINAURAL CLASSES =========================
class RTBinauralizer:
    """
    ================================
    Real time binauralizer
    ================================

    """
    def __init__(self, l_hrtf, r_hrtf, nb_bufsamp, grid_target=None, bypass=False):
        """
        Constructor
        """
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.nb_bufsamp = nb_bufsamp
        self.grid_target = grid_target
        self.bypass = bypass
        self.conv = ConvolutionIR(nb_bufsamp=self.nb_bufsamp, bypass=self.bypass)
        if grid_target is not None:
            self.update_positions(grid_target=grid_target)

    def process(self, sig_m):
        # DECLARE VAR
        sig_out_m = np.zeros((self.conv.nb_bufsamp, 2))
        sig_m = np.repeat(sig_m, 2, axis=1)
        # CONVOLUTION
        conv_out_m = self.conv.process(sig_m)
        # SUM LEFT-RIGHT
        idx_middle_n = int(conv_out_m.shape[1]/2)
        sig_out_m[:, 0] = np.sum(conv_out_m[:, 0:idx_middle_n], axis=1)
        sig_out_m[:, 1] = np.sum(conv_out_m[:, idx_middle_n:], axis=1)

        return sig_out_m

    def update_positions(self, grid_target):
        self.grid_target = grid_target
        l_hrtf_tmp = copy.deepcopy(self.l_hrtf)
        r_hrtf_tmp = copy.deepcopy(self.r_hrtf)
        ir_m = np.concatenate((l_hrtf_tmp.subset(grid_target).data_m, r_hrtf_tmp.subset(grid_target).data_m), axis=0)
        ir_m = ir_m.T
        self.conv.update_ir(ir_m)
        return

    def update_hrir(self, l_hrtf, r_hrtf):
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.update_positions(self.grid_target)
        return


class RTBinauralizerFFT:
    """
    ================================
    Real time binauralizer in FREQ domain
    ================================

    """
    def __init__(self, l_hrtf, r_hrtf, nb_fft, grid_target=None, bypass=False):
        """
        Constructor
        """
        # GENERAL
        self.bypass = bypass
        self.nb_fft = nb_fft
        self.nb_frq = int(self.nb_fft / 2) + 1
        #
        self.tf_m = 0
        self.idx_middle_n = 0
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.grid_target = grid_target
        if grid_target is not None:
            self.update_positions(grid_target=grid_target)

    def processfft(self, fft_inp):
        # CONVOLUTION
        fft_inp = np.concatenate((fft_inp, fft_inp), axis=1)
        fft_cnv = fft_inp * self.tf_m
        # SUM LEFT-RIGHT
        fft_out = np.zeros((self.nb_frq, 2), dtype=complex)
        fft_out[:, 0] = np.sum(fft_cnv[:, 0:self.idx_middle_n], axis=1)
        fft_out[:, 1] = np.sum(fft_cnv[:, self.idx_middle_n:], axis=1)
        return fft_out

    def update_positions(self, grid_target):
        self.grid_target = grid_target
        l_hrtf_tmp = copy.deepcopy(self.l_hrtf)
        r_hrtf_tmp = copy.deepcopy(self.r_hrtf)
        ir_m = np.concatenate((l_hrtf_tmp.subset(grid_target).data_m, r_hrtf_tmp.subset(grid_target).data_m), axis=0)
        ir_m = ir_m.T
        self.idx_middle_n = int(ir_m.shape[1]/2)
        self.tf_m = np.fft.fft(ir_m, self.nb_fft, axis=0)[0:self.nb_frq, :]
        return

    def update_hrir(self, l_hrtf, r_hrtf):
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.update_positions(self.grid_target)
        return

