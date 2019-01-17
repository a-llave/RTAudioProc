import numpy as np
import utils_llave as u
import matplotlib.pyplot as plt

class CompressorFftHa:
    """
    ================================
    Real time dynamic range compressor
    ================================
    Ref.:   - Digital Dynamic Range Compressor Design - A Tutorial and Analysis, Giannoulis et al. 2012
            - Principles of Digital Dynamic-Range Compression, James M. Kates 2005
    """
    def __init__(self, nb_fft, nb_channels, samp_freq=48000,
                 fq_ctr_v=[250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000],
                 Laud_v=[0, 0, 10, 20, 30, 40, 50, 50, 60],
                 Lpain_v=[100, 100, 100, 100, 100, 100, 100, 100, 100],
                 thr_ratio_v=[2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3],
                 knee_width=1, time_attack=0.01, time_release=0.06,
                 wet_f=1, bypass=False, verbose=False):
        # GENERAL
        self.bypass = bypass
        self.verbose = verbose
        self.nb_fft = nb_fft
        self.nb_freq = int(nb_fft/2+1)
        self.nb_channels = nb_channels
        self.samp_freq = samp_freq
        # FREQ SCALE
        self.fc_v = fq_ctr_v
        self.nb_band = len(self.fc_v)
        self.flim_v = np.zeros((self.nb_band + 1))
        for ii in range(self.nb_band - 1):
            self.flim_v[ii + 1] = np.round(np.sqrt(self.fc_v[ii] * self.fc_v[ii + 1]))
            self.flim_v[len(self.flim_v) - 1] = self.samp_freq / 2
        self.freq_v = np.linspace(0, self.samp_freq / 2, self.nb_freq)
        self.mask = [0] * self.nb_band
        for band in range(self.nb_band):
            mask1 = self.freq_v >= self.flim_v[band]
            mask2 = self.freq_v < self.flim_v[band + 1]
            self.mask[band] = mask1 * mask2
        # PARAMETERS
        self.gain_v = np.array(Laud_v)
        self.Laud_v = np.array(Laud_v)
        self.Lpain_v = np.array(Lpain_v)
        self.thr_ratio_v = np.array(thr_ratio_v)
        self.thrsh = 0.
        self.ratio = 0.
        self.audition2compression_param()
        self.kneeWidth = knee_width
        self.time_attack = time_attack
        self.time_release = time_release
        self.wet_f = wet_f
        # SMOOTH GR IN TIME
        self.coefA_f = np.exp(-1 / (self.time_attack * self.samp_freq))
        self.coefR_f = np.exp(-1 / (self.time_release * self.samp_freq))
        self.gr_prev_m = np.zeros((self.nb_band, self.nb_channels))
        # SMOOTH GR IN FREQ
        # N = self.nb_fft / 10.
        N = 1
        self.win_v = np.hanning(N) / np.sum(np.hanning(N))
        # FOR PLOT
        self.mag_inp = np.zeros((self.nb_freq, self.nb_channels))
        self.mag_out = np.zeros((self.nb_freq, self.nb_channels))
        self.gr_plot = np.zeros((self.nb_freq, self.nb_channels))
        self.gr2_plot = np.zeros((self.nb_freq, self.nb_channels))
        self.rms_band = np.zeros((self.nb_band, self.nb_channels))

    def audition2compression_param(self):
        thrsh_v = self.thr_ratio_v * self.Lpain_v + (1 - self.thr_ratio_v) * self.Laud_v - self.gain_v
        # thrsh_v = self.thr_ratio_v * self.Lpain_v + (1-self.thr_ratio_v)*self.Laud_v-100-self.gain_v
        ratio_v = 1 + self.Laud_v / ((1-self.thr_ratio_v)*(self.Lpain_v-self.Laud_v))
        self.thrsh = np.repeat(np.array(thrsh_v)[:, np.newaxis], self.nb_channels, axis=1)
        self.ratio = np.repeat(np.array(ratio_v)[:, np.newaxis], self.nb_channels, axis=1)
        return

    def processfft(self, fft_inp):
        magdb_inp = u.mag2db(fft_inp)
        if not self.bypass:
            rms_band = np.zeros((self.nb_band, self.nb_channels))
            for band in range(self.nb_band):
                mag_band = magdb_inp[self.mask[band], :]
                rms_band[band, :] = np.mean(mag_band, axis=0)
            self.rms_band = rms_band
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
            gr_band_m = gc_inp - gc_out
            # SMOOTH GR IN TIME
            mask_atk_b = gr_band_m >= self.gr_prev_m
            mask_rls_b = gr_band_m < self.gr_prev_m
            gr_band_m[mask_atk_b] = self.coefA_f * gr_band_m[mask_atk_b] + (1 - self.coefA_f) * self.gr_prev_m[mask_atk_b]
            gr_band_m[mask_rls_b] = self.coefR_f * gr_band_m[mask_rls_b] + (1 - self.coefR_f) * self.gr_prev_m[mask_rls_b]
            self.gr_prev_m = gr_band_m
            # ADD COMPENSATION GAIN
            gr_band_m = np.repeat(self.gain_v[:, np.newaxis], self.nb_channels, axis=1) - gr_band_m
            # INTERPOLATION GR
            gr_itrp_m = np.zeros(fft_inp.shape)
            for band in range(self.nb_band):
                gr_itrp_m[self.mask[band], :] = gr_band_m[band, :]
            self.gr2_plot = gr_itrp_m
            # ---- SMOOTH GR IN FREQ
            gr_m = np.zeros((self.nb_freq + len(self.win_v) - 1, self.nb_channels))
            for cc in range(self.nb_channels):
                # WEIGHTED MEAN FILTERING (HANN)
                gr_m[:, cc] = np.convolve(gr_itrp_m[:, cc], self.win_v)
            idx = int(len(self.win_v) / 2)
            gr_m = gr_m[idx:idx + self.nb_freq, :]
            # GAIN REDUCTION TO FILTER
            gain_lin_m = u.db2mag(gr_m)
            gain_sym_m = np.concatenate((gain_lin_m, np.flipud(gain_lin_m[1:self.nb_freq - 1, :])), axis=0)
            gain_min_m = u.min_phase_spectrum(gain_sym_m)
            gain_min_m = gain_min_m[0:self.nb_freq, :]
            # DRY/WET
            gain_min_m = gain_min_m * self.wet_f + (1 - self.wet_f)
            self.gr_plot = u.mag2db(gain_min_m)
            # APPLY GAIN REDUCTION
            fft_out = fft_inp * gain_min_m
        else:
            fft_out = fft_inp

        # FOR PLOT
        self.mag_inp = magdb_inp
        self.mag_out = u.mag2db(fft_out)
        return fft_out
