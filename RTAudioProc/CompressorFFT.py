import numpy as np
import utils_llave as u


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