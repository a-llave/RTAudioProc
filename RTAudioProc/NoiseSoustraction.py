import numpy as np
import scipy.signal as spsig
import utils_llave as u


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
