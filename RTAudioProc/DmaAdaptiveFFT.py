import numpy as np


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
