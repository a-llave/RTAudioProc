from .Butterworth import *


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
