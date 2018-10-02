import numpy as np
from .Butterworth import *


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