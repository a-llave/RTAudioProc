import numpy as np
import scipy.signal as spsig


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
    