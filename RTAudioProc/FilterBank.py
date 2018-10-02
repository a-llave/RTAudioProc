from .Butterworth import *


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