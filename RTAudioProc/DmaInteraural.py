from .BeamformerDMA import *


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
