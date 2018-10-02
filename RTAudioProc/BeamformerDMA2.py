import numpy as np
from .BeamformerDMA import *


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