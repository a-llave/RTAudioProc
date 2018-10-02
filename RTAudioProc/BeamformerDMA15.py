from .BeamformerDMA2 import *


class BeamformerDMA15:
    """

    """
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=np.array([180., 180., 90.]),
                 mic_dist=0.013, freq_cut=800., freq_cutlp_dma2=300., bypass=False):
        """

        :param samp_freq: <float>
        :param nb_buffsamp: <int>
        :param nullangle_v: <3x1 vector>
        :param mic_dist: <float>
        :param freq_cut: <float>
        :param freq_cutlp_dma2: <float>
        :param bypass: <bool>
        """
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.sub_dma_1 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=2*self.mic_dist)
        self.freq_cutlp_dma2 = freq_cutlp_dma2
        self.sub_dma_2 = BeamformerDMA2(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                        nullangle_v=self.nullangle_v[1:3], mic_dist=self.mic_dist, freq_cutlp=self.freq_cutlp_dma2)
        # CROSSOVER FILTER
        self.velocity = 343.
        self.freq_cut = freq_cut
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='low',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
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
            sig_inp_1 = np.concatenate((sig_inp[:, 0][:, np.newaxis], sig_inp[:, 2][:, np.newaxis]), axis=1)
            sig_subout_1 = self.sub_dma_1.process(sig_inp_1)
            sig_subout_1_lp = self.LPFilter.process(sig_subout_1)
            sig_subout_2 = self.sub_dma_2.process(sig_inp)
            sig_subout_2_hp = self.HPFilter.process(sig_subout_2)
            sig_out = sig_subout_1_lp + sig_subout_2_hp

        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out
