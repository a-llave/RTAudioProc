import numpy as np


class OverlapSave:
    def __init__(self, nb_bufsamp, nb_channels, nb_datain, bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.nb_overlap = nb_datain - nb_bufsamp
        self.data_overlap = np.zeros((self.nb_overlap, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig_inp):
        """"""
        # SPLIT OUTPUT AND OVERLAP TMP
        sig_out = sig_inp[0:self.nb_bufsamp, :]
        sig_2nextframe = sig_inp[self.nb_bufsamp:, :]
        # OVERLAP ADD
        if self.nb_bufsamp > self.data_overlap.shape[0]:  # N > L-1
            overlap_out = np.concatenate((self.data_overlap, np.zeros((self.nb_bufsamp - self.data_overlap.shape[0], self.nb_channels))), axis=0)
            overlap_2nextframe = np.zeros(self.data_overlap.shape)
        else:  # L-1 >= N
            overlap_out = self.data_overlap[0:self.nb_bufsamp]
            overlap_2nextframe = self.data_overlap[self.nb_bufsamp:]
            overlap_2nextframe = np.concatenate((overlap_2nextframe, np.zeros((self.nb_bufsamp, self.nb_channels))), axis=0)

        sig_out = sig_out + overlap_out
        # OVERLAP UPDATE
        self.data_overlap = sig_2nextframe + overlap_2nextframe

        return sig_out