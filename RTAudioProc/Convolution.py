import numpy as np


class Convolution:
    def __init__(self, nb_bufsamp, nb_channels, bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.data_overlap = np.zeros((self.nb_bufsamp-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1, sig2):
        """
        Process the convolution between sig 1 and sig2
        """
        if not self.bypass:
            # SHAPE
            sig1_shape = sig1.shape
            sig2_shape = sig2.shape
            # ASSERTION NCHANNELS
            nsample_ft = sig1_shape[0] + sig2_shape[0] - 1
            # FFT PARTY
            sig1_ft = np.fft.fft(sig1, nsample_ft, 0)
            sig2_ft = np.fft.fft(sig2, nsample_ft, 0)
            sig_out_ft = np.multiply(sig1_ft, sig2_ft)
            # COME BACK TO TIME DOMAIN
            sig_out = np.real(np.fft.ifft(sig_out_ft, nsample_ft, 0))
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:sig1_shape[0], :]
            overlap_tmp = sig_out[sig1_shape[0]:sig_out.shape[0], :]
            # OVERLAP ADD
            if sig1_shape[0] > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((sig1_shape[0] - self.data_overlap.shape[0], self.data_overlap.shape[1]))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:sig1_shape[0]]
                overlap_prev = self.data_overlap[sig1_shape[0]:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((sig1_shape[0], self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output