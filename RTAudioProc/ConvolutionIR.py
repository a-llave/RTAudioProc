import numpy as np


class ConvolutionIR:
    def __init__(self, nb_bufsamp, ir_m=np.concatenate((np.array([1.0])[:, np.newaxis], np.zeros((127, 1))),
                                                       axis=0), bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = ir_m.shape[1]
        self.IR_m = ir_m
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)
        self.data_overlap = np.zeros((self.IR_m.shape[0]-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1):
        """
        Process the convolution between sig 1 and IR
        """
        if not self.bypass:
            # CHECK
            if len(sig1.shape) == 1:
                sig1 = sig1[:, np.newaxis]

            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                      'columns i.e. the same number of channels'
            # CONVOLUTION
            sig_out = np.zeros((self.nsample_ft, self.nb_channels))
            for ch in range(self.nb_channels):
                sig_out[:, ch] = np.convolve(sig1[:, ch], self.IR_m[:, ch])
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:self.nb_bufsamp, :]
            overlap_tmp = sig_out[self.nb_bufsamp:self.nsample_ft, :]
            # OVERLAP ADD
            if self.nb_bufsamp > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((self.nb_bufsamp - self.data_overlap.shape[0], self.nb_channels))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:self.nb_bufsamp]
                overlap_prev = self.data_overlap[self.nb_bufsamp:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((self.nb_bufsamp, self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE FOR NEXT FRAME
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output

    def update_ir(self, ir_m):
        """
        Update impulse response
        """
        # TODO: remove discontinuity due to data_overlap size actualization when M < N or P < L
        old_size_ir_n = self.IR_m.shape[0]
        old_nb_ch_n = self.IR_m.shape[1]
        self.IR_m = ir_m
        self.nb_channels = self.IR_m.shape[1]
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)

        # OVERLAP VECTOR SIZE ACTUALIZATION
        # RESIZE NB CHANNELS
        if self.IR_m.shape[1] == old_nb_ch_n:  # M = N
            # print('M = N')
            pass
        elif self.IR_m.shape[1] > old_nb_ch_n:  # M > N
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((old_size_ir_n-1, self.nb_channels-old_nb_ch_n))), axis=1)
        elif self.IR_m.shape[1] < old_size_ir_n:  # M < N
            self.data_overlap = self.data_overlap[:, 0:self.nb_channels]
        else:
            print('PROBLEM IN UPDATE IR')

        # RESIZE IR SIZE
        if self.IR_m.shape[0] == old_size_ir_n:  # P = L
            # print('P = L')
            pass
        elif self.IR_m.shape[0] > old_size_ir_n:  # P > L
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((self.IR_m.shape[0]-old_size_ir_n, self.nb_channels))), axis=0)
        elif self.IR_m.shape[0] < old_size_ir_n:  # P < L
            self.data_overlap = self.data_overlap[0:self.IR_m.shape[0] - 1, :]
        else:
            print('PROBLEM IN UPDATE IR')

        return