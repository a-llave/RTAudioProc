import numpy as np
import copy


class RTBinauralizerFFT:
    """
    ================================
    Real time binauralizer in FREQ domain
    ================================

    """
    def __init__(self, l_hrtf, r_hrtf, nb_fft, grid_target=None, bypass=False):
        """
        Constructor
        """
        # GENERAL
        self.bypass = bypass
        self.nb_fft = nb_fft
        self.nb_frq = int(self.nb_fft / 2) + 1
        #
        self.tf_m = 0
        self.idx_middle_n = 0
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.grid_target = grid_target
        if grid_target is not None:
            self.update_positions(grid_target=grid_target)

    def processfft(self, fft_inp):
        # CONVOLUTION
        fft_inp = np.concatenate((fft_inp, fft_inp), axis=1)
        fft_cnv = fft_inp * self.tf_m
        # SUM LEFT-RIGHT
        fft_out = np.zeros((self.nb_frq, 2), dtype=complex)
        fft_out[:, 0] = np.sum(fft_cnv[:, 0:self.idx_middle_n], axis=1)
        fft_out[:, 1] = np.sum(fft_cnv[:, self.idx_middle_n:], axis=1)
        return fft_out

    def update_positions(self, grid_target):
        self.grid_target = grid_target
        l_hrtf_tmp = copy.deepcopy(self.l_hrtf)
        r_hrtf_tmp = copy.deepcopy(self.r_hrtf)
        ir_m = np.concatenate((l_hrtf_tmp.subset(grid_target).data_m, r_hrtf_tmp.subset(grid_target).data_m), axis=0)
        ir_m = ir_m.T
        self.idx_middle_n = int(ir_m.shape[1]/2)
        self.tf_m = np.fft.fft(ir_m, self.nb_fft, axis=0)[0:self.nb_frq, :]
        return

    def update_hrir(self, l_hrtf, r_hrtf):
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.update_positions(self.grid_target)
        return
