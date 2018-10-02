from .ConvolutionIR import *


class RTBinauralizer:
    """
    ================================
    Real time binauralizer
    ================================

    """
    def __init__(self, l_hrtf, r_hrtf, nb_bufsamp, grid_target=None, bypass=False):
        """
        Constructor
        """
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.nb_bufsamp = nb_bufsamp
        self.grid_target = grid_target
        self.bypass = bypass
        self.conv = ConvolutionIR(nb_bufsamp=self.nb_bufsamp, bypass=self.bypass)
        if grid_target is not None:
            self.update_positions(grid_target=grid_target)

    def process(self, sig_m):
        # DECLARE VAR
        sig_out_m = np.zeros((self.conv.nb_bufsamp, 2))
        sig_m = np.repeat(sig_m, 2, axis=1)
        # CONVOLUTION
        conv_out_m = self.conv.process(sig_m)
        # SUM LEFT-RIGHT
        idx_middle_n = int(conv_out_m.shape[1]/2)
        sig_out_m[:, 0] = np.sum(conv_out_m[:, 0:idx_middle_n], axis=1)
        sig_out_m[:, 1] = np.sum(conv_out_m[:, idx_middle_n:], axis=1)

        return sig_out_m

    def update_positions(self, grid_target):
        self.grid_target = grid_target
        l_hrtf_tmp = copy.deepcopy(self.l_hrtf)
        r_hrtf_tmp = copy.deepcopy(self.r_hrtf)
        ir_m = np.concatenate((l_hrtf_tmp.subset(grid_target).data_m, r_hrtf_tmp.subset(grid_target).data_m), axis=0)
        ir_m = ir_m.T
        self.conv.update_ir(ir_m)
        return

    def update_hrir(self, l_hrtf, r_hrtf):
        l_hrtf.freq2time()
        r_hrtf.freq2time()
        self.l_hrtf = l_hrtf
        self.r_hrtf = r_hrtf
        self.update_positions(self.grid_target)
        return
