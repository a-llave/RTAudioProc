import numpy as np


class Compressor:
    """
    ================================
    Real time dynamic range compressor
    ================================

    """
    def __init__(self, nb_bufsamp,
                 nb_channels,
                 thrsh=0,
                 ratio=1,
                 time_attack=0.005,
                 time_release=0.05,
                 knee_width=1,
                 fs_f=48000,
                 bypass=False,
                 verbose=False):
        """
        Constructor
        """

        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.gain_db = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.env_prev = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.thrsh = thrsh
        self.ratio = ratio
        self.coeff_attack = np.exp(-1/(time_attack*fs_f))
        self.coeff_release = np.exp(-1/(time_release*fs_f))
        self.kneeWidth = knee_width
        self.bypass = bypass
        self.verbose = verbose

    def process(self, npdata_in):
        """
        Process dynamic range compression
        """
        if not self.bypass:
            # GAIN COMPUTER
            inp_gr = 20 * np.log10(np.absolute(npdata_in) + np.finfo(float).eps)
            out_gr = np.zeros(npdata_in.shape)
            out_env = np.zeros(npdata_in.shape)
            mask_lin = 2 * (inp_gr - self.thrsh) < - self.kneeWidth
            mask_kne = 2 * np.absolute(inp_gr - self.thrsh) <= self.kneeWidth
            mask_cmp = 2 * (inp_gr - self.thrsh) > self.kneeWidth
            out_gr[mask_lin] = inp_gr[mask_lin]
            out_gr[mask_kne] = inp_gr[mask_kne] \
                               + (1 / self.ratio - 1) * (inp_gr[mask_kne] - self.thrsh + self.kneeWidth / 2) ** 2 \
                               / (2 * self.kneeWidth)
            out_gr[mask_cmp] = self.thrsh + (inp_gr[mask_cmp] - self.thrsh) / self.ratio
            # PEAK DETECTOR
            for cc in range(0, self.nb_channels):
                inp_env = inp_gr - out_gr
                if inp_env[0, cc] > self.env_prev[self.nb_bufsamp-1, cc]:
                    out_env[0, cc] = self.coeff_attack * self.env_prev[self.nb_bufsamp-1, cc] + (1 - self.coeff_attack) * inp_env[0, cc]
                else:
                    out_env[0, cc] = self.coeff_release * self.env_prev[self.nb_bufsamp-1, cc]

                for nn in range(1, self.nb_bufsamp):
                    if inp_env[nn, cc] > out_env[nn - 1, cc]:
                        out_env[nn, cc] = self.coeff_attack * out_env[nn - 1, cc] + (1 - self.coeff_attack) * inp_env[nn, cc]
                    else:
                        out_env[nn, cc] = self.coeff_release * out_env[nn - 1, cc]

            # BACK UP
            self.env_prev = out_env
            # GAIN REDUCTION
            self.gain_db = -out_env
            npdata_out = npdata_in * np.power(10, self.gain_db / 20)
            if np.min(self.gain_db) < -1 and self.verbose:
                gr_v = np.mean(self.gain_db, axis=0)
                print('GAIN REDUCTION:')
                for ch, gr in enumerate(gr_v):
                    print('CHANNEL %i: %.0f' % (ch, gr))
        else:
            npdata_out = npdata_in

        return npdata_out