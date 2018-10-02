import numpy as np


class BeamformerMVDR:
    def __init__(self, nb_fft, nb_mic=4, samp_freq=48000,
                 eps_f=1e-8, cov_timecste_f=0.5,
                 fq_ctr_v=[150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000],
                 adaptive_b=True, verbose=False, bypass=False):
        # TODO: CHECK IF fq_ctr_v IS SORTED
        # ---- MODE
        self.bypass = bypass
        self.verbose = verbose
        self.adaptive_b = adaptive_b
        # ---- PARAMETERS
        self.nb_mic = nb_mic
        self.nb_fft = nb_fft
        self.nb_freq = int(self.nb_fft/2)+1
        self.samp_freq = samp_freq
        self.eps_f = eps_f
        self.cov_timecste_f = cov_timecste_f
        self.cov_coef_f = 1 / (1 + self.samp_freq * self.cov_timecste_f / self.nb_fft)
        # ---- INIT VAR
        self.IR_m = np.zeros((self.nb_fft, self.nb_mic))
        self.IR_m[0, :] = 1.
        self.steer_m = np.ones((self.nb_freq, self.nb_mic), dtype=complex)
        self.cov_m = np.repeat(np.eye(self.nb_mic, dtype=complex)[:, :, np.newaxis], repeats=self.nb_freq, axis=2)
        self.cov_inv_m = np.repeat(np.eye(self.nb_mic, dtype=complex)[:, :, np.newaxis], repeats=self.nb_freq, axis=2)
        self.filt_m = np.zeros((self.nb_freq, self.nb_mic), dtype=complex)

    def processfft(self, fft_inp):
        if not self.bypass:
            self.compute_cov_mat(fft_inp)
            self.update_filter()
            # FILTERING
            sig_filt_m = np.zeros(fft_inp.shape, dtype=complex)
            for mm in range(self.nb_mic):
                sig_filt_m[:, mm] = fft_inp[:, mm] * self.filt_m[:, mm]
            # SUMMATION
            fft_out = np.sum(sig_filt_m, axis=1)[:, np.newaxis]
        else:
            fft_out = fft_inp[:, 0][:, np.newaxis]
        return fft_out

    def update_filter(self):
        """

        :return:
        """
        for freq in range(self.nb_freq):
            steer_v = self.steer_m[freq, :][np.newaxis, :].T
            self.cov_inv_m[:, :, freq] = np.linalg.inv(np.squeeze(self.cov_m[:, :, freq])
                                                       + self.eps_f * np.eye(self.nb_mic, dtype=complex))
            cov_inv_m = np.squeeze(self.cov_inv_m[:, :, freq])
            denom = np.matmul(np.conjugate(steer_v.T), np.matmul(cov_inv_m, steer_v))
            self.filt_m[freq, :] = np.squeeze(np.matmul(cov_inv_m, steer_v) / denom)
        self.filt_m = np.conjugate(self.filt_m)

        return

    def compute_cov_mat(self, fft_inp):
        """"""
        for freq in range(self.nb_freq):
            sig_tmp_v = fft_inp[freq, :][:, np.newaxis]
            self.cov_m[:, :, freq] = self.cov_coef_f * np.matmul(sig_tmp_v, np.conjugate(sig_tmp_v.T)) \
                                     + (1 - self.cov_coef_f) * np.squeeze(self.cov_m[:, :, freq])
        return

    def update_ir(self, ir_m, normalize=True, mic_id=4):
        """
        Update impulse response
        """
        self.IR_m = ir_m
        self.steer_m = np.fft.fft(self.IR_m, self.nb_fft, 0)[0:self.nb_freq]

        # NORMALIZE
        if normalize:
            self.steer_m = self.steer_m / np.repeat(self.steer_m[:, mic_id][:, np.newaxis], self.nb_mic, axis=1)

        return
