import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import binauralbox as bb


class SuperBeamformer:
    """
    Optimal Directivity Index (DI) beamformer
    c.f. Stadler and Rabinowitz 1993, Harder PhD dissertation 2015
    """
    def __init__(self, nb_buffsamp, nb_fft):
        self.nb_buffsamp = nb_buffsamp
        self.mic = None
        self.nb_mic = 0
        self.weight_v = 0
        self.nb_freq = int(nb_fft/2+1)
        self.grid = None
        self.nb_dir = 0
        self.dir_targ_v = np.array([1, 0, 0])
        self.w_m = 0
        self.DI_v = 0

    def update_mic_directivity(self, *args):
        """

        :param args: expect microphone directivity in format HrtfData object
        :return:
        """
        # CHECK INPUT DATA
        self.nb_mic = len(args)
        self.mic = [0] * self.nb_mic
        grids_tmp = [0] * self.nb_mic
        for ii, arg in enumerate(args):
            assert bb.is_hrtfdata(arg), 'INPUT ARGUMENT MUST BE HRTF DATA OBJECT'
            self.mic[ii] = copy.deepcopy(arg)
            self.mic[ii].time2freq(self.nb_freq)
            grids_tmp[ii] = arg.get_grid()
            if ii > 0:
                if grids_tmp[ii].norm_s == grids_tmp[0].norm_s:
                    grids_tmp[ii].convert_coordinates(grids_tmp[0].norm_s)
                assert grids_tmp[ii].coords_m.shape == grids_tmp[0].coords_m.shape, \
                    'GRID MUST HAVE THE SAME NUMBER OF POSITIONS'
                assert np.sum(np.sum(grids_tmp[ii].coords_m == grids_tmp[0].coords_m)), 'GRID NUMBER '+ii+' IS NOT EQUAL TO GRID 0'
        # GET GRID WEIGHTING
        self.grid = grids_tmp[0]
        self.nb_dir = self.grid.coords_m.shape[0]
        self.weight_v = self.grid.get_spherical_weighting_harder(self.grid)
        return

    def update_optimal_filter(self, dir_targ_v):
        """

        :param dir_targ_v: vector [1x3] float in format [radius (m), azim (deg), elev (deg)]
        :return:
        """
        # PREPARE
        self.dir_targ_v = dir_targ_v
        (trash, idx_targ, trash2) = self.grid.find_closest_point(self.dir_targ_v, norm_s='spherical_1')
        Szz = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        Szz_inv = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        self.w_m = np.zeros((self.nb_mic, self.nb_freq), dtype=complex)
        self.DI_v = np.zeros((self.nb_freq,))
        eps_f = 10**-8
        beampattern_m = np.zeros((self.nb_dir, self.nb_freq), dtype=complex)
        # COMPUTE COMPENSATION FILTERS
        for ff in range(self.nb_freq):
            # BUILD BIG MATRIX
            for mm, mic in enumerate(self.mic):
                if mm == 0:
                    MAT_m = mic.data_m[:, ff][:, np.newaxis].T * self.weight_v
                    steer_v = mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]
                else:
                    MAT_m = np.concatenate((MAT_m, mic.data_m[:, ff][:, np.newaxis].T * self.weight_v), axis=0)
                    steer_v = np.concatenate((steer_v, mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]), axis=0)

            # COVARIANCE MATRIX AND INVERSION
            Szz[:, :, ff] = MAT_m @ MAT_m.conjugate().T
            Szz_inv[:, :, ff] = np.linalg.inv(Szz[:, :, ff] + eps_f * np.eye(self.nb_mic))
            # COMPUTE COMPENSATION FILTERS
            denom = steer_v.conjugate().T @ Szz_inv[:, :, ff] @ steer_v
            self.w_m[:, ff] = steer_v.conjugate().T @ Szz_inv[:, :, ff] / denom

        # plt.semilogx(np.linspace(0, 44100/2, self.nb_freq), u.mag2db(np.abs(self.w_m.T)))
        # plt.show()
        # FILTER LOW AND HIGH PASS FOR SAFETY
        filter_v = np.concatenate((np.hanning(6)[0:3, np.newaxis].T,
                                   np.ones((1, self.nb_freq-3-50), dtype=complex),
                                   np.hanning(100)[50:, np.newaxis].T),
                                  axis=1)
        self.w_m = self.w_m * np.repeat(filter_v, self.nb_mic, axis=0)
        # FLIP IMPULSE RESPONSE
        ir_m = np.fft.ifft(np.concatenate((self.w_m.T, np.flipud(self.w_m[:, 1:self.nb_freq-1].conjugate().T)),  axis=0),
                           axis=0)
        ir_m = np.concatenate((ir_m[int(ir_m.shape[0] / 2):, :], ir_m[0:int(ir_m.shape[0] / 2), :]), axis=0)
        self.w_m = np.fft.fft(ir_m, axis=0)[0:self.nb_freq, :].T

        return

    def get_beampattern(self, plot_b=True):
        # PREPARE
        mag_lim = -40
        (trash, idx_targ, trash2) = self.grid.find_closest_point(self.dir_targ_v, norm_s='spherical_1')
        Szz = np.zeros((self.nb_mic, self.nb_mic, self.nb_freq), dtype=complex)
        self.DI_v = np.zeros((self.nb_freq,))
        beampattern_m = np.zeros((self.nb_dir, self.nb_freq), dtype=complex)
        Grid_hp_S = copy.deepcopy(self.grid)
        mask_hp = self.grid.coords_m[:, 2] == 0
        Grid_hp_S.coords_m = self.grid.coords_m[mask_hp, :]
        idx_order = np.argsort(Grid_hp_S.coords_m[:, 1])
        Grid_hp_S.coords_m = Grid_hp_S.coords_m[idx_order, :]
        Grid_hp_S.coords_m = np.concatenate((Grid_hp_S.coords_m, Grid_hp_S.coords_m[0, :][np.newaxis, :]), axis=0)

        # PREPARE PLOT
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        line1, = ax.plot(Grid_hp_S.coords_m[:, 0], Grid_hp_S.coords_m[:, 1])
        plt.axis([mag_lim, -mag_lim, mag_lim, -mag_lim])

        # COMPUTE COMPENSATION FILTERS
        for ff in range(self.nb_freq):
            # BUILD BIG MATRIX
            for mm, mic in enumerate(self.mic):
                if mm == 0:
                    MAT_m = mic.data_m[:, ff][:, np.newaxis].T * self.weight_v
                    steer_v = mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]
                else:
                    MAT_m = np.concatenate((MAT_m, mic.data_m[:, ff][:, np.newaxis].T * self.weight_v), axis=0)
                    steer_v = np.concatenate((steer_v, mic.data_m[idx_targ, ff][np.newaxis, np.newaxis]), axis=0)

            # COVARIANCE MATRIX AND INVERSION
            Szz[:, :, ff] = MAT_m @ MAT_m.conjugate().T
            # COMPUTE DIRECTIVITY INDEX
            denom_DI = self.w_m[:, ff][:, np.newaxis].T @ Szz[:, :, ff] @ np.conj(self.w_m[:, ff][:, np.newaxis]) / self.nb_dir
            self.DI_v[ff] = np.real(self.w_m[:, ff][:, np.newaxis].T
                                    @ steer_v
                                    @ steer_v.conjugate().T
                                    @ np.conj(self.w_m[:, ff][:, np.newaxis])
                                    / denom_DI)

            # COMPUTE BEAMPATTERN
            for dd in range(self.nb_dir):
                beampattern_m[dd, ff] = np.dot(self.w_m[:, ff], MAT_m[:, dd] / self.weight_v[dd])
            dataplot = u.mag2db(np.abs(beampattern_m[:, ff]))
            dataplot[dataplot < mag_lim] = mag_lim

            dataplot_hp = dataplot[mask_hp]
            dataplot_hp = dataplot_hp[idx_order]
            dataplot_hp = np.concatenate((dataplot_hp[:, np.newaxis],
                                          dataplot_hp[0][np.newaxis, np.newaxis]),
                                         axis=0)[:, 0]
            Grid_hp_S.coords_m[:, 0] = dataplot_hp - mag_lim
            Grid_hp_cart_S = copy.deepcopy(Grid_hp_S)
            Grid_hp_cart_S.convert_coordinates('cartesian')

            if plot_b:
                line1.set_xdata(Grid_hp_cart_S.coords_m[:, 0])
                line1.set_ydata(Grid_hp_cart_S.coords_m[:, 1])
                plt.title('Beampattern at ' + str(int(self.mic[0].xaxis_v[ff])) + ' Hz')
                fig.canvas.draw()
                time.sleep(0.1)

        return

    def processfft(self, fft_inp):
        """

        :param fft_inp: matrix [nb_freq x nb_mic] of complex spectra from input microphone
        :return: fft_out: vector [nb_freq x 1] of complex spectrum beamformer output
        """
        # FILTERING
        sig_filt_m = np.zeros(fft_inp.shape, dtype=complex)
        for mm in range(self.nb_mic):
            sig_filt_m[:, mm] = fft_inp[:, mm] * self.w_m[mm, :]
        # SUMMATION
        fft_out = np.sum(sig_filt_m, axis=1)[:, np.newaxis]

        return fft_out
