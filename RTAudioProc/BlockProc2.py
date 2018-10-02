"""
Description:    Frame management for real-time audio processing
                Ref: Kates 2005, Principles of Digital Dynamic-Range Compression

Author: Adrien Llave - CentraleSupélec
Date: 31/08/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/08/31  ALl     1.0       Initialization

"""

import numpy as np


class BlockProc2:

    def __init__(self, nb_buffsamp, nb_fft, nb_channels_inp, nb_channels_out, window_s='hann'):
        # PARAMETERS
        self.nb_buffsamp = nb_buffsamp
        self.nb_framesamp = self.nb_buffsamp * 2
        self.nb_fft = nb_fft
        self.overlap_size = self.nb_fft - self.nb_buffsamp
        self.nb_channels_inp = nb_channels_inp
        self.nb_channels_out = nb_channels_out
        # LINES
        self.frame_inp = np.zeros((self.nb_framesamp, self.nb_channels_inp))
        self.overlap_m = np.zeros((self.overlap_size, self.nb_channels_out))
        # WINDOW
        self.window_s = window_s
        self.window_m = np.zeros((self.nb_framesamp, self.nb_channels_inp))
        self.update_window(self.window_s)

    def input2frame(self, buffer_inp):
        self.frame_inp = np.concatenate((self.frame_inp[int(self.nb_framesamp/2):, :], buffer_inp), axis=0)
        frame_win = self.frame_inp * self.window_m
        return frame_win

    def frame2output(self, frame_out):
        # PREPARE BUFFER OUT
        frame2buffer = frame_out[0:self.nb_buffsamp, :]
        overlap2buffer = self.overlap_m[0:self.nb_buffsamp, :]
        buffer_out = frame2buffer + overlap2buffer

        # UPDATE OVERLAP SAVE ARRAY
        self.overlap_m = np.concatenate((self.overlap_m[self.nb_buffsamp:, :], np.zeros((self.nb_buffsamp, self.nb_channels_out))), axis=0)
        self.overlap_m += frame_out[self.nb_buffsamp:, :]
        return buffer_out

    def update_window(self, window_s='hann'):
        """

        :param window_s: 'blackman' cf. Kates 2005 // 'hann' ensure perfect overlap
        :return:
        """
        if window_s is 'blackman':
            self.window_m = np.repeat(np.blackman(self.nb_framesamp)[:, np.newaxis], self.nb_channels_inp, axis=1)
        elif window_s is 'hann':
            self.window_m = np.repeat(np.hanning(self.nb_framesamp)[:, np.newaxis], self.nb_channels_inp, axis=1)

        return
