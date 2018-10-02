import numpy as np


class BlockProc:
    def __init__(self, nb_buffsamp, nb_channels, window_s='hann'):
        # PARAMETERS
        self.buffer_count = 0
        self.nb_buffsamp = nb_buffsamp
        self.nb_framesamp = self.nb_buffsamp * 2
        self.nb_channels = nb_channels
        # LINES
        self.frame1_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.frame2_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.buf_out = np.zeros((self.nb_buffsamp, self.nb_channels))
        # WINDOW
        self.window_s = window_s
        self.window_m = np.zeros((self.nb_framesamp, self.nb_channels))
        self.update_window(self.window_s)

        self.toto = 0

    def input(self, buf_inp):
        self.buffer_count = np.mod(self.buffer_count+1, 4)

        if np.mod(self.buffer_count, 2):
            self.frame1_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] = buf_inp
            self.frame2_m[0:self.nb_buffsamp, :] = buf_inp
        else:
            self.frame1_m[0:self.nb_buffsamp, :] = buf_inp
            self.frame2_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] = buf_inp

        return

    def get_frame(self):
        if np.mod(self.buffer_count, 2):
            frame_m = self.frame1_m
        else:
            frame_m = self.frame2_m

        return frame_m

    def set_frame(self, frame_m):
        if np.mod(self.buffer_count, 2):
            self.frame1_m = frame_m * self.window_m
        else:
            self.frame2_m = frame_m * self.window_m

        return

    def output(self):
        if not np.mod(self.buffer_count, 2):
            buf_out = self.frame1_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] + \
                      self.frame2_m[0:self.nb_buffsamp, :]
        else:
            buf_out = self.frame1_m[0:self.nb_buffsamp, :] + \
                      self.frame2_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :]

        return buf_out

    def update_window(self, window_s='hann'):
        """

        :param window_s: 'blackman' cf. Kates 2005 // 'hann' ensure perfect overlap
        :return:
        """
        if window_s is 'blackman':
            self.window_m = np.repeat(np.blackman(self.nb_framesamp)[:, np.newaxis], self.nb_channels, axis=1)
        elif window_s is 'hann':
            self.window_m = np.repeat(np.hanning(self.nb_framesamp)[:, np.newaxis], self.nb_channels, axis=1)

        return
