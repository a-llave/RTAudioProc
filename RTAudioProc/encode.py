import numpy as np


def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio

    Signal should be a numpy array with shape (chunk_size, channels)
    """
    signal = signal * np.power(2.0, 16)
    signal = signal.astype(np.int16)
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype(np.int16).tostring()
    return out_data

