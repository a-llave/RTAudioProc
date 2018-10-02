import numpy as np


def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with
    shape (chunk_size, channels)

    Samples are interleaved, so for a stereo stream with left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
    is ordered as [L0, R0, L1, R1, ...]
    """
    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    result = np.fromstring(in_data, dtype=np.int16)
    chunk_length = int(len(result) / channels)
    assert chunk_length == int(chunk_length)
    result = np.reshape(result, (chunk_length, channels))
    result = result.astype(np.float32)
    result = result / np.power(2.0, 16)
    return result

