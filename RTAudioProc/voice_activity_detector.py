import numpy as np


def voice_activity_detector(signal, threshold=0.0):
    nb_xzero = 0
    nb_sample = len(signal)
    vad_b = False
    for nn in range(1, nb_sample):
        if not np.sign(signal[nn]) == np.sign(signal[nn-1]):
            nb_xzero += 1
    xzero_rate = nb_xzero / nb_sample
    if xzero_rate < threshold:
        vad_b = True
    return vad_b
