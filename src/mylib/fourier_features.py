import numpy as np
from scipy.fft import fft
import json


def is_clockwise(stroke):
    sum_ = 0
    for i in range(len(stroke) - 1):
        sum_ += np.prod(stroke[i + 1] - stroke[i])
    return sum_ >= 0

def strokes_transform(strokes, is_complex=False):
    result = []
    for stroke in strokes:
        if is_complex:
            stroke = np.array([stroke.real, stroke.imag]).T
        stroke -= np.mean(stroke, axis=0)
        if  np.allclose(stroke[0], stroke[-1]):
            stroke = stroke[:len(stroke) - 1]
            start = np.argmin(stroke[:, 0])
            stroke = list(stroke)
            stroke = stroke[start:] + stroke[:start + 1]
            stroke = np.array(stroke)
            if not is_clockwise(stroke):
                stroke = stroke[::-1]
        else:
            if stroke[0][0] > stroke[-1][0]:
                stroke = stroke[::-1]
        if is_complex:
            stroke = stroke[:, 0] + 1j * stroke[:, 1]
        result.append(stroke)
    return result

def strokes_to_numpy(strokes, return_radius=False, return_complex=False, min_length=0):
    result = []
    for stroke in strokes:
        tmp = []
        for point in stroke['Points']:
            if return_radius:
                tmp.append([float(point['x']), float(point['y']), float(point['r'])])
            else:
                if return_complex:
                    tmp.append(float(point['x']) + 1j * float(point['y']))
                else:
                    tmp.append([float(point['x']), float(point['y'])])
        if len(tmp) > min_length:
            result.append(np.array(tmp))
    return result


def complex_to_real(array):
    real = array.real
    imag = array.imag
    return np.hstack((real, imag))


def fft_features(filename, n_coefs):
    with open(filename) as f:
        strokes = strokes_to_numpy(json.load(f), return_complex=True)
    strokes = strokes_transform(strokes, is_complex=True)
    ffted = []
    for stroke in strokes:
        ffted.append(fft(stroke, n=23))
    ffted = np.array(ffted)
    abs_fft = np.abs(ffted)
    abs_fft = abs_fft.mean(axis=0)
    fourier_indexes = np.argsort(abs_fft)[::-1]
    coefs = ffted[:, fourier_indexes[:n_coefs]]
    return coefs
