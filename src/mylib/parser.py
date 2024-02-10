# This code belongs to Pazazia Alexander and used to compare his and my method

import numpy as np
import scipy as sp
import re
import io
from collections import Counter
from scipy.interpolate import make_interp_spline


def min_max(array):
    a_min = min(array) + 1e-6
    return (array - a_min) / (max(array) - a_min)

def parse_data(filename, size):
    strokes = []
    X = np.arange(0)
    with io.open(filename, 'r', encoding='latin1') as f:
        for line in f:
            line = line.replace(',', '.')
            if 'Ring' in line or 'Chain' in line or 'StickVert' in line:
                tp = 0 if 'Ring' in line else 1
                if strokes:
                    coords = np.array(strokes[-1][1:])

                    if len(coords) < 4:
                        k = 1
                    else:
                        k = 3

                    t = np.linspace(0, 10, coords.shape[0])
                    cdx = (t, min_max(coords[:, 0]).ravel())
                    cdy = (t, min_max(coords[:, 1]).ravel())
                    splx = make_interp_spline(*cdx, k=k)
                    sply = make_interp_spline(*cdy, k=k)
                    leng = coords.shape[0]
                    obj = np.zeros(2*size)
                    obj[:size] = splx(np.linspace(0, 10, size))
                    obj[size:2*size] = sply(np.linspace(0, 10, size))
                    X = obj[None, :] if X.shape[0] == 0 else np.vstack([X, obj])

                strokes.append([tp])
            else:
                coords = list(map(float, line.strip().split()[:2]))
                strokes[-1].append(coords)

        coords = np.array(strokes[-1][1:])

        if len(coords) < 4:
            k = 1
        else:
            k = 3

        t = np.linspace(0, 10, coords.shape[0])
        cdx = (t, min_max(coords[:, 0]).ravel())
        cdy = (t, min_max(coords[:, 1]).ravel())
        splx = make_interp_spline(*cdx, k=k)
        sply = make_interp_spline(*cdy, k=k)
        leng = coords.shape[0]
        obj = np.zeros(2*size)
        obj[:size] = splx(np.linspace(0, 10, size))
        obj[size:2*size] = sply(np.linspace(0, 10, size))
        X = obj[None, :] if X.shape[0] == 0 else np.vstack([X, obj])

        return X, strokes
