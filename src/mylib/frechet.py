# This code belongs to Pronina Natalia and used to compare her and my method


import numpy as np
from copy import copy
import os

from frechet_distance import *

def txt2streaks(name):
    f = open(f'{name}').read().strip()
    f = f.replace(",", ".")
    f = f.split('\n')

    list_Streaks = []
    num_stroke = 0

    for i, line in enumerate(f):
        words = line.strip().split()
        if words[0] == "?????":
            if words[1] != "0":
                len_stroke = len(np.array(l))
                if len_stroke >= 0:
                    list_Streaks += [Streak(num_stroke, style, np.array(l))]
                    num_stroke += 1
            style = words[2]
            l = []
        else:
            l += [[float(words[0]), float(words[1])]]

    len_stroke = len(np.array(l))
    if len_stroke >= 0:
        list_Streaks += [Streak(num_stroke, style, np.array(l))]

    return list_Streaks


def length(coords):
    l = 0
    for i in range(len(coords) - 1):
        l += euclidean(coords[i], coords[i + 1])

    return l


def shift(l, x, y):
    """
    двигает кривую так, чтобы точка x перешла в точку y
    """
    l_new = copy(l)
    A = x - y
    l_new -= A

    return l_new

def angle_segments(segment1, segment2):
    """
    выдаёт угол между сегментами, начальные точки отрезков перемещает в (0, 0)
    segment - np.array([[x1, y1], [x2, y2]])
    """
    segment1 = shift(segment1, segment1[0], np.zeros(2))
    segment2 = shift(segment2, segment2[0], np.zeros(2))

    A1, B1 = segment1
    A2, B2 = segment2

    cos_alpha = (B1 * B2).sum()
    cos_alpha /= euclidean(A1, B1) * euclidean(A2, B2)

    return cos_alpha


def make_segment(line, i, j):
    """
    line[i] - начало, line[j] - конец
    """
    A, B = line[i], line[j]
    segment = np.vstack((A, B))
    return segment


def two_remote_point(l):
    """Диаметр штриха"""
    max_dist = 0
    i_max = None
    j_max = None

    for i in range(len(l)):
        for j in range(i+1, len(l)):
            dist = euclidean(l[i], l[j])
            if dist > max_dist:
                max_dist, i_max, j_max = dist, i, j

    return i_max, j_max

def two_nearest_points(coords_1, coords_2):
    """Расстояние между двумя множествами точек"""
    min_dist = np.inf
    idx_1 = None
    idx_2 = None

    for i, point_1 in enumerate(coords_1):
        for j, point_2 in enumerate(coords_2):
            dist = euclidean(point_1, point_2)
            if dist < min_dist:
                min_dist, idx_1, idx_2 = dist, i, j

    return min_dist, idx_1, idx_2


def oriented_area(l):
    """
    Ориентированная площадь
    D = x1*y2 - x2*y1
    """
    Result = 0
    last = l[-1]
    for i, point in enumerate(l):
        Result += last[0] * point[1] - last[1] * point[0]
        last = point
    return Result / 2


def how_move_segment(segment1, segment2, line2):
    """
    direction = 1 или -1 (какой правильный обход второго отрезка)
    line2_new - изменённые координаты line2 (изменили мб обход и наложили середины)
    """
    # shift = np.array([[x1, y1], [x2, y2]]) (преобразование 2 отрезка, точку [x1, y1] переместить в [x2, y2])

    direction = 1
    if angle_segments(segment1, segment2) < 0:
        direction = -1
        # line2_new = line2_new[::-1]

    middle1 = segment1.mean(axis = 0) # координаты середины 1 отрезка
    middle2 = segment2.mean(axis = 0) # координаты середины 2 отрезка

    line2_new    = shift(line2, middle2, middle1)
    return direction, line2_new


def norm_chain(coords_1, coords_2):
    middle1 = coords_1.mean(axis = 0) # координаты центра тяжести 1 отрезка
    middle2 = coords_2.mean(axis = 0)
    coords_2    = shift(coords_2, middle2, middle1)

    if euclidean(coords_1[0], coords_2[-1]) < euclidean(coords_1[0], coords_2[0]):
        coords_2 = coords_2[::-1]

    return coords_1, coords_2


def norm_ring(coords_1, coords_2):
    segment1 = make_segment(coords_1, *two_remote_point(coords_1))
    segment2 = make_segment(coords_2, *two_remote_point(coords_2))

    direction, coords_2 = how_move_segment(segment1, segment2, coords_2)

    _, idx_1, idx_2 = two_nearest_points(coords_1, coords_2)
    coords_1 = np.vstack((coords_1[idx_1:], coords_1[:idx_1]))
    coords_2 = np.vstack((coords_2[idx_2:], coords_2[:idx_2]))

    return coords_1, coords_2


def normalization(streak_1, streak_2):
    coords_1 = streak_1.new_coords
    coords_2 = streak_2.new_coords

    if streak_1.style != streak_2.style:
        # raise TypeError("Error in normalization!!!")
        middle1 = coords_1.mean(axis = 0)
        middle2 = coords_2.mean(axis = 0)
        coords_2 = shift(coords_2, middle2, middle1)

    elif streak_1.style == "Chain":
        coords_1, coords_2 = norm_chain(coords_1, coords_2)

    elif streak_1.style == "Ring":
        coords_1, coords_2 = norm_ring(coords_1, coords_2)

    return coords_1, coords_2


def dist_two_streaks(streak_1, streak_2, type_f = "avr"):
    coords_1 = streak_1.new_coords
    coords_2 = streak_2.new_coords

    f = Frechet()
    f.cycle(coords_1, coords_2)
    dist1 = f.dist if type_f == "dist" else f.avr

    if streak_1.style != streak_2.style:
        return 999999
    else:
        coords_1, coords_2 = normalization(streak_1, streak_2)

        f = Frechet()
        f.cycle(coords_1, coords_2)
        dist2 = f.dist if type_f == "dist" else f.avr
        return min(dist1, dist2)


def Matrix_pairwise_distances(list_Streaks1, list_Streaks2):
    """Выводит матрицу размера len(list_Streaks1) x len(list_Streaks2)"""
    # Matrix = np.fromfunction(dist_two_streaks, (len(list_Streaks1), len(list_Streaks2)))

    Matrix = np.zeros((len(list_Streaks1), len(list_Streaks2)))
    for i, streak_i in enumerate(list_Streaks1):
        for j, streak_j in enumerate(list_Streaks2):
            Matrix[i, j] = dist_two_streaks(streak_i, streak_j)

    return Matrix


list_color = ['red', 'orange', 'limegreen', 'blue', 'fuchsia']


def delete(l, eps = 1):
    new_l = l[0].copy()

    i = 1
    while i != len(l):
        length = euclidean(l[i], new_l[-1])
        # print(length)
        if length > eps:
            new_l = np.vstack((new_l, l[i]))

        i += 1

    return new_l


def discretization (l, eps = 5):
    new_l = []
    for i in range(len(l) - 1):
        length = euclidean(l[i], l[i+1])
        vect = l[i+1] - l[i]
        new_l.append(l[i])
        if length > eps:
            num = int(length // eps) + 1
            dot = l[i]
            for n in range(num - 1):
                dot = dot + vect / num
                new_l.append(dot)
    new_l.append(l[i+1])
    return np.array(new_l)


class Streak(): # контур листа
    def __init__(self, idx, style, coords):
        self.idx = idx
        self.style = style
        self.cluster = None

        coords = delete(coords)
        coords = discretization(coords)

        self.coords = coords

        self.len = self.coords.shape[0]
        self.len_coords = length(self.coords)

        idx = np.argmin(coords[:, 1]) # нижняя точка
        self.new_coords = shift(coords, coords[idx], np.zeros(2))

        if self.style == "Ring" and oriented_area(self.new_coords) < 0:
            self.new_coords = self.new_coords[::-1]


