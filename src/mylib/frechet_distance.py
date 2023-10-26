import numpy as np

def euclidean(point1, point2):
    """
    point - np.array([x, y])
    """
    return np.sqrt(np.sum((point1 - point2)**2))

def euclidean_dot_segment (P, segment):
    """
    point   - np.array([x, y])
    segment - np.array([x1, y1], [x2, y2])

    alpha = < V, P - S1 > / < V, V >
    x = alpha * V
    """
    S1, S2 = segment
    v = S2 - S1
    if np.dot(v,v) == 0:
        # raise Exception("Error euclidean_dot_segment: длина segment = 0 !!!")
        return euclidean(S1, P)

    alpha = np.dot(v, P - S1) / np.dot(v,v)

    if alpha < 0:               # проекция до S1
        return euclidean(S1, P)
    elif 0 <= alpha <= 1:       # проекция попадаета отрезок [S1, S2]
        return euclidean(S1 + alpha * v, P)
    elif alpha > 1:             # проекция после S2
        return euclidean(S2, P)
    else:
        raise Exception("Error euclidean_dot_segment!!!")
    

def line_intersection(line1, line2):
    A, B = line1
    C, D = line2
    xdiff = (A[0] - B[0], C[0] - D[0])
    ydiff = (A[1] - B[1], C[1] - D[1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0: # lines do not intersect
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    I = np.array([x, y]) # точка пересечения

    I_A, B_A = I - A, B - A
    I_C, D_C = I - C, D - C
    # alpha = I_A / B_A
    # beta  = I_C / D_C
    if B_A[0] == 0:
        alpha = I_A[1] / B_A[1]
    else:
        alpha = I_A[0] / B_A[0]
    if D_C[0] == 0:
        beta = I_C[1] / D_C[1]
    else:
        beta = I_C[0] / D_C[0]

    # print(alpha, beta, I)
    # print(I_A, B_A, I_C, D_C)

    if 0 <= alpha <= 1 and 0 <= beta <= 1:
        return True
    else:
        return False
    

def euclidean_site (site1, site2):
    if len(site1.shape) == 1 and len(site2.shape) == 1:
        return euclidean(site1, site2)
    elif len(site1.shape) == 1 and len(site2.shape) == 2:
        return euclidean_dot_segment(site1, site2)
    elif len(site1.shape) == 2 and len(site2.shape) == 1:
        return euclidean_dot_segment(site2, site1)
    elif len(site1.shape) == 2 and len(site2.shape) == 2:
        if line_intersection(site1, site2):
            # print("Пересекаются")
            return 0
        else:
            A, B = site1
            C, D = site2
            return min(euclidean_dot_segment (A, site2),
                       euclidean_dot_segment (B, site2),
                       euclidean_dot_segment (C, site1),
                       euclidean_dot_segment (D, site1))
    else:
        raise Exception("Error in euclidean_site!!!")
    

class Frechet():
    def __init__(self, dist_func = euclidean_site):
        self.dist_func = dist_func
        self.n_p    = None
        self.n_q    = None
        self.D      = None # для подсчёта расстояния
        self.D0     = None # матрица попарных расстояний
        self.dist   = None

        self.stack  = None
        self.avr    = None

    def distance(self, p, q):
        self.n_p = p.shape[0]
        self.n_q = q.shape[0]
        self.D = np.zeros((self.n_p, self.n_q), dtype=np.float64)
        self.D0 = np.zeros((self.n_p, self.n_q), dtype=np.float64)

        for i in range(self.n_p):
            for j in range(self.n_q):
                d = self.dist_func(p[i], q[j])
                self.D0[i, j] = d

                if i > 0 and j > 0:
                    self.D[i, j] = max(min(self.D[i - 1, j], self.D[i - 1, j - 1], self.D[i, j - 1]), d)
                elif i > 0 and j == 0:
                    self.D[i, j] = max(self.D[i - 1, 0], d)
                elif i == 0 and j > 0:
                    self.D[i, j] = max(self.D[0, j - 1], d)
                elif i == 0 and j == 0:
                    self.D[i, j] = d

        self.dist = self.D[self.n_p - 1, self.n_q - 1]


    def backward(self):
        # обратный ход
        if self.D is None:
            raise Exception("Error in Frechet.backward(): self.D is None!!!")
        i, j = self.n_p - 1, self.n_q - 1
        self.stack = [[i, j]]
        
        while i != 0 and j != 0:
            m = np.argmin([self.D[i - 1, j], self.D[i - 1, j - 1], self.D[i, j - 1]])
            if m == 0:
                i, j = i - 1, j
            elif m == 1:
                i, j = i - 1, j - 1
            elif m == 2:
                i, j = i, j - 1
        
            self.stack.append([i, j])

        while j != 0:
            i, j = i, j - 1
            self.stack.append([i, j])

        while i != 0:
            i, j = i - 1, j
            self.stack.append([i, j])

        # self.stack = self.stack[::-1]
        # print(stack) # печать соответствующих точек

        self.avr = 0
        for i, j in self.stack:
            self.avr += self.D0[i, j]

        self.avr /= len(self.stack)

    def cycle(self, p, q):
        self.distance(p, q)
        self.backward()


class SiteFrechet(Frechet):
    def __init__(self, dist_func = euclidean_site):
        self.dist_func = dist_func
        self.n_p    = None
        self.n_q    = None
        self.D      = None # для подсчёта расстояния
        self.D0     = None # матрица попарных расстояний
        self.dist   = None

        self.stack  = None
        self.avr    = None

    def distance(self, p, q):
        self.n_p = p.shape[0] * 2 - 1
        self.n_q = q.shape[0] * 2 - 1
        self.D = np.zeros((self.n_p, self.n_q), dtype=np.float64)
        self.D0 = np.zeros((self.n_p, self.n_q), dtype=np.float64)

        for i in range(self.n_p):
            for j in range(self.n_q):
                if i % 2 == 0 and j % 2 == 0:
                    d = self.dist_func(p[i // 2], q[j // 2])
                elif i % 2 == 0 and j % 2 == 1:
                    # print(f"@1 {p[i // 2]} \n@2 {q[j // 2 : j // 2 + 2]}")
                    d = self.dist_func(p[i // 2], q[j // 2 : j // 2 + 2])
                elif i % 2 == 1 and j % 2 == 0:
                    # print(f"@1 {p[i // 2 : i // 2 + 2]} \n@2 {q[j // 2]}")
                    d = self.dist_func(p[i // 2 : i // 2 + 2], q[j // 2])
                elif i % 2 == 1 and j % 2 == 1:
                    d = self.dist_func(p[i // 2 : i // 2 + 2], q[j // 2 : j // 2 + 2])

                self.D0[i, j] = d
                    
                if i > 0 and j > 0:
                    self.D[i, j] = max(min(self.D[i - 1, j], self.D[i - 1, j - 1], self.D[i, j - 1]), d)
                elif i > 0 and j == 0:
                    self.D[i, j] = max(self.D[i - 1, 0], d)
                elif i == 0 and j > 0:
                    self.D[i, j] = max(self.D[0, j - 1], d)
                elif i == 0 and j == 0:
                    self.D[i, j] = d

        self.dist = self.D[self.n_p - 1, self.n_q - 1]
