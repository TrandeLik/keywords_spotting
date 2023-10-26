import numpy as np


def sample_from_square(x_0, y_0, a):
    x = np.random.uniform(low=x_0, high=x_0 + a, size=1)[0]
    y = np.random.uniform(low=y_0, high=y_0 + a, size=1)[0]
    return x, y


def upsample_segment(x_0, y_0, x_1, y_1, size=1):
    coef = np.random.uniform(low=0, high=1, size=size)
    coef = np.array([0] + list(coef) + [1])
    coef = np.sort(coef)
    x = x_0 + coef * (x_1 - x_0)
    x += np.random.normal(loc=0, scale=0.1, size=size+2)
    y = y_0 + coef * (y_1 - y_0)
    y += np.random.normal(loc=0, scale=0.1, size=size+2)
    return np.hstack((x[:, None], y[:, None]))


def generate_stroke(squares):
    main_points = []
    for square in squares:
        main_points.append(sample_from_square(square[0], square[1], square[2]))
    result = []
    for i in range(len(main_points) - 1):
        length = np.sqrt((main_points[i][0] - main_points[i + 1][0]) ** 2
                                     + (main_points[i][1] - main_points[i + 1][1]) ** 2)
        size = np.random.randint(low=0, high=int(length) + 1, size=1)[0]
        tmp = upsample_segment(main_points[i][0], main_points[i][1],
                               main_points[i + 1][0], main_points[i + 1][1], size)
        result += list(map(list, tmp))[:len(tmp) - 1]
    result += [list(main_points[-1])]
    return result


def write_data(strokes, directory='data', filename="syntetic", width=3, seed=42):
    np.random.seed(seed)
    classes = strokes[1]
    strokes = strokes[0]
    np.savez(f"{directory}/classes-{filename}", np.array(classes))
    with open(f"{directory}/{filename}.txt", "w") as f:
        for i, stroke in enumerate(strokes):
            f.write(f"   Штрих {i} {stroke['type']} Длина {len(stroke['points'])}\n")
            for point in stroke['points']:
                current_width = width + np.random.normal(loc=0, scale=2, size=1)[0]
                f.write(f"{point[0]} {point[1]} {current_width}\n")


def generate_data(objects_per_class, seed=42, size=5):
    np.random.seed(seed)
    result = []
    classes = []
    for _ in range(objects_per_class):
        classes += [0, 1, 2, 3, 4, 5]
        result.append({
            'type': 'Chain',
            'points': generate_stroke([
                [0, 2 * size, size],
                [size, 3 * size, size],
                [2 * size, 3 * size, size],
                [3 * size, 3 * size, size],
            ])
        })
        result.append({
            'type': 'Chain',
            'points': generate_stroke([
                [0, 3 * size, size],
                [size, 3 * size, size],
                [2 * size, 3 * size, size],
                [3 * size, 3 * size, size],
            ])
        })
        result.append({
            'type': 'Chain',
            'points': generate_stroke([
                [0, 3 * size, size],
                [size, 2 * size, size],
                [size, size, size],
                [0, 0, size],
            ])
        })
        result.append({
            'type': 'Chain',
            'points': generate_stroke([
                [0, 3 * size, size],
                [0, 2 * size, size],
                [0, size, size],
                [0, 0, size],
            ])
        })
        result.append({
            'type': 'Chain',
            'points': generate_stroke([
                [0, 2 * size, size],
                [size, 3 * size, size],
                [size, 2 * size, size],
                [size, size, size],
                [size, 0, size],
                [2 * size, size, size]
            ])
        })
        result.append({
            'type': 'Ring',
            'points': generate_stroke([
                [0, 0, size],
                [0, size, size],
                [size, 2 * size, size],
                [2 * size, 3 * size, size],
                [3 * size, 3 * size, size],
                [3 * size, 2 * size, size],
                [2 * size, 1 * size, size],
                [size, 0, size],
            ])
        })
    return result, classes