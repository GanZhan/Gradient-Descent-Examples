import csv
import numpy as np

def get_points(filepath):
    x = np.array([])
    y = np.array([])
    with open(filepath) as f:
        reader = csv.reader(f)
        _length = 0
        for row in reader:
            _length = _length + 1
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y, _length
filepath = 'data.csv'
get_points(filepath)
