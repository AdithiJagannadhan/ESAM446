import numpy as np

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class PeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Field:

    def __init__(self, grid, data):
        self.grid = grid
        self.data = data

    def data_padded(self, pad, bc='periodic'):
        start_pad, end_pad = pad
        new_data = np.zeros(start_pad + self.grid.N + end_pad)
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        new_data[s] = self.data
        if bc == 'periodic':
            if start_pad > 0:
                new_data[:start_pad] = self.data[-start_pad:]
            if end_pad > 0:
                new_data[-end_pad:] = self.data[:end_pad]
        else:
            raise ValueError("Only supports periodic BC")
        return new_data

