import numpy as np


class ToyQubicData:
    def __init__(self, points=20, x_0=-4, x_1=4, noise=9):
        self.x = np.random.uniform(x_0, x_1, points)
        self.y = np.power(self.x, 3)

    def dataset(self):
        return self.x, self.y
