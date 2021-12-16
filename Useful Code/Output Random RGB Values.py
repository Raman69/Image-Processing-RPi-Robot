import numpy as np

color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
print(color)