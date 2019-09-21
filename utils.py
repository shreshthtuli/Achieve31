import numpy as np

deck = list(range(1,11))+list(range(1, 11))+list(range(-10,0))

def draw():
    return np.random.choice(deck)