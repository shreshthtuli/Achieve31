import numpy as np

deck = list(range(1,11))+list(range(1, 11))+list(range(-10,0))

def draw():
    return np.random.choice(deck)

def initMC():
    val = np.zeros((4,32,10))
    times = np.zeros((4,32,10))
    return val, times