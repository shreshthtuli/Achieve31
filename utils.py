import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d

deck = list(range(1,11))+list(range(1, 11))+list(range(-10,0))

def draw():
    return np.random.choice(deck)

def initMC():
    val = np.zeros((4,32,10))
    times = np.zeros((4,32,10))
    return val, times

def basicPolicy(state):
    if state.score() >= 25:
        return "stick"
    return "hit"

def plot(val):
    nx, ny = 10, 32
    x = range(nx)
    y = range(ny)
    for i in range(4):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y) 
        ha.plot_wireframe(X, Y, val[i])
        ha.set_title('Wireframe for special cards = ' + str(i));
    plt.show()