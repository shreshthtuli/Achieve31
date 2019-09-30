import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cmap
from tqdm import tqdm
import seaborn as sns
import pandas as pd
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
        plt.xticks(range(nx), range(1,nx+1))
        X, Y = np.meshgrid(x, y) 
        ha.plot_wireframe(X, Y, val[i], color='black')
        # ha.plot_surface(X, Y, val[i], color='w', shade=False, alpha=0.5)
        ha.set_title('Wireframe for special cards = ' + str(i));
    plt.show()

def plotQ(q):
    nx, ny = 10, 32
    x = range(nx)
    y = range(ny)
    for i in range(4):
        for j in range(2):
            hf = plt.figure()
            ha = hf.add_subplot(111, projection='3d')
            plt.xticks(range(nx), range(1,nx+1))
            X, Y = np.meshgrid(x, y) 
            ha.plot_wireframe(X, Y, q[j,i])
            action = 'hit' if j == 0 else 'stick'
            ha.set_title('Wireframe for special cards = ' + str(i) + ', action = ' + action);
    plt.show()

def plotMap(q):
    nx, ny = 10, 32
    x = range(nx)
    y = range(ny)
    for i in range(4):
        hf = plt.figure()
        ha = hf.add_subplot(111)
        plt.xticks(range(nx), range(1,nx+1))
        ha.imshow(np.argmax(q[:,0,:,:], axis=0), cmap=cmap.hot)
        ha.set_title('Optimum action = ' + str(i));
    plt.show()

def plotPerf(df, hue):
    sns.lmplot(x='episode',y='reward',data=df, fit_reg=True, lpoly=True, hue=hue) 
    plt.show()

def plotPerfBar(df, hue, title):
    sns.set(style="whitegrid")
    sns.barplot(x='alpha',y='reward',data=df, hue=hue, ci=5).set_title(title)
    plt.show()
