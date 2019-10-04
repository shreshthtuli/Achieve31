import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cmap
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.ticker as ticker
from matplotlib import cm

deck = list(range(1,11))+list(range(1, 11))+list(range(-10,0))

def draw():
    return np.random.choice(deck)

def initMC():
    val = np.zeros((4,62,10))
    times = np.zeros((4,62,10))
    return val, times

def basicPolicy(state):
    if state.score() >= 25:
        return "stick"
    return "hit"

def plot(val, name=None):
    nx, ny = 10, 32
    x = range(nx)
    for i in range(4):
        y = range(-10*i,ny)
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        plt.xticks(x, range(1,nx+1))
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        ha.set_zlabel('Value Function')
        ha.yaxis.set_major_locator(ticker.MultipleLocator(5))
        X, Y = np.meshgrid(x, y) 
        ha.plot_wireframe(X, Y, val[i,0:ny+10*i,:], color='black')
        ha.plot_surface(X, Y, val[i,0:ny+10*i,:],cmap=cm.coolwarm)
        ha.set_title('Surface plot for special cards = ' + str(i));
        ha.view_init(elev=25, azim=-7)
        if name:
            hf.savefig(name+'-'+str(i)+'.png')
    if name == None:
        plt.show()

def plotQ(q, name=None):
    nx, ny = 10, 32
    x = range(nx)
    for i in range(4):
        y = range(-10*i,ny)
        for j in range(2):
            hf = plt.figure()
            ha = hf.add_subplot(111, projection='3d')
            plt.xticks(range(nx), range(1,nx+1))
            plt.xlabel('Dealer Card')
            plt.ylabel('Player Sum')
            ha.set_zlabel('Q Value Function')
            ha.yaxis.set_major_locator(ticker.MultipleLocator(5))
            X, Y = np.meshgrid(x, y) 
            ha.plot_wireframe(X, Y, q[j,i,0:ny+10*i,:], color='black')
            ha.plot_surface(X, Y, q[j,i,0:ny+10*i,:],cmap=cm.coolwarm)
            action = 'hit' if j == 0 else 'stick'
            ha.set_title('Surface plot for special cards = ' + str(i) + ', action = ' + action);
            if name:
                hf.savefig(name+'-'+str(i)+'-'+action+'.png')
    if name == None:
        plt.show()

def plotMap(q, name=None):
    nx, ny = 10, 32
    x = range(nx)
    for i in range(4):
        y = range(-10*i,ny)
        hf = plt.figure()
        ha = hf.add_subplot(111)
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        plt.xticks(range(nx), range(1,nx+1))
        ha.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ha.imshow(np.argmax(q[:,i,0:ny+10*i,:], axis=0), cmap=cmap.coolwarm)
        ha.set_title('Optimum action = ' + str(i));
        if name:
            hf.savefig(name+'-'+str(i)+'.png')
    if name == None:
        plt.show()

def plotPerf(df, hue, x, y):
    sns.lineplot(x=x,y=y,data=df, hue=hue) 
    plt.savefig('test'+str(np.random.random())+'.png')
    # plt.show()

def plotPerfBar(df, title, x, y):
    sns.set(style="whitegrid")
    sns.barplot(x=x,y=y,data=df).set_title(title)
    plt.show()

def simplePlot(rewards):
    plt.plot(rewards)
    plt.show()
