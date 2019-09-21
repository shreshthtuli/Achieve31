from simulator import *
from policies import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math

sim = Simulator()

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

def MonteCarlo(policy, episodes, everyVisit):
    val, times = initMC()
    for e in range(episodes):
        times_episode = np.zeros((4,32,10))
        state = sim.reset(); dealer = sim.dealerCard
        while True:
            action = policy(state)
            state, reward, done = sim.step(action)
            if (not done or (done and action == "stick")) and (everyVisit or times_episode[state.special, state.score(), dealer-1] == 0):
                times_episode[state.special, state.score(), dealer-1] += 1    
            if done: 
                val += reward*times_episode
                times += times_episode
                break
    return np.divide(val, times+0.001)

def updateTD(val, history, dealer, gamma, alpha, k):
    for i in range(len(history)):
        g = 0
        for j in range(k):
            g += math.pow(gamma, j) * (history[i+j][2] if i+j < len(history) else 0)
        g += math.pow(gamma, k) * (val[history[i+k][0],history[i+k][1],dealer-1] if i+k < len(history) else 0)
        try:
            val[history[i][0], history[i][1], dealer-1] += alpha*(g - val[history[i][0], history[i][1], dealer-1])
        except: pass
    return val

def TD(policy, episodes, gamma, alpha, k):
    val = np.zeros((4,32,10))
    for e in range(episodes):
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        while True:
            action = policy(state)
            sp, sc = state.special, state.score()
            state, reward, done = sim.step(action)
            history.append((sp, sc, reward))
            if done: 
                break
        val = updateTD(val, history, dealer, gamma, alpha, k)
    return val

# val = MonteCarlo(basicPolicy, 700000, True)
val = TD(basicPolicy, 700000, 0.7, 0.01, 100)
plot(val)
