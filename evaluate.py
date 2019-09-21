from simulator import *
import math

sim = Simulator()

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
            elif dealer > 0 and state.score() < 32:
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
                if dealer > 0 and state.score() < 32:
                    history.append((state.special, state.score(), 0))
                break
        val = updateTD(val, history, dealer, gamma, alpha, k)
    return val