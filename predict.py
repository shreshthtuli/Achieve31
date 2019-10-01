from simulator import *
import math

sim = Simulator()

def MonteCarlo(policy, episodes, everyVisit):
    val, times = initMC()
    for e in tqdm(range(episodes)):
        times_episode = np.zeros((4,62,10))
        state = sim.reset(); dealer = sim.dealerCard
        while True:
            sp, sh = state.special, state.sum
            action = policy(state)
            state, reward, done = sim.step(action)
            if dealer > 0 and sh < 32 and (everyVisit or times_episode[sp, sh+10*sp, dealer-1] == 0):
                times_episode[sp, sh+10*sp, dealer-1] += 1    
            if done:   
                val += reward*times_episode
                times += times_episode
                break
    return np.divide(val, times+0.00001)

def updateTD(val, history, dealer, gamma, alpha, k):
    for i in range(len(history)):
        g = 0
        for j in range(k):
            g += math.pow(gamma, j) * (history[i+j][2] if i+j < len(history) else 0)
        g += math.pow(gamma, k) * (val[history[i+k][0],history[i+k][1],dealer-1] if i+k < len(history) else 0)
        if dealer > 0:
            sp, sh = history[i][0], history[i][1]
            val[sp, sh+10*sp, dealer-1] += alpha*(g - val[sp, sh+10*sp, dealer-1])
    return val

def TD(policy, episodes, gamma, alpha, k):
    val = np.zeros((4,62,10))
    # val += 1
    for e in tqdm(range(episodes)):
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        while True:
            action = policy(state)
            sp, sh = state.special, state.sum
            state, reward, done = sim.step(action)
            history.append((sp, sh, reward))
            if done: 
                break
        val = updateTD(val, history, dealer, gamma, alpha, k)
    return val