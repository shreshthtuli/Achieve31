from simulator import *
from state import *
from random import random

sim = Simulator()

def convertAction(a):
    if a == 'hit':
        return 0
    return 1

def eGreedy(state, q, e, dealer):
    if random() < e:
        return np.random.choice(["hit", "stick"])
    if q[0, state.special, state.score(), dealer-1] >= q[1, state.special, state.score(), dealer-1]:
        return "hit"
    return "stick"

def updateQ(q, history, dealer, gamma, alpha, k):
    for i in range(len(history)):
        g = 0
        for j in range(k):
            g += math.pow(gamma, j) * (history[i+j][3] if i+j < len(history) else 0)
        g += math.pow(gamma, k) * (q[history[i+k][2],history[i+k][0],history[i+k][1],dealer-1] if i+k < len(history) else 0)
        try:
            q[history[i][2], history[i][0], history[i][1], dealer-1] += alpha*(g - q[history[i][2], history[i][0], history[i][1], dealer-1])
        except: pass
    return q

def sarsa(k, episodes, gamma, alpha, e, adaptive=False):
    q = np.zeros((2,4,32,10))
    e1 = e
    for episode in range(episodes):
        if adaptive:
            e = e1/(episode + 1)
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        a = eGreedy(state, q, e, dealer) if dealer > 0 else 'hit' # a
        while True:
            sp, sc = state.special, state.score() # s
            state, reward, done = sim.step(a) # s', r
            aPrime = eGreedy(state, q, e, dealer) if not done else 'hit' # a'
            history.append((sp, sc, convertAction(a), reward, state.special, state.score(), convertAction(aPrime))) # sars'a'
            a = aPrime
            if done: 
                break
        q = updateQ(q, history, dealer, gamma, alpha, k)
    return q

def Q(k, episodes, gamma, alpha, e):
    q = np.zeros((2,4,32,10))
    for episode in range(episodes):
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        while True:
            sp, sc = state.special, state.score() # s
            a = eGreedy(state, q, e, dealer) if dealer > 0 else 'hit' # a
            state, reward, done = sim.step(a) # s', r
            if (not done or (done and a == "stick")):
                q[convertAction(a), sp, sc, dealer-1] += alpha*(reward + gamma * np.max(q[:,state.special,state.score(),dealer-1]) - q[convertAction(a), sp, sc, dealer-1])
            elif dealer > 0:
                q[convertAction(a), sp, sc, dealer-1] += alpha*(reward - q[convertAction(a), sp, sc, dealer-1])
            if done: 
                break
    return q