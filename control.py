from simulator import *
from state import *
from random import random

sim = Simulator()

def convertAction(a):
    if a == 'hit':
        return 0
    return 1

def revertAction(a):
    if a == 0:
        return 'hit'
    return 'stick'

def eGreedy(state, q, e, dealer):
    if random() < e:
        return np.random.choice(["hit", "stick"])
    if q[0, state.special, state.sum+10*state.special, dealer-1] >= q[1, state.special, state.sum+10*state.special, dealer-1]:
        return "hit"
    return "stick"

def updateQ(q, history, dealer, gamma, alpha, k):
    for i in range(len(history)):
        g = 0
        for j in range(k):
            g += math.pow(gamma, j) * (history[i+j][3] if i+j < len(history) else 0)
        g += math.pow(gamma, k) * (q[history[i+k][2],history[i+k][0],history[i+k][1]+10*history[i+k][0],dealer-1] if i+k < len(history) else 0)
        if dealer > 0:
            sp, sh, a = history[i][0], history[i][1], history[i][2]
            q[a, sp, sh+10*sp, dealer-1] += alpha*(g - q[a, sp, sh+10*sp, dealer-1])
    return q

def updateQLambda(q, history, dealer, gamma, alpha, l):
    for i in range(len(history)):
        gLambda = 0; s = 0
        # print("i = ", i, "len = ", len(history))
        for k in range(1,len(history)-i+1):
            g = 0
            for j in range(k):
                g += math.pow(gamma, j) * (history[i+j][3] if i+j < len(history) else 0)
            g += math.pow(gamma, k-1) * (q[history[i+k][2],history[i+k][0],history[i+k][1]+10*history[i+k][0],dealer-1] if i+k < len(history) else 0)
            gLambda = (1 - l) * math.pow(l, k-1) * g; # s += (1 - l) * math.pow(l, k-1); print("left",k-1)
        g = 0; k = len(history)-i
        for j in range(k):
            g += math.pow(gamma, j) * (history[i+j][3] if i+j < len(history) else 0)
        if dealer > 0:
            g += math.pow(gamma, k) * (q[history[i+k][2],history[i+k][0],history[i+k][1],dealer-1] if i+k < len(history) else 0)
        gLambda += math.pow(l, k) * g; # s += math.pow(l, k); print("done",k)
        # print(s)
        try:
            sp, sh, a = history[i][0], history[i][1], history[i][2]
            q[a, sp, sh+10*sp, dealer-1] += alpha*(gLambda - q[a, sp, sh+10*sp, dealer-1])
        except: pass
    return q

def sarsa(k, episodes, gamma, alpha, e, adaptive=False):
    q = np.zeros((2,4,62,10))
    e1 = e
    rewards = []
    for episode in tqdm(range(episodes)):
        if adaptive:
            e = e1/(episode + 1)
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        a = eGreedy(state, q, e, dealer) if dealer > 0 else 'hit' # a
        while True:
            sp, sh = state.special, state.sum # s
            state, reward, done = sim.step(a) # s', r
            aPrime = eGreedy(state, q, e, dealer) if not done else 'hit' # a'
            history.append((sp, sh, convertAction(a), reward)) # sars'a'
            a = aPrime
            if done: 
                rewards.append(reward)
                break
        q = updateQ(q, history, dealer, gamma, alpha, k)
    return q, rewards

def Q(k, episodes, gamma, alpha, e):
    q = np.zeros((2,4,62,10))
    rewards = []
    for episode in tqdm(range(episodes)):
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        while True:
            sp, sh = state.special, state.sum # s
            a = eGreedy(state, q, e, dealer) if dealer > 0 else 'hit' # a
            state, reward, done = sim.step(a) # s', r
            if (not done or (done and a == "stick")):
                q[convertAction(a), sp, sh+10*sp, dealer-1] += alpha*(reward + gamma * np.max(q[:,state.special,state.sum+10*state.special, dealer-1]) - q[convertAction(a), sp, sh+10*sp, dealer-1])
            elif dealer > 0:
                q[convertAction(a), sp, sh+10*sp, dealer-1] += alpha*(reward - q[convertAction(a), sp, sh+10*sp, dealer-1])
            if done: 
                rewards.append(reward)
                break
    return q, rewards

def tdLambda(episodes, gamma, alpha, e, l, adaptive=False):
    q = np.zeros((2,4,62,10))
    rewards = []; e1 = e
    for episode in tqdm(range(episodes)):
        if adaptive:
            e = e1/(episode + 1)
        state = sim.reset(); dealer = sim.dealerCard
        history = []
        a = eGreedy(state, q, e, dealer) if dealer > 0 else 'hit' # a
        while True:
            sp, sh = state.special, state.sum # s
            state, reward, done = sim.step(a) # s', r
            aPrime = eGreedy(state, q, e, dealer) if not done else 'hit' # a'
            history.append((sp, sh, convertAction(a), reward)) # sars'a'
            a = aPrime
            if done: 
                rewards.append(reward)
                break
        q = updateQLambda(q, history, dealer, gamma, alpha, l)
    return q, rewards