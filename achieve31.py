from simulator import *
from policies import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


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

val = MonteCarlo(basicPolicy, 700000, True)
print(val)

plot(val)
