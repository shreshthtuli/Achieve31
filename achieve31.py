from predict import *
from control import *
from operator import add
import pandas as pd
from statistics import stdev, mean


def plotPredictionMC(runs):
    val = np.zeros((4,62,10))
    for i in range(runs):
        v = MonteCarlo(basicPolicy, 1000000, True)
        val += v
    plot(np.divide(val, runs))

def plotPredictionTD(runs):
    val = np.zeros((4,62,10))
    for i in range(runs):
        v = TD(basicPolicy, 1000000, 0.7, 0.01, 10)
        val += v
    plot(np.divide(val, runs))

def plotPerformance(algos, runs, episodes):
    df = pd.DataFrame() 
    for algo in algos:
        avgRewards = [0]*episodes
        for i in range(runs):
            rewards = []
            if algo[0] == 'sarsa':
                _, rewards = sarsa(algo[1], episodes, 0.7, 0.01, 0.1, algo[2])
            elif algo[0] == 'q':
                _, rewards = Q(5, episodes, 0.7, 0.01, 0.1)
            elif algo[0] == 'tdLambda':
                _, rewards = tdLambda(episodes, 0.7, 0.01, 0.1, algo[1])
            avgRewards = list(map(add, avgRewards, rewards))
        avgRewards[:] = [(x + 0.0) / runs for x in avgRewards]
        temp = pd.DataFrame(list(zip(avgRewards, range(episodes), [algo[0]]*episodes)), 
               columns =['reward', 'episode', 'algo']) 
        df = pd.concat([df, temp])
    print(df)
    plotPerf(df, 'algo')

def plotPerformanceAlpha(algo, train, test, alphas):
    df = pd.DataFrame() 
    for alpha in alphas:
        avgReward = 0; devReward = 0
        rewards = []
        if algo[0] == 'sarsa':
            _, rewards = sarsa(algo[1], train+test, 0.7, alpha, 0.1, algo[2])
        elif algo[0] == 'q':
            _, rewards = Q(5, train+test, 0.7, alpha, 0.1)
        elif algo[0] == 'tdLambda':
             _, rewards = tdLambda(train+test, 0.7, alpha, 0.1, algo[1])
        # avgReward, devReward = mean(rewards), stdev(rewards)
        temp = pd.DataFrame(list(zip(rewards[-test:], [str(alpha)]*test)), 
               columns =['reward', 'alpha']) 
        df = pd.concat([df, temp])
    print(df)
    plotPerfBar(df, 'alpha', algo[0])

def plotValueFunction(algo, episodes, alpha):
    val = np.zeros((4,32,10))
    if algo[0] == 'sarsa':
        q, _ = sarsa(algo[1], episodes, 0.7, alpha, 0.1, algo[2])
    elif algo[0] == 'q':
        q, _ = Q(5, episodes, 0.7, alpha, 0.1)
    elif algo[0] == 'tdLambda':
        q, _ = tdLambda(episodes, 0.7, alpha, 0.1, algo[1])
    val = np.amax(q, axis=0)
    print(val, np.shape(val))
    plot(val)
    

# val = MonteCarlo(basicPolicy, 1000000, True)
# val = TD(basicPolicy, 1000000, 0.7, 0.01, 10)
# plot(val)

# plotPredictionTD(1)

# q, _ = sarsa(5, 100000, 0.7, 0.01, 0.1, False)
# q, _ = Q(5, 1000000, 0.7, 0.01, 0.1)
# q, _ = tdLambda(1000000, 0.7, 0.01, 0.1, 0.5)
# plotQ(q)
# plotMap(q)


plotPerformance([['sarsa', 5, True], ['q'], ['tdLambda', 0.5]], 200, 100)
# plotPerformanceAlpha(['q', 5, True], 100000, 100, [0.1, 0.2, 0.3, 0.4, 0.5])
# plotValueFunction(['tdLambda', 0.5, True], 400000, 0.1)