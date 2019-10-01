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

def plotRewards(algos, runs, episodes):
    df = pd.DataFrame() 
    for algo in algos:
        avgRewards = [0]*episodes
        for i in range(runs):
            rewards = []
            if algo[0] == 'sarsa':
                _, rewards = sarsa(algo[1], episodes, 0.7, 0.05, 0.1, algo[2])
            elif algo[0] == 'q':
                _, rewards = Q(5, episodes, 0.7, 0.05, 0.1)
            elif algo[0] == 'tdLambda':
                _, rewards = tdLambda(episodes, 0.7, 0.01, 0.1, algo[1])
            avgRewards = list(map(add, avgRewards, rewards))
        avgRewards[:] = [(x + 0.0) / runs for x in avgRewards]
        simplePlot(avgRewards[::100])
        temp = pd.DataFrame(list(zip(avgRewards, range(episodes), [algo[0]]*episodes)), 
               columns =['reward', 'episode', 'algo']) 
        df = pd.concat([df, temp])
    print(df)
    plotPerf(df, 'algo', 'episode', 'reward')

def run(q, episodes, mycards, dealercards):
    totalReward = 0.0
    for i in tqdm(range(episodes)):
        state = sim.set(mycards[i], dealercards[i]); dealer = sim.dealerCard
        print('dealer =', dealer)
        while True:
            sp, sh = state.special, state.sum
            print(sp, sh)
            if dealer > 0:
                action = np.argmax(q[:,sp,sh,dealer-1])
            else:
                action = 'stick'
            state, reward, done = sim.step(revertAction(action))
            print(revertAction(action), state.special, state.sum)
            if done:   
                totalReward += reward
                print('reward = ', reward)
                break
    return totalReward / episodes

def plotPerformanceTest(algos, train, test):
    df = pd.DataFrame() 
    mycards = [draw() for i in range(test)]
    dealercards = [draw() for i in range(test)]
    for algo in algos:
        avgReward = 0; devReward = 0
        q = []
        if algo[0] == 'sarsa':
            q, _ = sarsa(algo[1], train, 0.7, 0.01, 0.1, algo[2])
        elif algo[0] == 'q':
            q, _ = Q(5, train, 0.7, 0.01, 0.1)
        elif algo[0] == 'tdLambda':
            q, _ = tdLambda(train, 0.7, 0.01, 0.1, algo[1])
        temp = pd.DataFrame(list(zip([run(q, test, mycards, dealercards)], [algo[0]])), 
               columns =['reward', 'algo']) 
        df = pd.concat([df, temp])
    print(df)
    plotPerfBar(df, 'Comparison of algos', 'algo', 'reward')

def plotPerformanceAlpha(algos, train, test, alphas):
    df = pd.DataFrame() 
    mycards = [draw() for i in range(test)]
    dealercards = [draw() for i in range(test)]
    for algo in algos:
        for alpha in alphas:
            avgReward = 0; devReward = 0
            rewards = []
            if algo[0] == 'sarsa':
                q, _ = sarsa(algo[1], train, 0.7, alpha, 0.1, algo[2])
            elif algo[0] == 'q':
                q, _ = Q(5, train, 0.7, alpha, 0.1)
            elif algo[0] == 'tdLambda':
                q, _ = tdLambda(train, 0.7, alpha, 0.1, algo[1])
            temp = pd.DataFrame(list(zip([run(q, test, mycards, dealercards)], [alpha], [algo[0]])), 
                columns =['reward', 'alpha', 'algo']) 
            df = pd.concat([df, temp])
    print(df)
    plotPerf(df, 'algo', 'alpha', 'reward')

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


# plotRewards([['sarsa', 5, True], ['q'], ['tdLambda', 0.5]], 200, 200)
# plotRewards([['sarsa', 1, False], ['q'], ['tdLambda', 0.5]], 10, 20000)
# plotPerformanceTest([['sarsa', 5, True], ['q'], ['tdLambda', 0.5]], 1000000, 100)
plotPerformanceAlpha([['sarsa', 1, False]], 100000, 10, [0.1, 0.2, 0.3, 0.4, 0.5])
# plotValueFunction(['tdLambda', 0.5, True], 400000, 0.1)