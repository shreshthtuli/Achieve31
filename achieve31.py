from predict import *
from control import *
from operator import add
import pandas as pd
from statistics import stdev, mean

def plotPredictionMC(runs, episodes, everyVisit, save):
    val = np.zeros((4,62,10))
    for i in range(runs):
        v = MonteCarlo(basicPolicy, episodes, everyVisit)
        val += v
    plot(np.divide(val, runs), 'mc'+('e-' if everyVisit else 'f-')+str(runs))

def plotPredictionTD(runs, episodes, k, save):
    val = np.zeros((4,62,10))
    for i in range(runs):
        v = TD(basicPolicy, episodes, 0.7, 0.1, k)
        val += v
    plot(np.divide(val, runs), 'td-'+str(k)+'-'+str(episodes) if save else None)

def plotRewards(algos, runs, episodes):
    df = pd.DataFrame() 
    for algo in algos:
        avgRewards = [0]*episodes
        for i in range(runs):
            rewards = []
            if 'sarsa' in algo[0]:
                _, rewards = sarsa(algo[1], episodes, 0.7, 0.1, 0.1, algo[2])
            elif algo[0] == 'q':
                _, rewards = Q(5, episodes, 0.7, 0.1, 0.1)
            elif algo[0] == 'tdLambda':
                _, rewards = tdLambda(episodes, 0.7, 0.1, 0.1, algo[1])
            avgRewards = list(map(add, avgRewards, rewards))
        avgRewards[:] = [(x + 0.0) / runs for x in avgRewards]
        temp = pd.DataFrame(list(zip(avgRewards, range(episodes), [algo[0]]*episodes)), 
               columns =['reward', 'episode', 'algo']) 
        temp['avgRewards'] = temp['reward'].rolling(90, min_periods=5).mean()
        df = pd.concat([df, temp])
    print(df)
    plotPerf(df, 'algo', 'episode', 'avgRewards')

def run(q, episodes, mycards, dealercards):
    totalReward = 0.0
    for i in tqdm(range(episodes)):
        state = sim.set(mycards[i], dealercards[i]); dealer = sim.dealerCard
        # print('dealer =', dealer)
        while True:
            sp, sh = state.special, state.sum
            # print(sp, sh)
            if dealer > 0:
                action = np.argmax(q[:,sp,sh+10*sp,dealer-1])
            else:
                action = 'stick'
            state, reward, done = sim.step(revertAction(action))
            # print(revertAction(action), state.special, state.sum)
            if done:   
                totalReward += reward
                # print('reward = ', reward)
                break
    return totalReward / episodes

def plotPerformanceTest(algos, train, test):
    df = pd.DataFrame() 
    mycards = [draw() for i in range(test)]
    dealercards = [draw() for i in range(test)]
    for algo in algos:
        avgReward = 0; devReward = 0
        q = []
        if 'sarsa' in algo[0]:
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
            if 'sarsa' in algo[0]:
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
    if 'sarsa' in algo[0]:
        q, _ = sarsa(algo[1], episodes, 0.7, alpha, 0.1, algo[2])
    elif algo[0] == 'q':
        q, _ = Q(5, episodes, 0.7, alpha, 0.1)
    elif algo[0] == 'tdLambda':
        q, _ = tdLambda(episodes, 0.7, alpha, 0.1, algo[1])
    val = np.amax(q, axis=0)
    print(val, np.shape(val))
    plot(val)
    
############################################################
############# Plot prediction of basic policy ##############
############################################################
# for episodes in [100, 1000, 10000]:
#     for k in [1, 3, 5, 10, 100, 1000]:
#         plotPredictionTD(1, episodes, k, False)
# plotPredictionMC(1, 1000000, False, False) # First visit MC
# plotPredictionMC(1, 1000000, True, False)  # Every visit MC


############################################################
############ Plot Q value functions and policy##############
############################################################
# q, _ = sarsa(10, 1000000, 0.7, 0.01, 0.1, False)
# q, _ = Q(5, 1000000, 0.7, 0.01, 0.1)
# q, _ = tdLambda(1000000, 0.7, 0.01, 0.1, 0.5)
# plotQ(q, 'ql-q-10')
# plotMap(q, 'ql-policy-10')
# plotQ(q)
# plotMap(q)


############################################################
############# Plot rewards for 100 episodes ################
############################################################
# plotRewards([['sarsa1', 1, False], 
#              ['sarsa10', 10, False], 
#              ['sarsa100', 100, False], 
#              ['sarsa1000', 1000, False]], 100, 100)

# plotRewards([['sarsa1', 1, True], 
#              ['sarsa10', 10, True], 
#              ['sarsa100', 100, True], 
#              ['sarsa1000', 1000, True]], 100, 100)

# plotRewards([['sarsa1', 1, False], 
#              ['sarsa1decay', 1, True], 
#              ['sarsa1000', 1000, False], 
#              ['sarsa1000decay', 1000, True], 
#              ['q'], 
#              ['tdLambda', 0.5, True]], 100, 100)


############################################################
############## Plot performance comparison #################
############################################################

# plotPerformanceTest([['sarsa1', 1, False], 
#              ['sarsa10', 10, False], 
#              ['sarsa100', 100, False], 
#              ['sarsa1000', 1000, False],
#              ['sarsa1decay', 1, True], 
#              ['sarsa10decay', 10, True], 
#              ['sarsa100decay', 100, True], 
#              ['sarsa1000decay', 1000, True],
#              ['q'], 
#              ['tdLambda', 0.5, True]], 500000, 100)


############################################################
# Plot performance after training 100k for different alpha #
############################################################
# plotPerformanceAlpha([['sarsa1', 1, False], 
#              ['sarsa10', 10, False], 
#              ['sarsa100', 100, False], 
#              ['sarsa1000', 1000, False]], 100000, 100, [0.1, 0.2, 0.3, 0.4, 0.5])

# plotPerformanceAlpha([['sarsa1', 1, True], 
#              ['sarsa10', 10, True], 
#              ['sarsa100', 100, True], 
#              ['sarsa1000', 1000, True]], 100000, 100, [0.1, 0.2, 0.3, 0.4, 0.5])

# plotPerformanceAlpha([['sarsa1', 1, False], 
#              ['sarsa1decay', 1, True], 
#              ['sarsa1000', 1000, False], 
#              ['sarsa1000decay', 1000, True], 
#              ['q'], 
#              ['tdLambda', 0.5, True]], 100000, 100, [0.1, 0.2, 0.3, 0.4, 0.5])



############################################################
################### Plot value function ####################
############################################################
# plotValueFunction(['tdLambda', 0.5, True], 1000000, 0.1)
# plotValueFunction(['sarsa10', 10, True], 1000000, 0.1)