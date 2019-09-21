from simulator import *
from policies import *

sim = Simulator()

def run(policy):
    state = sim.reset()
    while True:
        action = policy(state)
        state, reward, done = sim.step(action)
        print(str(state.special), ",", str(state.score()), "::" , str(state.dealer))
        if done: 
            print("Reward = ", reward)
            break

run(basicPolicy)
