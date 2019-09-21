from state import *

class Simulator:
    state = State()

    def reset(self):
        state = State(draw())
        return state
    
    def step(self, action):
        if action == "hit":
            state.action(draw())
        sc = state.score()
        if sc < 0 and state.dealer > 0:
            return state, -1, True
        elif sc < 0 and state.dealer < 0:
            return state, 0, True
        elif sc > 0 and state.dealer < 0:
            return state, 1, True
        return state, 0, action == "stick"
