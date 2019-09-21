
def basicPolicy(state):
    if state.score() >= 25:
        return "stick"
    return "hit"