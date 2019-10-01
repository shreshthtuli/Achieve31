from state import *

class Simulator:
    state = None

    def runDealer(self):
        dealerScore = self.dealer.score()
        while dealerScore < 25 and dealerScore >= 0:
            self.dealer.action(draw())
            dealerScore = self.dealer.score()
        return dealerScore

    def reset(self):
        dealerCard = draw()
        self.dealer = State(dealerCard, mySum=dealerCard)
        self.state = State(dealerCard)
        self.dealerCard = dealerCard
        return self.state
    
    def set(self, mycard, dealercard):
        self.dealer = State(dealercard, mySum=dealercard)
        self.state = State(dealercard, mySum=mycard)
        self.dealerCard = dealercard
        return self.state
    
    def step(self, action):
        if action == "hit":
            self.state.action(draw())
        elif action == "stick":
            self.state.dealer = self.runDealer()
        else:
            print("Invalid action!", action)
            exit(0)
        sc = self.state.score()
        if (sc < 0 or sc > 31) and self.state.dealer > 0:
            return self.state, -1, True
        elif sc < 0 and self.state.dealer < 0:
            return self.state, 0, True
        elif sc > 0 and self.state.dealer < 0:
            return self.state, 1, True
        elif action == "stick":
            reward = 1 if sc > self.state.dealer else -1 if sc < self.state.dealer else 0
            return self.state, reward, True
        return self.state, 0, False
