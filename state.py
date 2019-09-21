from utils import *

class State:
    def __init__(self, dealer, mySum=0):
        self.oneTwoThree = [False, False, False]
        self.special = 0
        self.sum = mySum
        self.dealer = dealer

    def action(self, card):
        # print("Card = ", card)
        if card <= 3 and card > 0 and not self.oneTwoThree[card-1]:
            self.oneTwoThree[card-1] = True
            self.special += 1
        self.sum += card
    
    def score(self):
        score = self.sum
        for i in range(self.special):
            if score <= 21:
                score += 10
        return score
    
    def runDealer(self):
        while self.dealer < 25 and self.dealer >= 0:
            self.dealer += draw()

    def __str__(self):
        return str(self.special) + ", " + str(self.sum) + ", " + str(self.score())
