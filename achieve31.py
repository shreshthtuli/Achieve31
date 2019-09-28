from predict import *
from control import *

# val = MonteCarlo(basicPolicy, 1000000, True)
# val = TD(basicPolicy, 1000000, 0.7, 0.01, 10)
# plot(val)


q = sarsa(5, 100000, 0.7, 0.01, 0.1, True)
# q = Q(5, 1000000, 0.7, 0.01, 0.1)
# q = tdLambda(100000, 0.7, 0.01, 0.1, 0.5)
plotQ(q)
plotMap(q)