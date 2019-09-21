from evaluate import *
from control import *

val = MonteCarlo(basicPolicy, 700000, True)
# val = TD(basicPolicy, 700000, 0.7, 0.01, 10)
plot(val)


# q = sarsa(5, 1000000, 0.7, 0.01, 0.1, True)
# q = Q(5, 1000000, 0.7, 0.01, 0.1)
# plotQ(q)
# plotMap(q)