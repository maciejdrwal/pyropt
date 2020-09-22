# Example

import numpy as np
import cvxpy as cp
from pyropt import Uncertain, RobustProblem

c = Uncertain(mid=np.array([-2.0, -3.0, -4.0]), width=3)
d = Uncertain(width=0)

Au = Uncertain(mid=np.array([[3,2,1],
                             [2,5,3]]), width=1)
Ac = np.array([[-1,0,0],
               [0,-1,0],
               [0,0,-1]])

bu = Uncertain(mid=np.array([10,15]), width=2)
bc = np.array([0,0,0])

x = cp.Variable(3)
objective = cp.Minimize(c@x + d)
constraints = [Au@x <= bu, Ac@x <= bc]
problem = RobustProblem(objective, constraints)

result = problem.solve()
# The optimal value for x is stored in `x.value`.
print("\nThe optimal value is", problem.value)
print("x=", problem.get_variables())
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(problem.constraints[0].dual_value)
