# Example 1

from context import pyropt as ro

import numpy as np
import cvxpy as cp

c = np.array([-2.0, -3.0, -4.0])
cu = ro.Uncertain(mid=c, width=np.array([0.1, 0.5, 0.2]))
d = 5.0

Au = ro.Uncertain(mid=np.array([[3,2,1],
                             [2,5,3]]), width=np.ones((2,3)))
Ac = np.array([[-1,0,0],
               [0,-1,0],
               [0,0,-1],
               [1,1,1]])

bu = ro.Uncertain(mid=np.array([10,15]), width=np.array([2.0, 2.0]))
bc = np.array([0,0,0,10])

x = cp.Variable(3)
objective = cu @ x + d
constraints = [Ac @ x <= bc, Au @ x <= np.array([2., 1.])]
rob = ro.RobustLinearProblem(cp.Minimize(objective), constraints)

rob.solve()
# The optimal value for x is stored in `x.value`.
print("\nThe optimal value is", rob.value)
print("x=", x.value)

# data = rob.rc_problem.get_problem_data(cp.SCS)
# import pdb; pdb.set_trace()
