from context import pyropt as ro

import numpy as np
import cvxpy as cp

c = ro.Uncertain(mid=np.array([-2.0, -3.0, -4.0]), width=np.array([1.0, 2.0, 3.0]))
A = np.array([[3,2,1],
              [2,5,3]])
b = np.array([10,15])

x = cp.Variable(3, boolean=True)

objective = c @ x
constraints = [A @ x <= b]

rob = ro.RegretProblem(cp.Minimize(objective), constraints)
rob.solve()
print("\nThe robust optimal value is", rob.value)
print("x=", x.value)
