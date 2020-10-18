from context import pyropt as ro

import numpy as np
import cvxpy as cp

# c = ro.Uncertain(mid=np.array([-2.0, -3.0, -4.0]), width=np.array([1.0, 2.0, 3.0]))
# A = np.array([[3,2,1],
#               [2,5,3]])
# b = np.array([10,15])

# x = cp.Variable(3, boolean=True)

# objective = c @ x
# constraints = [A @ x <= b]

C = ro.Uncertain(lbs=np.array([[67, 18, 58, 87, 48], [33, 47, 26, 37, 81], [50, 56, 3, 40, 48]]), 
                 ubs=np.array([[93, 99, 84, 98, 97], [74, 97, 84, 79, 97], [69, 68, 85, 67, 85]]))

X = cp.Variable((3,5), boolean=True)

objective = cp.sum(cp.multiply(C, X))

constraints  = [ X @ np.ones(5) == np.array([2, 2, 2]) ]
constraints += [ X[1,0] + X[2,4] <= 1 ]
constraints += [ X[1,3] + X[2,0] <= 1 ]
constraints += [ X[1,4] + X[2,3] <= 1 ]
constraints += [ X[0,0] + X[1,1] <= 1 ]
constraints += [ X[0,1] + X[2,3] <= 1 ]

rob = ro.RegretProblem(cp.Minimize(objective), constraints)
rob.solve()
print("\nThe robust optimal value is", rob.value)
print("x=", x.value)
