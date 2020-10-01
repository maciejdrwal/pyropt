import numpy as np
import cvxpy as cp
from pyropt import Uncertain, RobustProblem

c = np.array([100.0, 199.9, -5500.0, -6100.0])

A = np.array([[-0.01, -0.02, 0.5, 0.6],
              [1, 1, 0, 0],
              [0, 0, 90, 100],
              [0, 0, 40, 50],
              [100, 199.9, 700, 800]])

b = np.array([0.0, 1000.0, 2000.0, 800.0, 100000.0])

# RawI, RawII, DrugI, DrugII
x = cp.Variable(4, nonneg=True)

objective = c @ x
constraints = [A @ x <= b]
problem = cp.Problem(cp.Minimize(objective), constraints)

#problem.solve(solver=cp.CPLEX)     # solve via LP
problem.solve()                     # solve via convex programming
print("\nThe optimal value is", problem.value)
print("x=", problem.variables()[0].value)

# Expected:
# Opt = −8819.658; 
# RawI = 0, RawII = 438.789, DrugI = 17.552, DrugII = 0

Au = Uncertain(mid = A, width=np.array([[5e-5 * 2.0, 4e-4 * 2.0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]]))
uncertain_constraints = [Au @ x <= b]

rob = RobustProblem(cp.Minimize(objective), uncertain_constraints)

# data = rob.rc_problem.get_problem_data(cp.SCS)
# import pdb; pdb.set_trace()

rob.solve()
print("\nThe robust optimal value is", rob.value)
print("x=", x.value)

# Expected:
# RobOpt = −8294.567; 
# RawI = 877.732, RawII = 0, DrugI = 17.467, DrugII = 0.