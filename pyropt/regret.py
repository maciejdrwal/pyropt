import numpy as np
import cvxpy as cp

# Problem of the form:
#
#    min max [ f(x,s) - min f(y,s) ] 
#     x   s              y
#    
#    s.t.
#         A @ x <= b
#         A @ y <= b
#         s in U
# 
# where:
# f(x,s) = c(s) @ x   linear objective function with uncertain parameters c
# U is the uncertainty set (product of intervals)
# x is the vector of decision variables (continuous or discrete)
#
# The nominal problem min f(x) s.t. A @ x <= b is MIP.
#
class RegretProblem:

    def __init__(self, objective, constraints):
        pass