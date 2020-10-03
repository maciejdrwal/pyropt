import numpy as np
import cvxpy as cp

from pyropt.uncertain import Uncertain

# Supported problems for RobustLinearProblem:
#
#   min c @ x + d
#    x
# 
#   s.t.
#        A @ x <= b
#
# 1) only 1 vector of real-valued variables
# 2) linear objective function f(x) = c @ x + d, where c,d can be Uncertain or Constant
# 3) any number of linear constraints A @ x <= b, where A,b can be Uncertain or Constant, 
#    A can be either matrix or row vector

class RobustLinearProblem:

    def __init__(self, objective, constraints):
        
        assert len(objective.variables()) == 1, "Problem must have 1 group of variables"
        assert all([len(c.variables())==1 for c in constraints]), "Problem must have 1 group of variables"
        assert objective.args[0].is_affine(), "Unsupported type of objective function."
        assert all([c.expr.is_affine() for c in constraints]), "Unsupported type of constraints."

        self.rc_constraints = []
        self.rc_objective = objective
        self.rc_variables = objective.variables()

        tmp_constraints = constraints.copy()
        
        # If objective function contains uncertain parameters replace it
        # with new variable t, and add constraint f(x) <= t.
        # Otherwise, keep the original objective function.
        
        if RobustLinearProblem.is_uncertain(objective):
            print("transforming objective:", objective)
            
            # t = cp.Variable()
            # self.rc_objective = type(objective)(t)
            # obj_expr = objective.args[0]
            # tmp_constraints += [ obj_expr - t <= 0 ]

            # set the objective to: min/max t
            x = objective.variables()[0]
            t = cp.Variable()
            self.rc_objective = type(objective)(t)

            d = objective.constants()
            if len(d) > 0: d = -d[0].value
            else: d = 0.0

            # add new constraint: c @ x - t <= -d
            cu = objective.parameters()[0]
            tmp_constraints += [ cu @ x - t <= d]

        for constr in tmp_constraints:
            xs = constr.variables()
            x = xs[0]
            t = xs[1] if len(xs) > 1 else None
            N = x.size
            
            if RobustLinearProblem.is_uncertain(constr):
                print("transforming uncertain row:", constr)

                lhs, rhs = constr.args[0], constr.args[1]
                #import pdb; pdb.set_trace()

                is_matrix = lhs.ndim > 0
                for i in range(rhs.size):
                    if RobustLinearProblem.is_uncertain(lhs):
                        params = lhs.parameters()[0]
                        row_a = params._mid[i] if is_matrix else params._mid
                    else:
                        consts = lhs.constants()[0]
                        row_a = consts.value[i] if is_matrix else consts.value 

                    if RobustLinearProblem.is_uncertain(rhs):
                        row_b = rhs._mid[i] if is_matrix else rhs._mid 
                    else:
                        row_b = rhs.value[i] if is_matrix else rhs.value

                    join_constr = row_a @ x 
                    if t is not None: join_constr = row_a @ x - t

                    if RobustLinearProblem.is_uncertain(lhs):
                        params = lhs.parameters()[0]
                        uncertainty = sum(params._width[i]) if is_matrix else sum(params._width)
                        if uncertainty > 0.0:
                            ubs = params._ubs[i] if is_matrix else params._ubs
                            ah = ubs - row_a
                            A0 = row_a * np.eye(N)
                            A_hat = ah * np.eye(N)
                            us = cp.Variable(N)
                            self.rc_constraints += [-us <= A_hat @ x, A_hat @ x <= us]
                            join_constr = join_constr + sum(us)
                                            
                    if RobustLinearProblem.is_uncertain(rhs):
                        uncertainty = rhs._width[i] if is_matrix else rhs._width
                        if uncertainty > 0.0:
                            rhs_params = rhs.parameters()[0]
                            ubs = rhs_params._ubs[i] if is_matrix else rhs_params._ubs
                            bh =  - row_b
                            ub = cp.Variable()                        
                            self.rc_constraints += [-ub <= -bh, -bh <= ub]
                            join_constr = join_constr + ub

                    self.rc_constraints += [join_constr <= row_b]
                    
            else:
                print("skipping certain constraint:", constr)                
                self.rc_constraints += [constr]

        self.rc_problem = cp.Problem(self.rc_objective, self.rc_constraints)

    @staticmethod
    def is_uncertain(expr):
        if isinstance(expr, Uncertain):
            return True        
        if isinstance(expr, list):
            es = expr
        else:
            es = expr.args
        return any([RobustLinearProblem.is_uncertain(child) for child in es])

    def make_certain(self, expr):
        if isinstance(expr, Uncertain):
            expr.value = expr._ubs[0]
            return
        if isinstance(expr, list):
            for child in expr:
                self.make_certain(child)
        else:
            for child in expr.args:
                self.make_certain(child)
    
    @property
    def value(self):
        return self.rc_problem._value

    def solve(self, solver=None):
        self.rc_problem.solve(verbose=True)
