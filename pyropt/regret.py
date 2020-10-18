import numpy as np
import cvxpy as cp

from pyropt.uncertain import Uncertain

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
# x is the vector of decision variables (discrete or continuous [TODO])
#
# The nominal problem min f(x) s.t. A @ x <= b is MIP or LP.
#
class RegretProblem:

    def __init__(self, objective, constraints):
        
        assert len(objective.variables()) == 1, "Problem must have 1 group of variables"
        assert all([len(c.variables())==1 for c in constraints]), "Problem must have 1 group of variables"
        assert objective.args[0].is_affine(), "Unsupported type of objective function."
        assert all([c.expr.is_affine() for c in constraints]), "Unsupported type of constraints."
        assert objective.variables()[0].attributes["boolean"], "Currently only discrete problems are supported"
        assert Uncertain.is_uncertain(objective.parameters()), "Problem is not uncertain"

        self.objective = objective
        self.constraints = constraints

        self.x = objective.variables()[0]
        self.t = cp.Variable()

        self.N = self.x.size
        self.cu = objective.parameters()[0]
        self.cu.value = self.cu._ubs

        self.master_obj = self.objective.args[0] - self.t
        self.master_constrs = constraints
        self.u_cuts = []

    def solve_nominal(self, c):
        #x = cp.Variable(self.N, nonneg = self.x.is_nonneg())
        self.cu.value = c
        nominal_obj = type(self.objective)(self.objective.args[0])
        nominal_constrs = self.constraints
        nominal_prob = cp.Problem(nominal_obj, nominal_constrs)
        nominal_prob.solve()
        return self.x.value, nominal_prob.value

    def make_u_cut(self, y):
        s = sum([self.cu._lbs[ix] * y[ix] + (self.cu._ubs[ix] - self.cu._lbs[ix]) * y[ix] * self.x[ix]
                 for ix in np.ndindex(*y.shape)])
        cut = self.t <= s
        return cut

    def get_worst_case_scenario(self, x):
        # TODO: this is true only for nominal problem with all 0/1 variables
        s = [self.cu._ubs[ix] if x[ix] > 0 else self.cu._lbs[ix] for ix in np.ndindex(*x.shape)]
        return np.asarray(s).reshape(x.shape)

    def solve(self):
        # Initialization
        y_init, _ = self.solve_nominal(self.cu._ubs)
        self.master_constrs += [self.make_u_cut(y_init)]
        self.u_cuts.append(y_init)

        import pdb; pdb.set_trace()

        max_iters = 500
        iter_count = 1
        LB, UB = -1e9, 1e9
        x_opt = None
        R_opt = None
        epsilon=0.0001
        
        while iter_count <= max_iters:
            print("***Iteration", iter_count, "LB=", LB, "UB=", UB)
            # Solve Master
            self.cu.value = self.cu._ubs
            self.master_problem = cp.Problem(cp.Minimize(self.master_obj), self.master_constrs)
            self.master_problem.solve()
            LB = self.master_problem.value
            x_hat = self.x.value
            wcs = self.get_worst_case_scenario(x_hat)
            
            # Solve Slave
            y, v = self.solve_nominal(wcs)
            self.cu.value = wcs
            self.x.value = x_hat
            R = self.master_obj.args[0].value - v
            
            if LB >= R - epsilon:
                print("***Robust optimal found:", LB)
                #import pdb; pdb.set_trace()
                x_opt = x_hat
                R_opt = R
                break

            UB = min(UB, R)
            self.master_constrs += [self.make_u_cut(y)]
            self.u_cuts.append(y)

            iter_count += 1

        print("Opt.Regret =", R_opt)
        print("  Robust x =", x_opt)

    @property
    def value(self):
        return self.master_problem._value
