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

        self.objective = objective
        self.constraints = constraints

        self.x = objective.variables()[0]
        self.t = cp.Variable()

        self.N = self.x.size
        self.cu = objective.parameters()[0]

        self.master_obj = self.cu._ubs @ self.x - self.t
        self.master_constrs = constraints
        self.u_cuts = []

    def solve_nominal(self, c):
        #x = cp.Variable(self.N, nonneg = self.x.is_nonneg())
        nominal_obj = type(self.objective)(c @ self.x)
        nominal_constrs = self.constraints
        nominal_prob = cp.Problem(nominal_obj, nominal_constrs)
        nominal_prob.solve()
        return self.x.value, nominal_prob.value

    def make_u_cut(self, y):
        s = sum([self.cu._lbs[i] * y[i] + (self.cu._ubs[i] - self.cu._lbs[i]) * y[i] * self.x[i]
                 for i in range(self.N)])
        cut = self.t <= s
        return cut

    def get_worst_case_scenario(self, x):
        # TODO: this is true only for nominal problem with all 0/1 variables
        s = [self.cu._ubs[i] if x[i] > 0 else self.cu._lbs[i] for i in range(self.N)]
        return s

    def solve(self):
        # Initialization
        y_init, _ = self.solve_nominal(self.cu._ubs)
        self.master_constrs += [self.make_u_cut(y_init)]
        self.u_cuts.append(y_init)
        self.master_problem = cp.Problem(cp.Minimize(self.master_obj), self.master_constrs)

        max_iters = 500
        iter_count = 1
        LB, UB = -1e9, 1e9
        x_opt = None
        R_opt = None

        while iter_count <= max_iters:
            print("***Iteration", iter_count, "LB=", LB, "UB=", UB)
            # Solve Master
            self.master_problem.solve()
            x_hat = self.x.value
            wcs = self.get_worst_case_scenario(x_hat)
            
            # Solve Slave
            y, v = self.solve_nominal(wcs)
            R = wcs @ (x_hat - y)

            LB = self.cu._ubs @ x_hat - self.t.value
            if LB >= R:
                print("***Robust optimal found:", LB)
                #import pdb; pdb.set_trace()
                x_opt = x_hat
                R_opt = R
                break

            UB = min(UB, R)
            self.u_cuts.append(y)

            iter_count += 1

        print("Opt.Regret =", R_opt)
        print("  Robust x =", x_opt)

    @property
    def value(self):
        return self.master_problem._value
