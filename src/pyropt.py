import numpy as np
import cvxpy as cp
import cvxpy.expressions.constants.parameter as parameter
import cvxpy.expressions.constants.constant as constant
import cvxpy.expressions.leaf as leaf

class Uncertain(parameter.Parameter):
    
    next_id = 0

    def __init__(self, mid=0, width=0, lbs=None, ubs=None):
        self._mid = mid
        self._width = width
        self._shape = (1,)
        self._lbs = []
        self._ubs = []

        if isinstance(mid, np.ndarray):
            self._shape = mid.shape
            if self._shape[0] > 1:
                #assert self._shape == width.shape, "argument 'mid' shape must be the same as 'width' shape"
                self._lbs = self._mid - self._width/2.0
                self._ubs = self._mid + self._width/2.0
        elif isinstance(mid, int) or isinstance(mid, float):
            self._lbs = [self._mid - self._width/2.0]
            self._ubs = [self._mid + self._width/2.0]
        else:
            assert False, "invalid parameters"

        self._name = "uvar" + str(Uncertain.next_id)
        Uncertain.next_id += 1

        super(Uncertain, self).__init__(self._shape, self._name)

    def __repr__(self):
        s = "Uncertain(LB=" + str(self._lbs) + ",UB=" + str(self._ubs) + ")"
        return s

    
class RobustProblem:

    def __init__(self, objective, constraints):
        
        #import pdb; pdb.set_trace()

        assert objective.args[0].is_affine(), "Unsupported type of objective function."
        assert all([c.expr.is_affine() for c in constraints]), "Unsupported type of constraints."

        self.orig_variables = objective.variables()     # original problem's variables
        self.orig_constraints = constraints
        self.rc_variables = objective.variables()       # Robust Counterpart variables
        self.rc_constraints = []
        obj_constr = []

        if any([isinstance(c,Uncertain) for c in objective.args[0].args]):
            t = cp.Variable(name="t")
            if len(objective.args[0].args) > 1:
                cx, d = objective.args[0].args
                obj_constr = [cx - t <= -d]
            else:
                obj_constr = [objective.args[0] - t <= 0]
            self.rc_variables += [t]
            self.rc_objective = type(objective)(t)
            self.rc_constraints += obj_constr
        else:
            self.rc_objective = objective

        #self.make_certain(constraints)
        for constr in constraints + obj_constr:
            if RobustProblem.is_uncertain(constr):
                print("transforming uncertain constraint:", constr)
                lhs, rhs = constr.args[0], constr.args[1]
                xs = constr.variables()
                x = xs[0]
                N = x.shape[0]

                import pdb; pdb.set_trace()

                for i in range(lhs.shape[0]):
                    lhs_params = lhs.parameters()
                    a0 = []
                    if len(lhs_params) > 0:
                        a0 = lhs_params[0]._mid[i]
                        ah = lhs_params[0]._ubs[i] - a0
                    
                    rhs_params = rhs.parameters()
                    b0 = []
                    if len(rhs_params) > 0:
                        b0 = rhs_params[0]._mid[i]
                        bh = rhs_params[0]._ubs[i] - b0
                    
                    us = [cp.Variable() for _ in range(N+1)]
                    
                    assert N == len(a0), "dimensions error"

                    for j in range(N):
                        mask = np.zeros(N)
                        mask[j] = 1
                        a1 = ah * mask
                        self.rc_constraints += [-us[j] <= a1 @ x, a1 @ x <= us[j]]

                    self.rc_constraints += [us[j+1] >= bh, bh >= -us[j+1]]
                    self.rc_constraints += [a0@x + sum(us) <= b0]

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

        return any([RobustProblem.is_uncertain(child) for child in es])

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

    def get_variables(self):
        all_vars = [v.value for v in self.variables]
        return all_vars
    
    @property
    def value(self):
        return self.problem._value

    def solve(self):
        self.rc_problem.solve()
        