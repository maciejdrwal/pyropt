import numpy as np
import cvxpy as cp
import cvxpy.expressions.constants.parameter as parameter
import cvxpy.expressions.constants.constant as constant
import cvxpy.expressions.leaf as leaf

class Uncertain(parameter.Parameter):
    
    next_id = 0

    def __init__(self, mid=0, width=0, lbs=None, ubs=None):
        self._mid = mid if isinstance(mid, np.ndarray) else np.array([mid])
        self._width = width
        self._shape = self._mid.shape
        self._lbs = []
        self._ubs = []

        if isinstance(self._mid, np.ndarray):
            self._lbs = self._mid - self._width/2.0
            self._ubs = self._mid + self._width/2.0
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

        # If objective function contains uncertain parameters replace it
        # with new variable t, and add constraint f(x) <= t.
        if any([isinstance(c,Uncertain) for c in objective.args[0].args]):
            print("transforming objective:", objective)
            
            import pdb; pdb.set_trace()

            t = cp.Variable(name="t")
            if len(objective.args[0].args) > 1:
                cx, d = objective.args[0].args
                obj_constr = (cx - t <= -d)

                xs = obj_constr.variables()
                x = xs[0]
                N = x.size

                # TODO

            else:
                obj_constr = [objective.args[0] - t <= 0]
                
            self.rc_variables += [t]
            self.rc_objective = type(objective)(t)
        else:
            # Otherwise, leave the original objective function.
            self.rc_objective = objective

        for constr in constraints:
            xs = constr.variables()
            x = xs[0]
            N = x.size

            lhs, rhs = constr.args[0], constr.args[1]

            #import pdb; pdb.set_trace()

            if RobustProblem.is_uncertain(constr):
                for i in range(rhs.size):
                    if RobustProblem.is_uncertain(lhs):
                        row_a = lhs.args[0]._mid[i]
                    else:
                        row_a = lhs.args[0].value[i]

                    if RobustProblem.is_uncertain(rhs):
                        row_b = rhs._mid[i]
                    else:
                        row_b = rhs.value[i]

                    if RobustProblem.is_uncertain(lhs) and sum(lhs.args[0]._width[i]) > 0.0:
                        print("transforming uncertain row:", row_a, row_b)

                        ah = lhs.args[0]._ubs[i] - row_a
                        A0 = row_a * np.eye(N)
                        A_hat = ah * np.eye(N)
                        us = cp.Variable(N)
                        self.rc_constraints += [-us <= A_hat @ x, A_hat @ x <= us]
                        self.rc_variables += [us]
                        join_constr = row_a @ x + sum(us)
                        
                        rhs_params = rhs.parameters()
                        b0 = 0
                        if len(rhs_params) > 0:
                            bh = rhs_params[0]._ubs[i] - b0
                            ub = cp.Variable()                        
                            self.rc_variables += [ub]
                            self.rc_constraints += [-ub <= -bh, -bh <= ub]
                            join_constr = join_constr + ub
                        else:
                            b0 = rhs.value[i]

                        self.rc_constraints += [join_constr <= b0]

                    elif RobustProblem.is_uncertain(rhs) and rhs._width[i] > 0.0:
                        print("TODO!"); exit()
                    else:
                        print("the row has no uncertain variables:", row_a, row_b)
                        self.rc_constraints += [row_a @ x <= row_b]

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
        return self.rc_problem._value

    def solve(self):
        self.rc_problem.solve(solver=cp.CPLEX)
        