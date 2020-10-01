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
        
        assert objective.args[0].is_affine(), "Unsupported type of objective function."
        assert all([c.expr.is_affine() for c in constraints]), "Unsupported type of constraints."

        self.rc_constraints = []
        self.rc_objective = objective
        self.rc_variables = objective.variables()

        tmp_constraints = constraints.copy()
        
        # If objective function contains uncertain parameters replace it
        # with new variable t, and add constraint f(x) <= t.
        # Otherwise, keep the original objective function.
        
        if RobustProblem.is_uncertain(objective):
            print("transforming objective:", objective)

            # set the objective to: min/max t
            x = objective.variables()[0]
            #x._shape = (x.size + 1,)
            #c0 = np.zeros(x.size)
            #c0[-1] = 1.0
            #self.rc_objective = type(objective)(c0 @ x)
            t = cp.Variable()
            self.rc_objective = type(objective)(t)

            d = objective.constants()
            if len(d) > 0: d = -d[0].value
            else: d = 0.0

            # add new constraint: c @ x - t <= -d
            cu = objective.parameters()[0]
            #new_mid = np.append(cu._mid, -1.0)
            #new_width = np.append(cu._width, 0.0)
            #new_cu = Uncertain(mid=new_mid, width=new_width)
            #tmp_constraints += [ new_cu @ x <= d]
            tmp_constraints += [ cu @ x - t <= d]

        for constr in tmp_constraints:
            xs = constr.variables()
            x = xs[0]
            t = xs[1] if len(xs) > 1 else None
            N = x.size

            lhs, rhs = constr.args[0], constr.args[1]
            
            if RobustProblem.is_uncertain(constr):
                print("transforming uncertain row:", constr)

                is_matrix = lhs.ndim > 0
                for i in range(rhs.size):
                    if RobustProblem.is_uncertain(lhs):
                        params = lhs.parameters()[0]
                        row_a = params._mid[i] if is_matrix else params._mid
                    else:
                        consts = lhs.constants()[0]
                        row_a = consts.value[i] if is_matrix else consts.value 

                    if RobustProblem.is_uncertain(rhs):
                        row_b = rhs._mid[i] if is_matrix else rhs._mid 
                    else:
                        row_b = rhs.value[i] if is_matrix else rhs.value

                    join_constr = row_a @ x 
                    if t is not None: join_constr = row_a @ x - t

                    if RobustProblem.is_uncertain(lhs):
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
                                            
                    if RobustProblem.is_uncertain(rhs):
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

    def solve(self, solver=None):
        self.rc_problem.solve(solver=solver)
        