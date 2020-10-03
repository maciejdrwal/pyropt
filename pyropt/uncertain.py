import numpy as np
import cvxpy.expressions.constants.parameter as parameter

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
