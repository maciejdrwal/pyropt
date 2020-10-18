import numpy as np
import cvxpy.expressions.constants.parameter as parameter

INTERVAL = "interval"
ELLIPSOID = "ellipsoid"

class Uncertain(parameter.Parameter):
    
    next_id = 0

    def __init__(self, mid=0, width=0, lbs=None, ubs=None, utype=INTERVAL):
        self._utype = utype
        self._lbs = [] if lbs is None else lbs if isinstance(lbs, np.ndarray) else np.array([lbs])
        self._ubs = [] if ubs is None else ubs if isinstance(ubs, np.ndarray) else np.array([ubs])
        self._mid = mid if isinstance(mid, np.ndarray) else np.array([mid])
        self._width = width if isinstance(width, np.ndarray) else np.array([width])

        if isinstance(self._mid, np.ndarray) and (lbs is None or ubs is None):
            self._lbs = self._mid - self._width / 2.0
            self._ubs = self._mid + self._width / 2.0
        else:
            self._mid = (self._lbs + self._ubs) / 2.0
            self._width = self._ubs - self._lbs

        self._shape = self._mid.shape
        self._name = "uvar" + str(Uncertain.next_id)
        Uncertain.next_id += 1

        super(Uncertain, self).__init__(self._shape, self._name)
    
    @staticmethod
    def is_uncertain(expr):
        if isinstance(expr, Uncertain):
            return True        
        if isinstance(expr, list):
            es = expr
        else:
            es = expr.args
        return any([Uncertain.is_uncertain(child) for child in es])

    def __repr__(self):
        s = "Uncertain(None)"
        if self._utype == INTERVAL:
            s = "Uncertain(LB=" + str(self._lbs) + ",UB=" + str(self._ubs) + ")"
        return s
