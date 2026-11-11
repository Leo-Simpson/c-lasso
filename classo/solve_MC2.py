"""Joint row-Linf Huberized squared-hinge classification (MC2).

This uses the same event-driven engine as MC1; optional fixed ridge preserves
piecewise-affine segments. rho_classification is the lower margin, as in C2.
"""
from .elastic_net import problem_elastic_net
from .solve_MC1 import Classo_MC1, pathlasso_MC1


class problem_MC2(problem_elastic_net):
    def __init__(self, data, algo="Path-Alg", w=None, intercept=False,
                 covariates=None, ridge=0.0, rho_classification=-1.0):
        if algo != "Path-Alg":
            raise ValueError("MC2 supports only the Path-Alg method")
        super().__init__(data,typ="MC2",ridge=ridge,w=w,intercept=intercept,
                         covariates=covariates,rho_classification=rho_classification)
        self.type, self.name = algo, algo+" MC2"
        self.regpath = False


Classo_MC2 = Classo_MC1
pathlasso_MC2 = pathlasso_MC1
