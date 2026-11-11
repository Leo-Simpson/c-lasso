"""Small reproducible fixed-ridge regression and joint multiclass example."""
import numpy as np
from classo import pathlasso

rng = np.random.default_rng(42)
X = rng.normal(size=(60,8))
X[:,-1] = X[:,-2]  # ridge permits exactly collinear penalized features
C = np.ones((1,X.shape[1]))
y = X[:,0]-X[:,1]+.5*rng.normal(size=len(X))
labels = np.digitize(y,[-.5,.5])

for typ,target in [('R1',y),('C1',np.where(y>0,1,-1)),('MC1',labels),('MC2',labels)]:
    B,L = pathlasso((X,C,target),lambdas=[1.,.5,.2,.1],typ=typ,
                   ridge=2.,intercept=True)
    print(typ, B.shape, 'max constraint residual',float(np.max(abs(C@B[-1,1:]))))
