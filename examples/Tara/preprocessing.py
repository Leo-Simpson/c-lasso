# %%
import numpy as np
# import rpy in order to read data generated on R
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# %%
# this code code be used in order to import R library Matrix
# utils = importr('utils')    
# utils.chooseCRANmirror(ind=1) #
# utils.install_packages('Matrix')
# importr('Matrix')

# %%
#Open R file tara_sal_processed.RDS
file = 'tara_sal_processed.RDS'
rds = ro.r['readRDS'](file)

rA = ro.r["as.matrix"](rds[4])

# %%
x = np.array(rds[1])
y = np.array(rds[0])
A = np.array(rA)

label_OTU = rds[1].colnames
label_sample =rds[1].rownames
label_nodes = np.array(list(rA.colnames))
label_short = np.array([l.split("::")[-1] for l in label_nodes])
print(y.shape)
print(x.shape)
print(A.shape)

# %%
ro.r['load']("tara_sal_trac.RData")


cvfit = ro.r["cvfit"]
cv = cvfit.rx("cv")
lambda_1SE = cv.rx("lambda_1se")

tr = np.array(ro.r['tr']) - 1  # python index starts at 0 when R index starts at 1

# %%
np.save('A.npy', A)

# %%
np.savez('tara.npz', tr=tr, y=y, x=x, label=label_nodes)


