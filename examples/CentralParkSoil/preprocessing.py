# %%
# !pip install rpy2

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
file = 'cps_pH_processed.RDS'
rds = ro.r['readRDS'](file)

# %%
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
np.savez('cps.npz', y=y, x=x, label=label_nodes)
np.save('A.npy', A)

# %%



