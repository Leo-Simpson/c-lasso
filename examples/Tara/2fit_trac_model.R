library(Matrix)
library(tidyverse)
library(trac)

dat <- readRDS("tara_sal_processed.RDS")

set.seed(123)
ntot <- length(dat$y)
n <- round(2/3 * ntot)
tr <- sample(ntot, n)
ytr <- dat$y[tr]
yte <- dat$y[-tr]
log_pseudo <- function(x, pseudo_count = 1) log(x + pseudo_count)
ztr <- log_pseudo(dat$x[tr, ])
zte <- log_pseudo(dat$x[-tr, ])

fit <- trac(ztr, ytr, dat$A, w = NULL, min_frac = 1e-3)
cvfit <- cv_trac(fit, Z = ztr, y = ytr, A = dat$A)
save(tr, ytr, yte, ztr, zte, cvfit, fit, file = "tara_sal_trac.Rdata")

